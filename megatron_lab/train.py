"""Run a compact GPT training loop with Megatron Core local layers."""

from __future__ import annotations

import argparse
import atexit
import json
import os
import platform
import socket
import time
from pathlib import Path
from typing import Any

import torch
from megatron.core import __version__ as megatron_core_version
from megatron.core import parallel_state
from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig

from megatron_lab.topology import build_topology


def cleanup_distributed() -> None:
    if parallel_state.model_parallel_is_initialized():
        parallel_state.destroy_model_parallel()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--micro-batch-size", type=int, default=4)
    parser.add_argument("--seq-length", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-attention-heads", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output-dir", type=Path, default=Path("results/megatron"))
    return parser.parse_args()


def initialize_distributed(tp: int, seed: int) -> tuple[int, int, int]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Megatron Core training exercise")
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    build_topology(1, world_size, tp, 1, 1, 1)
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        device_id=torch.device("cuda", local_rank),
    )
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=tp,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        create_gloo_process_groups=False,
    )
    model_parallel_cuda_manual_seed(seed)
    atexit.register(cleanup_distributed)
    return rank, local_rank, world_size


def build_model(args: argparse.Namespace) -> GPTModel:
    if args.hidden_size % args.num_attention_heads != 0:
        raise ValueError("hidden-size must be divisible by num-attention-heads")
    if args.num_attention_heads % args.tp != 0:
        raise ValueError("num-attention-heads must be divisible by tp")
    if args.vocab_size % args.tp != 0:
        raise ValueError("vocab-size must be divisible by tp")
    config = TransformerConfig(
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_attention_heads,
        ffn_hidden_size=4 * args.hidden_size,
        tensor_model_parallel_size=args.tp,
        pipeline_model_parallel_size=1,
        bf16=True,
        params_dtype=torch.bfloat16,
        transformer_impl="local",
        hidden_dropout=0.0,
        attention_dropout=0.0,
        masked_softmax_fusion=False,
        bias_activation_fusion=False,
        bias_dropout_fusion=False,
    )
    model = GPTModel(
        config=config,
        transformer_layer_spec=get_gpt_layer_local_spec(normalization="LayerNorm"),
        vocab_size=args.vocab_size,
        max_sequence_length=args.seq_length,
        position_embedding_type="rope",
        share_embeddings_and_output_weights=True,
        parallel_output=True,
    )
    return model.to(device="cuda", dtype=torch.bfloat16)


def mock_batch(args: argparse.Namespace, step: int, dp_rank: int) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(args.seed + 10_000 * dp_rank + step)
    tokens = torch.randint(
        0,
        args.vocab_size,
        (args.micro_batch_size, args.seq_length),
        generator=generator,
        device="cuda",
    )
    labels = (tokens + 1) % args.vocab_size
    positions = torch.arange(args.seq_length, device="cuda").expand_as(tokens)
    attention_mask = torch.triu(
        torch.ones((1, 1, args.seq_length, args.seq_length), dtype=torch.bool, device="cuda"),
        diagonal=1,
    )
    return tokens, positions, attention_mask, labels


def average_data_parallel_gradients(model: torch.nn.Module) -> None:
    group = parallel_state.get_data_parallel_group()
    size = parallel_state.get_data_parallel_world_size()
    if size == 1:
        return
    for parameter in model.parameters():
        if parameter.grad is not None:
            torch.distributed.all_reduce(parameter.grad, group=group)
            parameter.grad.div_(size)


def reduced_loss(loss: torch.Tensor) -> float:
    value = loss.detach().clone()
    group = parallel_state.get_data_parallel_group()
    torch.distributed.all_reduce(value, group=group)
    value /= parallel_state.get_data_parallel_world_size()
    return float(value)


def main() -> None:
    args = parse_args()
    if args.steps < 1:
        raise ValueError("steps must be positive")
    rank, local_rank, world_size = initialize_distributed(args.tp, args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    model = build_model(args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    dp_rank = parallel_state.get_data_parallel_rank()
    losses: list[float] = []
    torch.cuda.reset_peak_memory_stats()
    torch.distributed.barrier()
    started = time.perf_counter()

    for step in range(args.steps):
        tokens, positions, attention_mask, labels = mock_batch(args, step, dp_rank)
        optimizer.zero_grad(set_to_none=True)
        token_losses = model(tokens, positions, attention_mask, labels=labels)
        loss = token_losses.float().mean()
        loss.backward()
        average_data_parallel_gradients(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(reduced_loss(loss))
        if rank == 0:
            print(f"step={step + 1} loss={losses[-1]:.6f}", flush=True)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    local_stats: dict[str, Any] = {
        "rank": rank,
        "local_rank": local_rank,
        "host": socket.gethostname(),
        "peak_allocated_memory_gib": torch.cuda.max_memory_allocated() / 1024**3,
    }
    rank_stats: list[dict[str, Any] | None] = [None] * world_size
    torch.distributed.all_gather_object(rank_stats, local_stats)
    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "configuration": vars(args) | {"output_dir": str(args.output_dir)},
            "topology": {
                "world_size": world_size,
                "tensor_parallel": args.tp,
                "data_parallel": parallel_state.get_data_parallel_world_size(),
                "pipeline_parallel": 1,
            },
            "metrics": {
                "first_loss": losses[0],
                "last_loss": losses[-1],
                "loss_change_percent": 100 * (losses[-1] - losses[0]) / losses[0],
                "elapsed_seconds": elapsed,
                "steps_per_second": args.steps / elapsed,
            },
            "ranks": rank_stats,
            "environment": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "megatron_core": megatron_core_version,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0),
            },
        }
        (args.output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(summary, indent=2), flush=True)

    cleanup_distributed()


if __name__ == "__main__":
    main()
