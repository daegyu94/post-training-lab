"""Small DDP workload for validating selected-rank PyTorch profiling."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from examples.pytorch.selected_rank_profiler import selected_rank_profile


def parse_ranks(value: str) -> set[int]:
    try:
        return {int(rank) for rank in value.split(",") if rank}
    except ValueError as error:
        raise argparse.ArgumentTypeError("Use comma-separated integer global ranks.") from error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--profile-ranks", type=parse_ranks, default={0})
    parser.add_argument("--trace-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("This profiling demo requires CUDA.")

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.manual_seed(17 + rank)

    model = nn.Sequential(
        nn.Embedding(4096, args.hidden_size),
        nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=args.hidden_size,
                nhead=8,
                dim_feedforward=args.hidden_size * 4,
                batch_first=True,
            ),
            num_layers=2,
        ),
        nn.Linear(args.hidden_size, 4096),
    ).cuda()
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    with selected_rank_profile(
        args.trace_dir,
        ranks=args.profile_ranks,
        skip_first=4,
        wait=1,
        warmup=1,
        active=2,
    ) as profiler:
        for step in range(args.steps):
            tokens = torch.randint(
                0,
                4096,
                (args.batch_size, args.sequence_length),
                device=local_rank,
            )
            target = torch.randint(0, 4096, tokens.shape, device=local_rank)
            logits = model(tokens)
            loss = nn.functional.cross_entropy(logits.flatten(0, 1), target.flatten())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            profiler.step()

            if rank == 0:
                print(f"step={step} loss={loss.item():.4f}", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
