"""Small DDP workload for validating selected-rank PyTorch profiling."""

from __future__ import annotations

import argparse
import os
import json
import math
import socket
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from examples.pytorch.selected_rank_profiler import selected_rank_profile
from profiling_lab.measurements import summarize_steps
from run_summary import make_run_summary


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
    parser.add_argument("--warmup-steps", type=int, default=4)
    parser.add_argument("--run-id", default="ddp-demo")
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 <= args.warmup_steps < args.steps:
        raise ValueError("Require 0 <= warmup-steps < steps")
    if args.profile_ranks and args.steps < 8:
        raise ValueError("Capture requires at least 8 steps")
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
    started_at = datetime.now(timezone.utc).isoformat()
    step_seconds = []
    losses = []
    torch.cuda.synchronize()
    dist.barrier()
    torch.cuda.reset_peak_memory_stats()

    with selected_rank_profile(
        args.trace_dir,
        ranks=args.profile_ranks,
        skip_first=4,
        wait=1,
        warmup=1,
        active=2,
    ) as profiler:
        for step in range(args.steps):
            started = time.perf_counter()
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
            torch.cuda.synchronize()
            step_seconds.append(time.perf_counter() - started)
            losses.append(loss.item())
            profiler.step()

            if rank == 0:
                print(f"step={step} loss={loss.item():.4f}", flush=True)

    local_steps = torch.tensor(step_seconds, device=local_rank, dtype=torch.float64)
    dist.all_reduce(local_steps, op=dist.ReduceOp.MAX)
    slowest_steps = local_steps.cpu().tolist()
    output_dir = args.output_dir or args.trace_dir.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = make_run_summary(
        configuration={
            "run_id": args.run_id, "rank": rank, "world_size": dist.get_world_size(),
            "steps": args.steps, "warmup_steps": args.warmup_steps,
            "batch_size": args.batch_size, "sequence_length": args.sequence_length,
            "hidden_size": args.hidden_size, "seed": 17 + rank,
            "profile_ranks": sorted(args.profile_ranks), "synthetic_data": True,
        },
        environment={
            "hostname": socket.gethostname(), "torch": torch.__version__,
            "cuda": torch.version.cuda, "nccl": torch.cuda.nccl.version(),
            "gpu": torch.cuda.get_device_name(), "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
        },
        quality={"losses": losses, "model_quality_evaluated": False},
        performance={
            "local": summarize_steps(step_seconds, args.warmup_steps),
            "slowest_rank": summarize_steps(slowest_steps, args.warmup_steps),
            "global_synthetic_tokens_per_second": (
                args.batch_size * args.sequence_length * dist.get_world_size()
                / summarize_steps(slowest_steps, args.warmup_steps)["mean_seconds"]
            ),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "local_step_seconds": step_seconds,
            "slowest_rank_step_seconds": slowest_steps,
        },
        artifacts={"trace_dir": str(args.trace_dir)},
        validation={"finite_losses": all(math.isfinite(v) for v in losses)},
    )
    (output_dir / f"rank-{rank}.json").write_text(json.dumps(summary, indent=2) + "\n")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
