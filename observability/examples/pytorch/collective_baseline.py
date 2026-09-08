"""Bounded PyTorch/NCCL all-reduce probe; not the nccl-tests benchmark."""

import argparse
import json
import os
import socket
import time
from pathlib import Path

import torch
import torch.distributed as dist


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    results = []
    for size in (4096, 1048576, 16777216, 67108864):
        value = torch.ones(size // 4, device=local_rank)
        dist.all_reduce(value)
        valid = bool(torch.all(value == dist.get_world_size()).item())
        for _ in range(5):
            value.zero_()
            dist.all_reduce(value)
        torch.cuda.synchronize()
        dist.barrier()
        started = time.perf_counter()
        for _ in range(20):
            dist.all_reduce(value)
        torch.cuda.synchronize()
        seconds = torch.tensor((time.perf_counter() - started) / 20, device=local_rank)
        dist.all_reduce(seconds, op=dist.ReduceOp.MAX)
        results.append({"bytes": size, "mean_seconds": seconds.item(),
                        "payload_GB_per_second": size / seconds.item() / 1e9,
                        "correct": valid})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f"rank-{dist.get_rank()}.json").write_text(json.dumps({
        "benchmark": "pytorch-nccl-all-reduce", "hostname": socket.gethostname(),
        "world_size": dist.get_world_size(), "warmup_iterations": 5,
        "measured_iterations": 20, "torch": torch.__version__,
        "nccl": torch.cuda.nccl.version(), "results": results,
    }, indent=2) + "\n")
    dist.destroy_process_group()
    if not all(row["correct"] for row in results):
        raise RuntimeError("all-reduce correctness check failed")


if __name__ == "__main__":
    main()
