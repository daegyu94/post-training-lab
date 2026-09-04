"""Simulate tensor- and data-parallel rank layouts without initializing GPUs."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Rank:
    global_rank: int
    tensor_parallel_rank: int
    data_parallel_rank: int


@dataclass(frozen=True)
class Layout:
    world_size: int
    tensor_parallel: int
    data_parallel: int
    ranks: tuple[Rank, ...]


def build_layout(world_size: int, tensor_parallel: int) -> Layout:
    if world_size < 1 or tensor_parallel < 1:
        raise ValueError("world-size and tensor-parallel must be positive")
    if world_size % tensor_parallel:
        raise ValueError("world-size must be divisible by tensor-parallel")
    data_parallel = world_size // tensor_parallel
    ranks = tuple(
        Rank(
            global_rank=rank,
            tensor_parallel_rank=rank % tensor_parallel,
            data_parallel_rank=rank // tensor_parallel,
        )
        for rank in range(world_size)
    )
    return Layout(world_size, tensor_parallel, data_parallel, ranks)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-size", type=int, default=2)
    args = parser.parse_args()
    layouts = {
        "tensor_parallel": asdict(build_layout(args.world_size, args.world_size)),
        "data_parallel": asdict(build_layout(args.world_size, 1)),
    }
    print(json.dumps(layouts, indent=2))


if __name__ == "__main__":
    main()
