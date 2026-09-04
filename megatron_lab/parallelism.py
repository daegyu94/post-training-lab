"""멀티 GPU 없이 Megatron 병렬 그룹의 논리적 배치를 살펴본다.

이 모듈은 torch.distributed, NCCL, CUDA를 초기화하지 않는다.
TP/PP/CP/DP 그룹은 학습에 사용되는 실제 process group이 아니라 설명용 계산 결과다.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Rank:
    global_rank: int
    tensor_parallel_rank: int
    pipeline_parallel_rank: int
    context_parallel_rank: int
    data_parallel_rank: int


def build_layout(
    world_size: int,
    tensor_parallel_size: int,
    pipeline_parallel_size: int,
    context_parallel_size: int,
) -> dict[str, object]:
    """TP/PP/CP/DP의 논리적 rank 배치를 계산한다."""
    for name, size in (
        ("world_size", world_size),
        ("tensor_parallel_size", tensor_parallel_size),
        ("pipeline_parallel_size", pipeline_parallel_size),
        ("context_parallel_size", context_parallel_size),
    ):
        if size < 1:
            raise ValueError(f"{name} must be at least 1")

    model_parallel_size = (
        tensor_parallel_size * pipeline_parallel_size * context_parallel_size
    )
    if world_size % model_parallel_size:
        raise ValueError(
            "world_size must be divisible by "
            "tensor_parallel_size * pipeline_parallel_size * context_parallel_size"
        )

    data_parallel_size = world_size // model_parallel_size
    ranks: list[Rank] = []
    for pp_rank in range(pipeline_parallel_size):
        for dp_rank in range(data_parallel_size):
            for cp_rank in range(context_parallel_size):
                for tp_rank in range(tensor_parallel_size):
                    global_rank = (
                        (((pp_rank * data_parallel_size) + dp_rank) * context_parallel_size)
                        + cp_rank
                    ) * tensor_parallel_size + tp_rank
                    ranks.append(
                        Rank(
                            global_rank=global_rank,
                            tensor_parallel_rank=tp_rank,
                            pipeline_parallel_rank=pp_rank,
                            context_parallel_rank=cp_rank,
                            data_parallel_rank=dp_rank,
                        )
                    )

    rank_index = {rank.global_rank: rank for rank in ranks}

    def group_for(rank: Rank, dimension: str) -> list[int]:
        matches: list[int] = []
        for candidate in ranks:
            same_except_dimension = (
                (dimension == "tp" or candidate.tensor_parallel_rank == rank.tensor_parallel_rank)
                and (dimension == "pp" or candidate.pipeline_parallel_rank == rank.pipeline_parallel_rank)
                and (dimension == "cp" or candidate.context_parallel_rank == rank.context_parallel_rank)
                and (dimension == "dp" or candidate.data_parallel_rank == rank.data_parallel_rank)
            )
            if same_except_dimension:
                matches.append(candidate.global_rank)
        return matches

    groups: dict[str, list[list[int]]] = {}
    for dimension, label in (
        ("tp", "tensor_parallel"),
        ("pp", "pipeline_parallel"),
        ("cp", "context_parallel"),
        ("dp", "data_parallel"),
    ):
        unique_groups: list[list[int]] = []
        for rank in ranks:
            group = group_for(rank, dimension)
            if group not in unique_groups:
                unique_groups.append(group)
        groups[label] = unique_groups

    return {
        "mode": "conceptual simulation only: no CUDA, NCCL, or torch.distributed initialization",
        "world_size": world_size,
        "tensor_parallel_size": tensor_parallel_size,
        "pipeline_parallel_size": pipeline_parallel_size,
        "context_parallel_size": context_parallel_size,
        "data_parallel_size": data_parallel_size,
        "ranks": [asdict(rank_index[index]) for index in range(world_size)],
        "groups": groups,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--pipeline-parallel-size", type=int, default=2)
    parser.add_argument("--context-parallel-size", type=int, default=1)
    args = parser.parse_args()

    layout = build_layout(
        world_size=args.world_size,
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        context_parallel_size=args.context_parallel_size,
    )
    print(json.dumps(layout, indent=2))


if __name__ == "__main__":
    main()
