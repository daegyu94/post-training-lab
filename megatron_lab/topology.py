"""Validate and explain Megatron parallel rank layouts without GPUs."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass


DEFAULT_ORDER = "tp-cp-ep-dp-pp"


@dataclass(frozen=True)
class Topology:
    nodes: int
    gpus_per_node: int
    tensor_parallel: int
    pipeline_parallel: int
    context_parallel: int
    expert_parallel: int
    data_parallel: int
    world_size: int
    order: str = DEFAULT_ORDER


def build_topology(
    nodes: int,
    gpus_per_node: int,
    tensor_parallel: int,
    pipeline_parallel: int,
    context_parallel: int,
    expert_parallel: int,
) -> Topology:
    values = {
        "nodes": nodes,
        "gpus_per_node": gpus_per_node,
        "tensor_parallel": tensor_parallel,
        "pipeline_parallel": pipeline_parallel,
        "context_parallel": context_parallel,
        "expert_parallel": expert_parallel,
    }
    if any(value < 1 for value in values.values()):
        raise ValueError("All topology sizes must be positive")
    world_size = nodes * gpus_per_node
    model_parallel = (
        tensor_parallel * pipeline_parallel * context_parallel * expert_parallel
    )
    if world_size % model_parallel != 0:
        raise ValueError(
            f"world_size={world_size} is not divisible by TP*PP*CP*EP={model_parallel}"
        )
    return Topology(
        nodes=nodes,
        gpus_per_node=gpus_per_node,
        tensor_parallel=tensor_parallel,
        pipeline_parallel=pipeline_parallel,
        context_parallel=context_parallel,
        expert_parallel=expert_parallel,
        data_parallel=world_size // model_parallel,
        world_size=world_size,
    )


def rank_coordinates(topology: Topology, rank: int) -> dict[str, int]:
    if rank < 0 or rank >= topology.world_size:
        raise ValueError(f"rank must be in [0, {topology.world_size})")
    sizes = {
        "tp": topology.tensor_parallel,
        "cp": topology.context_parallel,
        "ep": topology.expert_parallel,
        "dp": topology.data_parallel,
        "pp": topology.pipeline_parallel,
    }
    remaining = rank
    coordinates: dict[str, int] = {}
    for dimension in topology.order.split("-"):
        coordinates[dimension] = remaining % sizes[dimension]
        remaining //= sizes[dimension]
    coordinates["rank"] = rank
    coordinates["node"] = rank // topology.gpus_per_node
    coordinates["local_rank"] = rank % topology.gpus_per_node
    return coordinates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, default=2)
    parser.add_argument("--gpus-per-node", type=int, default=8)
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--pp", type=int, default=2)
    parser.add_argument("--cp", type=int, default=1)
    parser.add_argument("--ep", type=int, default=1)
    args = parser.parse_args()
    topology = build_topology(args.nodes, args.gpus_per_node, args.tp, args.pp, args.cp, args.ep)
    output = asdict(topology)
    output["ranks"] = [rank_coordinates(topology, rank) for rank in range(topology.world_size)]
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
