"""Validate the explicit topology used by the two-node Spark launcher."""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ClusterTopology:
    """The process and model-parallel dimensions for one torchrun job."""

    nnodes: int = 2
    node_rank: int = 0
    nproc_per_node: int = 1
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    expert_parallel_size: int = 2
    expert_tensor_parallel_size: int = 1
    context_parallel_size: int = 1
    micro_batch_size: int = 1
    global_batch_size: int = 8

    @property
    def world_size(self) -> int:
        return self.nnodes * self.nproc_per_node

    @property
    def model_parallel_size(self) -> int:
        return self.tensor_parallel_size * self.pipeline_parallel_size * self.context_parallel_size

    @property
    def data_parallel_size(self) -> int:
        return self.world_size // self.model_parallel_size

    @property
    def expert_data_parallel_size(self) -> int:
        return self.world_size // (
            self.expert_tensor_parallel_size
            * self.expert_parallel_size
            * self.pipeline_parallel_size
        )

    @property
    def gradient_accumulation_steps(self) -> int:
        return self.global_batch_size // (
            self.micro_batch_size * self.data_parallel_size
        )

    def validate(self) -> "ClusterTopology":
        for name in (
            "nnodes",
            "nproc_per_node",
            "tensor_parallel_size",
            "pipeline_parallel_size",
            "expert_parallel_size",
            "expert_tensor_parallel_size",
            "context_parallel_size",
            "micro_batch_size",
            "global_batch_size",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1")
        if self.node_rank < 0 or self.node_rank >= self.nnodes:
            raise ValueError("node_rank must be in [0, nnodes)")
        if self.world_size % self.model_parallel_size:
            raise ValueError(
                "world_size must be divisible by tensor_parallel_size * "
                "pipeline_parallel_size * context_parallel_size"
            )
        expert_model_parallel_size = (
            self.expert_tensor_parallel_size
            * self.expert_parallel_size
            * self.pipeline_parallel_size
        )
        if self.world_size % expert_model_parallel_size:
            raise ValueError(
                "world_size must be divisible by expert_tensor_parallel_size * "
                "expert_parallel_size * pipeline_parallel_size"
            )
        if self.global_batch_size % (
            self.micro_batch_size * self.data_parallel_size
        ):
            raise ValueError(
                "global_batch_size must be divisible by "
                "micro_batch_size * data_parallel_size"
            )
        return self

    def as_dict(self) -> dict[str, int]:
        self.validate()
        values = asdict(self)
        values.update(
            {
                "world_size": self.world_size,
                "model_parallel_size": self.model_parallel_size,
                "data_parallel_size": self.data_parallel_size,
                "expert_data_parallel_size": self.expert_data_parallel_size,
                "gradient_accumulation_steps": self.gradient_accumulation_steps,
            }
        )
        return values


def topology_from_environment(
    *,
    tp: int,
    pp: int,
    ep: int,
    micro_batch_size: int,
    global_batch_size: int,
) -> ClusterTopology:
    """Read the launcher contract without initializing torch.distributed."""

    def integer(name: str, default: str | None = None) -> int:
        value = os.environ.get(name, default)
        if value is None:
            raise ValueError(f"{name} is required")
        try:
            return int(value)
        except ValueError as exc:
            raise ValueError(f"{name} must be an integer: {value!r}") from exc

    topology = ClusterTopology(
        nnodes=integer("NNODES", "2"),
        node_rank=integer("NODE_RANK", "0"),
        nproc_per_node=integer("NPROC_PER_NODE", "1"),
        tensor_parallel_size=tp,
        pipeline_parallel_size=pp,
        expert_parallel_size=ep,
        expert_tensor_parallel_size=1,
        micro_batch_size=micro_batch_size,
        global_batch_size=global_batch_size,
    )
    return topology.validate()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nnodes", type=int, default=int(os.environ.get("NNODES", 2)))
    parser.add_argument("--node-rank", type=int, default=int(os.environ.get("NODE_RANK", 0)))
    parser.add_argument("--nproc-per-node", type=int, default=int(os.environ.get("NPROC_PER_NODE", 1)))
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--ep", type=int, default=2)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--global-batch-size", type=int, default=8)
    args = parser.parse_args()
    topology = ClusterTopology(
        nnodes=args.nnodes,
        node_rank=args.node_rank,
        nproc_per_node=args.nproc_per_node,
        tensor_parallel_size=args.tp,
        pipeline_parallel_size=args.pp,
        expert_parallel_size=args.ep,
        micro_batch_size=args.micro_batch_size,
        global_batch_size=args.global_batch_size,
    ).validate()
    import json

    print(json.dumps(topology.as_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
