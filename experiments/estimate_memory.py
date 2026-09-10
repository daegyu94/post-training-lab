"""Estimate per-rank training memory before spending GPU time on a real run.

Answers "will this configuration fit?" -- e.g. whether a 30B full-parameter SFT
has any chance on this cluster -- from the model's actual tensor sizes plus the
sharding the chosen backend applies. This is an ESTIMATE, not a measurement:
see ESTIMATE_CAVEATS and docs/verification.md's evidence separation.

Parameter counts come from the safetensors index/headers already on disk, not
from architecture-specific formulas, so MoE and dense models are handled the
same way and a new architecture needs no code change. Expert weights are
identified by tensor name so expert-parallel sharding is applied only to them.
"""

from __future__ import annotations

import argparse
import json
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

GIB = 1024**3

# Bytes per trainable parameter held by the optimizer, on top of the parameter
# itself. Mixed-precision Adam keeps an fp32 master copy plus two fp32 moments.
OPTIMIZER_STATE_BYTES = {"adam": 12, "adamw": 12, "sgd": 4, "none": 0}

ESTIMATE_CAVEATS = (
    "activation estimate is coarse (attention/MoE routing internals are not modelled)",
    "CUDA context, allocator fragmentation and framework workspaces are not included",
    "LoRA trainable share is approximated by --lora-trainable-fraction",
    "on unified-memory hardware this competes with host allocations for the same pool",
)


@dataclass(frozen=True)
class ModelWeights:
    """Actual on-disk tensor bytes, split by what expert parallelism can shard."""

    expert_bytes: int
    dense_bytes: int
    tensor_count: int

    @property
    def total_bytes(self) -> int:
        return self.expert_bytes + self.dense_bytes


@dataclass(frozen=True)
class Estimate:
    parameters_bytes: int
    gradients_bytes: int
    optimizer_bytes: int
    activations_bytes: int
    notes: list[str] = field(default_factory=list)

    @property
    def total_bytes(self) -> int:
        return self.parameters_bytes + self.gradients_bytes + self.optimizer_bytes + self.activations_bytes


def read_safetensors_weights(model_dir: Path) -> ModelWeights:
    """Sum real tensor byte ranges from each shard's header; no torch, no safetensors dep."""

    shards = sorted(model_dir.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no .safetensors shards under {model_dir}")
    expert_bytes = dense_bytes = tensor_count = 0
    for shard in shards:
        with shard.open("rb") as handle:
            (header_length,) = struct.unpack("<Q", handle.read(8))
            header = json.loads(handle.read(header_length))
        for name, entry in header.items():
            if name == "__metadata__":
                continue
            start, end = entry["data_offsets"]
            size = end - start
            tensor_count += 1
            if ".experts." in name or ".expert." in name:
                expert_bytes += size
            else:
                dense_bytes += size
    return ModelWeights(expert_bytes=expert_bytes, dense_bytes=dense_bytes, tensor_count=tensor_count)


def _activation_bytes(
    *, micro_batch_size: int, seq_len: int, hidden_size: int, layers: int, recompute: str, dtype_bytes: int
) -> int:
    """Coarse activation estimate: per-layer boundary tensors, times a recompute factor.

    'full' keeps roughly one layer's worth of internals plus per-layer inputs;
    'none' keeps the internals of every layer, which the multiplier stands in for.
    """
    per_layer_boundary = micro_batch_size * seq_len * hidden_size * dtype_bytes
    factors = {"full": 1, "selective": 4, "none": 12}
    if recompute not in factors:
        raise ValueError(f"recompute must be one of {sorted(factors)}")
    return per_layer_boundary * layers * factors[recompute]


def estimate_memory(
    weights: ModelWeights,
    *,
    backend: str,
    finetuning_mode: str,
    optimizer: str,
    world_size: int,
    tensor_parallel: int = 1,
    pipeline_parallel: int = 1,
    expert_parallel: int = 1,
    distributed_backend: str = "ddp",
    zero_stage: int = 0,
    offload: str = "none",
    micro_batch_size: int = 1,
    seq_len: int = 2048,
    hidden_size: int = 0,
    layers: int = 0,
    recompute: str = "full",
    dtype_bytes: int = 2,
    lora_trainable_fraction: float = 0.005,
) -> Estimate:
    """Per-rank resident bytes for the given configuration."""

    if world_size < 1 or micro_batch_size < 1 or seq_len < 1:
        raise ValueError("world_size, micro_batch_size and seq_len must be positive")
    notes: list[str] = []

    if backend == "megatron":
        model_parallel = tensor_parallel * pipeline_parallel
        dense_shard = model_parallel
        expert_shard = model_parallel * expert_parallel
        data_parallel = max(1, world_size // model_parallel)
        # Megatron's distributed optimizer shards optimizer state (and the
        # reduce-scattered main gradients) across the data-parallel group.
        optimizer_shard = data_parallel
        gradient_shard = data_parallel
        grad_bytes_per_param = 4  # fp32 main grads
    elif distributed_backend == "fsdp2":
        dense_shard = expert_shard = optimizer_shard = gradient_shard = world_size
        grad_bytes_per_param = dtype_bytes
    elif distributed_backend == "deepspeed":
        dense_shard = expert_shard = world_size if zero_stage >= 3 else 1
        gradient_shard = world_size if zero_stage >= 2 else 1
        optimizer_shard = world_size if zero_stage >= 1 else 1
        grad_bytes_per_param = dtype_bytes
    else:  # ddp
        dense_shard = expert_shard = optimizer_shard = gradient_shard = 1
        grad_bytes_per_param = dtype_bytes

    parameters_bytes = weights.dense_bytes // dense_shard + weights.expert_bytes // expert_shard

    if finetuning_mode == "lora":
        trainable_bytes = int(parameters_bytes * lora_trainable_fraction)
        notes.append(f"LoRA trainable share approximated at {lora_trainable_fraction:.3%} of resident parameters")
    elif finetuning_mode == "full":
        trainable_bytes = parameters_bytes
    else:
        raise ValueError("finetuning_mode must be lora or full")

    trainable_params = trainable_bytes // dtype_bytes
    if optimizer not in OPTIMIZER_STATE_BYTES:
        raise ValueError(f"optimizer must be one of {sorted(OPTIMIZER_STATE_BYTES)}")

    gradients_bytes = trainable_params * grad_bytes_per_param // gradient_shard
    optimizer_bytes = trainable_params * OPTIMIZER_STATE_BYTES[optimizer] // optimizer_shard

    if offload in {"cpu", "nvme"}:
        notes.append(f"optimizer and gradient state moved off-device by --offload {offload}; shown as 0 on-device")
        offloaded = gradients_bytes + optimizer_bytes
        gradients_bytes = optimizer_bytes = 0
        notes.append(f"{offloaded / GIB:.1f} GiB of state must fit in the {offload} tier instead")

    activations_bytes = (
        _activation_bytes(
            micro_batch_size=micro_batch_size,
            seq_len=seq_len,
            hidden_size=hidden_size,
            layers=layers,
            recompute=recompute,
            dtype_bytes=dtype_bytes,
        )
        if hidden_size and layers
        else 0
    )
    if not activations_bytes:
        notes.append("activations not estimated: pass --hidden-size and --layers")

    return Estimate(
        parameters_bytes=parameters_bytes,
        gradients_bytes=gradients_bytes,
        optimizer_bytes=optimizer_bytes,
        activations_bytes=activations_bytes,
        notes=notes,
    )


def format_report(estimate: Estimate, weights: ModelWeights, budget_gib: float | None) -> str:
    lines = [
        f"model tensors: {weights.tensor_count} ({weights.total_bytes / GIB:.1f} GiB on disk, "
        f"expert {weights.expert_bytes / GIB:.1f} GiB / dense {weights.dense_bytes / GIB:.1f} GiB)",
        "",
        "per-rank estimate:",
        f"  parameters   {estimate.parameters_bytes / GIB:8.1f} GiB",
        f"  gradients    {estimate.gradients_bytes / GIB:8.1f} GiB",
        f"  optimizer    {estimate.optimizer_bytes / GIB:8.1f} GiB",
        f"  activations  {estimate.activations_bytes / GIB:8.1f} GiB",
        f"  {'total':<12} {estimate.total_bytes / GIB:8.1f} GiB",
    ]
    if budget_gib is not None:
        headroom = budget_gib - estimate.total_bytes / GIB
        verdict = "FITS" if headroom > 0 else "DOES NOT FIT"
        lines += ["", f"budget {budget_gib:.1f} GiB -> {verdict} (headroom {headroom:+.1f} GiB)"]
    lines += ["", "estimate only, not a measurement:"]
    lines += [f"  - {caveat}" for caveat in (*estimate.notes, *ESTIMATE_CAVEATS)]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", type=Path, required=True, help="snapshot directory holding *.safetensors")
    parser.add_argument("--backend", choices=("trl", "megatron"), required=True)
    parser.add_argument("--finetuning-mode", choices=("lora", "full"), default="lora")
    parser.add_argument("--optimizer", choices=tuple(OPTIMIZER_STATE_BYTES), default="adam")
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--tp", type=int, default=1, help="megatron tensor parallel size")
    parser.add_argument("--pp", type=int, default=1, help="megatron pipeline parallel size")
    parser.add_argument("--ep", type=int, default=1, help="megatron expert parallel size")
    parser.add_argument("--distributed-backend", choices=("ddp", "fsdp2", "deepspeed"), default="ddp")
    parser.add_argument("--zero-stage", type=int, choices=(0, 1, 2, 3), default=0)
    parser.add_argument("--offload", choices=("none", "cpu", "nvme"), default="none")
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--recompute", choices=("none", "selective", "full"), default="full")
    parser.add_argument("--lora-trainable-fraction", type=float, default=0.005)
    parser.add_argument("--budget-gib", type=float, help="per-rank memory budget to judge against")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    weights = read_safetensors_weights(args.model_dir)
    config: dict[str, Any] = {}
    config_path = args.model_dir / "config.json"
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))

    estimate = estimate_memory(
        weights,
        backend=args.backend,
        finetuning_mode=args.finetuning_mode,
        optimizer=args.optimizer,
        world_size=args.world_size,
        tensor_parallel=args.tp,
        pipeline_parallel=args.pp,
        expert_parallel=args.ep,
        distributed_backend=args.distributed_backend,
        zero_stage=args.zero_stage,
        offload=args.offload,
        micro_batch_size=args.micro_batch_size,
        seq_len=args.seq_len,
        hidden_size=int(config.get("hidden_size", 0)),
        layers=int(config.get("num_hidden_layers", 0)),
        recompute=args.recompute,
        lora_trainable_fraction=args.lora_trainable_fraction,
    )

    if args.json:
        print(json.dumps({
            "model": {
                "tensor_count": weights.tensor_count,
                "expert_bytes": weights.expert_bytes,
                "dense_bytes": weights.dense_bytes,
            },
            "per_rank_bytes": {
                "parameters": estimate.parameters_bytes,
                "gradients": estimate.gradients_bytes,
                "optimizer": estimate.optimizer_bytes,
                "activations": estimate.activations_bytes,
                "total": estimate.total_bytes,
            },
            "budget_gib": args.budget_gib,
            "fits": None if args.budget_gib is None else estimate.total_bytes / GIB < args.budget_gib,
            "estimate_only": True,
            "caveats": [*estimate.notes, *ESTIMATE_CAVEATS],
        }, indent=2))
    else:
        print(format_report(estimate, weights, args.budget_gib))


if __name__ == "__main__":
    main()
