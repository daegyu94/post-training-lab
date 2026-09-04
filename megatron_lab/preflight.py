"""Fail early when the single-GPU Qwen2.5-14B run cannot start safely."""

from __future__ import annotations

import argparse
import importlib.metadata
import os
from pathlib import Path


MIN_MEMORY_GIB = 40.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--train-data", type=Path)
    parser.add_argument("--eval-data", type=Path)
    parser.add_argument("--hardware-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    import torch

    args = parse_args()
    errors: list[str] = []
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "not set")
    visible_gpus = torch.cuda.device_count()
    print(
        f"[preflight] CUDA_VISIBLE_DEVICES={visible_devices} "
        f"visible_gpus={visible_gpus}",
        flush=True,
    )
    if visible_gpus != 1:
        errors.append(f"expected exactly one visible GPU, found {visible_gpus}")
    elif not torch.cuda.is_bf16_supported():
        errors.append("the visible GPU does not support BF16")
    else:
        properties = torch.cuda.get_device_properties(0)
        memory_gib = properties.total_memory / 1024**3
        print(
            f"[preflight] gpu={properties.name} total_memory_gib={memory_gib:.2f}",
            flush=True,
        )
        if memory_gib < MIN_MEMORY_GIB:
            errors.append(
                f"Qwen2.5-14B Megatron LoRA requires at least "
                f"{MIN_MEMORY_GIB:.0f}GiB for this lab; found {memory_gib:.2f}GiB"
            )

    if not args.hardware_only:
        try:
            bridge_version = importlib.metadata.version("megatron-bridge")
            print(f"[preflight] megatron_bridge={bridge_version}", flush=True)
        except importlib.metadata.PackageNotFoundError:
            errors.append("megatron-bridge is not installed")
        for label, path in (
            ("model", args.model_dir),
            ("train data", args.train_data),
            ("evaluation data", args.eval_data),
        ):
            if path is None or not path.exists():
                errors.append(f"{label} path is missing: {path}")

    for error in errors:
        print(f"[blocked] {error}", flush=True)
    if errors:
        return 2
    print("[preflight] ready", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
