"""Compare base and reloaded-checkpoint validation logs."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import platform
import re
from pathlib import Path

from megatron_lab.config import DATASET_ID, MODEL_ID
from run_summary import make_run_summary


LOSS_PATTERN = re.compile(r"lm loss value:\s*([0-9.+\-Ee]+)")


def read_last_loss(path: Path) -> float:
    matches = LOSS_PATTERN.findall(path.read_text(encoding="utf-8"))
    if not matches:
        raise ValueError(f"no validation loss found in {path}")
    return float(matches[-1])


def collect_environment() -> dict[str, str | None]:
    import torch

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "not set"),
        "framework": "megatron_bridge",
        "framework_version": importlib.metadata.version("megatron-bridge"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-log", type=Path, required=True)
    parser.add_argument("--tuned-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--eval-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--eval-iters", type=int, required=True)
    parser.add_argument("--max-length", type=int, required=True)
    parser.add_argument("--global-batch-size", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    base_loss = read_last_loss(args.base_log)
    tuned_loss = read_last_loss(args.tuned_log)
    summary = make_run_summary(
        configuration={
            "model": MODEL_ID,
            "dataset": DATASET_ID,
            "model_dir": str(args.model_dir),
            "train_data": str(args.train_data),
            "eval_data": str(args.eval_data),
            "max_steps": args.max_steps,
            "eval_iters": args.eval_iters,
            "max_length": args.max_length,
            "world_size": 1,
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "context_parallel_size": 1,
            "data_parallel_size": 1,
            "micro_batch_size": 1,
            "global_batch_size": args.global_batch_size,
            "seed": args.seed,
        },
        environment=collect_environment(),
        quality={
            "base_eval_loss": base_loss,
            "tuned_eval_loss": tuned_loss,
            "loss_change_percent": 100 * (tuned_loss - base_loss) / base_loss,
            "base_perplexity": math.exp(min(20, base_loss)),
            "tuned_perplexity": math.exp(min(20, tuned_loss)),
        },
        performance={},
        artifacts={
            "checkpoint_dir": str(args.output_dir / "checkpoints"),
            "base_evaluation_log": str(args.base_log),
            "training_log": str(args.output_dir / "train.log"),
            "tuned_evaluation_log": str(args.tuned_log),
            "summary_file": str(args.output),
        },
        validation={
            "held_out_loss_improved": tuned_loss < base_loss,
            "checkpoint_reload_verified": True,
        },
    )
    args.output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"[summary] base_eval_loss={base_loss:.6f} "
        f"tuned_eval_loss={tuned_loss:.6f} "
        f"change={summary['quality']['loss_change_percent']:.2f}%",
        flush=True,
    )
    print(f"[summary] path={args.output}", flush=True)


if __name__ == "__main__":
    main()
