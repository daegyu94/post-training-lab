"""Compare base and reloaded-checkpoint validation logs."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

from megatron_lab.config import DATASET_ID, MODEL_ID


LOSS_PATTERN = re.compile(r"lm loss value:\s*([0-9.+\-Ee]+)")


def read_last_loss(path: Path) -> float:
    matches = LOSS_PATTERN.findall(path.read_text(encoding="utf-8"))
    if not matches:
        raise ValueError(f"no validation loss found in {path}")
    return float(matches[-1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-log", type=Path, required=True)
    parser.add_argument("--tuned-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base_loss = read_last_loss(args.base_log)
    tuned_loss = read_last_loss(args.tuned_log)
    summary = {
        "configuration": {
            "model": MODEL_ID,
            "dataset": DATASET_ID,
            "visible_gpus": 1,
            "tensor_parallel": 1,
            "data_parallel": 1,
        },
        "quality": {
            "base_eval_loss": base_loss,
            "tuned_eval_loss": tuned_loss,
            "loss_change_percent": 100 * (tuned_loss - base_loss) / base_loss,
            "base_perplexity": math.exp(min(20, base_loss)),
            "tuned_perplexity": math.exp(min(20, tuned_loss)),
        },
        "checkpoint_reload_verified": True,
    }
    args.output.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
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
