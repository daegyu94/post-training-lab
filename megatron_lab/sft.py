"""Run one stage of the Qwen2.5-7B Megatron Bridge SFT workflow."""

from __future__ import annotations

import argparse
from pathlib import Path

from megatron_lab.config import DATASET_ID, MODEL_ID, build_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("base", "train", "tuned"), required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--eval-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--eval-iters", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--global-batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    from megatron.bridge.training.finetune import finetune
    from megatron.bridge.training.gpt_step import forward_step

    args = parse_args()
    print(
        f"[stage] name={args.stage} model={MODEL_ID} dataset={DATASET_ID}",
        flush=True,
    )
    print(
        f"[stage] world_size=1 tensor_parallel=1 data_parallel=1 "
        f"max_length={args.max_length}",
        flush=True,
    )
    config = build_config(args)
    finetune(config=config, forward_step_func=forward_step)


if __name__ == "__main__":
    main()
