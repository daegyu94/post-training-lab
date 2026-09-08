"""Prepare a bounded, pinned public dataset as canonical conversational JSONL."""

from __future__ import annotations

import argparse
from pathlib import Path

from sft_lab.public_data import get_preset, iter_hub_rows, resolve_revision, select_bounded, write_public_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=("ultrachat", "no_robots", "self_oss", "xlam"), required=True)
    parser.add_argument("--revision")
    parser.add_argument("--train-count", type=int, default=32)
    parser.add_argument("--eval-count", type=int, default=8)
    parser.add_argument("--max-scan", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    spec = get_preset(args.preset)
    revision = resolve_revision(spec, args.revision)
    if spec.gated:
        print(f"[data] {spec.dataset_id} is gated; access terms must be accepted outside this script", flush=True)
    train, evaluation = select_bounded(iter_hub_rows(spec, revision), spec, args.train_count, args.eval_count, args.seed, args.max_scan)
    manifest = write_public_jsonl(args.output_dir, spec, revision, train, evaluation, args.seed, args.max_scan)
    print(f"[data] preset={spec.key} revision={revision} train={manifest['train_count']} validation={manifest['eval_count']} overlap=0", flush=True)


if __name__ == "__main__":
    main()
