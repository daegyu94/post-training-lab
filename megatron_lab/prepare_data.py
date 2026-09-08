"""Materialize small, disjoint UltraChat train and evaluation subsets."""

from __future__ import annotations

import argparse
import json
from itertools import islice
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from megatron_lab.data import (
    DATASET_NAME,
    EVAL_SPLIT,
    TRAIN_SPLIT,
    select_conversations,
    write_jsonl,
)


DEFAULT_OUTPUT_DIR = Path("data/ultrachat_200k")
DATASET_VIEWER_URL = "https://datasets-server.huggingface.co/rows"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-samples", type=int, default=32)
    parser.add_argument("--eval-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--dataset-revision",
        help="Pinned Hugging Face dataset commit; required by the Spark-cluster script",
    )
    return parser.parse_args()


def fetch_rows(split: str, offset: int, length: int) -> list[dict]:
    query = urlencode(
        {
            "dataset": DATASET_NAME,
            "config": "default",
            "split": split,
            "offset": offset,
            "length": length,
        }
    )
    request = Request(
        f"{DATASET_VIEWER_URL}?{query}",
        headers={"User-Agent": "megatron-sft-lab/1.0"},
    )
    with urlopen(request, timeout=60) as response:
        payload = json.load(response)
    return payload["rows"]


def load_split(split: str, seed: int, count: int) -> list[dict]:
    rows: list[dict] = []
    offset = seed
    while len(rows) < count:
        page = fetch_rows(split, offset, count - len(rows))
        if not page:
            raise ValueError(f"{split} ended after {len(rows)} selected rows")
        rows.extend(item["row"] for item in page)
        offset = int(page[-1]["row_idx"]) + 1
    return rows


def load_pinned_split(split: str, revision: str, seed: int, count: int) -> list[dict]:
    """Stream a pinned Hub revision without using the mutable dataset viewer."""

    from datasets import load_dataset

    dataset = load_dataset(
        DATASET_NAME,
        "default",
        split=split,
        revision=revision,
        streaming=True,
    )
    rows: list[dict] = []
    for row in islice(dataset.skip(seed), count * 4):
        rows.append(row)
        if len(rows) >= count:
            break
    if len(rows) < count:
        raise ValueError(f"{split} ended after {len(rows)} selected rows")
    return rows


def main() -> None:
    args = parse_args()
    print(
        f"[data] dataset={DATASET_NAME} train_split={TRAIN_SPLIT} "
        f"eval_split={EVAL_SPLIT}",
        flush=True,
    )
    if args.dataset_revision:
        train_source = load_pinned_split(
            TRAIN_SPLIT, args.dataset_revision, args.seed, args.train_samples
        )
        eval_source = load_pinned_split(
            EVAL_SPLIT, args.dataset_revision, args.seed, args.eval_samples
        )
    else:
        train_source = load_split(TRAIN_SPLIT, args.seed, args.train_samples)
        eval_source = load_split(EVAL_SPLIT, args.seed, args.eval_samples)
    train_rows = select_conversations(train_source, args.train_samples)
    eval_rows = select_conversations(eval_source, args.eval_samples)
    overlap = {row["prompt_id"] for row in train_rows}.intersection(
        row["prompt_id"] for row in eval_rows
    )
    if overlap:
        raise RuntimeError("train and evaluation subsets overlap")

    training_path = args.output_dir / "training.jsonl"
    validation_path = args.output_dir / "validation.jsonl"
    write_jsonl(training_path, train_rows)
    write_jsonl(validation_path, eval_rows)
    manifest = {
        "dataset": DATASET_NAME,
        "dataset_revision": args.dataset_revision or "viewer-default-unpinned",
        "seed": args.seed,
        "train_split": TRAIN_SPLIT,
        "eval_split": EVAL_SPLIT,
        "train_count": len(train_rows),
        "eval_count": len(eval_rows),
        "train_prompt_ids": [row["prompt_id"] for row in train_rows],
        "eval_prompt_ids": [row["prompt_id"] for row in eval_rows],
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"[data] saved train={len(train_rows)} path={training_path}", flush=True)
    print(
        f"[data] saved evaluation={len(eval_rows)} path={validation_path}",
        flush=True,
    )
    print(f"[data] revision={manifest['dataset_revision']}", flush=True)
    print("[data] overlap=0", flush=True)


if __name__ == "__main__":
    main()
