"""Materialize small, disjoint UltraChat train and evaluation subsets."""

from __future__ import annotations

import argparse
import json
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


def main() -> None:
    args = parse_args()
    print(
        f"[data] dataset={DATASET_NAME} train_split={TRAIN_SPLIT} "
        f"eval_split={EVAL_SPLIT}",
        flush=True,
    )
    train_rows = select_conversations(
        load_split(TRAIN_SPLIT, args.seed, args.train_samples), args.train_samples
    )
    eval_rows = select_conversations(
        load_split(EVAL_SPLIT, args.seed, args.eval_samples), args.eval_samples
    )
    overlap = {row["prompt_id"] for row in train_rows}.intersection(
        row["prompt_id"] for row in eval_rows
    )
    if overlap:
        raise RuntimeError("train and evaluation subsets overlap")

    training_path = args.output_dir / "training.jsonl"
    validation_path = args.output_dir / "validation.jsonl"
    write_jsonl(training_path, train_rows)
    write_jsonl(validation_path, eval_rows)
    print(f"[data] saved train={len(train_rows)} path={training_path}", flush=True)
    print(
        f"[data] saved evaluation={len(eval_rows)} path={validation_path}",
        flush=True,
    )
    print("[data] overlap=0", flush=True)


if __name__ == "__main__":
    main()
