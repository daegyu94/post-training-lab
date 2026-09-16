"""Convert the shared math smoke data to Verl's parquet schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def rows(source: Path) -> list[dict[str, object]]:
    items = [
        json.loads(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line
    ]
    return [
        {
            "data_source": "math-smoke",
            "prompt": [{"role": "user", "content": item["input"]}],
            "reward_model": {"style": "rule", "ground_truth": item["output"]},
        }
        for item in items
    ]


def main(argv: list[str] | None = None) -> int:
    from datasets import Dataset

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    dataset = Dataset.from_list(rows(args.source))
    dataset.to_parquet(args.output / "train.parquet")
    dataset.to_parquet(args.output / "val.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
