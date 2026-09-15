"""Write a tiny local calculator dataset in Verl RL parquet format."""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import Dataset


EXAMPLES = (("(7 - 4) * 3", "9"), ("12 + 5 * 6", "42"))
SYSTEM_PROMPT = (
    "Solve the arithmetic question. Use the calculator tool exactly once before answering. "
    "Finish with the exact form `#### <integer>`."
)


def rows() -> list[dict[str, object]]:
    return [
        {
            "data_source": "calculator",
            "agent_name": "tool_agent",
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"What is {expression}?"},
            ],
            "reward_model": {"style": "rule", "ground_truth": answer},
        }
        for expression, answer in EXAMPLES
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    dataset = Dataset.from_list(rows())
    dataset.to_parquet(args.output / "train.parquet")
    dataset.to_parquet(args.output / "val.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
