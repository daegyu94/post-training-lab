"""Build a fixed, model-tokenized cohort from prepared canonical JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
MEGATRON_ROOT = ROOT / "backends" / "megatron"
if str(MEGATRON_ROOT) not in sys.path:
    sys.path.insert(0, str(MEGATRON_ROOT))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _quantile(values: list[int], probability: float) -> int:
    return values[min(len(values) - 1, math.ceil(probability * len(values)) - 1)]


def _scan(
    path: Path,
    *,
    limit: int,
    max_length: int,
    renderer: Callable[[dict[str, Any]], dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected = []
    lengths = []
    failures = 0
    source_rows = 0
    over = {threshold: 0 for threshold in (1024, 2048, 4096)}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            source_rows += 1
            row = json.loads(line)
            try:
                rendered = renderer(row)
            except ValueError:
                failures += 1
                continue
            length = int(rendered["prompt_tokens"]) + int(rendered["supervised_tokens"])
            lengths.append(length)
            for threshold in over:
                over[threshold] += length > threshold
            if length <= max_length and len(selected) < limit:
                selected.append(row)
    if len(selected) != limit:
        raise ValueError(f"{path} has only {len(selected)} valid rows at max_length={max_length}; need {limit}")
    lengths.sort()
    return selected, {
        "source_rows": source_rows,
        "valid_tokenization": len(lengths),
        "tokenization_failures": failures,
        "minimum": lengths[0],
        "p50": _quantile(lengths, 0.5),
        "p90": _quantile(lengths, 0.9),
        "p95": _quantile(lengths, 0.95),
        "p99": _quantile(lengths, 0.99),
        "p99_5": _quantile(lengths, 0.995),
        "maximum": lengths[-1],
        "over": {str(key): value for key, value in over.items()},
    }


def build_cohort(
    source_dir: Path,
    output_dir: Path,
    *,
    dataset_id: str,
    dataset_revision: str,
    model_revision: str,
    max_length: int,
    train_count: int,
    eval_count: int,
    renderer: Callable[[dict[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse cohort output: {output_dir}")
    source_manifest = json.loads((source_dir / "manifest.json").read_text())
    if source_manifest.get("dataset") != dataset_id or source_manifest.get("dataset_revision") != dataset_revision:
        raise ValueError("source dataset manifest does not match requested dataset and revision")
    train, train_stats = _scan(
        source_dir / "training.jsonl", limit=train_count, max_length=max_length, renderer=renderer
    )
    validation, validation_stats = _scan(
        source_dir / "validation.jsonl", limit=eval_count, max_length=max_length, renderer=renderer
    )
    train_ids = {str(row["prompt_id"]) for row in train}
    eval_ids = {str(row["prompt_id"]) for row in validation}
    if train_ids & eval_ids:
        raise ValueError("selected train and validation prompt IDs overlap")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    # Build under a temp sibling and rename into place so a crash/timeout mid-build
    # never leaves output_dir existing-but-manifest-less (which would permanently
    # block retries via the exists() guard above).
    # ponytail: an interrupt before the final rename leaves this temp dir orphaned on disk;
    # add periodic cleanup of `.*.tmp-*` siblings if orphaned dirs start piling up.
    tmp_dir = Path(tempfile.mkdtemp(dir=output_dir.parent, prefix=f".{output_dir.name}.tmp-"))
    paths = {"training": tmp_dir / "training.jsonl", "validation": tmp_dir / "validation.jsonl"}
    for name, rows in (("training", train), ("validation", validation)):
        paths[name].write_text(
            "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
    manifest = {
        "manifest_version": 1,
        "adapter_version": "checkpoint-memory-30b-v1",
        "preset": "ultrachat",
        "dataset": dataset_id,
        "dataset_revision": dataset_revision,
        "model_revision": model_revision,
        "seed": source_manifest.get("seed"),
        "max_length": max_length,
        "train_count": len(train),
        "eval_count": len(validation),
        "train_prompt_ids": [str(row["prompt_id"]) for row in train],
        "eval_prompt_ids": [str(row["prompt_id"]) for row in validation],
        "length_distribution": {"training": train_stats, "validation": validation_stats},
        "source_manifest_sha256": _sha256(source_dir / "manifest.json"),
        "source_selection_sha256": hashlib.sha256(
            json.dumps(
                {"train": sorted(train_ids), "validation": sorted(eval_ids)},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest(),
        "files": {name: {"path": path.name, "sha256": _sha256(path)} for name, path in paths.items()},
    }
    (tmp_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
    )
    tmp_dir.rename(output_dir)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--dataset-id", default="HuggingFaceH4/ultrachat_200k")
    parser.add_argument("--dataset-revision", default="8049631c405ae6576f93f445c6b8166f76f5505a")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--train-count", type=int, default=32)
    parser.add_argument("--eval-count", type=int, default=8)
    args = parser.parse_args()
    if min(args.max_length, args.train_count, args.eval_count) < 1:
        raise SystemExit("length and cohort counts must be positive")
    from transformers import AutoTokenizer
    from megatron_lab.cluster_data import render_prompt_completion

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    manifest = build_cohort(
        args.source_dir,
        args.output_dir,
        dataset_id=args.dataset_id,
        dataset_revision=args.dataset_revision,
        model_revision=args.model_revision,
        max_length=args.max_length,
        train_count=args.train_count,
        eval_count=args.eval_count,
        renderer=lambda row: render_prompt_completion(row, tokenizer),
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
