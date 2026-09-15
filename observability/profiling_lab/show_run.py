"""Print a serverless summary of one training run from its output directory.

No collector, no database: everything here already sits on disk in the
run's output-dir -- Megatron's run-metadata-<stage>.json / TRL's
summary-<stage>.json, and the last application metric snapshot per worker.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _flatten(data: dict[str, Any]) -> list[str]:
    lines = []
    for key, value in data.items():
        if isinstance(value, dict):
            lines.append(f"  {key}: {{...{len(value)} fields}}")
        elif isinstance(value, list):
            lines.append(f"  {key}: [...{len(value)} items]")
        else:
            lines.append(f"  {key}: {value}")
    return lines


def summarize(output_dir: Path) -> str:
    lines = [f"Run: {output_dir}"]

    metadata_files = sorted(
        p for p in output_dir.glob("run-metadata-*.json") if "-rank-" not in p.stem
    ) + sorted(output_dir.glob("summary-*.json"))
    if not metadata_files:
        lines.append("\n(no run-metadata-*.json or summary-*.json found in this directory)")
    for path in metadata_files:
        data = _load(path)
        if data is None:
            continue
        lines.append(f"\n[{path.name}]")
        lines.extend(_flatten(data))

    metrics_dir = output_dir / "observatory-metrics"
    metrics_files = sorted(metrics_dir.glob("*.json")) if metrics_dir.is_dir() else []
    if not metrics_files:
        lines.append("\n(no observatory-metrics -- metrics were not enabled for this run)")
    for path in metrics_files:
        snapshot = _load(path)
        if snapshot is None or snapshot.get("schema_version") != 2:
            continue
        metrics = " ".join(
            f"{sample.get('name')}={sample.get('value')}"
            for sample in snapshot.get("samples", [])
            if isinstance(sample, dict)
        )
        lines.append(
            f"\n[{snapshot.get('producer')}/{snapshot.get('role')} worker "
            f"{snapshot.get('worker_id')}] step {snapshot.get('step')}: {metrics}"
        )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path, help="a run's output-dir (same path the launcher used)")
    args = parser.parse_args()
    if not args.output_dir.is_dir():
        parser.error(f"not a directory: {args.output_dir}")
    print(summarize(args.output_dir))


if __name__ == "__main__":
    main()
