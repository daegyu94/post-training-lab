"""Republish application metric snapshots through Node Exporter textfiles."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from observatory_metrics.prometheus import GaugeSample, write_gauges


def _iter_snapshots(metrics_dir: Path) -> list[dict]:
    snapshots = []
    for path in sorted(metrics_dir.glob("*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if value.get("schema_version") == 2:
                snapshots.append(value)
        except (AttributeError, OSError, json.JSONDecodeError):
            continue
    return snapshots


def build_metrics(snapshots: list[dict]) -> list[GaugeSample]:
    metrics = []
    for snapshot in snapshots:
        labels = {
            "run_id": str(snapshot.get("run_id", "")),
            "producer": str(snapshot.get("producer", "")),
            "role": str(snapshot.get("role", "")),
            "worker_id": str(snapshot.get("worker_id", "")),
            "node": str(snapshot.get("node", "")),
        }
        local_rank = snapshot.get("local_rank")
        if type(local_rank) is int:
            labels["local_rank"] = str(local_rank)
        visible = [device.strip() for device in str(snapshot.get("cuda_visible_devices") or "").split(",")]
        if (type(local_rank) is int and 0 <= local_rank < len(visible)
                and visible[local_rank] and visible[local_rank] != "-1"):
            metrics.append(GaugeSample(
                "training_gpu_allocation",
                "Application worker assignment from CUDA_VISIBLE_DEVICES.",
                1,
                {**labels, "gpu": visible[local_rank]},
            ))
        observed_at = snapshot.get("observed_at")
        if isinstance(observed_at, (int, float)):
            metrics.append(GaugeSample(
                "training_sample_timestamp_seconds",
                "Application-reported sample timestamp.",
                observed_at,
                labels,
            ))
        step = snapshot.get("step")
        if type(step) is int:
            metrics.append(GaugeSample("training_step", "Latest reported training step.", step, labels))
        for sample in snapshot.get("samples", []):
            if not isinstance(sample, dict):
                continue
            metrics.append(GaugeSample(
                str(sample.get("name", "")),
                f"Application-reported {sample.get('name', '')}.",
                sample.get("value", 0),
                {**labels, **sample.get("labels", {})},
                str(sample.get("kind", "gauge")),
            ))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-dir", required=True, type=Path)
    parser.add_argument("--textfile-dir", required=True, type=Path)
    parser.add_argument("--interval", type=float, default=2.0)
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("interval must be positive")
    while True:
        write_gauges(args.textfile_dir, "observatory.prom", build_metrics(_iter_snapshots(args.metrics_dir)))
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
