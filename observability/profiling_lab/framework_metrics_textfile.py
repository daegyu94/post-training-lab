"""Republish local framework-metrics JSON snapshots as Node Exporter textfile gauges.

Reads the same <framework>-rank-<rank>.json files write_framework_metrics()
produces (see framework_metrics.py) and writes them into the same
--collector.textfile.directory the node's own node_exporter already serves.
This puts training metrics (loss, tokens/s, step time) on the same
Prometheus/Grafana stack as host/GPU/network metrics for that node, so there
is one dashboard instead of a separate training-metrics viewer.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from profiling_lab.prometheus_textfile import GaugeSample, write_gauges


def _iter_samples(metrics_dir: Path) -> list[dict]:
    samples = []
    for path in sorted(metrics_dir.glob("*-rank-*.json")):
        try:
            samples.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return samples


def build_gauges(samples: list[dict]) -> list[GaugeSample]:
    gauges = []
    for sample in samples:
        labels = {
            "run_id": str(sample.get("run_id", "")),
            "framework": str(sample.get("framework", "")),
            "node": str(sample.get("node", "")),
            "rank": str(sample.get("rank", "")),
            "local_rank": str(sample.get("local_rank", "")),
        }
        visible = [device.strip() for device in str(sample.get("cuda_visible_devices", "")).split(",")]
        local_rank = sample.get("local_rank")
        if (type(local_rank) is int and 0 <= local_rank < len(visible)
                and visible[local_rank] and visible[local_rank] != "-1"):
            gauges.append(GaugeSample(
                "training_gpu_allocation",
                "Framework rank assignment from CUDA_VISIBLE_DEVICES.",
                1,
                {**labels, "gpu": visible[local_rank]},
            ))
        observed_at = sample.get("observed_at")
        if isinstance(observed_at, (int, float)):
            gauges.append(GaugeSample(
                "training_sample_timestamp_seconds",
                "Framework-reported sample timestamp.",
                observed_at,
                labels,
            ))
        gauges.append(GaugeSample("training_step", "Latest reported training step.", sample.get("step", 0), labels))
        for name, value in sample.get("metrics", {}).items():
            gauges.append(GaugeSample(name, f"Framework-reported {name}.", value, labels))
        for timer, value in sample.get("timers", {}).items():
            gauges.append(GaugeSample(
                "training_timer_seconds", "Framework rank-local timer.", value, {**labels, "timer": timer},
            ))
    return gauges


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-dir", required=True, type=Path)
    parser.add_argument("--textfile-dir", required=True, type=Path)
    parser.add_argument("--interval", type=float, default=2.0)
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("interval must be positive")
    while True:
        write_gauges(args.textfile_dir, "framework.prom", build_gauges(_iter_samples(args.metrics_dir)))
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
