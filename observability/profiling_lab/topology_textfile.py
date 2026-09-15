"""Publish supplied compute and storage topology JSON as Prometheus gauges."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from observatory_metrics.prometheus import GaugeSample, write_gauges


def build_gauges(directory: Path) -> list[GaugeSample]:
    gauges = []
    for kind in ("compute", "storage"):
        path = directory / f"{kind}-topology.json"
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for component in payload.get("components", []):
            gauges.append(GaugeSample(
                "profiling_topology_component_info", "Supplied topology component.", 1,
                {"kind": kind, "component": str(component["id"]), "role": str(component.get("role", ""))},
            ))
        for edge in payload.get("edges", []):
            gauges.append(GaugeSample(
                "profiling_topology_edge_info", "Supplied topology edge.", 1,
                {"kind": kind, "source": str(edge["source"]), "destination": str(edge["destination"]), "relation": str(edge.get("relation", ""))},
            ))
    return gauges


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology-dir", type=Path, required=True)
    parser.add_argument("--textfile-dir", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=10)
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("interval must be positive")
    while True:
        write_gauges(args.textfile_dir, "topology.prom", build_gauges(args.topology_dir))
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
