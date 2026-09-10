"""Sample Linux host counters and push them to the local collector."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import time
from typing import Any
from urllib.request import Request, urlopen


def counters() -> tuple[float, int, int, dict[str, int], dict[str, tuple[int, int]]]:
    cpu = list(map(int, Path("/proc/stat").read_text().splitlines()[0].split()[1:9]))
    memory = {line.split(":")[0]: int(line.split()[1]) for line in Path("/proc/meminfo").read_text().splitlines()}
    network = {}
    for line in Path("/proc/net/dev").read_text().splitlines()[2:]:
        name, raw = line.split(":")
        name = name.strip()
        if name == "lo" or name.startswith(("veth", "docker", "br-")):
            continue
        values = list(map(int, raw.split()))
        network[name] = (values[0], values[8])
    return time.monotonic(), sum(cpu), cpu[3] + cpu[4], memory, network


def metrics(before: tuple, after: tuple) -> dict[str, float]:
    elapsed = after[0] - before[0]
    total = after[1] - before[1]
    if elapsed <= 0 or total <= 0:
        raise ValueError("sampling interval did not advance")
    shared_interfaces = before[4].keys() & after[4].keys()
    network = [sum(max(0, after[4][name][index] - before[4][name][index]) for name in shared_interfaces) * 8 / elapsed / 1e9 for index in (0, 1)]
    memory = after[3]
    return {
        "cpu_utilization_percent": round(100 * (1 - (after[2] - before[2]) / total), 3),
        "memory_used_gib": (memory["MemTotal"] - memory["MemAvailable"]) / 1024**2,
        "memory_total_gib": memory["MemTotal"] / 1024**2,
        "memory_available_gib": memory["MemAvailable"] / 1024**2,
        "swap_used_gib": (memory["SwapTotal"] - memory["SwapFree"]) / 1024**2,
        "swap_total_gib": memory["SwapTotal"] / 1024**2,
        "nic_receive_gbps": network[0],
        "nic_transmit_gbps": network[1],
        "interval_seconds": elapsed,
    }


def post(endpoint: str, token: str, path: str, data: dict[str, Any]) -> None:
    request = Request(
        endpoint.rstrip("/") + path,
        data=json.dumps(data).encode(),
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )
    with urlopen(request, timeout=10) as response:
        if response.status != 201:
            raise RuntimeError("collector rejected sample")


def framework_samples(directory: Path | None, seen: dict[Path, Any]) -> list[dict[str, Any]]:
    if directory is None:
        return []
    samples = []
    for path in sorted(directory.glob("*-rank-*.json")):
        try:
            sample = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        marker = sample.get("observed_at")
        if marker != seen.get(path):
            seen[path] = marker
            samples.append(sample)
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8001")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--node", default=socket.gethostname())
    parser.add_argument("--interval", type=float, default=2)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--framework-metrics-dir", type=Path)
    args = parser.parse_args()
    if args.interval <= 0 or args.samples <= 0:
        parser.error("interval and samples must be positive")
    token = os.environ.get("OBSERVATORY_TOKEN", "")
    if len(token) < 24:
        parser.error("set OBSERVATORY_TOKEN to the collector token")
    seen: dict[Path, Any] = {}
    before = counters()
    for _ in range(args.samples):
        time.sleep(args.interval)
        after = counters()
        sample = {"run_id": args.run_id, "node": args.node, "metrics": metrics(before, after)}
        post(args.endpoint, token, "/api/telemetry", sample)
        for framework_sample in framework_samples(args.framework_metrics_dir, seen):
            if framework_sample.get("run_id") != args.run_id:
                raise ValueError("framework metric run_id must match the agent")
            post(args.endpoint, token, "/api/framework-metrics", {**framework_sample, "node": args.node})
        print(json.dumps(sample), flush=True)
        before = after


if __name__ == "__main__":
    main()
