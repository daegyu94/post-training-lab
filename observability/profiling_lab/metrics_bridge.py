"""Forward local framework-metrics JSON samples to the local collector.

Reads the same <framework>-rank-<rank>.json files write_framework_metrics()
produces (see framework_metrics.py) from a directory this process can read
directly - typically the NFS-shared path FRAMEWORK_METRICS_DIR was pointed at
- and POSTs each new sample to a collector's /api/framework-metrics endpoint.

This has no SSH, tunnel, or remote agent involved: it only ever reads a local
(or NFS-shared) directory and makes an HTTP POST, so it can run entirely on
controller. The collector defaults to 127.0.0.1 (same host); pass --endpoint
to forward to a collector running elsewhere instead.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def _post(endpoint: str, token: str, sample: dict[str, Any]) -> int:
    body = json.dumps(sample).encode("utf-8")
    request = urllib.request.Request(
        f"{endpoint.rstrip('/')}/api/framework-metrics",
        data=body,
        method="POST",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return response.status


def _iter_samples(metrics_dir: Path) -> list[tuple[str, dict[str, Any]]]:
    samples = []
    for path in sorted(metrics_dir.glob("*-rank-*.json")):
        try:
            samples.append((path.name, json.loads(path.read_text(encoding="utf-8"))))
        except (OSError, json.JSONDecodeError):
            continue
    return samples


def bridge_once(metrics_dir: Path, endpoint: str, token: str, last_sent: dict[str, Any]) -> dict[str, Any]:
    """Post any sample whose (step, observed_at) hasn't been sent yet; return updated last_sent."""
    for name, sample in _iter_samples(metrics_dir):
        fingerprint = (sample.get("step"), sample.get("observed_at"))
        if last_sent.get(name) == fingerprint:
            continue
        try:
            _post(endpoint, token, sample)
            last_sent[name] = fingerprint
        except (urllib.error.URLError, OSError) as exc:
            print(f"[metrics-bridge] post failed for {name}: {exc}")
    return last_sent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-dir", required=True, type=Path)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8001", help="collector base URL; point at a non-local host to publish elsewhere")
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--once", action="store_true", help="poll once and exit instead of looping forever")
    args = parser.parse_args()

    token = os.environ.get("OBSERVATORY_TOKEN", "")
    if len(token) < 24:
        parser.error("set OBSERVATORY_TOKEN to the collector's token")

    last_sent: dict[str, Any] = {}
    while True:
        last_sent = bridge_once(args.metrics_dir, args.endpoint, token, last_sent)
        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
