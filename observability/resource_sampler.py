"""Sample node memory and the block device backing a run output."""

from __future__ import annotations

import argparse
import json
import os
import signal
import time
from pathlib import Path


_stop = False


def _request_stop(*_: object) -> None:
    global _stop
    _stop = True


def _memory() -> dict[str, int]:
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        name, raw = line.split(":", 1)
        if name in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}:
            values[name.lower() + "_bytes"] = int(raw.split()[0]) * 1024
    return values


def _device(target: Path) -> tuple[str, Path]:
    device = os.stat(target).st_dev
    major_minor = f"{os.major(device)}:{os.minor(device)}"
    stat_path = Path("/sys/dev/block") / major_minor / "stat"
    if not stat_path.is_file():
        raise RuntimeError(f"cannot find block statistics for {target}: {major_minor}")
    return major_minor, stat_path


def _disk(stat_path: Path) -> dict[str, int]:
    fields = [int(value) for value in stat_path.read_text().split()]
    if len(fields) < 11:
        raise RuntimeError(f"invalid block statistics: {stat_path}")
    return {
        "read_operations": fields[0],
        "read_bytes": fields[2] * 512,
        "read_time_ms": fields[3],
        "write_operations": fields[4],
        "write_bytes": fields[6] * 512,
        "write_time_ms": fields[7],
        "in_flight": fields[8],
        "busy_time_ms": fields[9],
        "weighted_busy_time_ms": fields[10],
    }


def sample(target: Path) -> dict[str, object]:
    major_minor, stat_path = _device(target)
    return {
        "monotonic_seconds": time.monotonic(),
        "wall_time_ns": time.time_ns(),
        "device_major_minor": major_minor,
        **_memory(),
        **_disk(stat_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=0.2)
    args = parser.parse_args()
    if args.interval <= 0:
        raise SystemExit("--interval must be positive")
    if not args.target.is_dir():
        raise SystemExit(f"--target must be an existing directory: {args.target}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)
    with args.output.open("x", encoding="utf-8") as stream:
        while not _stop:
            stream.write(json.dumps(sample(args.target), allow_nan=False) + "\n")
            stream.flush()
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
