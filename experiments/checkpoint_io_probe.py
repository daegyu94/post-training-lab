"""Measure buffered cache states and Direct I/O for one local checkpoint shard."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Callable


def latest_iteration(checkpoint_dir: Path) -> Path:
    candidates = []
    for path in checkpoint_dir.glob("iter_*"):
        try:
            iteration = int(path.name.removeprefix("iter_"))
        except ValueError:
            continue
        if path.is_dir():
            candidates.append((iteration, path))
    if not candidates:
        raise FileNotFoundError(f"no iteration checkpoint under {checkpoint_dir}")
    return max(candidates)[1]


def shard_files(iteration_dir: Path) -> list[Path]:
    files = sorted(iteration_dir.glob("*.distcp"))
    if not files:
        raise FileNotFoundError(f"no rank-local .distcp shard under {iteration_dir}")
    return files


def _device_reader(path: Path) -> tuple[str, Callable[[], dict[str, int]]]:
    device = os.stat(path).st_dev
    major_minor = f"{os.major(device)}:{os.minor(device)}"
    stat_path = Path("/sys/dev/block") / major_minor / "stat"
    if not stat_path.is_file():
        raise RuntimeError(f"cannot find block statistics for {path}: {major_minor}")

    def read() -> dict[str, int]:
        fields = [int(value) for value in stat_path.read_text().split()]
        return {
            "read_operations": fields[0],
            "read_bytes": fields[2] * 512,
            "write_operations": fields[4],
            "write_bytes": fields[6] * 512,
            "busy_time_ms": fields[9],
        }

    return major_minor, read


def _delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    return {name: after[name] - value for name, value in before.items()}


def _safe_rate(amount: int, elapsed: float) -> float | None:
    return amount / elapsed if elapsed > 0 else None


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def buffered_read(files: list[Path], read_device: Callable[[], dict[str, int]]) -> dict[str, object]:
    buffer = bytearray(4 * 1024 * 1024)
    before = read_device()
    started = time.perf_counter()
    logical = 0
    for path in files:
        with path.open("rb", buffering=0) as stream:
            while count := stream.readinto(buffer):
                logical += count
    elapsed = time.perf_counter() - started
    device = _delta(before, read_device())
    return {
        "seconds": elapsed,
        "logical_bytes": logical,
        "logical_bytes_per_second": _safe_rate(logical, elapsed),
        "physical_read_bytes": device["read_bytes"],
        "physical_read_operations": device["read_operations"],
        "device_busy_time_ms": device["busy_time_ms"],
        "physical_to_logical_ratio": _safe_ratio(device["read_bytes"], logical),
    }


def evict(files: list[Path]) -> None:
    if not hasattr(os, "posix_fadvise"):
        raise RuntimeError("POSIX_FADV_DONTNEED is unavailable")
    for path in files:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(descriptor, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(descriptor)


def evict_checkpoint(checkpoint_dir: Path) -> dict[str, object]:
    files = sorted(path for path in checkpoint_dir.rglob("*") if path.is_file())
    if not files:
        raise FileNotFoundError(f"no checkpoint files under {checkpoint_dir}")
    evict(files)
    return {
        "scope": "advisory checkpoint page-cache eviction",
        "file_count": len(files),
        "logical_bytes": sum(path.stat().st_size for path in files),
    }


def direct_read(
    files: list[Path],
    read_device: Callable[[], dict[str, int]],
    *,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, object]:
    dd = shutil.which("dd")
    if dd is None:
        raise RuntimeError("dd is required for the Direct I/O baseline")
    before = read_device()
    started = time.perf_counter()
    logical = 0
    for path in files:
        result = run(
            [dd, f"if={path}", "of=/dev/null", "bs=4M", "iflag=direct", "status=none"],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode:
            raise RuntimeError(f"Direct I/O failed for {path}: {result.stderr.strip()}")
        logical += path.stat().st_size
    elapsed = time.perf_counter() - started
    device = _delta(before, read_device())
    return {
        "seconds": elapsed,
        "logical_bytes": logical,
        "logical_bytes_per_second": _safe_rate(logical, elapsed),
        "physical_read_bytes": device["read_bytes"],
        "physical_read_operations": device["read_operations"],
        "device_busy_time_ms": device["busy_time_ms"],
        "physical_to_logical_ratio": _safe_ratio(device["read_bytes"], logical),
    }


def run_probe(checkpoint_dir: Path, *, include_direct: bool = True) -> dict[str, object]:
    iteration_dir = latest_iteration(checkpoint_dir)
    files = shard_files(iteration_dir)
    major_minor, read_device = _device_reader(checkpoint_dir)
    inventory = {
        "iteration": int(iteration_dir.name.removeprefix("iter_")),
        "directory": str(iteration_dir),
        "file_count": len(files),
        "logical_bytes": sum(path.stat().st_size for path in files),
        "allocated_bytes": sum(path.stat().st_blocks * 512 for path in files),
        "files": [{"name": path.name, "bytes": path.stat().st_size} for path in files],
    }
    warm_after_write = buffered_read(files, read_device)
    evict(files)
    cold = buffered_read(files, read_device)
    warm = buffered_read(files, read_device)
    cold["classification"] = (
        "valid" if cold["physical_to_logical_ratio"] is not None and cold["physical_to_logical_ratio"] >= 0.9
        else "cache-contaminated"
    )
    warm["classification"] = (
        "valid" if warm["physical_to_logical_ratio"] is not None and warm["physical_to_logical_ratio"] <= 0.1
        else "unexpected-cache-miss"
    )
    return {
        "scope": "raw local-shard read; not Megatron restore",
        "device_major_minor": major_minor,
        "inventory": inventory,
        "reads": {
            "warm_after_write": warm_after_write,
            "cold_buffered": cold,
            "warm_buffered": warm,
            "direct": direct_read(files, read_device) if include_direct else None,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--skip-direct", action="store_true")
    parser.add_argument("--evict-only", action="store_true")
    args = parser.parse_args()
    if args.evict_only:
        print(json.dumps(evict_checkpoint(args.checkpoint_dir), sort_keys=True))
        return
    if args.output is None:
        raise SystemExit("--output is required unless --evict-only is used")
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite output: {args.output}")
    result = run_probe(args.checkpoint_dir, include_direct=not args.skip_direct)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
