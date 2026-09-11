"""Write the latest framework metrics without blocking training on the collector."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import socket
import time
from typing import Mapping


_IDENTIFIER = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
_TIMER = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
_METRICS = {
    "training_loss",
    "training_step_time_seconds",
    "training_tokens_per_second",
}


def configured_output() -> tuple[Path, str] | None:
    """Return the launcher-owned output location when live metrics are enabled."""
    directory = os.environ.get("FRAMEWORK_METRICS_DIR")
    run_id = os.environ.get("OBSERVATORY_RUN_ID")
    return (Path(directory), run_id) if directory and run_id else None


def write_framework_metrics(
    directory: Path,
    *,
    run_id: str,
    framework: str,
    rank: int,
    step: int,
    metrics: Mapping[str, float],
    timers: Mapping[str, float] | None = None,
) -> Path:
    """Atomically replace one rank's latest normalized metric sample."""
    timers = timers or {}
    if not _IDENTIFIER.fullmatch(run_id) or framework not in {"trl", "megatron"}:
        raise ValueError("invalid run_id or framework")
    if rank < 0 or step < 0:
        raise ValueError("rank and step must be nonnegative")
    if not metrics or set(metrics) - _METRICS:
        raise ValueError("unsupported framework metric")
    if len(timers) > 32 or any(not _TIMER.fullmatch(name) for name in timers):
        raise ValueError("invalid framework timer")
    values = [*metrics.values(), *timers.values()]
    if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("framework metrics must be finite nonnegative numbers")

    sample = {
        "schema_version": 1,
        "run_id": run_id,
        "framework": framework,
        "node": socket.gethostname(),
        "rank": rank,
        "local_rank": int(os.environ.get("LOCAL_RANK", "0")),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "step": step,
        "observed_at": time.time(),
        "metrics": dict(metrics),
        "timers": dict(timers),
    }
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{framework}-rank-{rank}.json"
    temporary = directory / f".{destination.name}.{os.getpid()}.tmp"
    temporary.write_text(json.dumps(sample, separators=(",", ":")) + "\n", encoding="utf-8")
    os.replace(temporary, destination)
    return destination
