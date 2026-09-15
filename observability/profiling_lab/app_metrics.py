"""Portable, best-effort application metric snapshots."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import re
import socket
import sys
import time
from typing import Callable, Iterable, Mapping


_IDENTIFIER = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
_METRIC_NAME = re.compile(r"^[a-zA-Z_:][a-zA-Z0-9_:]*$")
_LABEL_NAME = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")
_CONTEXT_LABELS = {"run_id", "producer", "role", "worker_id", "node", "local_rank"}


@dataclass(frozen=True)
class Metric:
    name: str
    value: float
    kind: str = "gauge"
    labels: Mapping[str, str] = field(default_factory=dict)


class MetricEmitter:
    """Write one producer worker's latest metrics without blocking its workload."""

    def __init__(
        self,
        directory: Path,
        *,
        run_id: str,
        producer: str,
        role: str,
        worker_id: str,
        node: str | None = None,
        local_rank: int | None = None,
        cuda_visible_devices: str | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        identifiers = {"run_id": run_id, "producer": producer, "role": role, "worker_id": worker_id}
        for name, value in identifiers.items():
            if not _IDENTIFIER.fullmatch(value):
                raise ValueError(f"invalid {name}: {value!r}")
        if local_rank is not None and local_rank < 0:
            raise ValueError("local_rank must be nonnegative")
        self.directory = directory
        self.run_id = run_id
        self.producer = producer
        self.role = role
        self.worker_id = worker_id
        self.node = node or socket.gethostname()
        self.local_rank = local_rank
        self.cuda_visible_devices = cuda_visible_devices
        self.clock = clock
        self.disabled = False

    @classmethod
    def from_env(
        cls,
        *,
        producer: str,
        role: str,
        worker_id: str | None = None,
    ) -> MetricEmitter | None:
        directory = os.environ.get("OBSERVATORY_METRICS_DIR")
        run_id = os.environ.get("OBSERVATORY_RUN_ID")
        if not directory and not run_id:
            return None
        if not directory or not run_id:
            print(
                "[metrics] OBSERVATORY_METRICS_DIR and OBSERVATORY_RUN_ID must be set together",
                file=sys.stderr,
            )
            return None
        try:
            local_rank = int(os.environ["LOCAL_RANK"]) if "LOCAL_RANK" in os.environ else None
            return cls(
                Path(directory),
                run_id=run_id,
                producer=producer,
                role=role,
                worker_id=worker_id or os.environ.get("RANK", "0"),
                local_rank=local_rank,
                cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
            )
        except ValueError as exc:
            print(f"[metrics] export disabled: {exc}", file=sys.stderr)
            return None

    def emit(self, *, step: int | None, samples: Iterable[Metric]) -> Path | None:
        if self.disabled:
            return None
        try:
            if step is not None and (type(step) is not int or step < 0):
                raise ValueError("step must be a nonnegative integer or None")
            encoded = [self._encode(sample) for sample in samples]
            if not encoded:
                raise ValueError("at least one metric is required")
            snapshot = {
                "schema_version": 2,
                "run_id": self.run_id,
                "producer": self.producer,
                "role": self.role,
                "worker_id": self.worker_id,
                "node": self.node,
                "local_rank": self.local_rank,
                "cuda_visible_devices": self.cuda_visible_devices,
                "step": step,
                "observed_at": self.clock(),
                "samples": encoded,
            }
            self.directory.mkdir(parents=True, exist_ok=True)
            filename = f"{self.producer}-{self.role}-{self.worker_id}.json"
            destination = self.directory / filename
            temporary = self.directory / f".{filename}.{os.getpid()}.tmp"
            temporary.write_text(json.dumps(snapshot, separators=(",", ":")) + "\n", encoding="utf-8")
            os.replace(temporary, destination)
            return destination
        except (OSError, ValueError) as exc:
            print(f"[metrics] export disabled: {exc}", file=sys.stderr)
            self.disabled = True
            return None

    @staticmethod
    def _encode(metric: Metric) -> dict[str, object]:
        if not _METRIC_NAME.fullmatch(metric.name):
            raise ValueError(f"invalid metric name: {metric.name!r}")
        if metric.kind not in {"gauge", "counter"}:
            raise ValueError(f"invalid metric kind: {metric.kind!r}")
        if type(metric.value) not in (int, float) or not math.isfinite(metric.value):
            raise ValueError(f"metric {metric.name!r} must be finite")
        if metric.kind == "counter" and metric.value < 0:
            raise ValueError(f"counter {metric.name!r} must be nonnegative")
        if any(not _LABEL_NAME.fullmatch(name) for name in metric.labels):
            raise ValueError(f"metric {metric.name!r} has an invalid label name")
        if set(metric.labels) & _CONTEXT_LABELS:
            raise ValueError(f"metric {metric.name!r} overrides a context label")
        return {
            "name": metric.name,
            "kind": metric.kind,
            "value": metric.value,
            "labels": {name: str(value) for name, value in metric.labels.items()},
        }
