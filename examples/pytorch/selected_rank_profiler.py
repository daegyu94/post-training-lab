"""Bounded PyTorch Profiler context for Megatron or custom distributed loops."""

from __future__ import annotations

import itertools
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Protocol

import torch
import torch.distributed as dist


class StepProfiler(Protocol):
    def step(self) -> None: ...


class _NoOpProfiler:
    def step(self) -> None:
        pass


def _global_rank() -> int:
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


@contextmanager
def selected_rank_profile(
    output_dir: Path,
    ranks: set[int],
    *,
    skip_first: int = 0,
    wait: int = 1,
    warmup: int = 1,
    active: int = 2,
    repeat: int = 1,
    record_shapes: bool = False,
    profile_memory: bool = False,
    with_stack: bool = False,
) -> Iterator[StepProfiler]:
    """Profile a bounded schedule on selected global ranks only."""
    rank = _global_rank()
    if rank not in ranks:
        yield _NoOpProfiler()
        return

    rank_dir = output_dir / f"rank-{rank}"
    rank_dir.mkdir(parents=True, exist_ok=True)
    sequence = itertools.count()

    def save_trace(profiler: torch.profiler.profile) -> None:
        profiler.export_chrome_trace(str(rank_dir / f"trace-{next(sequence)}.json"))

    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(
            skip_first=skip_first,
            wait=wait,
            warmup=warmup,
            active=active,
            repeat=repeat,
        ),
        on_trace_ready=save_trace,
        record_shapes=record_shapes,
        profile_memory=profile_memory,
        with_stack=with_stack,
    ) as profiler:
        yield profiler
