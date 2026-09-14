"""Opt-in host timings around Bridge checkpoint calls; never infer durability."""
from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from functools import wraps
from pathlib import Path


def _process_io() -> dict[str, int]:
    try:
        return {
            name: int(value)
            for name, value in (
                line.split(":", 1) for line in Path("/proc/self/io").read_text().splitlines()
            )
        }
    except (OSError, ValueError):
        return {}


@contextmanager
def measure_execution(
    output_dir: Path,
    stage: str,
    *,
    manager_class=None,
    cuda=None,
    process_io=_process_io,
    callback_class=None,
):
    """Measure checkpoint load/save calls and whole-stage CUDA allocation.

    The wrappers preserve return values and exceptions and are restored on exit.
    Async save call time is enqueue overhead, not background I/O completion time.
    Finalization records include whether the call blocks; this is not fsync or
    crash-durability evidence. Step timings come from the native iteration log.
    """
    if manager_class is None:
        from megatron.bridge.training.checkpointing import DefaultCheckpointManager
        manager_class = DefaultCheckpointManager
    if callback_class is None:
        from megatron.bridge.training.callbacks import Callback
        callback_class = Callback
    if cuda is None:
        from torch import cuda
    path = Path(output_dir) / 'measurements' / f'rank-{os.environ.get("RANK", "0")}-{stage}.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    originals = {}
    with path.open('x', encoding='utf-8') as stream:
        def emit(record):
            stream.write(json.dumps(record, allow_nan=False) + '\n')
            stream.flush()

        def wrap(name, original):
            @wraps(original)
            def measured(*args, **kwargs):
                started = time.perf_counter()
                started_wall = time.time_ns()
                io_before = process_io()
                success = False
                try:
                    result = original(*args, **kwargs)
                    success = True
                    return result
                finally:
                    event = {'event': name, 'seconds': time.perf_counter() - started,
                             'started_monotonic_seconds': started,
                             'started_wall_time_ns': started_wall,
                             'ended_monotonic_seconds': time.perf_counter(),
                             'ended_wall_time_ns': time.time_ns(),
                             'success': success, 'stage': stage}
                    io_after = process_io()
                    event['process_logical_read_bytes'] = max(
                        0, io_after.get('rchar', 0) - io_before.get('rchar', 0)
                    )
                    event['process_storage_read_bytes'] = max(
                        0, io_after.get('read_bytes', 0) - io_before.get('read_bytes', 0)
                    )
                    if name == 'finalize_async_saves':
                        event['blocking'] = kwargs.get('blocking', args[2] if len(args) > 2 else False)
                        event['terminate'] = kwargs.get('terminate', args[3] if len(args) > 3 else False)
                    emit(event)
            return measured

        available = cuda.is_available()
        if available:
            cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        started_wall = time.time_ns()

        class ModelReadyCallback(callback_class):
            def on_data_init_start(self, _context):
                emit({
                    'event': 'model_ready', 'stage': stage, 'success': True,
                    'seconds': time.perf_counter() - started,
                    'started_monotonic_seconds': started,
                    'started_wall_time_ns': started_wall,
                    'ended_monotonic_seconds': time.perf_counter(),
                    'ended_wall_time_ns': time.time_ns(),
                })

        success = False
        try:
            for name in ('load', 'save', 'finalize_async_saves'):
                originals[name] = getattr(manager_class, name)
                setattr(manager_class, name, wrap(name, originals[name]))
            yield ModelReadyCallback()
            success = True
        finally:
            for name, original in originals.items():
                setattr(manager_class, name, original)
            emit({'event': 'stage', 'stage': stage, 'success': success,
                  'seconds': time.perf_counter() - started,
                  'started_monotonic_seconds': started,
                  'started_wall_time_ns': started_wall,
                  'ended_monotonic_seconds': time.perf_counter(),
                  'ended_wall_time_ns': time.time_ns(),
                  'peak_cuda_allocated_bytes': cuda.max_memory_allocated() if available else None,
                  'peak_cuda_reserved_bytes': cuda.max_memory_reserved() if available else None})
