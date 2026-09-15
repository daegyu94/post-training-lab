"""Hugging Face Trainer adapter for portable application metrics."""

from __future__ import annotations

import time
from typing import Any, Callable

from profiling_lab.app_metrics import Metric, MetricEmitter


class _MetricsCallback:
    def __init__(
        self,
        emitter: MetricEmitter,
        *,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.emitter = emitter
        self.clock = clock
        self.previous_time = clock()
        self.previous_tokens = 0

    def on_train_begin(self, args: Any, state: Any, control: Any, **_: Any) -> None:
        self.previous_time = self.clock()
        self.previous_tokens = int(getattr(state, "num_input_tokens_seen", 0))

    def on_log(self, args: Any, state: Any, control: Any, logs: dict[str, Any] | None = None, **_: Any) -> None:
        if not getattr(state, "is_world_process_zero", True) or not logs or "loss" not in logs:
            return
        now = self.clock()
        elapsed = now - self.previous_time
        tokens = int(getattr(state, "num_input_tokens_seen", self.previous_tokens))
        samples = [
            Metric("training_loss", float(logs["loss"])),
            Metric("training_step_time_seconds", elapsed),
        ]
        if elapsed > 0 and tokens > self.previous_tokens:
            samples.append(Metric("training_tokens_per_second", (tokens - self.previous_tokens) / elapsed))
        self.previous_time, self.previous_tokens = now, tokens
        self.emitter.emit(step=int(state.global_step), samples=samples)


def make_trainer_callback(*, producer: str, role: str = "trainer") -> Any | None:
    emitter = MetricEmitter.from_env(producer=producer, role=role)
    if emitter is None:
        return None
    from transformers import TrainerCallback

    class LocalMetricsCallback(_MetricsCallback, TrainerCallback):
        pass

    return LocalMetricsCallback(emitter)
