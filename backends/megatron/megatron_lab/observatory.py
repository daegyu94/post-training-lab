"""Megatron Bridge callback for the live metric spool."""

from __future__ import annotations

import time
from typing import Any, Callable

from profiling_lab.app_metrics import Metric, MetricEmitter


TIMER_NAMES = (
    "forward-backward",
    "all-grads-sync",
    "params-all-gather",
    "optimizer",
    "batch-generator",
)


def _timer_values(state: Any) -> dict[str, float]:
    timers = getattr(state.timers, "_timers", {})
    return {
        name: float(timers[name].active_time())
        for name in TIMER_NAMES
        if name in timers
    }


def _loss(losses: dict[str, Any] | None) -> float | None:
    if not losses:
        return None
    value = losses.get("lm loss", next(iter(losses.values())))
    if hasattr(value, "item"):
        value = value.item()
    return float(value)


class _MetricsCallback:
    def __init__(
        self,
        emitter: MetricEmitter,
        tokens_per_step: int,
        *,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.emitter = emitter
        self.tokens_per_step = tokens_per_step
        self.clock = clock
        self.started = clock()
        self.timer_start: dict[str, float] = {}

    def on_train_step_start(self, context: Any) -> None:
        self.started = self.clock()
        self.timer_start = _timer_values(context.state)

    def on_train_step_end(self, context: Any) -> None:
        elapsed = self.clock() - self.started
        if elapsed <= 0:
            return
        current = _timer_values(context.state)
        timers = {
            name: max(0.0, value - self.timer_start.get(name, 0.0))
            for name, value in current.items()
        }
        samples = [
            Metric("training_step_time_seconds", elapsed),
            Metric("training_tokens_per_second", self.tokens_per_step / elapsed),
        ]
        loss = _loss(context.loss_dict)
        if loss is not None:
            samples.append(Metric("training_loss", loss))
        samples.extend(
            Metric("training_timer_seconds", value, labels={"timer": name})
            for name, value in timers.items()
        )
        self.emitter.emit(step=int(context.state.train_state.step) + 1, samples=samples)


def make_megatron_callback(tokens_per_step: int) -> Any | None:
    emitter = MetricEmitter.from_env(producer="megatron", role="trainer")
    if emitter is None:
        return None
    from megatron.bridge.training.callbacks import Callback

    class LocalMetricsCallback(_MetricsCallback, Callback):
        pass

    return LocalMetricsCallback(emitter, tokens_per_step)
