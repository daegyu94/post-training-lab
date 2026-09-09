"""Megatron Bridge callback for the Observatory live metric spool."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import time
from typing import Any, Callable

from profiling_lab.framework_metrics import configured_output, write_framework_metrics


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
        output: tuple[Path, str],
        tokens_per_step: int,
        *,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.directory, self.run_id = output
        self.tokens_per_step = tokens_per_step
        self.clock = clock
        self.started = clock()
        self.timer_start: dict[str, float] = {}
        self.warned = False

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
        metrics = {
            "training_step_time_seconds": elapsed,
            "training_tokens_per_second": self.tokens_per_step / elapsed,
        }
        loss = _loss(context.loss_dict)
        if loss is not None:
            metrics["training_loss"] = loss
        try:
            write_framework_metrics(
                self.directory,
                run_id=self.run_id,
                framework="megatron",
                rank=int(os.environ.get("RANK", "0")),
                step=int(context.state.train_state.step) + 1,
                metrics=metrics,
                timers=timers,
            )
        except (OSError, ValueError, ZeroDivisionError) as exc:
            if not self.warned:
                print(f"[observatory] metric export disabled after error: {exc}", file=sys.stderr)
                self.warned = True


def make_megatron_callback(tokens_per_step: int) -> Any | None:
    output = configured_output()
    if output is None:
        return None
    from megatron.bridge.training.callbacks import Callback

    class ObservatoryCallback(_MetricsCallback, Callback):
        pass

    return ObservatoryCallback(output, tokens_per_step)
