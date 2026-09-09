"""TRL callback for the Observatory live framework metric spool."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import time
from typing import Any, Callable

from profiling_lab.framework_metrics import configured_output, write_framework_metrics


class _MetricsCallback:
    def __init__(
        self,
        output: tuple[Path, str],
        *,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.directory, self.run_id = output
        self.clock = clock
        self.previous_time = clock()
        self.previous_tokens = 0
        self.warned = False

    def on_train_begin(self, args: Any, state: Any, control: Any, **_: Any) -> None:
        self.previous_time = self.clock()
        self.previous_tokens = int(getattr(state, "num_input_tokens_seen", 0))

    def on_log(self, args: Any, state: Any, control: Any, logs: dict[str, Any] | None = None, **_: Any) -> None:
        if not getattr(state, "is_world_process_zero", True) or not logs or "loss" not in logs:
            return
        now = self.clock()
        elapsed = now - self.previous_time
        tokens = int(getattr(state, "num_input_tokens_seen", self.previous_tokens))
        metrics = {
            "training_loss": float(logs["loss"]),
            "training_step_time_seconds": elapsed,
        }
        if elapsed > 0 and tokens > self.previous_tokens:
            metrics["training_tokens_per_second"] = (tokens - self.previous_tokens) / elapsed
        self.previous_time, self.previous_tokens = now, tokens
        try:
            write_framework_metrics(
                self.directory,
                run_id=self.run_id,
                framework="trl",
                rank=int(os.environ.get("RANK", "0")),
                step=int(state.global_step),
                metrics=metrics,
            )
        except (OSError, ValueError) as exc:
            if not self.warned:
                print(f"[observatory] metric export disabled after error: {exc}", file=sys.stderr)
                self.warned = True


def make_trl_callback() -> Any | None:
    output = configured_output()
    if output is None:
        return None
    from transformers import TrainerCallback

    class ObservatoryCallback(_MetricsCallback, TrainerCallback):
        pass

    return ObservatoryCallback(output)
