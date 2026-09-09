import json
from pathlib import Path
import types

from megatron_lab.observatory import _MetricsCallback


class Timer:
    def __init__(self, value: float) -> None:
        self.value = value

    def active_time(self) -> float:
        return self.value


def test_callback_writes_rank_loss_throughput_and_timer_deltas(tmp_path: Path) -> None:
    times = iter((10.0, 10.0, 12.0))
    callback = _MetricsCallback((tmp_path, "run-1"), 200, clock=lambda: next(times))
    timer = Timer(3.0)
    state = types.SimpleNamespace(
        timers=types.SimpleNamespace(_timers={"forward-backward": timer}),
        train_state=types.SimpleNamespace(step=4),
    )
    context = types.SimpleNamespace(state=state, loss_dict={"lm loss": types.SimpleNamespace(item=lambda: 1.25)})
    callback.on_train_step_start(context)
    timer.value = 4.5
    callback.on_train_step_end(context)

    sample = json.loads((tmp_path / "megatron-rank-0.json").read_text(encoding="utf-8"))
    assert sample["step"] == 5
    assert sample["metrics"] == {
        "training_step_time_seconds": 2.0,
        "training_tokens_per_second": 100.0,
        "training_loss": 1.25,
    }
    assert sample["timers"] == {"forward-backward": 1.5}
