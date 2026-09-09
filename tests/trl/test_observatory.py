import json
from pathlib import Path
import types

from trl_lab.observatory import _MetricsCallback


def test_callback_writes_loss_step_time_and_token_rate(tmp_path: Path, monkeypatch) -> None:
    times = iter((10.0, 10.0, 12.0))
    callback = _MetricsCallback((tmp_path, "run-1"), clock=lambda: next(times))
    state = types.SimpleNamespace(global_step=0, num_input_tokens_seen=0, is_world_process_zero=True)
    callback.on_train_begin(None, state, None)
    state.global_step = 1
    state.num_input_tokens_seen = 100
    callback.on_log(None, state, None, {"loss": 1.5})

    sample = json.loads((tmp_path / "trl-rank-0.json").read_text(encoding="utf-8"))
    assert sample["metrics"] == {
        "training_loss": 1.5,
        "training_step_time_seconds": 2.0,
        "training_tokens_per_second": 50.0,
    }


def test_callback_ignores_non_training_and_non_main_process_logs(tmp_path: Path) -> None:
    callback = _MetricsCallback((tmp_path, "run-1"), clock=lambda: 1.0)
    state = types.SimpleNamespace(global_step=1, num_input_tokens_seen=10, is_world_process_zero=True)
    callback.on_log(None, state, None, {"eval_loss": 1.0})
    state.is_world_process_zero = False
    callback.on_log(None, state, None, {"loss": 1.0})
    assert not list(tmp_path.iterdir())
