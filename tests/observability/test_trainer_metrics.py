import json
from pathlib import Path
import types

from observatory_metrics import MetricEmitter
from profiling_lab.adapters.hf_trainer import _MetricsCallback


def test_callback_writes_loss_step_time_and_token_rate(tmp_path: Path) -> None:
    times = iter((10.0, 10.0, 12.0))
    emitter = MetricEmitter(
        tmp_path,
        run_id="run-1",
        producer="trl",
        role="trainer",
        worker_id="0",
        node="spark1",
        clock=lambda: 100.0,
    )
    callback = _MetricsCallback(emitter, clock=lambda: next(times))
    state = types.SimpleNamespace(global_step=0, num_input_tokens_seen=0, is_world_process_zero=True)
    callback.on_train_begin(None, state, None)
    state.global_step = 1
    state.num_input_tokens_seen = 100
    callback.on_log(None, state, None, {"loss": 1.5})

    snapshot = json.loads((tmp_path / "trl-trainer-0.json").read_text(encoding="utf-8"))
    assert {sample["name"]: sample["value"] for sample in snapshot["samples"]} == {
        "training_loss": 1.5,
        "training_step_time_seconds": 2.0,
        "training_tokens_per_second": 50.0,
    }


def test_callback_ignores_non_training_and_non_main_process_logs(tmp_path: Path) -> None:
    emitter = MetricEmitter(
        tmp_path,
        run_id="run-1",
        producer="trl",
        role="trainer",
        worker_id="0",
    )
    callback = _MetricsCallback(emitter, clock=lambda: 1.0)
    state = types.SimpleNamespace(global_step=1, num_input_tokens_seen=10, is_world_process_zero=True)
    callback.on_log(None, state, None, {"eval_loss": 1.0})
    state.is_world_process_zero = False
    callback.on_log(None, state, None, {"loss": 1.0})
    assert not list(tmp_path.iterdir())
