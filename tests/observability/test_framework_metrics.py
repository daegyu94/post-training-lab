import json
from pathlib import Path

import pytest

from profiling_lab.framework_metrics import write_framework_metrics


def test_write_framework_metrics_replaces_rank_sample_atomically(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,5")
    path = write_framework_metrics(
        tmp_path,
        run_id="run-1",
        framework="megatron",
        rank=3,
        step=7,
        metrics={"training_loss": 1.25, "training_tokens_per_second": 42.0},
        timers={"forward-backward": 0.5},
    )

    sample = json.loads(path.read_text(encoding="utf-8"))
    assert sample | {"observed_at": 0, "node": ""} == {
        "schema_version": 1,
        "run_id": "run-1",
        "framework": "megatron",
        "node": "",
        "rank": 3,
        "local_rank": 1,
        "cuda_visible_devices": "2,5",
        "step": 7,
        "observed_at": 0,
        "metrics": {"training_loss": 1.25, "training_tokens_per_second": 42.0},
        "timers": {"forward-backward": 0.5},
    }
    assert not list(tmp_path.glob(".*.tmp"))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"run_id": "bad/id"},
        {"metrics": {"unknown": 1.0}},
        {"timers": {"Bad Timer": 1.0}},
        {"metrics": {"training_loss": float("nan")}},
    ],
)
def test_write_framework_metrics_rejects_invalid_samples(tmp_path: Path, kwargs) -> None:
    values = dict(run_id="run-1", framework="trl", rank=0, step=1, metrics={"training_loss": 1.0})
    values.update(kwargs)
    with pytest.raises(ValueError):
        write_framework_metrics(tmp_path, **values)
