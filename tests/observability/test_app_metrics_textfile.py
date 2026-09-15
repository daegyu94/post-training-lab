import json
import sys
from pathlib import Path

import pytest

from observatory_metrics import textfile
from observatory_metrics.textfile import _iter_snapshots, build_metrics


SNAPSHOT = {
    "schema_version": 2,
    "run_id": "run-1",
    "producer": "megatron",
    "role": "trainer",
    "worker_id": "0",
    "node": "node-a",
    "local_rank": 0,
    "cuda_visible_devices": "GPU-abc",
    "step": 7,
    "observed_at": 100.0,
    "samples": [
        {"name": "training_loss", "kind": "gauge", "value": 1.25, "labels": {}},
        {
            "name": "agent_tool_call_errors_total",
            "kind": "counter",
            "value": 2,
            "labels": {"tool": "python"},
        },
        {
            "name": "training_timer_seconds",
            "kind": "gauge",
            "value": 0.5,
            "labels": {"timer": "forward-backward"},
        },
    ],
}


def test_build_metrics_emits_context_labels_and_metric_kinds() -> None:
    metrics = build_metrics([SNAPSHOT])
    by_name = {(metric.name, metric.labels.get("timer")): metric for metric in metrics}
    labels = {
        "run_id": "run-1",
        "producer": "megatron",
        "role": "trainer",
        "worker_id": "0",
        "node": "node-a",
        "local_rank": "0",
    }

    assert by_name[("training_loss", None)].labels == labels
    assert by_name[("agent_tool_call_errors_total", None)].kind == "counter"
    assert by_name[("agent_tool_call_errors_total", None)].labels == {**labels, "tool": "python"}
    assert by_name[("training_timer_seconds", "forward-backward")].value == 0.5
    assert by_name[("training_gpu_allocation", None)].labels == {**labels, "gpu": "GPU-abc"}
    assert by_name[("training_step", None)].value == 7
    assert by_name[("training_sample_timestamp_seconds", None)].value == 100.0


def test_collector_runs_multiple_refreshes(tmp_path: Path, monkeypatch) -> None:
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    snapshot_path = metrics_dir / "megatron-trainer-0.json"
    snapshot_path.write_text(json.dumps(SNAPSHOT), encoding="utf-8")
    output = tmp_path / "textfile"
    monkeypatch.setattr(sys, "argv", [
        "collector", "--metrics-dir", str(metrics_dir),
        "--textfile-dir", str(output), "--interval", "0.01",
    ])
    observed = []

    def tick(interval):
        observed.append((output / "observatory.prom").read_text())
        if len(observed) == 2:
            raise KeyboardInterrupt
        snapshot_path.write_text(json.dumps(SNAPSHOT | {"step": 8}), encoding="utf-8")

    monkeypatch.setattr(textfile.time, "sleep", tick)
    with pytest.raises(KeyboardInterrupt):
        textfile.main()
    assert "# TYPE agent_tool_call_errors_total counter" in observed[0]
    assert any(line.endswith(" 7") for line in observed[0].splitlines() if line.startswith("training_step{"))
    assert any(line.endswith(" 8") for line in observed[1].splitlines() if line.startswith("training_step{"))


def test_iter_snapshots_ignores_malformed_and_old_schema(tmp_path: Path) -> None:
    (tmp_path / "valid.json").write_text(json.dumps(SNAPSHOT), encoding="utf-8")
    (tmp_path / "malformed.json").write_text("not json", encoding="utf-8")
    (tmp_path / "old.json").write_text(json.dumps({"schema_version": 1}), encoding="utf-8")

    assert _iter_snapshots(tmp_path) == [SNAPSHOT]
