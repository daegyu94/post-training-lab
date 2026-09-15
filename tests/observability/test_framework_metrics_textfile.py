import json
import sys
from pathlib import Path

from profiling_lab.framework_metrics_textfile import build_gauges, _iter_samples
from profiling_lab import framework_metrics_textfile
import pytest


SAMPLE = {
    "schema_version": 1, "run_id": "run-1", "framework": "megatron", "node": "node-a",
    "rank": 0, "local_rank": 0, "cuda_visible_devices": "GPU-abc", "step": 7, "observed_at": 100.0,
    "metrics": {"training_loss": 1.25, "training_tokens_per_second": 42.0},
    "timers": {"forward-backward": 0.5},
}


def test_build_gauges_emits_metrics_timers_and_step_with_shared_labels() -> None:
    gauges = build_gauges([SAMPLE])
    by_name = {(g.name, g.labels.get("timer")): g for g in gauges}

    labels = {"run_id": "run-1", "framework": "megatron", "node": "node-a", "rank": "0", "local_rank": "0"}
    assert by_name[("training_loss", None)].value == 1.25
    assert by_name[("training_loss", None)].labels == labels
    assert by_name[("training_tokens_per_second", None)].value == 42.0
    assert by_name[("training_step", None)].value == 7
    assert by_name[("training_gpu_allocation", None)].labels == {**labels, "gpu": "GPU-abc"}
    assert by_name[("training_timer_seconds", "forward-backward")].value == 0.5
    assert by_name[("training_timer_seconds", "forward-backward")].labels == {**labels, "timer": "forward-backward"}
    assert by_name[("training_sample_timestamp_seconds", None)].value == 100.0
    assert by_name[("training_sample_timestamp_seconds", None)].labels == labels


def test_build_gauges_handles_no_samples() -> None:
    gauges = build_gauges([])
    assert gauges == []


@pytest.mark.parametrize("visible,rank,expected", [
    ("2, 5", 1, "5"), ("GPU-a,GPU-b", 0, "GPU-a"),
    ("", 0, None), ("-1", 0, None), ("0", 1, None),
])
def test_allocation_uses_explicit_visible_device(visible, rank, expected) -> None:
    gauges = build_gauges([SAMPLE | {"cuda_visible_devices": visible, "local_rank": rank}])
    allocations = [g.labels["gpu"] for g in gauges if g.name == "training_gpu_allocation"]
    assert allocations == ([] if expected is None else [expected])


def test_collector_runs_multiple_refreshes(tmp_path: Path, monkeypatch) -> None:
    metrics = tmp_path / "metrics"
    metrics.mkdir()
    sample_path = metrics / "megatron-rank-0.json"
    sample_path.write_text(json.dumps(SAMPLE))
    output = tmp_path / "textfile"
    monkeypatch.setattr(sys, "argv", ["collector", "--metrics-dir", str(metrics),
                                    "--textfile-dir", str(output), "--interval", "0.01"])
    observed = []

    def tick(interval):
        observed.append((output / "framework.prom").read_text())
        if len(observed) == 2:
            raise KeyboardInterrupt
        sample_path.write_text(json.dumps(SAMPLE | {"step": 8}))

    monkeypatch.setattr(framework_metrics_textfile.time, "sleep", tick)
    with pytest.raises(KeyboardInterrupt):
        framework_metrics_textfile.main()
    assert any(line.endswith(" 7") for line in observed[0].splitlines() if line.startswith("training_step{"))
    assert any(line.endswith(" 8") for line in observed[1].splitlines() if line.startswith("training_step{"))


def test_iter_samples_skips_malformed_files_and_ignores_other_names(tmp_path: Path) -> None:
    (tmp_path / "megatron-rank-0.json").write_text(json.dumps(SAMPLE), encoding="utf-8")
    (tmp_path / "megatron-rank-1.json").write_text("not json", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("ignore me", encoding="utf-8")

    samples = _iter_samples(tmp_path)

    assert samples == [SAMPLE]
