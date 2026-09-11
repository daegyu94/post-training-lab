import json
from pathlib import Path

from profiling_lab.framework_metrics_textfile import build_gauges, _iter_samples


SAMPLE = {
    "schema_version": 1, "run_id": "run-1", "framework": "megatron", "node": "spark1",
    "rank": 0, "local_rank": 0, "cuda_visible_devices": "GPU-abc", "step": 7, "observed_at": 100.0,
    "metrics": {"training_loss": 1.25, "training_tokens_per_second": 42.0},
    "timers": {"forward-backward": 0.5},
}


def test_build_gauges_emits_metrics_timers_and_step_with_shared_labels() -> None:
    gauges = build_gauges([SAMPLE])
    by_name = {(g.name, g.labels.get("timer")): g for g in gauges}

    labels = {"run_id": "run-1", "framework": "megatron", "node": "spark1", "rank": "0", "local_rank": "0"}
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


def test_iter_samples_skips_malformed_files_and_ignores_other_names(tmp_path: Path) -> None:
    (tmp_path / "megatron-rank-0.json").write_text(json.dumps(SAMPLE), encoding="utf-8")
    (tmp_path / "megatron-rank-1.json").write_text("not json", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("ignore me", encoding="utf-8")

    samples = _iter_samples(tmp_path)

    assert samples == [SAMPLE]
