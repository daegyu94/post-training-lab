import json
from pathlib import Path

from profiling_lab.show_run import summarize


def test_summarize_reads_metadata_and_framework_metrics_without_rank_duplicates(tmp_path: Path) -> None:
    (tmp_path / "run-metadata-train.json").write_text(
        json.dumps({"model_id": "zai-org/GLM-4.7-Flash", "configuration": {"max_steps": 1}}), encoding="utf-8"
    )
    (tmp_path / "run-metadata-train-rank-0.json").write_text(json.dumps({"model_id": "duplicate"}), encoding="utf-8")
    (tmp_path / "summary-train.json").write_text(json.dumps({"model_id": "trl-model", "train_seconds": 12.5}), encoding="utf-8")
    metrics_dir = tmp_path / "framework-metrics"
    metrics_dir.mkdir()
    (metrics_dir / "megatron-rank-0.json").write_text(
        json.dumps({"framework": "megatron", "rank": 0, "step": 7, "metrics": {"training_loss": 1.5}}),
        encoding="utf-8",
    )

    output = summarize(tmp_path)

    assert "zai-org/GLM-4.7-Flash" in output
    assert "duplicate" not in output
    assert "trl-model" in output
    assert "[megatron rank 0] step 7: loss=1.5" in output


def test_summarize_notes_missing_sources(tmp_path: Path) -> None:
    output = summarize(tmp_path)
    assert "no run-metadata" in output
    assert "no framework-metrics" in output
