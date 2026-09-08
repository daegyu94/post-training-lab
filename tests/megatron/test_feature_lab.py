from pathlib import Path

import pytest

from megatron_lab.feature_lab import parse_megatron_log


def _log(*iterations: int, checkpoint: bool = True) -> str:
    lines = []
    for iteration in iterations:
        lines.append(
            f"[stage] iteration {iteration}/4 | elapsed time per iteration (ms): {100 + iteration}.0 | "
            f"lm loss: {2 + iteration / 10:.6f} | grad norm: {iteration + 1}.0 | "
            f"number of skipped iterations: 0 | number of nan iterations: 0 |"
        )
    if checkpoint:
        lines.append("    save-checkpoint ................................: (12.5, 18.5)")
    return "\n".join(lines) + "\n"


def test_parse_megatron_log_separates_warmup_steady_and_checkpoint(tmp_path: Path):
    path = tmp_path / "rank-1.log"
    path.write_text(_log(1, 2, 3, 4), encoding="utf-8")
    parsed = parse_megatron_log(
        path,
        feature="overlap-grad-reduce",
        variant="on",
        run_index=2,
    )
    assert parsed["warmup_step_ids"] == [1, 2]
    assert parsed["steady_step_ids"] == [3, 4]
    assert parsed["steady_step_median_ms"] == 103.5
    assert parsed["steps"][2]["loss"] == 2.3
    assert parsed["steps"][3]["grad_norm"] == 5.0
    assert parsed["steps"][0]["nan_iterations"] == 0
    assert parsed["checkpoint_save_rank_range_ms"] == [{"min_ms": 12.5, "max_ms": 18.5}]
    assert len(parsed["raw_log_sha256"]) == 64
    assert "speedup" in parsed["interpretation"]


def test_parse_rejects_missing_iteration(tmp_path: Path):
    path = tmp_path / "missing.log"
    path.write_text(_log(1, 2, 4), encoding="utf-8")
    with pytest.raises(ValueError, match="missing iterations"):
        parse_megatron_log(path, feature="recompute", variant="full", run_index=0)


def test_parse_rejects_duplicate_iteration(tmp_path: Path):
    path = tmp_path / "duplicate.log"
    path.write_text(_log(1, 2, 2, 4), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        parse_megatron_log(path, feature="recompute", variant="full", run_index=0)


def test_parse_rejects_nonfinite_metric(tmp_path: Path):
    path = tmp_path / "nan.log"
    path.write_text(
        _log(1, 2, 3, 4).replace("lm loss: 2.300000", "lm loss: nan"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="non-finite"):
        parse_megatron_log(path, feature="recompute", variant="full", run_index=0)


@pytest.mark.parametrize(
    ("needle", "replacement", "message"),
    [
        ("elapsed time per iteration (ms): 101.0", "elapsed time per iteration (ms): 0.0", "non-positive elapsed"),
        ("elapsed time per iteration (ms): 101.0", "elapsed time per iteration (ms): -1.0", "non-positive elapsed"),
        ("grad norm: 2.0", "grad norm: -1.0", "negative grad"),
    ],
)
def test_parse_rejects_invalid_positive_timing_metrics(
    tmp_path: Path, needle: str, replacement: str, message: str
):
    path = tmp_path / "invalid-timing.log"
    path.write_text(_log(1, 2, 3, 4).replace(needle, replacement), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        parse_megatron_log(path, feature="recompute", variant="full", run_index=0)


@pytest.mark.parametrize("field", ["number of skipped iterations", "number of nan iterations"])
def test_parse_rejects_skipped_or_nan_updates_for_benchmark(tmp_path: Path, field: str):
    path = tmp_path / "invalid-update.log"
    path.write_text(_log(1, 2, 3, 4).replace(f"{field}: 0", f"{field}: 1"), encoding="utf-8")
    with pytest.raises(ValueError, match="benchmark timing is invalid"):
        parse_megatron_log(path, feature="recompute", variant="full", run_index=0)


def test_parse_does_not_infer_stage_completion(tmp_path: Path):
    path = tmp_path / "partial-stage.log"
    path.write_text(_log(1, 2, 3, 4), encoding="utf-8")
    parsed = parse_megatron_log(
        path,
        feature="checkpoint",
        variant="sync",
        run_index=3,
        exit_code=1,
    )
    assert parsed["completion_unverified"] is True
    assert parsed["completion_verified"] is False
    assert parsed["process_exit_code"] == 1


def test_parse_marks_completion_only_with_zero_exit(tmp_path: Path):
    path = tmp_path / "complete.log"
    path.write_text(_log(1, 2, 3, 4), encoding="utf-8")
    parsed = parse_megatron_log(
        path,
        feature="checkpoint",
        variant="async",
        run_index=4,
        exit_code=0,
        completed=True,
    )
    assert parsed["completion_unverified"] is False
    assert parsed["completion_verified"] is True

    with pytest.raises(ValueError, match="exit code 0"):
        parse_megatron_log(
            path,
            feature="checkpoint",
            variant="async",
            run_index=4,
            exit_code=1,
            completed=True,
        )


def test_raw_log_provenance_requires_explicit_run_index(tmp_path: Path):
    # The parser API itself accepts an explicit run index; this fixture also
    # documents that raw log provenance is not inferred from a filename.
    path = tmp_path / "rank.log"
    path.write_text(_log(1, 2, 3, 4), encoding="utf-8")
    result = parse_megatron_log(path, feature="checkpoint", variant="sync", run_index=7)
    assert result["run_index"] == 7
