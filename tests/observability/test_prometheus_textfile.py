from pathlib import Path

import pytest

from profiling_lab.prometheus_textfile import GaugeSample, write_gauges


def test_write_gauges_emits_labels_and_replaces_atomically(tmp_path: Path) -> None:
    destination = write_gauges(
        tmp_path,
        "rank_0.prom",
        [
            GaugeSample(
                "llm_step_time_seconds",
                "Step time.",
                1.25,
                {"rank": "0", "run_id": 'run-"one"'},
            )
        ],
    )

    assert destination.read_text() == (
        "# HELP llm_step_time_seconds Step time.\n"
        "# TYPE llm_step_time_seconds gauge\n"
        'llm_step_time_seconds{rank="0",run_id="run-\\"one\\""} 1.25\n'
    )
    assert list(tmp_path.glob(".*.tmp")) == []


def test_write_gauges_rejects_path_traversal(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="basename"):
        write_gauges(tmp_path, "../rank.prom", [])
