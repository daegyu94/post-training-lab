from pathlib import Path

import pytest

from observatory_metrics.prometheus import GaugeSample, write_gauges


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


def test_write_gauges_preserves_counter_type(tmp_path: Path) -> None:
    destination = write_gauges(
        tmp_path,
        "agent.prom",
        [GaugeSample("agent_errors_total", "Agent errors.", 2, kind="counter")],
    )

    assert "# TYPE agent_errors_total counter" in destination.read_text()
