import json
from argparse import Namespace
from pathlib import Path

import pytest

from profiling_lab import observability


def write_targets(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "applications.json").write_text("[]\n", encoding="utf-8")
    for name, target in (
        ("gpus.json", "node-a:9400"),
        ("nodes.json", "node-a:9100"),
    ):
        (directory / name).write_text(
            json.dumps([{"targets": [target], "labels": {"cluster": "lab"}}]),
            encoding="utf-8",
        )


def test_validate_target_files_counts_valid_groups(tmp_path: Path) -> None:
    write_targets(tmp_path)

    assert observability.validate_target_files(tmp_path) == {
        "files": 3,
        "groups": 2,
        "targets": 2,
    }


def test_validate_target_files_rejects_non_string_labels(tmp_path: Path) -> None:
    write_targets(tmp_path)
    (tmp_path / "nodes.json").write_text(
        json.dumps([{"targets": ["node-a:9100"], "labels": {"rank": 0}}]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="labels must be strings"):
        observability.validate_target_files(tmp_path)


def test_summarize_prometheus_targets_counts_health() -> None:
    payload = {
        "status": "success",
        "data": {
            "activeTargets": [
                {"health": "up"},
                {"health": "down"},
                {"health": "unknown"},
            ]
        },
    }

    assert observability.summarize_prometheus_targets(payload) == {
        "configured": 3,
        "up": 1,
        "down": 2,
    }


def test_validate_stack_writes_failure_summary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    target_dir = tmp_path / "targets"
    write_targets(target_dir)
    output = tmp_path / "summary.json"

    def fail_readiness(url: str, timeout: float) -> None:
        raise RuntimeError(f"unavailable: {url}")

    monkeypatch.setattr(observability, "_wait_until_ready", fail_readiness)
    args = Namespace(
        target_dir=target_dir,
        prometheus_url="http://127.0.0.1:9090",
        grafana_url="http://127.0.0.1:3000",
        output=output,
        timeout=0.01,
        require_targets_up=False,
    )

    assert observability.validate_stack(args) == 1
    summary = json.loads(output.read_text(encoding="utf-8"))
    assert summary["schema_version"] == 1
    assert not summary["validation"]["stack_valid"]
    assert len(summary["validation"]["errors"]) == 2


@pytest.mark.parametrize(
    ("require_targets_up", "expected_status"),
    ((False, 0), (True, 1)),
)
def test_validate_stack_applies_target_health_policy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    require_targets_up: bool,
    expected_status: int,
) -> None:
    target_dir = tmp_path / "targets"
    write_targets(target_dir)
    output = tmp_path / "summary.json"

    monkeypatch.setattr(
        observability,
        "_wait_until_ready",
        lambda url, timeout: None,
    )

    def read_json(url: str) -> dict:
        if url.endswith("/api/health"):
            return {"database": "ok"}
        if "/api/v1/targets" in url:
            return {
                "status": "success",
                "data": {
                    "activeTargets": [
                        {"health": "up"},
                        {"health": "down"},
                    ]
                },
            }
        return {"status": "success", "data": {"result": []}}

    monkeypatch.setattr(observability, "_read_json", read_json)
    args = Namespace(
        target_dir=target_dir,
        prometheus_url="http://127.0.0.1:9090",
        grafana_url="http://127.0.0.1:3000",
        output=output,
        timeout=0.01,
        require_targets_up=require_targets_up,
    )

    assert observability.validate_stack(args) == expected_status
    summary = json.loads(output.read_text(encoding="utf-8"))
    assert summary["validation"]["stack_valid"] is (expected_status == 0)
    assert summary["validation"]["prometheus_targets"] == {
        "configured": 2,
        "up": 1,
        "down": 1,
    }
