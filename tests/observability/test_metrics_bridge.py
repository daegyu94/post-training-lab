import json
from pathlib import Path

from profiling_lab import metrics_bridge


SAMPLE = {
    "schema_version": 1, "run_id": "run-1", "framework": "trl", "node": "spark1",
    "rank": 0, "local_rank": 0, "step": 1, "observed_at": 100.0,
    "metrics": {"training_loss": 1.0}, "timers": {},
}


def test_bridge_once_posts_new_samples_and_skips_unchanged(tmp_path: Path, monkeypatch) -> None:
    posted = []
    monkeypatch.setattr(metrics_bridge, "_post", lambda endpoint, token, sample: posted.append((endpoint, token, sample)) or 201)
    (tmp_path / "trl-rank-0.json").write_text(json.dumps(SAMPLE), encoding="utf-8")

    last_sent = metrics_bridge.bridge_once(tmp_path, "http://127.0.0.1:8001", "token", {})
    assert posted == [("http://127.0.0.1:8001", "token", SAMPLE)]

    metrics_bridge.bridge_once(tmp_path, "http://127.0.0.1:8001", "token", last_sent)
    assert len(posted) == 1

    next_sample = {**SAMPLE, "step": 2, "observed_at": 101.0}
    (tmp_path / "trl-rank-0.json").write_text(json.dumps(next_sample), encoding="utf-8")
    metrics_bridge.bridge_once(tmp_path, "http://127.0.0.1:8001", "token", last_sent)
    assert len(posted) == 2
    assert posted[1][2] == next_sample


def test_bridge_once_skips_unreadable_or_malformed_files(tmp_path: Path, monkeypatch) -> None:
    posted = []
    monkeypatch.setattr(metrics_bridge, "_post", lambda endpoint, token, sample: posted.append(sample) or 201)
    (tmp_path / "trl-rank-0.json").write_text("not json", encoding="utf-8")

    metrics_bridge.bridge_once(tmp_path, "http://127.0.0.1:8001", "token", {})

    assert posted == []


def test_bridge_once_reports_post_failures_without_raising(tmp_path: Path, monkeypatch, capsys) -> None:
    def failing_post(endpoint, token, sample):
        raise OSError("connection refused")

    monkeypatch.setattr(metrics_bridge, "_post", failing_post)
    (tmp_path / "trl-rank-0.json").write_text(json.dumps(SAMPLE), encoding="utf-8")

    last_sent = metrics_bridge.bridge_once(tmp_path, "http://127.0.0.1:8001", "token", {})

    assert last_sent == {}
    assert "post failed" in capsys.readouterr().out
