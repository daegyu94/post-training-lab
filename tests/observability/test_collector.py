from pathlib import Path
import sqlite3
import types

import pytest

from profiling_lab import collector


TOKEN = "x" * 32
HOST_SAMPLE = {
    "run_id": "run-1",
    "node": "spark1",
    "metrics": {
        "cpu_utilization_percent": 25.0,
        "memory_used_gib": 1.0,
        "memory_total_gib": 2.0,
        "memory_available_gib": 1.0,
        "swap_used_gib": 0.0,
        "swap_total_gib": 0.0,
        "nic_receive_gbps": 0.1,
        "nic_transmit_gbps": 0.2,
        "interval_seconds": 2.0,
    },
}


def test_collector_initializes_local_ui_and_database(tmp_path: Path, monkeypatch) -> None:
    created = {}

    def fake_server(address, handler):
        created.update(address=address, handler=handler)
        return types.SimpleNamespace()

    monkeypatch.setattr(collector, "ThreadingHTTPServer", fake_server)
    database = tmp_path / "metrics.sqlite3"

    collector.make_server(("127.0.0.1", 8001), database, TOKEN)

    assert created["address"] == ("127.0.0.1", 8001)
    assert b"Post-Training Lab" in collector.VIEWER.read_bytes()
    assert collector._web_file("/assets/charts.js")[1] == "text/javascript; charset=utf-8"
    assert collector._web_file("/api/telemetry.json")[1] == "application/json"
    assert collector._web_file("/../profiling_lab/collector.py") is None
    with sqlite3.connect(database) as db:
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert tables == {"samples", "framework_samples"}


def test_collector_validates_host_metrics() -> None:
    assert collector.validate_host_sample(HOST_SAMPLE) == HOST_SAMPLE
    invalid = {**HOST_SAMPLE, "metrics": {**HOST_SAMPLE["metrics"], "cpu_utilization_percent": 101}}
    with pytest.raises(ValueError, match="range"):
        collector.validate_host_sample(invalid)


def test_collector_validates_framework_metrics() -> None:
    sample = {
        "schema_version": 1,
        "run_id": "run-1",
        "framework": "trl",
        "node": "spark1",
        "rank": 0,
        "local_rank": 0,
        "step": 1,
        "observed_at": 1.0,
        "metrics": {"training_loss": 1.25},
        "timers": {},
    }

    assert collector.validate_framework_sample(sample) == sample
    with pytest.raises(ValueError, match="framework"):
        collector.validate_framework_sample({**sample, "framework": "unknown"})
