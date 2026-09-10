from pathlib import Path

from profiling_lab.node_agent import framework_samples, metrics


def test_metrics_uses_interval_deltas() -> None:
    before = (1.0, 100, 50, {}, {"eth0": (10, 20)})
    after = (
        3.0,
        200,
        75,
        {"MemTotal": 2097152, "MemAvailable": 1048576, "SwapTotal": 1048576, "SwapFree": 524288},
        {"eth0": (250000010, 500000020)},
    )

    result = metrics(before, after)

    assert result["cpu_utilization_percent"] == 75
    assert result["memory_used_gib"] == 1
    assert result["swap_used_gib"] == 0.5
    assert result["nic_receive_gbps"] == 1
    assert result["nic_transmit_gbps"] == 2


def test_framework_samples_skips_unchanged_and_invalid_files(tmp_path: Path) -> None:
    sample = tmp_path / "trl-rank-0.json"
    sample.write_text('{"observed_at": 1, "step": 1}', encoding="utf-8")
    (tmp_path / "bad-rank-1.json").write_text("invalid", encoding="utf-8")
    seen = {}

    assert framework_samples(tmp_path, seen) == [{"observed_at": 1, "step": 1}]
    assert framework_samples(tmp_path, seen) == []
