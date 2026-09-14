from pathlib import Path
from types import SimpleNamespace

from experiments import checkpoint_io_probe


def test_latest_iteration_and_shards(tmp_path: Path) -> None:
    (tmp_path / "iter_0000002").mkdir()
    latest = tmp_path / "iter_0000008"
    latest.mkdir()
    (latest / "__0_0.distcp").write_bytes(b"data")

    assert checkpoint_io_probe.latest_iteration(tmp_path) == latest
    assert checkpoint_io_probe.shard_files(latest) == [latest / "__0_0.distcp"]


def test_buffered_read_reports_physical_delta(tmp_path: Path) -> None:
    path = tmp_path / "shard.distcp"
    path.write_bytes(b"x" * 1024)
    samples = iter([
        {"read_operations": 1, "read_bytes": 512, "write_operations": 0, "write_bytes": 0, "busy_time_ms": 2},
        {"read_operations": 3, "read_bytes": 1536, "write_operations": 0, "write_bytes": 0, "busy_time_ms": 5},
    ])

    result = checkpoint_io_probe.buffered_read([path], lambda: next(samples))

    assert result["logical_bytes"] == 1024
    assert result["physical_read_bytes"] == 1024
    assert result["physical_to_logical_ratio"] == 1


def test_direct_read_invokes_dd_with_direct_flag(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "shard.distcp"
    path.write_bytes(b"x" * 512)
    monkeypatch.setattr(checkpoint_io_probe.shutil, "which", lambda _: "/bin/dd")
    seen = []
    samples = iter([
        {"read_operations": 1, "read_bytes": 0, "write_operations": 0, "write_bytes": 0, "busy_time_ms": 0},
        {"read_operations": 2, "read_bytes": 512, "write_operations": 0, "write_bytes": 0, "busy_time_ms": 1},
    ])

    def fake_run(command, **kwargs):
        seen.append(command)
        return SimpleNamespace(returncode=0, stderr="")

    result = checkpoint_io_probe.direct_read([path], lambda: next(samples), run=fake_run)

    assert "iflag=direct" in seen[0]
    assert result["physical_read_bytes"] == 512


def test_evict_checkpoint_covers_metadata_and_shards(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "metadata.json").write_text("{}", encoding="utf-8")
    iteration = tmp_path / "iter_1"
    iteration.mkdir()
    (iteration / "shard.distcp").write_bytes(b"abc")
    seen = []
    monkeypatch.setattr(checkpoint_io_probe, "evict", lambda files: seen.extend(files))

    result = checkpoint_io_probe.evict_checkpoint(tmp_path)

    assert result["file_count"] == 2
    assert set(seen) == {tmp_path / "metadata.json", iteration / "shard.distcp"}


def test_tree_inventory_is_generic_and_non_reading(tmp_path: Path) -> None:
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "shard.distcp").write_bytes(b"abc")

    result = checkpoint_io_probe.tree_inventory(tmp_path)

    assert result["logical_bytes"] == 3
    assert result["files"] == [{"name": "nested/shard.distcp", "bytes": 3}]
