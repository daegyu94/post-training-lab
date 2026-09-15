from pathlib import Path

from experiments import checkpoint_io_probe


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
