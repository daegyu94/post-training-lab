import json
from pathlib import Path

import pytest

from megatron_lab.model_cache import (
    default_hf_home,
    require_immutable_revision,
    snapshot_path,
    validate_snapshot,
)


REVISION = "a" * 40


def make_snapshot(tmp_path: Path) -> Path:
    snapshot = tmp_path / REVISION
    snapshot.mkdir()
    (snapshot / "config.json").write_text(
        json.dumps({"model_type": "qwen3_moe"}), encoding="utf-8"
    )
    (snapshot / "model-00001-of-00002.safetensors").write_bytes(b"first")
    (snapshot / "model-00002-of-00002.safetensors").write_bytes(b"second")
    (snapshot / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "layer.0": "model-00001-of-00002.safetensors",
                    "layer.1": "model-00002-of-00002.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )
    return snapshot


def test_default_hf_home_follows_standard_precedence(monkeypatch) -> None:
    monkeypatch.setenv("HF_HOME", "/cache/hf")
    monkeypatch.setenv("XDG_CACHE_HOME", "/cache/xdg")
    assert default_hf_home() == Path("/cache/hf")
    monkeypatch.delenv("HF_HOME")
    assert default_hf_home() == Path("/cache/xdg/huggingface")


def test_snapshot_path_uses_node_local_hub_layout(tmp_path: Path) -> None:
    assert snapshot_path("Qwen/Qwen3-30B-A3B", REVISION, hf_home=tmp_path) == (
        tmp_path
        / "hub"
        / "models--Qwen--Qwen3-30B-A3B"
        / "snapshots"
        / REVISION
    )


def test_validate_snapshot_records_complete_weight_set(tmp_path: Path) -> None:
    snapshot = make_snapshot(tmp_path)
    manifest = validate_snapshot(snapshot, REVISION)
    assert manifest["shard_count"] == 2
    assert manifest["tensor_count"] == 2
    assert manifest["weight_bytes"] == 11
    assert len(str(manifest["config_sha256"])) == 64


def test_validate_snapshot_rejects_missing_or_unsafe_shards(tmp_path: Path) -> None:
    snapshot = make_snapshot(tmp_path)
    (snapshot / "model-00002-of-00002.safetensors").unlink()
    with pytest.raises(ValueError, match="missing 1 shard"):
        validate_snapshot(snapshot, REVISION)

    index = snapshot / "model.safetensors.index.json"
    index.write_text(
        json.dumps({"weight_map": {"layer.0": "../outside.safetensors"}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unsafe shard paths"):
        validate_snapshot(snapshot, REVISION)


def test_revisions_must_be_immutable_commits() -> None:
    assert require_immutable_revision(REVISION) == REVISION
    with pytest.raises(ValueError, match="40-character lowercase"):
        require_immutable_revision("main")
