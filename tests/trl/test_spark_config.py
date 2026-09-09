import json
from pathlib import Path

import pytest

from trl_lab.spark_config import (
    SparkConfig,
    detect_model_family,
    estimate_optimizer_bytes,
    huggingface_hub_cache,
    resolve_model_snapshot,
    validate_config,
    validate_deepspeed_config,
    validate_model_snapshot,
)


def make_inputs(tmp_path: Path, model_type: str = "qwen3_moe") -> SparkConfig:
    model = tmp_path / ("a" * 40)
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"model_type": model_type}), encoding="utf-8")
    (model / "model-00001-of-00001.safetensors").write_bytes(b"weights")
    (model / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.weight": "model-00001-of-00001.safetensors"}}),
        encoding="utf-8",
    )
    data = tmp_path / "data"
    data.mkdir()
    (data / "training.jsonl").write_text("{}\n", encoding="utf-8")
    (data / "validation.jsonl").write_text("{}\n", encoding="utf-8")
    (data / "manifest.json").write_text(json.dumps({"dataset": "public/data", "dataset_revision": "b" * 40}), encoding="utf-8")
    return SparkConfig("Qwen/Qwen3-30B-A3B", model, "a" * 40, "public/data", "b" * 40, data, tmp_path / "out", "train", "lora", "adamw", 1e-5, 5, 512, 8, 42)


def test_model_family_and_manifest_are_explicit(tmp_path: Path) -> None:
    config = make_inputs(tmp_path)
    assert detect_model_family(config.model_dir) == "qwen3_moe"
    assert validate_config(config) == "qwen3_moe"


def test_max_steps_accepts_epoch_driven_sentinel(tmp_path: Path) -> None:
    config = make_inputs(tmp_path)
    validate_config(SparkConfig(**{**config.__dict__, "max_steps": -1}))
    with pytest.raises(ValueError, match="epoch-driven"):
        validate_config(SparkConfig(**{**config.__dict__, "max_steps": 0}))


def test_standard_hub_cache_resolution_honors_environment(monkeypatch, tmp_path: Path) -> None:
    revision = "a" * 40
    hub = tmp_path / "hub"
    snapshot = hub / "models--Qwen--Qwen3-30B-A3B" / "snapshots" / revision
    snapshot.mkdir(parents=True)
    monkeypatch.setenv("HF_HUB_CACHE", str(hub))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "ignored"))
    assert huggingface_hub_cache() == hub
    assert resolve_model_snapshot("Qwen/Qwen3-30B-A3B", revision) == snapshot


def test_snapshot_preflight_reports_indexed_shards(tmp_path: Path) -> None:
    config = make_inputs(tmp_path)
    evidence = validate_model_snapshot(config.model_dir, config.model_revision, require_index=True)
    assert evidence["path_revision_matches_request"] is True
    assert evidence["indexed_tensor_count"] == 1
    assert evidence["indexed_shard_count"] == 1
    assert evidence["indexed_shard_bytes"] == len(b"weights")
    assert len(evidence["weight_index_sha256"]) == 64


def test_snapshot_preflight_rejects_missing_or_partial_shards(tmp_path: Path) -> None:
    config = make_inputs(tmp_path)
    shard = config.model_dir / "model-00001-of-00001.safetensors"
    shard.unlink()
    with pytest.raises(ValueError, match="missing 1 indexed shard"):
        validate_model_snapshot(config.model_dir, config.model_revision, require_index=True)
    shard.write_bytes(b"weights")
    (config.model_dir / "download.incomplete").write_bytes(b"partial")
    with pytest.raises(ValueError, match="incomplete downloads"):
        validate_model_snapshot(config.model_dir, config.model_revision, require_index=True)


def test_small_feature_model_can_use_a_single_unindexed_weight(tmp_path: Path) -> None:
    revision = "c" * 40
    model = tmp_path / revision
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"model_type": "qwen2"}), encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"small")
    evidence = validate_model_snapshot(model, revision)
    assert evidence["unindexed_weight_file_count"] == 1


def test_30b_config_requires_revision_named_snapshot(tmp_path: Path) -> None:
    config = make_inputs(tmp_path)
    renamed = tmp_path / "model"
    config.model_dir.rename(renamed)
    config = SparkConfig(**{**config.__dict__, "model_dir": renamed})
    with pytest.raises(ValueError, match="immutable Hugging Face snapshot"):
        validate_config(config)


def test_model_identity_mismatch_fails(tmp_path: Path) -> None:
    config = make_inputs(tmp_path, "glm4_moe_lite")
    with pytest.raises(ValueError, match="disagrees"):
        validate_config(config)


def test_full_optimizer_estimates_are_not_fit_claims() -> None:
    assert estimate_optimizer_bytes(30_000_000_000, "sgd") == 0
    assert estimate_optimizer_bytes(30_000_000_000, "adamw", fp32_states=False) == 120_000_000_000


def test_manifest_hash_and_count_claims_are_verified(tmp_path: Path) -> None:
    config = make_inputs(tmp_path)
    manifest_path = config.data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["train_count"] = 2
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="train_count"):
        validate_config(config)


def test_requested_sample_counts_must_be_positive(tmp_path: Path) -> None:
    config = make_inputs(tmp_path)
    config = SparkConfig(**{**config.__dict__, "train_samples": 0})
    with pytest.raises(ValueError, match="train_samples"):
        validate_config(config)


def test_fsdp2_is_an_explicit_backend(tmp_path: Path) -> None:
    config = make_inputs(tmp_path)
    config = SparkConfig(**{**config.__dict__, "distributed_backend": "fsdp2"})
    assert validate_config(config) == "qwen3_moe"


def test_deepspeed_requires_a_bounded_zero_profile(tmp_path: Path) -> None:
    config = make_inputs(tmp_path)
    profile = tmp_path / "zero3.json"
    profile.write_text(json.dumps({
        "bf16": {"enabled": "auto"},
        "zero_optimization": {"stage": 3},
        "train_micro_batch_size_per_gpu": "auto",
        "train_batch_size": "auto",
        "gradient_accumulation_steps": "auto",
    }), encoding="utf-8")
    config = SparkConfig(**{
        **config.__dict__,
        "distributed_backend": "deepspeed",
        "deepspeed_config": profile,
    })
    assert validate_config(config) == "qwen3_moe"
    assert validate_deepspeed_config(profile)["zero_optimization"]["stage"] == 3

    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps({
        "bf16": {"enabled": True},
        "zero_optimization": {"stage": 1},
        "train_micro_batch_size_per_gpu": "auto",
        "train_batch_size": "auto",
        "gradient_accumulation_steps": "auto",
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="stage 2 or 3"):
        validate_deepspeed_config(invalid)


def test_backend_specific_options_do_not_leak(tmp_path: Path) -> None:
    config = make_inputs(tmp_path)
    profile = tmp_path / "zero2.json"
    profile.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="valid only"):
        validate_config(SparkConfig(**{**config.__dict__, "deepspeed_config": profile}))
    with pytest.raises(ValueError, match="distributed_backend"):
        validate_config(SparkConfig(**{**config.__dict__, "distributed_backend": "unknown"}))


def test_sharded_backend_rejects_unverified_tuned_reload(tmp_path: Path) -> None:
    config = make_inputs(tmp_path)
    with pytest.raises(ValueError, match="export/reload"):
        validate_config(SparkConfig(**{
            **config.__dict__,
            "distributed_backend": "fsdp2",
            "stage": "tuned",
        }))


def test_deepspeed_backend_allows_tuned_reload(tmp_path: Path) -> None:
    config = make_inputs(tmp_path)
    (config.output_dir / "model").mkdir(parents=True)
    deepspeed_config = Path(__file__).parents[2] / "backends" / "trl" / "configs" / "deepspeed-zero3-nvme.json"

    validate_config(SparkConfig(**{
        **config.__dict__,
        "distributed_backend": "deepspeed",
        "deepspeed_config": deepspeed_config,
        "stage": "tuned",
        "finetuning_mode": "full",
    }))


@pytest.mark.parametrize("name, stage", [
    ("deepspeed-zero2.json", 2),
    ("deepspeed-zero3.json", 3),
    ("deepspeed-zero3-nvme.json", 3),
    ("deepspeed-zero3-cpu.json", 3),
])
def test_shipped_deepspeed_profiles_are_valid(name: str, stage: int) -> None:
    profile = Path(__file__).parents[2] / "backends" / "trl" / "configs" / name
    assert validate_deepspeed_config(profile)["zero_optimization"]["stage"] == stage
