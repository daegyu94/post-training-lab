"""Configuration and provenance checks for the non-quantized Spark path."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
import json
from pathlib import Path
import re
import hashlib
from typing import Any


MODEL_FAMILIES = {"qwen2": "qwen2", "qwen3_moe": "qwen3_moe", "glm4_moe_lite": "glm4_moe_lite"}
DISTRIBUTED_BACKENDS = {"ddp", "fsdp2", "deepspeed"}
QWEN3_ID = "Qwen/Qwen3-30B-A3B"
GLM_ID = "zai-org/GLM-4.7-Flash"
SHA_RE = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class SparkConfig:
    model_id: str
    model_dir: Path
    model_revision: str
    dataset_id: str
    dataset_revision: str
    data_dir: Path
    output_dir: Path
    stage: str
    finetuning_mode: str
    optimizer: str
    learning_rate: float
    max_steps: int
    max_length: int
    gradient_accumulation_steps: int
    seed: int
    distributed_backend: str = "ddp"
    deepspeed_config: Path | None = None
    train_samples: int | None = None
    eval_samples: int | None = None
    pad_to_max_length: bool = False


def huggingface_hub_cache() -> Path:
    """Return the effective Hugging Face Hub cache without importing the Hub SDK."""

    if value := os.environ.get("HF_HUB_CACHE"):
        return Path(value).expanduser()
    if value := os.environ.get("HF_HOME"):
        return Path(value).expanduser() / "hub"
    if value := os.environ.get("XDG_CACHE_HOME"):
        return Path(value).expanduser() / "huggingface" / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def resolve_model_snapshot(model_id: str, revision: str, cache_dir: Path | None = None) -> Path:
    """Resolve an immutable revision from the standard per-node Hub cache."""

    validate_revision(revision, "model_revision")
    if model_id not in {QWEN3_ID, GLM_ID}:
        raise ValueError(f"automatic cache resolution is unsupported for model ID: {model_id}")
    hub = cache_dir.expanduser() if cache_dir is not None else huggingface_hub_cache()
    repository = "models--" + model_id.replace("/", "--")
    snapshot = hub / repository / "snapshots" / revision
    if not snapshot.is_dir():
        raise ValueError(
            f"model snapshot is not present in this node's Hugging Face cache: {snapshot}"
        )
    return snapshot.absolute()


def detect_model_family(model_dir: Path) -> str:
    path = model_dir / "config.json"
    if not path.is_file():
        raise ValueError(f"model config is missing: {path}")
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid model config: {path}") from exc
    model_type = config.get("model_type")
    if model_type not in MODEL_FAMILIES:
        raise ValueError(f"unsupported Spark model_type: {model_type!r}")
    return MODEL_FAMILIES[model_type]


def validate_model_identity(model_id: str, family: str) -> None:
    expected = {QWEN3_ID: "qwen3_moe", GLM_ID: "glm4_moe_lite"}.get(model_id)
    if expected and family != expected:
        raise ValueError(f"model id {model_id!r} disagrees with model_type family {family!r}")


def validate_revision(value: str, field: str) -> str:
    if not SHA_RE.fullmatch(value):
        raise ValueError(f"{field} must be an immutable 40-hex revision")
    return value


def model_snapshot_evidence(model_dir: Path, requested_revision: str | None = None) -> dict[str, Any]:
    """Report the files and immutable cache revision observed on this node."""

    indexes = list(model_dir.glob("*.safetensors.index.json"))
    weights = list(model_dir.glob("*.safetensors"))
    config_path = model_dir / "config.json"
    path_revision = model_dir.name if SHA_RE.fullmatch(model_dir.name) else None
    index_path = indexes[0] if len(indexes) == 1 else None
    index_sha256 = hashlib.sha256(index_path.read_bytes()).hexdigest() if index_path else None
    indexed_tensor_count = None
    indexed_shard_count = None
    if index_path:
        try:
            weight_map = json.loads(index_path.read_text(encoding="utf-8")).get("weight_map")
        except (json.JSONDecodeError, AttributeError):
            weight_map = None
        if isinstance(weight_map, dict) and all(isinstance(value, str) for value in weight_map.values()):
            indexed_tensor_count = len(weight_map)
            indexed_shard_count = len(set(weight_map.values()))
    evidence = {
        "config_json_present": (model_dir / "config.json").is_file(),
        "config_json_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest() if config_path.is_file() else None,
        "weight_index_present": bool(indexes),
        "weight_index_sha256": index_sha256,
        "indexed_tensor_count": indexed_tensor_count,
        "indexed_shard_count": indexed_shard_count,
        "weight_file_count": len(weights),
        "snapshot_path_revision": path_revision,
        "verification": "complete local files observed",
    }
    if requested_revision is not None:
        evidence["requested_revision"] = requested_revision
        evidence["path_revision_matches_request"] = path_revision == requested_revision
    return evidence


def validate_model_snapshot(
    model_dir: Path, revision: str, *, require_index: bool = False
) -> dict[str, Any]:
    """Fail before model load when an indexed safetensors snapshot is incomplete."""

    if not model_dir.is_dir():
        raise ValueError(f"model snapshot directory is missing: {model_dir}")
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise ValueError(f"model config is missing: {config_path}")
    incomplete = sorted(model_dir.rglob("*.incomplete"))
    if incomplete:
        raise ValueError(f"model snapshot contains incomplete downloads: {incomplete[0]}")
    indexes = sorted(model_dir.glob("*.safetensors.index.json"))
    if len(indexes) > 1 or (require_index and len(indexes) != 1):
        raise ValueError(
            f"model snapshot must contain exactly one safetensors index; found {len(indexes)}"
        )
    if not indexes:
        weights = sorted(model_dir.glob("*.safetensors"))
        if not weights:
            raise ValueError("model snapshot has no safetensors weights")
        empty = [path for path in weights if path.stat().st_size == 0]
        if empty:
            raise ValueError(f"model snapshot contains {len(empty)} empty shard(s): {empty[0].name}")
        evidence = model_snapshot_evidence(model_dir, revision)
        evidence.update(
            indexed_tensor_count=None,
            indexed_shard_count=None,
            indexed_shard_bytes=None,
            unindexed_weight_file_count=len(weights),
            unindexed_weight_bytes=sum(path.stat().st_size for path in weights),
        )
        return evidence
    try:
        index = json.loads(indexes[0].read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid model weight index: {indexes[0]}") from exc
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"model weight index has no weight_map: {indexes[0]}")
    shard_names = sorted(set(weight_map.values()))
    if not all(
        isinstance(name, str)
        and name.endswith(".safetensors")
        and Path(name).name == name
        for name in shard_names
    ):
        raise ValueError("model weight index contains an invalid shard name")
    missing = []
    empty = []
    for name in shard_names:
        shard = model_dir / name
        if not shard.is_file():
            missing.append(name)
        elif shard.stat().st_size == 0:
            empty.append(name)
    if missing:
        raise ValueError(f"model snapshot is missing {len(missing)} indexed shard(s): {missing[0]}")
    if empty:
        raise ValueError(f"model snapshot contains {len(empty)} empty shard(s): {empty[0]}")
    evidence = model_snapshot_evidence(model_dir, revision)
    evidence.update(
        weight_index_sha256=hashlib.sha256(indexes[0].read_bytes()).hexdigest(),
        indexed_tensor_count=len(weight_map),
        indexed_shard_count=len(shard_names),
        indexed_shard_bytes=sum((model_dir / name).stat().st_size for name in shard_names),
    )
    return evidence


def validate_dataset_manifest(data_dir: Path, dataset_id: str, revision: str) -> dict[str, Any]:
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"dataset manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dataset") != dataset_id:
        raise ValueError("dataset ID does not match manifest")
    if manifest.get("dataset_revision") != revision:
        raise ValueError("dataset revision does not match manifest")
    split_specs = (("training", "training.jsonl", "train_count"), ("validation", "validation.jsonl", "eval_count"))
    files = manifest.get("files", {})
    for split, name, count_field in split_specs:
        path = data_dir / name
        if not path.is_file():
            raise ValueError(f"dataset split is missing: {path}")
        actual_count = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
        if count_field in manifest and manifest[count_field] != actual_count:
            raise ValueError(f"manifest {count_field} does not match {name}")
        declared = files.get(split, {}) if isinstance(files, dict) else {}
        if isinstance(declared, dict) and "sha256" in declared:
            actual_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
            if declared["sha256"] != actual_sha256:
                raise ValueError(f"manifest SHA-256 does not match {name}")
    return manifest


def validate_config(config: SparkConfig) -> str:
    if config.stage not in {"base", "train", "tuned"}:
        raise ValueError(f"unsupported stage: {config.stage}")
    if config.finetuning_mode not in {"lora", "full"}:
        raise ValueError("finetuning_mode must be lora or full")
    if config.optimizer not in {"adamw", "sgd"}:
        raise ValueError("optimizer must be adamw or sgd")
    if (config.max_steps < 1 and config.max_steps != -1) or config.max_length < 1 or config.gradient_accumulation_steps < 1:
        raise ValueError("steps, sequence length, and gradient accumulation must be positive (-1 means epoch-driven)")
    if config.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if config.train_samples is not None and config.train_samples < 1:
        raise ValueError("train_samples must be positive when specified")
    if config.eval_samples is not None and config.eval_samples < 1:
        raise ValueError("eval_samples must be positive when specified")
    if config.distributed_backend not in DISTRIBUTED_BACKENDS:
        raise ValueError(
            "distributed_backend must be one of ddp, fsdp2, or deepspeed"
        )
    if config.distributed_backend == "deepspeed":
        deepspeed = validate_deepspeed_config(config.deepspeed_config)
        zero = deepspeed["zero_optimization"]
        if config.finetuning_mode == "lora" and any(
            zero.get(name, {}).get("device") == "nvme"
            for name in ("offload_param", "offload_optimizer")
        ):
            raise ValueError("LoRA with DeepSpeed NVMe offload is unsupported; use DDP or full fine-tuning")
    elif config.deepspeed_config is not None:
        raise ValueError("deepspeed_config is valid only with distributed_backend=deepspeed")
    if config.distributed_backend == "fsdp2" and config.stage == "tuned":
        raise ValueError(
            "tuned stage is disabled for fsdp2 until export/reload is verified"
        )
    validate_revision(config.model_revision, "model_revision")
    validate_revision(config.dataset_revision, "dataset_revision")
    is_30b_target = config.model_id in {QWEN3_ID, GLM_ID}
    snapshot = validate_model_snapshot(
        config.model_dir, config.model_revision, require_index=is_30b_target
    )
    if is_30b_target and not snapshot["path_revision_matches_request"]:
        raise ValueError(
            "30B model_dir must be the immutable Hugging Face snapshot directory "
            f"named by model_revision: {config.model_revision}"
        )
    family = detect_model_family(config.model_dir)
    validate_model_identity(config.model_id, family)
    validate_dataset_manifest(config.data_dir, config.dataset_id, config.dataset_revision)
    if config.stage == "tuned" and config.finetuning_mode == "lora":
        if not (config.output_dir / "adapter").is_dir():
            raise ValueError("tuned LoRA stage requires output_dir/adapter")
    if config.stage == "tuned" and config.finetuning_mode == "full":
        if not (config.output_dir / "model").is_dir():
            raise ValueError("tuned full stage requires output_dir/model")
    return family


def validate_deepspeed_config(path: Path | None) -> dict[str, Any]:
    """Validate the bounded ZeRO profile used by the Trainer integration."""

    if path is None:
        raise ValueError("deepspeed backend requires --deepspeed-config")
    if not path.is_file():
        raise ValueError(f"DeepSpeed config is missing: {path}")
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid DeepSpeed config: {path}") from exc
    zero = config.get("zero_optimization")
    if not isinstance(zero, dict) or zero.get("stage") not in {2, 3}:
        raise ValueError("DeepSpeed exercise supports only ZeRO stage 2 or 3")
    for field in (
        "train_micro_batch_size_per_gpu",
        "train_batch_size",
        "gradient_accumulation_steps",
    ):
        if config.get(field) != "auto":
            raise ValueError(f"DeepSpeed {field} must be 'auto' to match SFTConfig")
    bf16 = config.get("bf16")
    if not isinstance(bf16, dict) or bf16.get("enabled") not in {True, "auto"}:
        raise ValueError("DeepSpeed bf16.enabled must be true or 'auto'")
    return config


def estimate_optimizer_bytes(parameters: int, optimizer: str, *, fp32_states: bool = True) -> int:
    """Return optimizer-state bytes only, for two AdamW moment tensors."""

    if parameters < 0:
        raise ValueError("parameters must be non-negative")
    if optimizer == "sgd":
        return 0
    if optimizer == "adamw":
        return parameters * (8 if fp32_states else 4)
    raise ValueError(f"unknown optimizer: {optimizer}")


def _main() -> None:
    parser = argparse.ArgumentParser(description="Resolve a Spark model from the local Hugging Face cache")
    parser.add_argument("--resolve-model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args()
    print(resolve_model_snapshot(args.resolve_model_id, args.revision, args.cache_dir))


if __name__ == "__main__":
    _main()
