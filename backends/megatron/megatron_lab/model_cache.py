"""Resolve and validate node-local Hugging Face model snapshots."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path


IMMUTABLE_REVISION = re.compile(r"[0-9a-f]{40}")


def require_immutable_revision(value: str, *, name: str = "revision") -> str:
    """Require a full lowercase Git commit used as an immutable Hub revision."""

    if not IMMUTABLE_REVISION.fullmatch(value):
        raise ValueError(f"{name} must be a full 40-character lowercase commit SHA")
    return value


def default_hf_home() -> Path:
    """Return the Hugging Face cache root using the library's standard precedence."""

    explicit = os.environ.get("HF_HOME")
    if explicit:
        return Path(explicit).expanduser()
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache:
        return Path(xdg_cache).expanduser() / "huggingface"
    return Path.home() / ".cache" / "huggingface"


def snapshot_path(model_id: str, revision: str, *, hf_home: Path | None = None) -> Path:
    """Map a Hub model ID and pinned revision to its node-local cache snapshot."""

    require_immutable_revision(revision, name="model revision")
    parts = model_id.split("/")
    if len(parts) != 2 or not all(parts):
        raise ValueError("model id must have the form organization/name")
    cache_root = hf_home if hf_home is not None else default_hf_home()
    return cache_root / "hub" / f"models--{parts[0]}--{parts[1]}" / "snapshots" / revision


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_snapshot(model_dir: Path, revision: str) -> dict[str, object]:
    """Validate metadata and every weight referenced by a local HF snapshot."""

    require_immutable_revision(revision, name="model revision")
    if not model_dir.is_dir():
        raise ValueError(f"model snapshot directory is missing: {model_dir}")
    config_path = model_dir / "config.json"
    index_path = model_dir / "model.safetensors.index.json"
    if not config_path.is_file():
        raise ValueError(f"model config is missing: {config_path}")
    if index_path.is_file():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid safetensors index: {index_path}") from exc
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"safetensors index has no weight_map: {index_path}")
        shards = sorted({str(value) for value in weight_map.values()})
        tensor_count = len(weight_map)
        index_sha256: str | None = _sha256(index_path)
    else:
        single_weight = model_dir / "model.safetensors"
        if not single_weight.is_file():
            raise ValueError(
                f"model weights are missing; expected {index_path} or {single_weight}"
            )
        shards = [single_weight.name]
        tensor_count = None
        index_sha256 = None
    unsafe = [name for name in shards if Path(name).name != name]
    if unsafe:
        raise ValueError(f"safetensors index contains unsafe shard paths: {unsafe[:3]}")
    missing = [name for name in shards if not (model_dir / name).is_file()]
    if missing:
        raise ValueError(
            f"model snapshot is incomplete; missing {len(missing)} shard(s): {missing[:3]}"
        )
    return {
        "path": str(model_dir.resolve()),
        "revision": revision,
        "config_sha256": _sha256(config_path),
        "index_sha256": index_sha256,
        "shard_count": len(shards),
        "tensor_count": tensor_count,
        "weight_bytes": sum((model_dir / name).stat().st_size for name in shards),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    model_dir = args.model_dir or snapshot_path(args.model_id, args.revision)
    manifest = validate_snapshot(model_dir, args.revision)
    if args.json:
        print(json.dumps(manifest, sort_keys=True))
    else:
        print(manifest["path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
