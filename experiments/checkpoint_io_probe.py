"""Inventory checkpoint files or evict them from the page cache."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def evict(files: list[Path]) -> None:
    if not hasattr(os, "posix_fadvise"):
        raise RuntimeError("POSIX_FADV_DONTNEED is unavailable")
    for path in files:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(descriptor, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(descriptor)


def evict_checkpoint(checkpoint_dir: Path) -> dict[str, object]:
    files = sorted(path for path in checkpoint_dir.rglob("*") if path.is_file())
    if not files:
        raise FileNotFoundError(f"no checkpoint files under {checkpoint_dir}")
    evict(files)
    return {
        "scope": "advisory checkpoint page-cache eviction",
        "file_count": len(files),
        "logical_bytes": sum(path.stat().st_size for path in files),
    }


def tree_inventory(checkpoint_dir: Path) -> dict[str, object]:
    files = sorted(path for path in checkpoint_dir.rglob("*") if path.is_file())
    if not files:
        raise FileNotFoundError(f"no checkpoint files under {checkpoint_dir}")
    return {
        "directory": str(checkpoint_dir),
        "file_count": len(files),
        "logical_bytes": sum(path.stat().st_size for path in files),
        "allocated_bytes": sum(path.stat().st_blocks * 512 for path in files),
        "files": [
            {"name": str(path.relative_to(checkpoint_dir)), "bytes": path.stat().st_size}
            for path in files
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--evict-only", action="store_true")
    action.add_argument("--inventory-only", action="store_true")
    args = parser.parse_args()
    if args.inventory_only:
        print(json.dumps(tree_inventory(args.checkpoint_dir), sort_keys=True))
        return
    print(json.dumps(evict_checkpoint(args.checkpoint_dir), sort_keys=True))


if __name__ == "__main__":
    main()
