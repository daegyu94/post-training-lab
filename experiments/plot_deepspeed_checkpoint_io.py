"""Plot the trl-zero3-nvme-checkpoint-io condition's manifest.json.

Reads the output of:
  python experiments/checkpoint_memory_30b.py --phase memory \
    --condition trl-zero3-nvme-checkpoint-io --output results/<name> --execute

and writes three PNGs: checkpoint size by file group, save/restore call time,
and checkpoint-window vs. rest-of-run device write bytes. Requires matplotlib,
which is not otherwise a dependency of this repo (pip install matplotlib).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def measured_records(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        item for item in manifest["records"]
        if item.get("status") == "passed" and not item.get("warmup") and item.get("deepspeed_checkpoint_io")
    ]


def per_run_bytes(record: dict[str, Any]) -> dict[str, int]:
    per_rank = record["deepspeed_checkpoint_io"].values()
    return {
        "metadata": sum(r["checkpoint_io"]["inventory"]["groups"]["metadata"]["logical_bytes"] for r in per_rank),
        "offloaded_tensors": sum(
            r["checkpoint_io"]["inventory"]["groups"]["offloaded_tensors"]["logical_bytes"] for r in per_rank
        ),
    }


def per_run_seconds(record: dict[str, Any], event: str) -> float | None:
    per_rank = record["deepspeed_checkpoint_io"].values()
    values = [r["timing"][event]["host_seconds"] for r in per_rank if r["timing"].get(event)]
    return max(values) if values else None


def per_run_window_bytes(record: dict[str, Any], event: str, direction: str) -> int | None:
    per_rank = record["deepspeed_checkpoint_io"].values()
    values = [
        r["windows"][event][f"device_{direction}_bytes_delta"] for r in per_rank
        if r["windows"].get(event) and not r["windows"][event]["insufficient_resolution"]
    ]
    return sum(values) if values else None


def plot(manifest_path: Path, output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = measured_records(manifest)
    if not records:
        raise SystemExit(f"no measured trl-zero3-nvme-checkpoint-io records in {manifest_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = [f"run-{item['run']:02d}" for item in records]

    sizes = [per_run_bytes(item) for item in records]
    metadata_mib = [s["metadata"] / 2**20 for s in sizes]
    offloaded_mib = [s["offloaded_tensors"] / 2**20 for s in sizes]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(runs, metadata_mib, label="metadata (torch.save, no fsync)")
    ax.bar(runs, offloaded_mib, bottom=metadata_mib, label="offloaded_tensors (optimizer NVMe swap copy)")
    ax.set_ylabel("Checkpoint size (MiB)")
    ax.set_title("DeepSpeed ZeRO-3 NVMe checkpoint size by file group")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "deepspeed-checkpoint-size.png", dpi=150)
    plt.close(fig)

    save_seconds = [per_run_seconds(item, "save") for item in records]
    restore_seconds = [per_run_seconds(item, "restore") for item in records]
    fig, ax = plt.subplots(figsize=(6, 4))
    x = range(len(runs))
    width = 0.35
    ax.bar([i - width / 2 for i in x], [v if v is not None else 0 for v in save_seconds], width, label="save")
    ax.bar([i + width / 2 for i in x], [v if v is not None else 0 for v in restore_seconds], width, label="restore")
    ax.set_xticks(list(x), runs)
    ax.set_ylabel("Host call seconds (max across ranks)")
    ax.set_title("DeepSpeed checkpoint save/restore time (sync-only; no async split)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "deepspeed-checkpoint-timing.png", dpi=150)
    plt.close(fig)

    save_window = [per_run_window_bytes(item, "save", "write") for item in records]
    whole_run = [sum(r["device_write_bytes_delta"] for r in item["resources"].values()) for item in records]
    save_window_mib = [v / 2**20 if v is not None else 0 for v in save_window]
    whole_run_mib = [v / 2**20 for v in whole_run]
    offload_only_mib = [max(w - s, 0) for w, s in zip(whole_run_mib, save_window_mib)]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(runs, save_window_mib, label="checkpoint-save window")
    ax.bar(runs, offload_only_mib, bottom=save_window_mib, label="rest of run (NVMe offload + other)")
    ax.set_ylabel("Device write bytes (MiB)")
    ax.set_title("Checkpoint-save I/O vs. rest-of-run NVMe offload I/O")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "deepspeed-checkpoint-vs-offload-io.png", dpi=150)
    plt.close(fig)

    print(f"wrote 3 PNGs to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    plot(args.manifest, args.output_dir)


if __name__ == "__main__":
    main()
