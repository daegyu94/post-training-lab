"""Plot the 2026-09-14 I/O experiment directly from raw manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
BLUE, ORANGE = "#31688e", "#d87828"


def load(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def single_node(root: Path) -> list[tuple[str, list[float], list[float]]]:
    runs = {
        "Qwen r=8": ["single-node-io-qwen-lora-r8-20260914"],
        "Qwen r=16": ["single-node-io-qwen-lora-r16-20260914"],
        "Qwen r=32": ["single-node-io-qwen-lora-r32-20260914"],
        "GLM r=8": ["single-node-io-glm-lora-r8-20260914"],
        "GLM r=16": ["single-node-io-glm-lora-r16-20260914"],
        "GLM r=32": [
            "single-node-io-glm-lora-r32-part1-20260914",
            "single-node-io-glm-lora-r32-part2-20260914",
        ],
    }
    result = []
    for label, directories in runs.items():
        records = [record for directory in directories
                   for record in load(root / directory / "manifest.json")["records"]]
        assert len(records) == 3, (label, len(records))
        result.append((label,
                       [record["cold"]["model_restore_seconds"] for record in records],
                       [record["warm"]["model_restore_seconds"] for record in records]))
    return result


def distributed(root: Path) -> list[tuple[str, str, float]]:
    manifests = {
        "Qwen": [
            "distributed-io-qwen-n1-v4-20260914",
            "distributed-io-qwen-phase2-n1-v5-20260914",
        ],
        "GLM": ["distributed-io-glm-n1-v2-20260914"],
    }
    result = []
    for model, directories in manifests.items():
        records = [record for directory in directories
                   for record in load(root / directory / "manifest.json")["records"]
                   if record.get("status") == "passed"]
        for record in records:
            seconds = (record["metrics"]["save_call_host_seconds_max_across_ranks"]
                       + record["metrics"]["blocking_finalization_host_seconds_max_across_ranks"])
            result.append((model, record["variant"], seconds))

        if model == "Qwen":
            dcp = load(root / "distributed-io-trl-dcp-qwen-n1-v3-20260914" / "manifest.json")
            record = dcp["records"][0]
            assert record["status"] == "passed"
            result.append((model, "trl-fsdp2-dcp",
                           record["trl_summary"]["checkpoint_save_seconds"]))
    return result


def full_sft(root: Path) -> list[tuple[str, float, float]]:
    runs = {
        "Qwen": "trl-ultrachat-fullsft-2node-20260914",
        "GLM": "trl-ultrachat-glm-fullsft-2node-retry1-20260914",
    }
    result = []
    for model, directory in runs.items():
        path = root / directory
        assert load(path / "manifest.json")["status"] == "passed"
        train = load(path / "summary-train.json")
        tuned = load(path / "summary-tuned.json")
        result.append((model, train["checkpoint_save_seconds"], tuned["model_restore_seconds"]))
    return result


def save(
    fig, output: Path, name: str, title: str, note: str, axes_shift_y: float = 0.0
) -> None:
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.text(0.5, 0.02, note, ha="center", fontsize=9, color="#444444")
    fig.tight_layout(rect=(0.02, 0.10, 0.98, 0.93))
    if axes_shift_y:
        position = fig.axes[0].get_position()
        fig.axes[0].set_position(
            (position.x0, position.y0 + axes_shift_y, position.width, position.height)
        )
    path = output / name
    fig.savefig(path, metadata={"Date": None})
    plt.close(fig)
    path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=ROOT / "results")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs/figures")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "svg.fonttype": "none", "svg.hashsalt": "io-experiment-results",
                         "figure.facecolor": "white"})

    rows = single_node(args.results_root)
    labels = [row[0] for row in rows]
    cold, warm = [[mean(row[index]) for row in rows] for index in (1, 2)]
    cold_err, warm_err = [[stdev(row[index]) for row in rows] for index in (1, 2)]
    x = list(range(len(rows)))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([value - 0.19 for value in x], cold, 0.38, yerr=cold_err, label="Cold", color=BLUE, capsize=3)
    ax.bar([value + 0.19 for value in x], warm, 0.38, yerr=warm_err, label="Warm", color=ORANGE, capsize=3)
    ax.set(xticks=x, xticklabels=labels, ylabel="Restore time (seconds)")
    ax.legend(); ax.grid(axis="y", alpha=0.2); ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    save(
        fig,
        args.output_dir,
        "single-node-model-restore.svg",
        "Single-node 30B model restore",
        "Mean +/- sample standard deviation, n=3; every restore starts a new process.",
        axes_shift_y=0.03,
    )

    rows = distributed(args.results_root)
    labels = [f"{model}\n{variant}" for model, variant, _ in rows]
    values = [value for _, _, value in rows]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    bars = ax.barh(labels, values, color=[BLUE if model == "Qwen" else ORANGE for model, _, _ in rows])
    ax.bar_label(bars, labels=[f"{value:.3f}" for value in values], padding=3)
    ax.invert_yaxis(); ax.set(xlabel="Checkpoint completion latency (seconds)", xlim=(0, max(values) * 1.18))
    ax.grid(axis="x", alpha=0.2); ax.set_axisbelow(True); ax.spines[["top", "right"]].set_visible(False)
    save(fig, args.output_dir, "distributed-checkpoint-write.svg", "Distributed checkpoint completion latency",
         "Exploratory n=1; save/enqueue plus blocking finalization. No restore or error bars.")

    rows = full_sft(args.results_root)
    labels = [row[0] for row in rows]
    save_seconds = [row[1] for row in rows]
    restore_seconds = [row[2] for row in rows]
    x = list(range(len(rows)))
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    save_bars = ax.bar([value - 0.19 for value in x], save_seconds, 0.38, label="Checkpoint save", color=BLUE)
    restore_bars = ax.bar([value + 0.19 for value in x], restore_seconds, 0.38, label="New-process restore", color=ORANGE)
    ax.bar_label(save_bars, labels=[f"{value:.1f}" for value in save_seconds], padding=3)
    ax.bar_label(restore_bars, labels=[f"{value:.1f}" for value in restore_seconds], padding=3)
    ax.set(xticks=x, xticklabels=labels, ylabel="Elapsed time (seconds)", ylim=(0, max(restore_seconds) * 1.18))
    ax.legend(); ax.grid(axis="y", alpha=0.2); ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, args.output_dir, "full-sft-checkpoint-restore.svg", "UltraChat full-SFT checkpoint lifecycle",
         "Two nodes, ZeRO-3 NVMe, one optimizer step, exploratory n=1. Both one-node runs OOM before checkpoint.")


if __name__ == "__main__":
    main()
