"""Render selected published result tables; does not run or aggregate experiments.

The Markdown tables remain the source of truth. Requires Matplotlib 3.10.8.
Run from any directory with --output-dir to preview without changing docs.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
BLUE, ORANGE = "#31688e", "#d87828"


def table(path: Path, heading: str) -> list[list[str]]:
    """Read the first pipe table following one exact heading, failing on drift."""
    text = path.read_text(encoding="utf-8")
    marker = heading + "\n"
    if text.count(marker) != 1:
        raise ValueError(f"Expected one heading {heading!r} in {path}")
    section = re.split(r"\n#{1,6} ", text.split(marker, 1)[1], maxsplit=1)[0]
    lines = re.findall(r"^\|.*\|$", section, flags=re.MULTILINE)
    if len(lines) < 3:
        raise ValueError(f"Missing result table after {heading}")
    return [[cell.strip().strip("`") for cell in line.strip("|").split("|")]
            for line in lines[2:]]


def number(cell: str) -> float:
    return float(re.search(r"\d+(?:\.\d+)?", cell.replace(",", "")).group())


def save(fig, output: Path, name: str, title: str, note: str):
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.98)
    fig.text(0.5, 0.025, note, ha="center", va="bottom", fontsize=9, color="#444444")
    fig.tight_layout(rect=(0.02, 0.16, 0.98, 0.90), w_pad=3)
    fig.savefig(output / f"{name}.svg", metadata={"Date": None})
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs/figures")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "svg.hashsalt": "post-training-results", "figure.facecolor": "white"})
    doc = ROOT / "docs/experiments.md"

    rows = table(doc, "#### Memory footprint")
    assert len(rows) == 5
    fig, ax = plt.subplots(figsize=(9, 5.2))
    labels = ["Megatron LoRA\n4096 tokens", "Megatron LoRA\n8192 tokens",
              "TRL DDP\nLoRA", "TRL FSDP2\nLoRA", "TRL ZeRO-3 NVMe\nFull + SGD"]
    cuda = [number(r[2]) for r in rows]
    marks = ax.barh(labels, cuda, color=BLUE, height=0.58)
    ax.bar_label(marks, labels=[f"{value:.1f} GB" for value in cuda], padding=4)
    ax.invert_yaxis()
    ax.set(xlabel="CUDA peak allocated (GB)", xlim=(0, max(cuda) * 1.18))
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, args.output_dir, "memory-footprint", "30B CUDA memory footprint",
         "Peak CUDA memory allocated; medians of 3 runs per condition.\n"
         "Workloads differ by framework, tuning mode, and sequence length; compare only matching conditions.")

    rows = table(doc, "### LoRA trainable-ratio가 checkpoint I/O에 미치는 영향")
    assert [int(r[1]) for r in rows] == [25, 126, 251]
    measured_x = [number(r[2]) for r in rows]
    display_x = [0.1, 0.5, 1.0]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    for ax, column, title, unit, color in zip(axes, [3, 4], ["Checkpoint size", "Save host-call time"], ["MB", "Seconds"], [BLUE, ORANGE]):
        values = [number(r[column]) for r in rows]
        ax.plot(display_x, values, "o-", color=color, linewidth=1.8, markersize=6)
        for xv, yv in zip(display_x, values):
            ax.annotate(f"{yv:,.2f}" if column == 4 else f"{yv:,.1f}", (xv, yv),
                        xytext=(0, 9), textcoords="offset points", ha="center")
        ax.set(title=title, xlabel="Trainable parameters / rank-local shard (%)",
               ylabel=unit, xlim=(0, 1.12), ylim=(0, max(values) * 1.25))
        ax.set_xticks(display_x, ["0.1", "0.5", "1.0"])
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    save(fig, args.output_dir, "lora-ratio-checkpoint", "LoRA parameter ratio and checkpoint I/O",
         "Sync only; medians of 3 measured runs per condition; LORA_DIM = 25 / 126 / 251.\n"
         "The x-axis shows target ratios; measured ratios were "
         + " / ".join(f"{value:.4f}%" for value in measured_x) + ".\n"
         "Ratios use the rank-local shard reference, not the full 30B model. Lines connect measured points.")

if __name__ == "__main__":
    main()
