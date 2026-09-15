"""Render the tracked 30B measurement tables without re-running experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs/experiments/measured-results.md"
BLUE, ORANGE = "#31688e", "#d87828"


def table(heading: str) -> list[list[str]]:
    text = DOC.read_text(encoding="utf-8")
    marker = f"## {heading}\n"
    if text.count(marker) != 1:
        raise ValueError(f"expected one heading: {heading}")
    section = re.split(r"\n## ", text.split(marker, 1)[1], maxsplit=1)[0]
    lines = re.findall(r"^\|.*\|$", section, flags=re.MULTILINE)
    if len(lines) < 3:
        raise ValueError(f"missing table after: {heading}")
    return [[cell.strip() for cell in line.strip("|").split("|")] for line in lines[2:]]


def save(figure, output: Path, name: str) -> None:
    figure.tight_layout(w_pad=3)
    target = output / name
    figure.savefig(target, format="svg", metadata={"Date": None})
    plt.close(figure)
    target.write_text(
        "\n".join(line.rstrip() for line in target.read_text().splitlines()) + "\n",
        encoding="utf-8",
    )


def plot_memory(output: Path) -> None:
    sequence = table("Sequence memory")
    recompute = table("Recompute")
    assert len(sequence) == len(recompute) == 4

    figure, axis = plt.subplots(figsize=(5.4, 4.2))
    x = range(2)
    for offset, model, color in ((-0.18, "Qwen", BLUE), (0.18, "GLM", ORANGE)):
        rows = [row for row in sequence if row[0] == model]
        axis.bar(
            [value + offset for value in x],
            [float(row[2]) for row in rows],
            0.36,
            color=color,
            label=model,
        )
    axis.set(title="Sequence memory", xlabel="Sequence length", ylabel="Peak allocated (GiB)")
    axis.set_xticks(list(x), ("4096", "8192"))
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    save(figure, output, "measured-sequence-memory.svg")

    figure, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    lengths = (2048, 4096)
    x = range(len(lengths))
    for offset, mode, color in ((-0.18, "full", BLUE), (0.18, "selective", ORANGE)):
        rows = [row for row in recompute if row[1] == mode]
        axes[0].bar([value + offset for value in x], [float(row[2]) for row in rows], 0.36, label=mode, color=color)
        axes[1].bar([value + offset for value in x], [float(row[3]) for row in rows], 0.36, label=mode, color=color)
    axes[0].set(title="Recompute step time", ylabel="Steady step (ms)")
    axes[1].set(title="Recompute memory", ylabel="Peak allocated (GiB)")
    for axis in axes:
        axis.set_xticks(list(x), [str(value) for value in lengths])
        axis.set_xlabel("Sequence length")
        axis.legend()
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    save(figure, output, "measured-recompute.svg")


def plot_checkpoint(output: Path) -> None:
    checkpoint = table("Async checkpoint scaling")
    ratios = table("LoRA ratio")
    assert [row[0] for row in checkpoint] == ["8", "8", "251", "251"]
    assert [int(row[1]) for row in ratios] == [25, 126, 251]

    figure, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    payloads = ("rank 8\n73 MB/save", "rank 251\n2.25 GB/save")
    x = range(2)
    for axis, column, title in (
        (axes[0], 6, "Direct checkpoint wait"),
        (axes[1], 7, "100-step time after model ready"),
    ):
        for offset, mode, color in ((-0.18, "sync", BLUE), (0.18, "async", ORANGE)):
            rows = [row for row in checkpoint if row[2] == mode]
            bars = axis.bar(
                [value + offset for value in x],
                [float(row[column]) for row in rows],
                0.36,
                label=mode,
                color=color,
            )
            axis.bar_label(bars, fmt="%.2f", padding=3)
        axis.set_xticks(list(x), payloads)
        axis.set(title=title, ylabel="Seconds")
        axis.legend()
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    save(figure, output, "measured-async-checkpoint.svg")

    figure, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    ratio_labels = [row[0] for row in ratios]
    for axis, column, title, unit, color in (
        (axes[0], 3, "Checkpoint size by LoRA ratio", "MB", BLUE),
        (axes[1], 4, "Save time by LoRA ratio", "Seconds", ORANGE),
    ):
        values = [float(row[column]) for row in ratios]
        bars = axis.bar(ratio_labels, values, color=color, width=0.6)
        axis.bar_label(bars, fmt="%.2f" if column == 4 else "%.1f", padding=3)
        axis.set(title=title, xlabel="Target trainable ratio", ylabel=unit)
    for axis in axes.flat:
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    save(figure, output, "measured-lora-ratio.svg")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs/figures")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "font.size": 9, "svg.hashsalt": "post-training-measured-results"}
    )
    plot_memory(args.output_dir)
    plot_checkpoint(args.output_dir)


if __name__ == "__main__":
    main()
