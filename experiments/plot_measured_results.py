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
    assert len(sequence) == 4 and len(recompute) == 8

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

    figure, axes = plt.subplots(2, 2, figsize=(9, 7.5))
    lengths = (2048, 4096)
    x = range(len(lengths))
    for model_index, model in enumerate(("Qwen", "GLM")):
        for offset, mode, color in ((-0.18, "full", BLUE), (0.18, "selective", ORANGE)):
            rows = [row for row in recompute if row[0] == model and row[2] == mode]
            axes[model_index, 0].bar([value + offset for value in x], [float(row[3]) for row in rows], 0.36, label=mode, color=color)
            axes[model_index, 1].bar([value + offset for value in x], [float(row[4]) for row in rows], 0.36, label=mode, color=color)
        axes[model_index, 0].set(title=f"{model}: step time", ylabel="Steady step (ms)")
        axes[model_index, 1].set(title=f"{model}: memory", ylabel="Peak allocated (GiB)")
    for axis in axes.flat:
        axis.set_xticks(list(x), [str(value) for value in lengths])
        axis.set_xlabel("Sequence length")
        axis.legend()
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    save(figure, output, "measured-recompute.svg")


def plot_checkpoint(output: Path) -> None:
    checkpoint = table("Async checkpoint scaling")
    ratios = table("LoRA ratio")
    assert [(row[0], row[1]) for row in checkpoint] == [
        (model, rank) for model in ("Qwen", "GLM") for rank in ("8", "8", "251", "251")
    ]
    assert [(row[0], int(row[2])) for row in ratios] == [
        (model, rank) for model in ("Qwen", "GLM") for rank in (25, 126, 251)
    ]

    figure, axes = plt.subplots(2, 2, figsize=(9, 7.5))
    x = range(2)
    for model_index, model in enumerate(("Qwen", "GLM")):
        sync_rows = [row for row in checkpoint if row[0] == model and row[3] == "sync"]
        payloads = []
        for row in sync_rows:
            per_save_gb = float(row[4]) / 10
            size = f"{per_save_gb * 1000:.0f} MB" if per_save_gb < 1 else f"{per_save_gb:.2f} GB"
            payloads.append(f"rank {row[1]}\n{size}/save")
        for column, title, axis in (
            (7, "Direct checkpoint wait", axes[model_index, 0]),
            (8, "100-step time after model ready", axes[model_index, 1]),
        ):
            for offset, mode, color in ((-0.18, "sync", BLUE), (0.18, "async", ORANGE)):
                rows = [row for row in checkpoint if row[0] == model and row[3] == mode]
                bars = axis.bar(
                    [value + offset for value in x],
                    [float(row[column]) for row in rows],
                    0.36,
                    label=mode,
                    color=color,
                )
                axis.bar_label(bars, fmt="%.2f", padding=3)
            axis.set_xticks(list(x), payloads)
            axis.set(title=f"{model}: {title}", ylabel="Seconds")
            axis.legend()
    for axis in axes.flat:
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    save(figure, output, "measured-async-checkpoint.svg")

    figure, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    ratio_labels = [row[2] for row in ratios if row[0] == "Qwen"]
    x = range(len(ratio_labels))
    for axis, column, title, unit in (
        (axes[0], 4, "Checkpoint size by LoRA rank", "MB"),
        (axes[1], 5, "Save time by LoRA rank", "Seconds"),
    ):
        for offset, model, color in ((-0.18, "Qwen", BLUE), (0.18, "GLM", ORANGE)):
            rows = [row for row in ratios if row[0] == model]
            bars = axis.bar([value + offset for value in x], [float(row[column]) for row in rows], 0.36, color=color, label=model)
            axis.bar_label(bars, fmt="%.2f" if column == 5 else "%.1f", padding=3)
        axis.set_xticks(list(x), ratio_labels)
        axis.set(title=title, xlabel="LoRA rank", ylabel=unit)
        axis.legend()
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
