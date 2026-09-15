"""Generate the capacity chart used by docs/experiments/scaling-estimates.md."""

from __future__ import annotations

import argparse
from pathlib import Path


GIB = 1024**3
TIB = 1024**4
ADAPTER_FRACTION = 0.001
MODEL_SIZES_B = (30.5, 100, 500, 1000)


def training_state_bytes(parameters: float, method: str) -> float:
    base_bytes = {"Full FT": 0.0, "LoRA": 2.0, "QLoRA": 0.5}[method]
    trainable_fraction = 1.0 if method == "Full FT" else ADAPTER_FRACTION
    return parameters * (base_bytes + 16.0 * trainable_fraction)


def restart_checkpoint_bytes(parameters: float, method: str) -> float:
    trainable_fraction = 1.0 if method == "Full FT" else ADAPTER_FRACTION
    return parameters * 14.0 * trainable_fraction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parents[1] / "docs/figures/scaling-estimates.svg",
    )
    args = parser.parse_args()

    import matplotlib.pyplot as plt

    plt.rcParams["svg.hashsalt"] = "post-training-lab"
    methods = ("Full FT", "LoRA", "QLoRA")
    colors = {"Full FT": "#d95f02", "LoRA": "#1b9e77", "QLoRA": "#7570b3"}
    parameters = [size * 1e9 for size in MODEL_SIZES_B]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    for method in methods:
        axes[0].plot(
            MODEL_SIZES_B,
            [training_state_bytes(value, method) / TIB for value in parameters],
            marker="o",
            color=colors[method],
            label=method,
        )
    axes[0].set_title("Training model state")
    axes[0].set_ylabel("Capacity (TiB, log scale)")
    axes[0].legend()

    axes[1].plot(
        MODEL_SIZES_B,
        [restart_checkpoint_bytes(value, "Full FT") / TIB for value in parameters],
        marker="o",
        color=colors["Full FT"],
        label="Full FT",
    )
    axes[1].plot(
        MODEL_SIZES_B,
        [restart_checkpoint_bytes(value, "LoRA") / TIB for value in parameters],
        marker="o",
        color=colors["LoRA"],
        label="LoRA / QLoRA adapter",
    )
    axes[1].set_title("Restart checkpoint (base excluded)")
    axes[1].legend()

    for axis in axes:
        axis.set_xlabel("Model parameters (billions)")
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xticks(MODEL_SIZES_B, labels=("30.5B", "100B", "500B", "1T"))
        axis.grid(True, which="both", alpha=0.25)

    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, format="svg", metadata={"Date": None})
    plt.close(figure)
    args.output.write_text(
        "\n".join(line.rstrip() for line in args.output.read_text().splitlines()) + "\n",
        encoding="utf-8",
    )

    assert round(training_state_bytes(1e12, "QLoRA") / GIB, 1) == 480.6
    assert round(restart_checkpoint_bytes(1e12, "Full FT") / TIB, 2) == 12.73


if __name__ == "__main__":
    main()
