"""Day 20 visualizations: parameter counts and architecture comparison."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_param_comparison() -> Path:
    # Sample param numbers (in millions) for illustration
    names = ["VGG16", "ResNet50", "EfficientNet-B0", "ConvNeXt-T"]
    params = [138, 25, 5.3, 28]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(names, params)
    ax.set_ylabel("Parameters (millions)")
    ax.set_title("Architecture Parameter Comparison")
    for i, p in enumerate(params):
        ax.text(i, p + 1, f"{p}", ha="center")
    path = OUT_DIR / "00_param_comparison.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main() -> None:
    print("Saved parameter comparison plot →", plot_param_comparison())


if __name__ == "__main__":
    main()
