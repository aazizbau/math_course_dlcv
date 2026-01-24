"""Day 47 visualizations: variance shrink with averaging."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .expectation_variance import variance_of_mean

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_variance_shrink() -> Path:
    batch_sizes = np.array([1, 5, 10, 20, 50, 100, 200, 500])
    variances = [variance_of_mean(n) for n in batch_sizes]

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(batch_sizes, variances, marker="o")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Variance of the Mean vs Batch Size")
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Var(mean)")
    ax.grid(True, alpha=0.3)

    path = OUT_DIR / "00_variance_shrink.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main() -> None:
    path = plot_variance_shrink()
    print("Saved variance shrink plot →", path)


if __name__ == "__main__":
    main()
