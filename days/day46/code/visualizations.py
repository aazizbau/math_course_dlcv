"""Day 46 visualizations: distributions and law of large numbers."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .random_variables import running_mean, sample_gaussian

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_distributions() -> Path:
    rng = np.random.default_rng(0)
    gauss = rng.normal(0, 1, size=10000)
    bern = rng.binomial(1, 0.3, size=10000)

    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    axes[0].hist(gauss, bins=50, density=True, color="#4c72b0", alpha=0.8)
    axes[0].set_title("Gaussian Samples")
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(bern, bins=2, density=True, color="#c44e52", alpha=0.8)
    axes[1].set_title("Bernoulli Samples")
    axes[1].set_xticks([0, 1])
    axes[1].grid(True, alpha=0.3)

    path = OUT_DIR / "00_distributions.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_running_mean() -> Path:
    samples = sample_gaussian(4000)
    mean_curve = running_mean(samples)

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(mean_curve, color="#55a868")
    ax.axhline(0.0, color="black", linewidth=1, linestyle="--")
    ax.set_title("Running Mean (Law of Large Numbers)")
    ax.set_xlabel("Samples")
    ax.set_ylabel("Mean")
    ax.grid(True, alpha=0.3)

    path = OUT_DIR / "01_running_mean.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main() -> None:
    print("Saved distributions →", plot_distributions())
    print("Saved running mean →", plot_running_mean())


if __name__ == "__main__":
    main()
