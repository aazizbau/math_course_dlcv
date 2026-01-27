"""Day 49 visualizations: entropy and cross-entropy curves."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_entropy_curve() -> Path:
    ps = np.linspace(1e-4, 1 - 1e-4, 400)
    h = -(ps * np.log(ps) + (1 - ps) * np.log(1 - ps))

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(ps, h)
    ax.set_xlabel("Probability p")
    ax.set_ylabel("Entropy")
    ax.set_title("Bernoulli Entropy")
    ax.grid(True, alpha=0.3)

    path = OUT_DIR / "00_entropy_curve.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_cross_entropy_curve() -> Path:
    ps = np.linspace(1e-4, 1 - 1e-4, 400)
    ce_y1 = -np.log(ps)
    ce_y0 = -np.log(1 - ps)

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(ps, ce_y1, label="CE (y=1)")
    ax.plot(ps, ce_y0, label="CE (y=0)")
    ax.set_xlabel("Predicted probability p")
    ax.set_ylabel("Cross-Entropy")
    ax.set_title("Binary Cross-Entropy Curves")
    ax.grid(True, alpha=0.3)
    ax.legend()

    path = OUT_DIR / "01_cross_entropy_curve.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main() -> None:
    path1 = plot_entropy_curve()
    path2 = plot_cross_entropy_curve()
    print("Saved plots →", path1, path2)


if __name__ == "__main__":
    main()
