"""Day 55 visualizations: reliability diagram and confidence hist."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_reliability() -> Path:
    conf = np.array([0.2, 0.4, 0.6, 0.8, 0.9])
    acc = np.array([0.1, 0.35, 0.55, 0.75, 0.85])

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(conf, acc, marker='o')
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
    ax.set_xlabel("Predicted confidence")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title("Reliability Diagram")
    ax.grid(True, alpha=0.3)

    path = OUT_DIR / "00_reliability_diagram.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_confidence_hist() -> Path:
    rng = np.random.default_rng(0)
    conf = rng.beta(5, 2, size=500)

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist(conf, bins=20, alpha=0.7)
    ax.set_title("Confidence Distribution")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Count")

    path = OUT_DIR / "01_confidence_hist.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main() -> None:
    path1 = plot_reliability()
    path2 = plot_confidence_hist()
    print("Saved plots →", path1, path2)


if __name__ == "__main__":
    main()
