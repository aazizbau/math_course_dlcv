"""Day 61 visualizations: view alignment and negative separation."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_view_alignment() -> Path:
    z1 = np.array([0.8, 0.1])
    z2 = np.array([0.75, 0.15])
    z3 = np.array([-0.6, 0.2])

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter([z1[0]], [z1[1]], c='green', label='View 1')
    ax.scatter([z2[0]], [z2[1]], c='blue', label='View 2')
    ax.scatter([z3[0]], [z3[1]], c='red', label='Negative')
    ax.plot([z1[0], z2[0]], [z1[1], z2[1]], 'g--')
    ax.plot([z1[0], z3[0]], [z1[1], z3[1]], 'r--')
    ax.set_title("SSL Alignment")
    ax.set_aspect('equal')
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = OUT_DIR / "00_ssl_alignment.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_similarity_distribution() -> Path:
    rng = np.random.default_rng(0)
    positives = rng.normal(0.8, 0.05, size=200)
    negatives = rng.normal(0.2, 0.05, size=200)

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist(positives, bins=20, alpha=0.6, label='Positive pairs')
    ax.hist(negatives, bins=20, alpha=0.6, label='Negative pairs')
    ax.set_title("Similarity Distribution")
    ax.legend()

    path = OUT_DIR / "01_similarity_distribution.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main() -> None:
    path1 = plot_view_alignment()
    path2 = plot_similarity_distribution()
    print("Saved plots →", path1, path2)


if __name__ == "__main__":
    main()
