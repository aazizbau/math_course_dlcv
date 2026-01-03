"""Day 35 visualizations: embedding distances and collapse spectrum."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .embedding_geometry import detect_collapse

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_distance_comparison() -> Path:
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, size=(200, 8))
    y = x + rng.normal(0, 0.3, size=x.shape)

    l2 = np.linalg.norm(x - y, axis=1)
    cos = np.sum(x * y, axis=1) / (
        np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1) + 1e-6
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(l2, cos, alpha=0.6)
    ax.set_title("L2 Distance vs Cosine Similarity")
    ax.set_xlabel("L2 distance")
    ax.set_ylabel("Cosine similarity")
    ax.grid(True, alpha=0.3)

    path = OUT_DIR / "00_distance_vs_cosine.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_collapse_spectrum() -> Path:
    rng = np.random.default_rng(2)
    embeddings = rng.normal(0, 1, size=(200, 32))
    eigvals = detect_collapse(embeddings)

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(eigvals, marker="o")
    ax.set_title("Covariance Eigenvalues")
    ax.set_xlabel("Index")
    ax.set_ylabel("Eigenvalue")
    ax.grid(True, alpha=0.3)

    path = OUT_DIR / "01_covariance_eigenvalues.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main() -> None:
    print("Saved distance vs cosine →", plot_distance_comparison())
    print("Saved covariance spectrum →", plot_collapse_spectrum())


if __name__ == "__main__":
    main()
