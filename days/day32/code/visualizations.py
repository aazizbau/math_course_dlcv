"""Day 32 visualizations: PCA directions and reconstruction."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .pca_demo import explained_variance_ratio, pca_fit, project, reconstruct

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _make_data(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, size=(300, 2))
    X[:, 1] = 0.7 * X[:, 0] + 0.3 * rng.normal(0, 1, size=300)
    return X


def plot_pca_directions() -> Path:
    X = _make_data()
    mean, eigvals, eigvecs = pca_fit(X)
    Xc = X - mean

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(Xc[:, 0], Xc[:, 1], alpha=0.3, s=10)
    for i in range(2):
        vec = eigvecs[:, i] * np.sqrt(eigvals[i]) * 3
        ax.plot([0, vec[0]], [0, vec[1]], linewidth=3)
    ax.set_title("PCA Directions")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    path = OUT_DIR / "00_pca_directions.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_reconstruction(k: int = 1) -> Path:
    X = _make_data(1)
    mean, eigvals, eigvecs = pca_fit(X)
    Xc = X - mean
    Z = project(Xc, eigvecs, k)
    Xr = reconstruct(Z, eigvecs, k)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(Xc[:, 0], Xc[:, 1], alpha=0.2, s=10, label="Original")
    ax.scatter(Xr[:, 0], Xr[:, 1], alpha=0.5, s=10, label="Reconstruction")
    ax.set_title(f"PCA Reconstruction (k={k})")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()

    path = OUT_DIR / "01_pca_reconstruction.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_explained_variance() -> Path:
    X = _make_data(2)
    _, eigvals, _ = pca_fit(X)
    evr = explained_variance_ratio(eigvals)

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(range(1, len(evr) + 1), evr, color="#55a868")
    ax.set_title("Explained Variance Ratio")
    ax.set_xlabel("Component")
    ax.set_ylabel("Ratio")
    ax.set_ylim(0, 1)
    ax.grid(True, axis="y", alpha=0.3)

    path = OUT_DIR / "02_explained_variance.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main() -> None:
    print("Saved PCA directions →", plot_pca_directions())
    print("Saved PCA reconstruction →", plot_reconstruction())
    print("Saved explained variance →", plot_explained_variance())


if __name__ == "__main__":
    main()
