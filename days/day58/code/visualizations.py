"""Day 58 visualizations: input vs compressed representation."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_compression() -> Path:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((500, 2))
    Y = (X[:, 0] + X[:, 1] > 0).astype(int)
    T = X @ np.array([[1.0], [1.0]])

    fig, axes = plt.subplots(1, 2, figsize=(7, 3))
    axes[0].scatter(X[:, 0], X[:, 1], c=Y, cmap='coolwarm', alpha=0.6)
    axes[0].set_title('Input Space')
    axes[0].set_xlabel('x1')
    axes[0].set_ylabel('x2')

    axes[1].scatter(T[:, 0], np.zeros_like(T[:, 0]), c=Y, cmap='coolwarm', alpha=0.6)
    axes[1].set_title('Compressed Representation')
    axes[1].set_xlabel('t')
    axes[1].set_yticks([])

    path = OUT_DIR / "00_compressed_representation.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_information_tradeoff() -> Path:
    beta = np.linspace(0.1, 2.0, 30)
    i_tx = np.exp(-beta) + 0.05
    i_ty = 1 - np.exp(-beta) * 0.8

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(i_tx, i_ty, marker='o')
    ax.set_xlabel("I(T;X)")
    ax.set_ylabel("I(T;Y)")
    ax.set_title("Information Bottleneck Tradeoff")
    ax.grid(True, alpha=0.3)

    path = OUT_DIR / "01_ib_tradeoff.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main() -> None:
    path1 = plot_compression()
    path2 = plot_information_tradeoff()
    print("Saved plots →", path1, path2)


if __name__ == "__main__":
    main()
