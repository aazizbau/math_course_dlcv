"""Day 62 visualizations: collapse and geometry comparisons."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_collapse_variance() -> Path:
    rng = np.random.default_rng(0)
    Z_good = rng.normal(size=(200, 64))
    Z_bad = np.ones((200, 64))

    var_good = Z_good.var(axis=0)
    var_bad = Z_bad.var(axis=0)

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(var_good, label='Healthy embedding variance')
    ax.plot(var_bad, label='Collapsed variance')
    ax.set_title('Embedding Variance per Dimension')
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Variance')
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = OUT_DIR / '00_collapse_variance.png'
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_contrastive_vs_noncontrastive() -> Path:
    rng = np.random.default_rng(1)
    # Contrastive: more globally spread
    contrastive = rng.multivariate_normal([0, 0], [[1.6, 0.0], [0.0, 0.8]], size=200)
    # Non-contrastive: tighter local alignment
    noncontrastive = rng.multivariate_normal([0, 0], [[0.7, 0.0], [0.0, 0.4]], size=200)

    fig, axes = plt.subplots(1, 2, figsize=(7, 3))
    axes[0].scatter(contrastive[:, 0], contrastive[:, 1], s=8, alpha=0.6)
    axes[0].set_title('Contrastive: spread geometry')
    axes[0].grid(True, alpha=0.2)

    axes[1].scatter(noncontrastive[:, 0], noncontrastive[:, 1], s=8, alpha=0.6)
    axes[1].set_title('Non-contrastive: local invariance')
    axes[1].grid(True, alpha=0.2)

    path = OUT_DIR / '01_ssl_geometry_compare.png'
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main() -> None:
    path1 = plot_collapse_variance()
    path2 = plot_contrastive_vs_noncontrastive()
    print('Saved plots ->', path1, path2)


if __name__ == '__main__':
    main()
