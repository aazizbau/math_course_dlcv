"""Day 64 visualizations: covariate shift and embedding drift."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_covariate_shift() -> Path:
    rng = np.random.default_rng(0)
    train = rng.normal(0, 1, 1200)
    test = rng.normal(2, 1, 1200)

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist(train, bins=35, alpha=0.55, label='Train', density=True)
    ax.hist(test, bins=35, alpha=0.55, label='Test', density=True)
    ax.set_title('Covariate Shift Example')
    ax.set_xlabel('Feature value')
    ax.set_ylabel('Density')
    ax.legend()

    path = OUT_DIR / '00_covariate_shift.png'
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_embedding_drift() -> Path:
    rng = np.random.default_rng(1)
    src = rng.multivariate_normal([0, 0], [[1.0, 0.2], [0.2, 0.9]], size=300)
    tgt = rng.multivariate_normal([1.5, 0.8], [[1.1, -0.1], [-0.1, 0.8]], size=300)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(src[:, 0], src[:, 1], s=10, alpha=0.55, label='Source domain')
    ax.scatter(tgt[:, 0], tgt[:, 1], s=10, alpha=0.55, label='Target domain')
    ax.set_title('Embedding Drift Across Domains')
    ax.set_xlabel('dim-1')
    ax.set_ylabel('dim-2')
    ax.legend()
    ax.grid(True, alpha=0.25)

    path = OUT_DIR / '01_embedding_drift.png'
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main() -> None:
    p1 = plot_covariate_shift()
    p2 = plot_embedding_drift()
    print('Saved plots ->', p1, p2)


if __name__ == '__main__':
    main()
