"""Day 69 visualizations: overfitting and gap trend."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).resolve().parent.parent / 'outputs'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_overfitting_example() -> Path:
    rng = np.random.default_rng(0)
    x = np.linspace(-1, 1, 20)
    y = x**2 + rng.normal(0, 0.1, size=x.shape)

    coeffs = np.polyfit(x, y, 15)
    poly = np.poly1d(coeffs)
    x_test = np.linspace(-1, 1, 200)

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.scatter(x, y, label='Train samples', s=20)
    ax.plot(x_test, poly(x_test), label='High-capacity fit')
    ax.set_title('Optimization vs Generalization')
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = OUT_DIR / '00_overfitting_curve.png'
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_gap_vs_capacity() -> Path:
    cap = np.linspace(1, 100, 160)
    train = 1.0 / (cap + 2)
    test = train + 0.05 + 0.13 * np.exp(-((cap - 28) ** 2) / 280) - 0.00025 * cap
    gap = test - train

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(cap, gap)
    ax.set_title('Conceptual Generalization Gap vs Capacity')
    ax.set_xlabel('Capacity')
    ax.set_ylabel('Test - Train')
    ax.grid(True, alpha=0.3)

    path = OUT_DIR / '01_generalization_gap.png'
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main() -> None:
    p1 = plot_overfitting_example()
    p2 = plot_gap_vs_capacity()
    print('Saved plots ->', p1, p2)


if __name__ == '__main__':
    main()
