"""Day 67 visualizations: conceptual double descent and scaling laws."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).resolve().parent.parent / 'outputs'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_double_descent() -> Path:
    capacity = np.linspace(1, 100, 300)
    error = 1.0 / capacity + 0.02 * (capacity - 50.0) ** 2 / 1000.0

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(capacity, error)
    ax.axvline(50, linestyle='--', color='gray', label='Interpolation threshold')
    ax.set_title('Conceptual Double Descent')
    ax.set_xlabel('Model capacity')
    ax.set_ylabel('Test error')
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = OUT_DIR / '00_double_descent.png'
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_scaling_law() -> Path:
    n = np.logspace(3, 7, 120)
    loss = 1.2 * n ** (-0.28)

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(n, loss)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title('Conceptual Scaling Law')
    ax.set_xlabel('Scale (params/data/compute)')
    ax.set_ylabel('Loss')
    ax.grid(True, which='both', alpha=0.3)

    path = OUT_DIR / '01_scaling_law.png'
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main() -> None:
    p1 = plot_double_descent()
    p2 = plot_scaling_law()
    print('Saved plots ->', p1, p2)


if __name__ == '__main__':
    main()
