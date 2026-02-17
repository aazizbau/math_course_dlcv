"""Day 68 visualizations: sharp vs flat basins and batch-size noise effects."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).resolve().parent.parent / 'outputs'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_sharp_vs_flat() -> Path:
    x = np.linspace(-3, 3, 400)
    sharp = x**4 + 0.1 * x**2
    flat = 0.2 * x**4 + 0.1 * x**2

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(x, sharp, label='Sharp basin')
    ax.plot(x, flat, label='Flat basin')
    ax.set_title('Sharp vs Flat Minima (1D Slice)')
    ax.set_xlabel('Parameter slice')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = OUT_DIR / '00_sharp_vs_flat.png'
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_batch_noise_effect() -> Path:
    steps = np.arange(1, 101)
    theta_small = np.zeros_like(steps, dtype=float)
    theta_large = np.zeros_like(steps, dtype=float)

    t_small = 1.8
    t_large = 1.8
    rng = np.random.default_rng(0)

    for i, _ in enumerate(steps):
        grad_small = 4 * t_small**3 + 0.2 * t_small
        grad_large = 4 * t_large**3 + 0.2 * t_large

        # Small batch: higher noise, more exploration
        t_small = t_small - 0.01 * grad_small + rng.normal(0, 0.02)
        # Large batch: lower noise, steadier but less exploratory
        t_large = t_large - 0.01 * grad_large + rng.normal(0, 0.004)

        theta_small[i] = t_small
        theta_large[i] = t_large

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(steps, theta_small, label='Small batch (high noise)')
    ax.plot(steps, theta_large, label='Large batch (low noise)')
    ax.set_title('Batch Noise as Geometric Force')
    ax.set_xlabel('Step')
    ax.set_ylabel('Parameter value')
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = OUT_DIR / '01_batch_noise_effect.png'
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main() -> None:
    p1 = plot_sharp_vs_flat()
    p2 = plot_batch_noise_effect()
    print('Saved plots ->', p1, p2)


if __name__ == '__main__':
    main()
