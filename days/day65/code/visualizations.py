"""Day 65 visualizations: in-vs-ood geometry and energy scores."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).resolve().parent.parent / 'outputs'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_in_vs_ood() -> Path:
    rng = np.random.default_rng(0)
    in_data = rng.normal(0, 1, (250, 2))
    ood_data = rng.normal(5, 1, (60, 2))

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(in_data[:, 0], in_data[:, 1], alpha=0.5, label='In-distribution', s=14)
    ax.scatter(ood_data[:, 0], ood_data[:, 1], alpha=0.8, label='OOD', s=18)
    ax.set_title('In vs OOD in Feature Space')
    ax.legend()
    ax.grid(True, alpha=0.2)

    path = OUT_DIR / '00_in_vs_ood.png'
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_energy_hist() -> Path:
    rng = np.random.default_rng(1)
    logits_in = rng.normal(2.0, 0.7, (600, 4))
    logits_ood = rng.normal(0.1, 0.6, (600, 4))

    e_in = -np.log(np.sum(np.exp(logits_in), axis=1))
    e_ood = -np.log(np.sum(np.exp(logits_ood), axis=1))

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist(e_in, bins=30, alpha=0.6, label='In-distribution')
    ax.hist(e_ood, bins=30, alpha=0.6, label='OOD')
    ax.set_title('Energy Score Distribution')
    ax.set_xlabel('Energy')
    ax.set_ylabel('Count')
    ax.legend()

    path = OUT_DIR / '01_energy_hist.png'
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main() -> None:
    p1 = plot_in_vs_ood()
    p2 = plot_energy_hist()
    print('Saved plots ->', p1, p2)


if __name__ == '__main__':
    main()
