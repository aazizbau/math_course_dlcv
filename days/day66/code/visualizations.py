"""Day 66 visualizations: shortcut failure under shift."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).resolve().parent.parent / 'outputs'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_shortcut_correlation() -> Path:
    rng = np.random.default_rng(0)
    n = 1200
    x_true = rng.standard_normal(n)
    x_spur = x_true + 0.1 * rng.standard_normal(n)
    y = (x_true > 0).astype(int)

    fig, axes = plt.subplots(1, 2, figsize=(7, 3))
    axes[0].scatter(x_true, y + 0.02 * rng.standard_normal(n), s=6, alpha=0.4)
    axes[0].set_title('Causal signal vs label')
    axes[0].set_xlabel('x_true')

    axes[1].scatter(x_spur, y + 0.02 * rng.standard_normal(n), s=6, alpha=0.4, color='tab:orange')
    axes[1].set_title('Spurious feature vs label')
    axes[1].set_xlabel('x_spurious')

    path = OUT_DIR / '00_shortcut_correlation.png'
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_shift_failure() -> Path:
    rng = np.random.default_rng(1)
    n = 500

    # Train env: spurious aligns with label
    x_true_train = rng.standard_normal(n)
    x_spur_train = x_true_train + 0.08 * rng.standard_normal(n)
    y_train = (x_true_train > 0).astype(int)

    # Test env: spurious relation breaks
    x_true_test = rng.standard_normal(n)
    x_spur_test = -x_true_test + 0.08 * rng.standard_normal(n)
    y_test = (x_true_test > 0).astype(int)

    # Naive shortcut classifier uses spurious feature sign
    pred_train = (x_spur_train > 0).astype(int)
    pred_test = (x_spur_test > 0).astype(int)

    acc_train = (pred_train == y_train).mean()
    acc_test = (pred_test == y_test).mean()

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(['Train env', 'Shifted test env'], [acc_train, acc_test], color=['tab:green', 'tab:red'])
    ax.set_ylim(0, 1)
    ax.set_title('Shortcut Classifier Under Shift')
    ax.set_ylabel('Accuracy')
    ax.grid(True, axis='y', alpha=0.25)

    path = OUT_DIR / '01_shift_failure.png'
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main() -> None:
    p1 = plot_shortcut_correlation()
    p2 = plot_shift_failure()
    print('Saved plots ->', p1, p2)


if __name__ == '__main__':
    main()
