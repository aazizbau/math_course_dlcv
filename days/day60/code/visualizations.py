"""Day 60 visualizations: metric learning geometry."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_metric_forces() -> Path:
    z_pos = np.array([[0.0, 0.0], [1.0, 1.0]])
    z_neg = np.array([[3.0, 3.0], [4.0, 4.0]])

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(z_pos[:, 0], z_pos[:, 1], c='green', label='Positive')
    ax.scatter(z_neg[:, 0], z_neg[:, 1], c='red', label='Negative')
    ax.plot([0, 1], [0, 1], 'g--')
    ax.plot([1, 3], [1, 3], 'r--')
    ax.set_aspect('equal')
    ax.set_title('Pull Positives, Push Negatives')
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = OUT_DIR / "00_metric_forces.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_triplet_margin() -> Path:
    d_ap = np.linspace(0, 2, 200)
    margin = 0.5
    d_an = 1.5
    loss = np.maximum(0, d_ap**2 - d_an**2 + margin)

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(d_ap, loss)
    ax.set_title("Triplet Loss vs Positive Distance")
    ax.set_xlabel("||a-p||")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)

    path = OUT_DIR / "01_triplet_loss.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main() -> None:
    path1 = plot_metric_forces()
    path2 = plot_triplet_margin()
    print("Saved plots →", path1, path2)


if __name__ == "__main__":
    main()
