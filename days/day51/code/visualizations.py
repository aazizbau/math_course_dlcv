"""Day 51 visualizations: L1 vs L2 geometry and weight decay effect."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_l1_l2_geometry() -> Path:
    x = np.linspace(-1.5, 1.5, 400)
    X, Y = np.meshgrid(x, x)
    L1 = np.abs(X) + np.abs(Y)
    L2 = X**2 + Y**2

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.contour(X, Y, L1, levels=[1.0], colors='r')
    ax.contour(X, Y, L2, levels=[1.0], colors='b')
    ax.set_title("L1 (Red) vs L2 (Blue) Geometry")
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)

    path = OUT_DIR / "00_l1_l2_geometry.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_weight_decay() -> Path:
    steps = 50
    theta = 2.0
    lr = 0.1
    decay = 0.2
    vals = []

    for _ in range(steps):
        theta = theta - lr * decay * theta
        vals.append(theta)

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(range(1, steps + 1), vals)
    ax.set_title("Weight Decay Shrinks Parameters")
    ax.set_xlabel("Step")
    ax.set_ylabel("Theta")
    ax.grid(True, alpha=0.3)

    path = OUT_DIR / "01_weight_decay.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main() -> None:
    path1 = plot_l1_l2_geometry()
    path2 = plot_weight_decay()
    print("Saved plots →", path1, path2)


if __name__ == "__main__":
    main()
