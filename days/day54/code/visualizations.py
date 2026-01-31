"""Day 54 visualizations: loss shapes and margins."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_loss_shapes() -> Path:
    r = np.linspace(-3, 3, 400)
    mse = r**2
    l1 = np.abs(r)
    huber = np.where(np.abs(r) <= 1, 0.5 * r**2, np.abs(r) - 0.5)

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(r, mse, label="MSE")
    ax.plot(r, l1, label="L1")
    ax.plot(r, huber, label="Huber")
    ax.set_title("Loss Geometry")
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = OUT_DIR / "00_loss_shapes.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_hinge_margin() -> Path:
    f = np.linspace(-2, 2, 400)
    y = 1.0
    hinge = np.maximum(0, 1 - y * f)

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(f, hinge)
    ax.axvline(1.0, color='gray', linestyle='--', linewidth=1)
    ax.set_title("Hinge Loss with Margin")
    ax.set_xlabel("Score f(x)")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)

    path = OUT_DIR / "01_hinge_margin.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main() -> None:
    path1 = plot_loss_shapes()
    path2 = plot_hinge_margin()
    print("Saved plots →", path1, path2)


if __name__ == "__main__":
    main()
