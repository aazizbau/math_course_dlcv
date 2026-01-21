"""Day 44 visualizations: GD vs SGD on a saddle."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .sgd_saddle import run_gd, run_sgd

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_gd_vs_sgd() -> Path:
    x0 = np.array([1.0, 1.0])
    gd = run_gd(x0, lr=0.1, steps=40)
    sgd = run_sgd(x0, lr=0.1, steps=40, noise=0.15)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(gd[:, 0], gd[:, 1], marker="o", label="GD")
    ax.plot(sgd[:, 0], sgd[:, 1], marker="o", label="SGD")
    ax.set_title("GD vs SGD near a Saddle")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    ax.legend()

    path = OUT_DIR / "00_gd_vs_sgd.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main() -> None:
    path = plot_gd_vs_sgd()
    print("Saved GD vs SGD plot →", path)


if __name__ == "__main__":
    main()
