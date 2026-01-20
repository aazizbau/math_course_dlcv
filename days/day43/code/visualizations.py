"""Day 43 visualizations: saddle contours and GD path."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_saddle_contours() -> Path:
    x = np.linspace(-2, 2, 200)
    y = np.linspace(-2, 2, 200)
    X, Y = np.meshgrid(x, y)
    Z = X**2 - Y**2

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.contour(X, Y, Z, levels=20)
    ax.scatter(0, 0, color="red", label="Saddle")
    ax.set_title("Saddle Point at (0,0)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()

    path = OUT_DIR / "00_saddle_contours.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_gd_near_saddle(lr: float = 0.1, steps: int = 20) -> Path:
    x = np.array([1.0, 1.0])
    path = [x.copy()]
    for _ in range(steps):
        grad = np.array([2 * x[0], -2 * x[1]])
        x = x - lr * grad
        path.append(x.copy())
    path = np.array(path)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(path[:, 0], path[:, 1], marker="o")
    ax.set_title("Gradient Descent near Saddle")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    path_out = OUT_DIR / "01_gd_saddle_path.png"
    fig.tight_layout()
    fig.savefig(path_out, dpi=200)
    plt.close(fig)
    return path_out


def main() -> None:
    print("Saved saddle contours →", plot_saddle_contours())
    print("Saved GD saddle path →", plot_gd_near_saddle())


if __name__ == "__main__":
    main()
