"""Day 41 visualizations: Hessian curvature and saddle."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_saddle_surface() -> Path:
    xs = np.linspace(-2, 2, 120)
    ys = np.linspace(-2, 2, 120)
    X, Y = np.meshgrid(xs, ys)
    Z = X**2 - Y**2

    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, cmap="coolwarm", alpha=0.9)
    ax.set_title("Saddle Surface: x^2 - y^2")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("f")

    path = OUT_DIR / "00_saddle_surface.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_hessian_eigenvalues() -> Path:
    eigvals = np.array([2.0, -2.0])
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(["+", "-"], eigvals, color=["#55a868", "#c44e52"])
    ax.set_title("Hessian Eigenvalues")
    ax.set_ylabel("Value")
    ax.axhline(0, color="black", linewidth=1)
    ax.grid(True, axis="y", alpha=0.3)

    path = OUT_DIR / "01_hessian_eigenvalues.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main() -> None:
    print("Saved saddle surface →", plot_saddle_surface())
    print("Saved Hessian eigenvalues →", plot_hessian_eigenvalues())


if __name__ == "__main__":
    main()
