"""Day 37 visualizations: partial derivative slices and surface."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _surface(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return x**2 + y**2


def plot_surface() -> Path:
    xs = np.linspace(-2, 2, 80)
    ys = np.linspace(-2, 2, 80)
    X, Y = np.meshgrid(xs, ys)
    Z = _surface(X, Y)

    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.85)
    ax.set_title("f(x,y)=x^2+y^2 Surface")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("f")

    path = OUT_DIR / "00_surface.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_partial_slices() -> Path:
    xs = np.linspace(-2, 2, 200)
    ys = np.linspace(-2, 2, 200)
    y0 = -1.0
    x0 = 1.0

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(xs, _surface(xs, y0), label=f"y={y0}")
    ax.plot(ys, _surface(x0, ys), label=f"x={x0}")
    ax.set_title("Partial Derivative Slices")
    ax.set_xlabel("input")
    ax.set_ylabel("f")
    ax.grid(True, alpha=0.3)
    ax.legend()

    path = OUT_DIR / "01_partial_slices.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main() -> None:
    print("Saved surface plot →", plot_surface())
    print("Saved partial slices →", plot_partial_slices())


if __name__ == "__main__":
    main()
