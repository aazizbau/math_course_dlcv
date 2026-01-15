"""Day 38 visualizations: gradient field and descent path."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _grad(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return 2 * x, 2 * y


def plot_gradient_field() -> Path:
    xs = np.linspace(-2, 2, 15)
    ys = np.linspace(-2, 2, 15)
    X, Y = np.meshgrid(xs, ys)
    U, V = _grad(X, Y)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.quiver(X, Y, U, V, color="#4c72b0")
    ax.set_title("Gradient Field for f(x,y)=x^2+y^2")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    path = OUT_DIR / "00_gradient_field.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_descent_path(lr: float = 0.2, steps: int = 15) -> Path:
    x = np.array([1.5, -1.0])
    path = [x.copy()]
    for _ in range(steps):
        x = x - lr * np.array([2 * x[0], 2 * x[1]])
        path.append(x.copy())
    path = np.array(path)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(path[:, 0], path[:, 1], marker="o")
    ax.set_title("Gradient Descent Path")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    path_out = OUT_DIR / "01_descent_path.png"
    fig.tight_layout()
    fig.savefig(path_out, dpi=200)
    plt.close(fig)
    return path_out


def main() -> None:
    print("Saved gradient field →", plot_gradient_field())
    print("Saved descent path →", plot_descent_path())


if __name__ == "__main__":
    main()
