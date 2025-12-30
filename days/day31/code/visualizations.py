"""Day 31 visualizations: circle-to-ellipse SVD geometry."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_circle_ellipse() -> Path:
    theta = np.linspace(0, 2 * np.pi, 200)
    circle = np.stack([np.cos(theta), np.sin(theta)])
    A = np.array([[2.0, 1.0], [0.0, 1.0]])
    transformed = A @ circle

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(circle[0], circle[1], label="Unit circle", color="#4c72b0")
    ax.plot(transformed[0], transformed[1], label="Transformed", color="#c44e52")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_title("SVD Geometry: Circle to Ellipse")
    ax.legend()

    path = OUT_DIR / "00_circle_to_ellipse.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_singular_values() -> Path:
    A = np.array([[3.0, 1.0], [1.0, 3.0], [0.0, 2.0]])
    _, S, _ = np.linalg.svd(A)

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(range(1, len(S) + 1), S, color="#55a868")
    ax.set_title("Singular Values")
    ax.set_xlabel("Index")
    ax.set_ylabel("Value")
    ax.grid(True, axis="y", alpha=0.3)

    path = OUT_DIR / "01_singular_values.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main() -> None:
    print("Saved circle-ellipse plot →", plot_circle_ellipse())
    print("Saved singular values plot →", plot_singular_values())


if __name__ == "__main__":
    main()
