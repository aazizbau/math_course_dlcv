"""Day 59 visualizations: swiss roll and embedding flattening."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def make_swiss_roll(n: int = 1000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = 1.5 * np.pi * (1 + 2 * rng.random(n))
    x = t * np.cos(t)
    y = 21 * rng.random(n)
    z = t * np.sin(t)
    return np.stack([x, y, z], axis=1)


def plot_swiss_roll() -> Path:
    X = make_swiss_roll(1200)
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=3, c=X[:, 0], cmap='viridis')
    ax.set_title("Swiss Roll (2D Manifold in 3D)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    path = OUT_DIR / "00_swiss_roll.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_flattened() -> Path:
    X = make_swiss_roll(800)
    t = np.arctan2(X[:, 2], X[:, 0])
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.scatter(t, X[:, 1], s=4, c=X[:, 0], cmap='viridis')
    ax.set_title("Unrolled Coordinates (Approx)")
    ax.set_xlabel("angle")
    ax.set_ylabel("height")

    path = OUT_DIR / "01_unrolled.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main() -> None:
    path1 = plot_swiss_roll()
    path2 = plot_flattened()
    print("Saved plots →", path1, path2)


if __name__ == "__main__":
    main()
