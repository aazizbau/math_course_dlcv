"""Animated visualizations for Day 1: arrows, projections, and matrix warps."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")  # Ensure compatibility with headless environments.

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import FancyArrowPatch
import numpy as np

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def arrow(ax: plt.Axes, start: np.ndarray, vec: np.ndarray, **kwargs) -> FancyArrowPatch:
    """Utility to draw a vector arrow on the provided axes."""
    end = start + vec
    arr = FancyArrowPatch(
        posA=tuple(start),
        posB=tuple(end),
        arrowstyle="->",
        mutation_scale=12,
        lw=2,
        **kwargs,
    )
    ax.add_patch(arr)
    return arr


def anim_vector_addition(frames: int = 50, fps: int = 20) -> Path:
    """Animate vector scaling and addition as a growing parallelogram."""
    a = np.array([3.0, 1.2])
    b = np.array([1.4, 3.3])

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(-1, 6)
    ax.set_ylim(-1, 6)
    ax.set_aspect("equal")
    ax.grid(True)
    ax.set_title("Vector Scaling & Addition")

    arr_a = arrow(ax, np.zeros(2), a, color="r")
    arr_b = arrow(ax, np.zeros(2), b, color="b")
    arr_sum = arrow(ax, np.zeros(2), a + b, color="g")

    def update(i: int) -> Iterable[FancyArrowPatch]:
        t = i / (frames - 1)
        arr_a.set_positions((0, 0), tuple(a * t))
        arr_b.set_positions(tuple(a * t), tuple(a * t + b * t))
        arr_sum.set_positions((0, 0), tuple(a * t + b * t))
        return arr_a, arr_b, arr_sum

    anim = animation.FuncAnimation(fig, update, frames=frames, interval=60)
    path = OUT_DIR / "01_vector_add_scale.gif"
    anim.save(path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    return path


def anim_projection_cosine(frames: int = 60, fps: int = 20) -> Path:
    """Animate projection and cosine similarity as a rotating vector."""
    a = np.array([3.0, 0.0])

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect("equal")
    ax.grid(True)
    ax.set_title("Projection & Cosine Similarity")

    arr_a = arrow(ax, np.zeros(2), a, color="r")
    arr_b = arrow(ax, np.zeros(2), np.zeros(2), color="b")
    proj_line, = ax.plot([], [], "g", lw=2)
    text_box = ax.text(0.05, 0.95, "", transform=ax.transAxes, va="top")

    def update(i: int):
        theta = i / (frames - 1) * np.pi
        b = np.array([3 * np.cos(theta), 3 * np.sin(theta)])
        arr_b.set_positions((0, 0), tuple(b))
        cos_theta = (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))
        proj = cos_theta * np.linalg.norm(b) * a / np.linalg.norm(a)
        proj_line.set_data([0, proj[0]], [0, proj[1]])
        text_box.set_text(f"cos(θ) = {cos_theta:.2f}")
        return arr_b, proj_line, text_box

    anim = animation.FuncAnimation(fig, update, frames=frames, interval=60)
    path = OUT_DIR / "02_projection_cosine.gif"
    anim.save(path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    return path


def anim_matrix_transform(frames: int = 40, fps: int = 20) -> Path:
    """Animate grid warping as a target matrix gradually replaces the identity."""
    A = np.array([[1.0, 0.5], [0.0, 1.2]])
    xs = np.linspace(-2, 2, 11)
    ys = np.linspace(-2, 2, 11)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect("equal")
    ax.grid(True)
    ax.set_title("Matrix Transformation")

    h_lines = [ax.plot(xs, np.full_like(xs, y), color="gray", lw=0.5)[0] for y in ys]
    v_lines = [ax.plot(np.full_like(ys, x), ys, color="gray", lw=0.5)[0] for x in xs]

    def update(i: int):
        t = i / (frames - 1)
        M = np.eye(2) * (1 - t) + A * t
        for idx, y in enumerate(ys):
            pts = np.vstack([xs, np.full_like(xs, y)]).T
            transformed = pts @ M.T
            h_lines[idx].set_data(transformed[:, 0], transformed[:, 1])
        for idx, x in enumerate(xs):
            pts = np.vstack([np.full_like(ys, x), ys]).T
            transformed = pts @ M.T
            v_lines[idx].set_data(transformed[:, 0], transformed[:, 1])
        return h_lines + v_lines

    anim = animation.FuncAnimation(fig, update, frames=frames, interval=80)
    path = OUT_DIR / "03_matrix_transform.gif"
    anim.save(path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    return path


def main() -> None:
    generators = [
        ("Vector addition", anim_vector_addition),
        ("Projection", anim_projection_cosine),
        ("Matrix transform", anim_matrix_transform),
    ]
    for label, fn in generators:
        gif_path = fn()
        print(f"Saved {label} animation → {gif_path}")


if __name__ == "__main__":
    main()
