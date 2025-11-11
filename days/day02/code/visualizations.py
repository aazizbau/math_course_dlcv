"""Day 2 visualizations: cosine similarity, normalization, and orthogonality."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import FancyArrowPatch
import numpy as np

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def arrow(ax: plt.Axes, start: np.ndarray, vec: np.ndarray, **kwargs) -> FancyArrowPatch:
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


def anim_cosine_similarity(frames: int = 100, fps: int = 25) -> Path:
    a = np.array([3.0, 0.0])

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect("equal")
    ax.grid(True)
    ax.set_title("Cosine Similarity as Angle Changes")

    arr_a = arrow(ax, np.zeros(2), a, color="r")
    arr_b = arrow(ax, np.zeros(2), np.zeros(2), color="b")
    proj_line, = ax.plot([], [], "g", lw=2)
    text_box = ax.text(0.05, 0.95, "", transform=ax.transAxes, va="top")

    def update(i: int):
        theta = i / (frames - 1) * 2 * np.pi
        b = np.array([3 * np.cos(theta), 3 * np.sin(theta)])
        arr_b.set_positions((0, 0), tuple(b))
        cos_theta = (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))
        proj = cos_theta * np.linalg.norm(b) * a / np.linalg.norm(a)
        proj_line.set_data([0, proj[0]], [0, proj[1]])
        text_box.set_text(f"cos(θ)={cos_theta:+.2f}")
        return arr_b, proj_line, text_box

    anim = animation.FuncAnimation(fig, update, frames=frames, interval=60)
    path = OUT_DIR / "01_cosine_similarity.gif"
    anim.save(path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    return path


def anim_vector_norms(frames: int = 50, fps: int = 20) -> Path:
    rng = np.random.default_rng(1)
    vecs = rng.normal(size=(8, 2)) * 2

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_aspect("equal")
    ax.grid(True)
    ax.set_title("Normalizing Vectors")

    base_arrows = [arrow(ax, np.zeros(2), v, color="gray", alpha=0.4) for v in vecs]
    norm_arrows = [arrow(ax, np.zeros(2), v, color="b", alpha=0.0) for v in vecs]

    def update(i: int) -> Iterable[FancyArrowPatch]:
        t = i / (frames - 1)
        for j, v in enumerate(vecs):
            nv = v / np.linalg.norm(v)
            inter = (1 - t) * v + t * nv
            norm_arrows[j].set_positions((0, 0), tuple(inter))
            norm_arrows[j].set_alpha(0.6 * t)
        return norm_arrows + base_arrows

    anim = animation.FuncAnimation(fig, update, frames=frames, interval=80)
    path = OUT_DIR / "02_vector_norms.gif"
    anim.save(path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    return path


def render_orthogonality_diagram() -> Path:
    a = np.array([3.0, 0.0])
    b = np.array([0.0, 3.0])

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect("equal")
    ax.grid(True)
    ax.set_title("Orthogonality — Independent Directions")

    arrow(ax, np.zeros(2), a, color="r")
    arrow(ax, np.zeros(2), b, color="b")
    ax.text(0.1, 0.9, "a·b = 0 → no overlap", transform=ax.transAxes, fontsize=11)

    path = OUT_DIR / "03_orthogonality.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main() -> None:
    generators = [
        ("Cosine similarity", anim_cosine_similarity),
        ("Vector norms", anim_vector_norms),
        ("Orthogonality diagram", render_orthogonality_diagram),
    ]
    for label, fn in generators:
        asset = fn()
        print(f"Saved {label} asset → {asset}")


if __name__ == "__main__":
    main()
