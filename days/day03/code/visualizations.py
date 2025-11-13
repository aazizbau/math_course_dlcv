"""Day 3 visualizations for gradient descent intuition."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def bowl_loss(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return 0.5 * (3 * x**2 + 0.8 * x * y + y**2)


def bowl_grad(w: np.ndarray) -> np.ndarray:
    x, y = w
    return np.array([3 * x + 0.4 * y, 0.4 * x + y])


def _contour_setup(ax: plt.Axes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xg = np.linspace(-3, 3, 200)
    yg = np.linspace(-3, 3, 200)
    XX, YY = np.meshgrid(xg, yg)
    ZZ = bowl_loss(XX, YY)
    ax.contour(XX, YY, ZZ, levels=20)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect("equal")
    ax.grid(True)
    return XX, YY, ZZ


def anim_gradient_descent(init=(2.5, -2.0), lr=0.15, frames: int = 60, fps: int = 15) -> Path:
    fig, ax = plt.subplots(figsize=(6, 6))
    _contour_setup(ax)
    ax.set_title("Gradient Descent — Rolling Down the Hill")

    point, = ax.plot([], [], "ro", markersize=6)
    trail, = ax.plot([], [], "r-", lw=2)

    w = np.array(init, dtype=float)
    path_x: List[float] = [w[0]]
    path_y: List[float] = [w[1]]

    def update(_frame: int):
        nonlocal w
        grad = bowl_grad(w)
        w = w - lr * grad
        path_x.append(w[0])
        path_y.append(w[1])
        point.set_data([w[0]], [w[1]])
        trail.set_data(path_x, path_y)
        return point, trail

    anim = animation.FuncAnimation(fig, update, frames=frames, interval=120, blit=False)
    path = OUT_DIR / "01_gradient_descent.gif"
    anim.save(path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    return path


def anim_learning_rates(init=(2.5, -2.0), lrs: Iterable[float] = (0.05, 0.15, 0.35), frames: int = 50, fps: int = 15) -> Path:
    fig, ax = plt.subplots(figsize=(6, 6))
    XX, YY, ZZ = _contour_setup(ax)
    ax.set_title("Learning Rate Comparison")

    colors = ["b", "r", "g"]
    labels = ["small (slow)", "medium (good)", "large (oscillates)"]
    trails = [ax.plot([], [], color=c, label=f"η={lr} ({lab})")[0] for lr, c, lab in zip(lrs, colors, labels)]
    points = [ax.plot([], [], "o", color=c)[0] for c in colors]
    ax.legend(loc="upper right")

    paths = []
    max_steps = frames
    for lr in lrs:
        w = np.array(init, dtype=float)
        path = [w.copy()]
        for _ in range(max_steps):
            w = w - lr * bowl_grad(w)
            path.append(w.copy())
        paths.append(np.stack(path))

    def update(frame: int):
        idx = min(frame, max_steps)
        artists = []
        for path, trail, point in zip(paths, trails, points):
            subpath = path[: idx + 1]
            trail.set_data(subpath[:, 0], subpath[:, 1])
            point.set_data([subpath[-1, 0]], [subpath[-1, 1]])
            artists.extend([trail, point])
        return artists

    anim = animation.FuncAnimation(fig, update, frames=frames, interval=120, blit=False)
    path = OUT_DIR / "02_learning_rate.gif"
    anim.save(path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    return path


def render_gradient_field() -> Path:
    fig, ax = plt.subplots(figsize=(6, 6))
    XX, YY, ZZ = _contour_setup(ax)
    ax.set_title("Gradient Field — Direction of Descent")
    U, V = np.gradient(-ZZ)
    step = 10
    ax.quiver(XX[::step, ::step], YY[::step, ::step], U[::step, ::step], V[::step, ::step], color="gray")
    path = OUT_DIR / "03_gradient_field.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main() -> None:
    generators = [
        ("Gradient descent", anim_gradient_descent),
        ("Learning rate comparison", anim_learning_rates),
        ("Gradient field", render_gradient_field),
    ]
    for label, fn in generators:
        asset = fn()
        print(f"Saved {label} asset → {asset}")


if __name__ == "__main__":
    main()
