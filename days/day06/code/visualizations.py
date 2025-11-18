"""Day 6 visualizations: convex and non-convex landscapes."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - needed for 3D projection

from days.day06.code.landscapes import (
    GDConfig,
    banana,
    banana_grad,
    convex_bowl,
    gd_path,
    waves,
)

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _grid(lim: float = 2.0, num: int = 200) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(-lim, lim, num)
    y = np.linspace(-lim, lim, num)
    return np.meshgrid(x, y)


def render_surface(func, title: str, filename: str, cmap: str = "coolwarm",
                   elev: int = 40, azim: int = 20) -> Path:
    X, Y = _grid()
    Z = func(X, Y)

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(title)
    surf = ax.plot_surface(X, Y, Z, cmap=cmap, alpha=0.85)
    ax.view_init(elev, azim)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("Loss")

    path = OUT_DIR / filename
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def anim_banana_descent(init=(-1.5, 1.5), lr: float = 1e-3, steps: int = 2000, fps: int = 20) -> Path:
    X, Y = _grid(lim=2.0, num=400)
    Z = banana(X, Y)

    config = GDConfig(lr=lr, steps=steps)
    path_points = gd_path(init, banana_grad, config)

    fig, ax = plt.subplots(figsize=(6, 6))
    contour = ax.contour(X, Y, Z, levels=40)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-1, 3)
    ax.set_title("Gradient Descent on Banana Valley")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    point, = ax.plot([], [], "ro")
    trail, = ax.plot([], [], "r-", lw=2)

    def update(i: int):
        idx = min(i, len(path_points) - 1)
        trail.set_data(path_points[:idx, 0], path_points[:idx, 1])
        point.set_data([path_points[idx, 0]], [path_points[idx, 1]])
        return trail, point

    frames = min(len(path_points), 400)
    anim = animation.FuncAnimation(fig, update, frames=frames, interval=25)
    path = OUT_DIR / "04_banana_gd.gif"
    anim.save(path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    return path


def main() -> None:
    assets = [
        ("Convex bowl", lambda: render_surface(convex_bowl, "Convex Bowl — Easy Surface", "01_convex_bowl.png", cmap="coolwarm", azim=10)),
        ("Banana valley", lambda: render_surface(banana, "Rosenbrock Banana Valley", "02_banana_valley.png", cmap="viridis", azim=20)),
        ("Wavy surface", lambda: render_surface(waves, "Non-Convex Wavy Landscape", "03_wavy_surface.png", cmap="plasma", azim=30)),
        ("Banana GD", anim_banana_descent),
    ]
    for label, fn in assets:
        artifact = fn()
        print(f"Saved {label} asset → {artifact}")


if __name__ == "__main__":
    main()
