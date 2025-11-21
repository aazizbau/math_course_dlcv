"""Day 8 visualizations: Hessian curvature surfaces and Newton vs GD."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np

from days.day08.code.hessian_demo import grad, hessian, loss, newton_step

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _grid(lim: float = 3.0, num: int = 300) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(-lim, lim, num)
    y = np.linspace(-lim, lim, num)
    return np.meshgrid(x, y)


def render_surface(title: str, filename: str) -> Path:
    X, Y = _grid()
    Z = loss(np.array([X, Y]))
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(title)
    ax.plot_surface(X, Y, Z, cmap="coolwarm", alpha=0.85)
    ax.view_init(35, 30)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("Loss")
    path = OUT_DIR / filename
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def anim_newton_vs_gd(init=(2.5, -2.0), gd_lr: float = 0.15, frames: int = 12, fps: int = 4) -> Path:
    X, Y = _grid()
    Z = loss(np.array([X, Y]))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.contour(X, Y, Z, levels=40)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect("equal")
    ax.grid(True)
    ax.set_title("Gradient Descent vs Newton's Method")

    gd_point, = ax.plot([], [], "ro", label="GD")
    newt_point, = ax.plot([], [], "bo", label="Newton")
    gd_trail, = ax.plot([], [], "r-", lw=2)
    newt_trail, = ax.plot([], [], "b-", lw=2)
    ax.legend(loc="upper right")

    w_gd = np.array(init, dtype=float)
    w_newt = np.array(init, dtype=float)
    gd_path = [w_gd.copy()]
    newt_path = [w_newt.copy()]

    def update(_frame: int):
        nonlocal w_gd, w_newt
        w_gd = w_gd - gd_lr * grad(w_gd)
        gd_path.append(w_gd.copy())

        w_newt = newton_step(w_newt)
        newt_path.append(w_newt.copy())

        gd_arr = np.array(gd_path)
        newt_arr = np.array(newt_path)

        gd_trail.set_data(gd_arr[:, 0], gd_arr[:, 1])
        newt_trail.set_data(newt_arr[:, 0], newt_arr[:, 1])
        gd_point.set_data([w_gd[0]], [w_gd[1]])
        newt_point.set_data([w_newt[0]], [w_newt[1]])
        return gd_point, newt_point, gd_trail, newt_trail

    anim = animation.FuncAnimation(fig, update, frames=frames, interval=300)
    path = OUT_DIR / "01_newton_vs_gd.gif"
    anim.save(path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    return path


def main() -> None:
    assets = [
        ("Quadratic surface", lambda: render_surface("Quadratic Loss Surface", "00_loss_surface.png")),
        ("Newton vs GD", anim_newton_vs_gd),
    ]
    for label, fn in assets:
        artifact = fn()
        print(f"Saved {label} asset → {artifact}")


if __name__ == "__main__":
    main()
