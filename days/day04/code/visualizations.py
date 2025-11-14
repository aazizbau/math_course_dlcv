"""Day 4 visualizations: momentum vs. gradient descent."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np

from days.day04.code.momentum_methods import Bowl, OptimizerConfig, gradient_descent, momentum, nesterov

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


bowl = Bowl()


def _contour_axes(ax: plt.Axes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xg = np.linspace(-3, 3, 200)
    yg = np.linspace(-3, 3, 200)
    XX, YY = np.meshgrid(xg, yg)
    ZZ = 0.5 * (3 * XX**2 + 0.8 * XX * YY + YY**2)
    ax.contour(XX, YY, ZZ, levels=20)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect("equal")
    ax.grid(True)
    return XX, YY, ZZ


def anim_gd_vs_momentum(init: Sequence[float] = (2.5, -2.0), lr: float = 0.15, beta: float = 0.9,
                         steps: int = 40, fps: int = 15) -> Path:
    gd_path = gradient_descent(init, bowl, lr=lr, steps=steps)
    mom_path = momentum(init, bowl, OptimizerConfig(lr=lr, beta=beta, steps=steps))

    fig, ax = plt.subplots(figsize=(6, 6))
    _contour_axes(ax)
    ax.set_title("Momentum vs Gradient Descent")

    gd_point, = ax.plot([], [], "ro", label="Gradient Descent")
    gd_trail, = ax.plot([], [], "r-", lw=2)
    mom_point, = ax.plot([], [], "bo", label="Momentum")
    mom_trail, = ax.plot([], [], "b-", lw=2)
    ax.legend()

    def update(i: int):
        idx = min(i, len(gd_path) - 1)
        gd_trail.set_data(gd_path[:idx, 0], gd_path[:idx, 1])
        mom_trail.set_data(mom_path[:idx, 0], mom_path[:idx, 1])
        gd_point.set_data([gd_path[idx, 0]], [gd_path[idx, 1]])
        mom_point.set_data([mom_path[idx, 0]], [mom_path[idx, 1]])
        return gd_trail, mom_trail, gd_point, mom_point

    anim = animation.FuncAnimation(fig, update, frames=len(gd_path), interval=120, blit=False)
    path = OUT_DIR / "01_momentum_vs_gd.gif"
    anim.save(path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    return path


def anim_momentum_beta(init: Sequence[float] = (2.5, -2.0), lr: float = 0.15,
                        betas: Iterable[float] = (0.5, 0.8, 0.95), steps: int = 40,
                        fps: int = 15) -> Path:
    fig, ax = plt.subplots(figsize=(6, 6))
    _contour_axes(ax)
    ax.set_title("Momentum Strength (β) Effect")

    colors = ["g", "orange", "b"]
    labels = [f"β={b}" for b in betas]
    trails = [ax.plot([], [], color=c, label=lab)[0] for c, lab in zip(colors, labels)]
    ax.legend()

    paths = []
    for beta in betas:
        config = OptimizerConfig(lr=lr, beta=beta, steps=steps)
        paths.append(momentum(init, bowl, config))

    def update(frame: int):
        idx = min(frame, steps)
        artists = []
        for path, trail in zip(paths, trails):
            subpath = path[: idx + 1]
            trail.set_data(subpath[:, 0], subpath[:, 1])
            artists.append(trail)
        return artists

    anim = animation.FuncAnimation(fig, update, frames=steps + 1, interval=120, blit=False)
    path = OUT_DIR / "02_momentum_beta.gif"
    anim.save(path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    return path


def anim_momentum_vs_nesterov(init: Sequence[float] = (2.5, -2.0), lr: float = 0.15,
                               beta: float = 0.9, steps: int = 40, fps: int = 15) -> Path:
    mom_path = momentum(init, bowl, OptimizerConfig(lr=lr, beta=beta, steps=steps))
    nag_path = nesterov(init, bowl, OptimizerConfig(lr=lr, beta=beta, steps=steps))

    fig, ax = plt.subplots(figsize=(6, 6))
    _contour_axes(ax)
    ax.set_title("Momentum vs Nesterov")

    mom_point, = ax.plot([], [], "bo", label="Momentum")
    mom_trail, = ax.plot([], [], "b-", lw=2)
    nag_point, = ax.plot([], [], "go", label="Nesterov")
    nag_trail, = ax.plot([], [], "g-", lw=2)
    ax.legend()

    def update(i: int):
        idx = min(i, len(mom_path) - 1)
        mom_trail.set_data(mom_path[:idx, 0], mom_path[:idx, 1])
        nag_trail.set_data(nag_path[:idx, 0], nag_path[:idx, 1])
        mom_point.set_data([mom_path[idx, 0]], [mom_path[idx, 1]])
        nag_point.set_data([nag_path[idx, 0]], [nag_path[idx, 1]])
        return mom_trail, nag_trail, mom_point, nag_point

    anim = animation.FuncAnimation(fig, update, frames=len(mom_path), interval=120, blit=False)
    path = OUT_DIR / "03_momentum_vs_nesterov.gif"
    anim.save(path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    return path


def main() -> None:
    generators = [
        ("Momentum vs GD", anim_gd_vs_momentum),
        ("Momentum beta sweep", anim_momentum_beta),
        ("Momentum vs Nesterov", anim_momentum_vs_nesterov),
    ]
    for label, fn in generators:
        asset = fn()
        print(f"Saved {label} asset → {asset}")


if __name__ == "__main__":
    main()
