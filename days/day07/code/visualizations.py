"""Day 7 visualizations: Jacobian local linearization and heatmap."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np

from days.day07.code.jacobian_demo import TinyNetwork, build_default_network

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def anim_local_linearization(samples: int = 200, eps: float = 0.3, fps: int = 20) -> Path:
    net = build_default_network()
    x0 = np.array([0.6, -0.4])
    J = net.jacobian(x0)
    _, y0 = net.forward(x0)

    noise = np.random.randn(samples, 2) * eps

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_title("Jacobian as Local Linear Approximation")
    ax.set_xlabel("y₁")
    ax.set_ylabel("y₂")
    ax.grid(True)

    real_scatter = ax.scatter([], [], color="blue", s=30, label="Real f(x)")
    lin_scatter = ax.scatter([], [], color="red", s=30, label="J·Δx + f(x₀)")
    ax.legend(loc="upper left")

    def update(i: int):
        dx = noise[i]
        _, y_real = net.forward(x0 + dx)
        y_lin = y0 + J @ dx
        real_scatter.set_offsets([y_real])
        lin_scatter.set_offsets([y_lin])
        return real_scatter, lin_scatter

    anim = animation.FuncAnimation(fig, update, frames=samples, interval=80)
    path = OUT_DIR / "01_jacobian_local_linearization.gif"
    anim.save(path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    return path


def render_jacobian_heatmap() -> Path:
    net = build_default_network()
    x = np.array([0.6, -0.4])
    J = net.jacobian(x)

    fig, ax = plt.subplots(figsize=(4, 4))
    cax = ax.imshow(J, cmap="coolwarm")
    ax.set_title("Jacobian Heatmap — Sensitivity")
    ax.set_xlabel("Input dimension")
    ax.set_ylabel("Output dimension")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    fig.colorbar(cax, ax=ax, shrink=0.8)

    path = OUT_DIR / "02_jacobian_heatmap.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main() -> None:
    assets = [
        ("Local linearization", anim_local_linearization),
        ("Jacobian heatmap", render_jacobian_heatmap),
    ]
    for label, fn in assets:
        artifact = fn()
        print(f"Saved {label} asset → {artifact}")


if __name__ == "__main__":
    main()
