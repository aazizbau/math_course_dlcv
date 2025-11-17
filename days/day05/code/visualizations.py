"""Day 5 visualizations: gradient flow through a tiny chain."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def f1(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def df1(x: np.ndarray) -> np.ndarray:
    return 1 - np.tanh(x) ** 2


def f2(x: np.ndarray) -> np.ndarray:
    return x**2


def df2(x: np.ndarray) -> np.ndarray:
    return 2 * x


def f3(x: np.ndarray) -> np.ndarray:
    return x


def df3(x: np.ndarray) -> np.ndarray:
    return np.ones_like(x)


def forward_chain(x: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    h1 = f1(x)
    h2 = f2(h1)
    h3 = f3(h2)
    L = 0.5 * h3**2
    return h1, h2, h3, L


def backprop_chain(x: float) -> tuple[list[float], list[float]]:
    h1, h2, h3, L = forward_chain(x)

    dL_dh3 = h3
    dL_dh2 = dL_dh3 * float(df3(h2))
    dL_dh1 = dL_dh2 * float(df2(h1))
    dL_dx = dL_dh1 * float(df1(x))
    return [dL_dh3, dL_dh2, dL_dh1, dL_dx], [h1, h2, h3, L]


def anim_backprop_chain(fps: int = 20) -> Path:
    xs = np.linspace(-2, 2, 200)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-1, 4)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["h1", "h2", "h3", "L"])
    ax.set_title("Backprop: Gradient Flow Through Layers")
    ax.grid(True)

    forward_line, = ax.plot([], [], "r-o", label="Forward values")
    gradient_line, = ax.plot([], [], "b-o", label="Gradient magnitudes")
    ax.legend()

    def update(i: int):
        x = xs[i]
        grads, vals = backprop_chain(float(x))
        forward_line.set_data(range(4), vals)
        gradient_line.set_data(range(4), np.abs(grads))
        return forward_line, gradient_line

    anim = animation.FuncAnimation(fig, update, frames=len(xs), interval=80)
    path = OUT_DIR / "01_backprop_chain.gif"
    anim.save(path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    return path


def main() -> None:
    asset = anim_backprop_chain()
    print(f"Saved backprop animation → {asset}")


if __name__ == "__main__":
    main()
