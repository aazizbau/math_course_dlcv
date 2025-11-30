"""Day 10 visualizations: activation functions and derivatives."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np

from days.day10.code.activations import build_activations

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_activations(x: np.ndarray) -> Path:
    acts = build_activations()
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, activation in acts.items():
        ax.plot(x, activation.fn(x), label=name)
    ax.set_title("Activation Functions")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.grid(True)
    ax.legend()
    path = OUT_DIR / "00_activation_curves.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def anim_activation_derivatives(x: np.ndarray, fps: int = 30) -> Path:
    acts = build_activations()
    selected = ["Sigmoid", "Tanh", "ReLU", "GELU"]
    colors = ["blue", "green", "red", "purple"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(0, 1.2)
    ax.set_title("Derivative of Activation Functions")
    ax.set_xlabel("x")
    ax.set_ylabel("Derivative")
    ax.grid(True)

    lines = [ax.plot([], [], color=c, label=f"{name}'")[0] for name, c in zip(selected, colors)]
    ax.legend()

    def update(i: int):
        xi = x[:i]
        for line, name in zip(lines, selected):
            act = acts[name]
            line.set_data(xi, act.derivative(xi))
        return lines

    anim = animation.FuncAnimation(fig, update, frames=len(x), interval=20)
    path = OUT_DIR / "01_activation_derivatives.gif"
    anim.save(path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    return path


def main() -> None:
    x = np.linspace(-5, 5, 400)
    curve_path = plot_activations(x)
    print(f"Saved activation curve plot → {curve_path}")
    gif_path = anim_activation_derivatives(x)
    print(f"Saved derivative animation → {gif_path}")


if __name__ == "__main__":
    main()
