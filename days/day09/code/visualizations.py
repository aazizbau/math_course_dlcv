"""Day 9 visualizations for vanishing/exploding gradients."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def anim_gradient_evolution(steps: int = 30, fps: int = 10) -> Path:
    factors = [0.7, 1.0, 1.3]
    labels = ["Vanishing (0.7)", "Stable (1.0)", "Exploding (1.3)"]
    colors = ["blue", "green", "red"]

    trajectories = {a: [] for a in factors}
    g = {a: 1.0 for a in factors}

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_xlim(0, steps)
    ax.set_ylim(1e-5, 1e5)
    ax.set_yscale("log")
    ax.set_title("Gradient Magnitude Evolution (log scale)")
    ax.set_xlabel("Layer (depth)")
    ax.set_ylabel("|g|")
    ax.grid(True)

    lines = [ax.plot([], [], color=c, label=lab)[0] for c, lab in zip(colors, labels)]
    ax.legend()

    def update(_frame: int):
        for idx, a in enumerate(factors):
            g[a] *= a
            trajectories[a].append(g[a])
            xs = np.arange(len(trajectories[a]))
            lines[idx].set_data(xs, trajectories[a])
        return lines

    anim = animation.FuncAnimation(fig, update, frames=steps, interval=200)
    path = OUT_DIR / "01_gradients_evolution.gif"
    anim.save(path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    return path


def main() -> None:
    artifact = anim_gradient_evolution()
    print(f"Saved gradient evolution animation → {artifact}")


if __name__ == "__main__":
    main()
