"""Day 12 visualizations: variance propagation under different initializations."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import animation

from days.day12.code.initialization import VarianceSimulator

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def anim_variance_evolution(layers: int = 50, fps: int = 10) -> Path:
    sim = VarianceSimulator(layers=layers, width=200)
    data = {
        "Xavier": sim.run("xavier"),
        "He": sim.run("he"),
        "Bad init": sim.run("bad"),
    }

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.set_title("Variance Across Layers")
    ax.set_xlim(0, layers)
    ax.set_ylim(1e-4, 1e4)
    ax.set_yscale("log")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Variance")
    ax.grid(True)

    colors = {
        "Xavier": "blue",
        "He": "green",
        "Bad init": "red",
    }
    lines = {name: ax.plot([], [], color=colors[name], label=name)[0] for name in data}
    ax.legend()

    def update(frame: int):
        for name, values in data.items():
            lines[name].set_data(range(frame), values[:frame])
        return list(lines.values())

    anim = animation.FuncAnimation(fig, update, frames=layers, interval=120)
    path = OUT_DIR / "01_variance_evolution.gif"
    anim.save(path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    return path


def main() -> None:
    artifact = anim_variance_evolution()
    print(f"Saved variance animation → {artifact}")


if __name__ == "__main__":
    main()
