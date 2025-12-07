"""Day 14 visualizations: receptive-field growth and multi-scale context."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import animation, patches
import numpy as np

from days.day14.code.receptive_field import RFSimulator

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_rf_progression(layers: list[tuple[int, int]]) -> Path:
    simulator = RFSimulator(layers=len(layers))
    history = simulator.run()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, len(history) + 1), history, marker="o")
    ax.set_title("Receptive Field Growth")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Receptive field (pixels)")
    ax.grid(True)
    path = OUT_DIR / "00_rf_progression.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def anim_rf_growth(size: int = 32, layers: int = 6, kernel: int = 3, fps: int = 2) -> Path:
    img = np.random.rand(size, size)
    simulator = RFSimulator(layers=layers, kernel=kernel)
    rf_sizes = simulator.run()

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(img, cmap="gray")
    ax.set_title("Growing Receptive Field")
    ax.axis("off")
    rect = patches.Rectangle((size // 2, size // 2), 1, 1, fill=False, edgecolor="red", linewidth=2)
    ax.add_patch(rect)

    def update(i: int):
        size_rf = rf_sizes[i]
        # clamp to image borders
        size_rf = min(size_rf, size)
        rect.set_width(size_rf)
        rect.set_height(size_rf)
        rect.set_xy((size // 2 - size_rf / 2, size // 2 - size_rf / 2))
        ax.set_title(f"Layer {i+1}: RF {size_rf}x{size_rf}")
        return (rect,)

    anim = animation.FuncAnimation(fig, update, frames=len(rf_sizes), interval=600)
    path = OUT_DIR / "01_receptive_field_growth.gif"
    anim.save(path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    return path


def main() -> None:
    plot_path = plot_rf_progression([(3, 1)] * 6)
    print(f"Saved RF plot → {plot_path}")
    gif_path = anim_rf_growth()
    print(f"Saved RF gif → {gif_path}")


if __name__ == "__main__":
    main()
