"""Day 17 visualizations: conv window + gradient accumulation animations."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import animation, patches
import numpy as np

from days.day17.code.conv_backprop import conv2d_forward

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


IMG = np.random.rand(8, 8)
KERNEL = np.array([[1.0, -1.0], [2.0, 0.0]])
OUTPUT = conv2d_forward(IMG, KERNEL)


def anim_conv_backprop(dY: np.ndarray | None = None, fps: int = 15) -> Path:
    if dY is None:
        dY = np.ones_like(OUTPUT)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].set_title("Forward Pass Window")
    axes[1].set_title("Weight Gradient Accumulation")

    axes[0].imshow(IMG, cmap="gray")
    grad_map = axes[1].imshow(np.zeros_like(KERNEL), cmap="plasma", vmin=0, vmax=np.sum(np.abs(OUTPUT)) + 1)
    rect = patches.Rectangle((0, 0), KERNEL.shape[1], KERNEL.shape[0], fill=False, edgecolor="red", linewidth=2)
    axes[0].add_patch(rect)

    dW_accum = np.zeros_like(KERNEL)
    positions = [(i, j) for i in range(OUTPUT.shape[0]) for j in range(OUTPUT.shape[1])]

    def update(frame: int):
        i, j = positions[frame]
        rect.set_xy((j, i))
        patch = IMG[i : i + KERNEL.shape[0], j : j + KERNEL.shape[1]]
        dW_accum[:] += dY[i, j] * patch
        grad_map.set_data(dW_accum)
        axes[1].set_title(f"Accumulated dW\nstep {frame+1}/{len(positions)}")
        return rect, grad_map

    anim = animation.FuncAnimation(fig, update, frames=len(positions), interval=120)
    path = OUT_DIR / "01_conv_backprop.gif"
    anim.save(path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    return path


def main() -> None:
    gif_path = anim_conv_backprop()
    print(f"Saved convolution backprop animation → {gif_path}")


if __name__ == "__main__":
    main()
