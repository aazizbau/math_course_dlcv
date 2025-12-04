"""Day 13 visualizations: pooling window and variance reduction."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import animation, patches
import numpy as np

from days.day13.code.pooling import pool2d

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def anim_sliding_maxpool(size: int = 6, kernel: int = 2, fps: int = 5) -> Path:
    img = np.random.randn(size, size)
    positions = []
    for i in range(size - kernel + 1):
        for j in range(size - kernel + 1):
            positions.append((i, j))

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(img, cmap="viridis")
    ax.set_title("Max Pooling Window Movement")
    rect = patches.Rectangle((0, 0), kernel, kernel, fill=False, edgecolor="red", linewidth=2)
    ax.add_patch(rect)

    def update(idx: int):
        i, j = positions[idx]
        rect.set_xy((j, i))
        return (rect,)

    anim = animation.FuncAnimation(fig, update, frames=len(positions), interval=300)
    path = OUT_DIR / "01_maxpool_sliding.gif"
    anim.save(path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    return path


def plot_pooling_example() -> Path:
    img = np.array(
        [
            [1, 3, 2, 8],
            [4, 6, 5, 2],
            [7, 1, 0, 3],
            [2, 9, 4, 1],
        ]
    )
    maxpooled = pool2d(img, mode="max")
    avgpooled = pool2d(img, mode="avg")

    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    for ax, data, title in zip(axes, [img, maxpooled, avgpooled], ["Original", "MaxPool", "AvgPool"]):
        im = ax.imshow(data, cmap="magma")
        ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.046)
    path = OUT_DIR / "00_pooling_examples.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main() -> None:
    img_path = plot_pooling_example()
    print(f"Saved pooling example plot → {img_path}")
    gif_path = anim_sliding_maxpool()
    print(f"Saved sliding max-pool animation → {gif_path}")


if __name__ == "__main__":
    main()
