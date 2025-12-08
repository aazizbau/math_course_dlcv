"""Day 15 visualizations: stride movement and padding effects."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import animation, patches
import numpy as np

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


IMG = np.random.rand(20, 20)


def anim_stride(stride: int = 1, kernel: int = 3, fps: int = 10) -> Path:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(IMG, cmap="gray")
    ax.set_title(f"Stride = {stride}")
    ax.axis("off")

    rect = patches.Rectangle((0, 0), kernel, kernel, fill=False, edgecolor="red", linewidth=2)
    ax.add_patch(rect)

    positions = [
        (i, j)
        for i in range(0, IMG.shape[0] - kernel + 1, stride)
        for j in range(0, IMG.shape[1] - kernel + 1, stride)
    ]

    def update(k: int):
        i, j = positions[k]
        rect.set_xy((j, i))
        return (rect,)

    anim = animation.FuncAnimation(fig, update, frames=len(positions), interval=100)
    path = OUT_DIR / f"01_stride_{stride}.gif"
    anim.save(path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    return path


def padding_visual(pad: int) -> Path:
    img = np.random.rand(10, 10)
    padded = np.pad(img, pad, mode="constant")

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(padded, cmap="gray")
    ax.set_title(f"Padding = {pad}")
    ax.axis("off")
    path = OUT_DIR / f"02_padding_{pad}.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main() -> None:
    path_stride1 = anim_stride(1)
    print(f"Saved stride=1 animation → {path_stride1}")
    path_stride2 = anim_stride(2)
    print(f"Saved stride=2 animation → {path_stride2}")
    for pad in (0, 1, 2):
        print(f"Saved padding visual → {padding_visual(pad)}")


if __name__ == "__main__":
    main()
