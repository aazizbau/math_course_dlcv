"""Day 18 visualizations: FC vs Conv gradient coverage."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import animation, patches
import numpy as np

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


IMG = np.random.rand(10, 10)
GLOBAL_GRAD = np.random.rand(10, 10)


def anim_fc_vs_conv(window: int = 3, fps: int = 10) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(GLOBAL_GRAD, cmap="viridis")
    axes[0].set_title("FC Gradient\n(global influence)")
    axes[0].axis("off")

    axes[1].imshow(IMG, cmap="gray")
    axes[1].set_title("Conv Gradient\n(local sliding influence)")
    axes[1].axis("off")

    rect = patches.Rectangle((0, 0), window, window, fill=False, edgecolor="red", linewidth=2)
    axes[1].add_patch(rect)

    positions = [(i, j) for i in range(IMG.shape[0] - window + 1) for j in range(IMG.shape[1] - window + 1)]

    def update(frame: int):
        i, j = positions[frame]
        rect.set_xy((j, i))
        return (rect,)

    anim = animation.FuncAnimation(fig, update, frames=len(positions), interval=80)
    path = OUT_DIR / "01_fc_vs_conv.gif"
    anim.save(path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    return path


def main() -> None:
    print("Saved FC vs Conv animation →", anim_fc_vs_conv())


if __name__ == "__main__":
    main()
