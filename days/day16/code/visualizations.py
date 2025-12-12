"""Day 16 visualizations: receptive-field expansion via dilation."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import animation, patches
import numpy as np

from days.day16.code.dilated_conv import effective_kernel

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


IMG = np.random.rand(25, 25)


def anim_dilation_growth(k: int = 3, dilations: list[int] | None = None, fps: int = 1) -> Path:
    if dilations is None:
        dilations = [1, 2, 3, 4]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(IMG, cmap="gray")
    ax.axis("off")
    rect = patches.Rectangle((0, 0), k, k, fill=False, edgecolor="red", linewidth=2)
    ax.add_patch(rect)

    def update(i: int):
        d = dilations[i]
        eff = effective_kernel(k, d)
        rect.set_width(eff)
        rect.set_height(eff)
        rect.set_xy((12 - eff / 2, 12 - eff / 2))
        ax.set_title(f"Dilation = {d}, Effective RF = {eff}×{eff}")
        return (rect,)

    anim = animation.FuncAnimation(fig, update, frames=len(dilations), interval=800)
    path = OUT_DIR / "01_dilation_rf_growth.gif"
    anim.save(path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    return path


def plot_effective_kernel(k: int = 3, max_d: int = 6) -> Path:
    ds = np.arange(1, max_d + 1)
    eff = [effective_kernel(k, d) for d in ds]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ds, eff, marker="o")
    ax.set_xlabel("Dilation")
    ax.set_ylabel("Effective kernel size")
    ax.set_title("Dilated Kernel Growth")
    ax.grid(True)
    path = OUT_DIR / "00_effective_kernel.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main() -> None:
    print("Saved kernel plot →", plot_effective_kernel())
    print("Saved dilation gif →", anim_dilation_growth())


if __name__ == "__main__":
    main()
