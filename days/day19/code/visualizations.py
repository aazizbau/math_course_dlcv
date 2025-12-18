"""Day 19 visualizations: feature maps and filter dreaming animations."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_feature_maps(img: np.ndarray, kernel_bank: np.ndarray) -> Path:
    fig, axes = plt.subplots(1, kernel_bank.shape[0], figsize=(12, 4))
    for ax, kernel in zip(axes, kernel_bank):
        k = kernel.reshape(2, 2)
        fm = feature_map(img, k)
        ax.imshow(fm, cmap="viridis")
        ax.axis("off")
    path = OUT_DIR / "00_feature_maps.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def anim_filter_dreaming(steps: int = 50, lr: float = 0.1) -> Path:
    w = np.random.randn(3, 3)
    imgs = []
    for _ in range(steps):
        grad = np.random.randn(3, 3)
        w += lr * grad
        normed = (w - w.min()) / (w.max() - w.min() + 1e-6)
        imgs.append(normed.copy())

    fig, ax = plt.subplots(figsize=(3, 3))
    ax.axis("off")

    def update(i: int):
        ax.imshow(imgs[i], cmap="magma")
        return (ax,)

    anim = animation.FuncAnimation(fig, update, frames=len(imgs), interval=120)
    path = OUT_DIR / "01_filter_dreaming.gif"
    anim.save(path, writer=animation.PillowWriter(fps=6))
    plt.close(fig)
    return path


def feature_map(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    h, w = img.shape
    k_h, k_w = kernel.shape
    out = np.zeros((h - k_h + 1, w - k_w + 1))
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i, j] = np.sum(img[i : i + k_h, j : j + k_w] * kernel)
    return out


def main() -> None:
    img = np.random.rand(6, 6)
    kernel_bank = np.random.randn(3, 4)
    print("Saved feature maps →", plot_feature_maps(img, kernel_bank))
    print("Saved filter dreaming gif →", anim_filter_dreaming())


if __name__ == "__main__":
    main()
