"""Day 53 visualizations: simple augmentation examples."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def make_square() -> np.ndarray:
    img = np.zeros((100, 100))
    img[30:70, 40:60] = 1
    return img


def plot_simple_augmentations() -> Path:
    img = make_square()
    translated = np.roll(img, 10, axis=1)
    rotated = np.rot90(img)

    fig, axes = plt.subplots(1, 3, figsize=(7, 2.5))
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('Original')
    axes[1].imshow(translated, cmap='gray')
    axes[1].set_title('Translated')
    axes[2].imshow(rotated, cmap='gray')
    axes[2].set_title('Rotated')
    for ax in axes:
        ax.axis('off')

    path = OUT_DIR / "00_simple_augmentations.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_distribution_shift() -> Path:
    rng = np.random.default_rng(0)
    samples = rng.normal(0, 1, size=1000)
    augmented = samples + rng.normal(0, 0.5, size=1000)

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist(samples, bins=30, alpha=0.6, label='Original')
    ax.hist(augmented, bins=30, alpha=0.6, label='Augmented')
    ax.set_title('Augmentation Expands Distribution')
    ax.legend()

    path = OUT_DIR / "01_distribution_shift.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main() -> None:
    path1 = plot_simple_augmentations()
    path2 = plot_distribution_shift()
    print("Saved plots →", path1, path2)


if __name__ == "__main__":
    main()
