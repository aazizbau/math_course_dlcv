"""Day 25 visualizations: post-processing steps on a noisy mask."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .postprocessing import closing, opening, remove_small_components

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _make_noisy_mask(seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mask = np.zeros((96, 96), dtype=np.uint8)
    rr, cc = np.ogrid[:96, :96]
    circle = (rr - 40) ** 2 + (cc - 40) ** 2 <= 18**2
    mask[circle] = 1
    mask[10:14, 70:74] = 1
    mask[70:72, 10:12] = 1
    noise = rng.random(mask.shape) > 0.985
    mask[noise] = 1
    holes = rng.random(mask.shape) > 0.995
    mask[holes] = 0
    return mask


def plot_postprocessing() -> Path:
    raw = _make_noisy_mask()
    opened = opening(raw, radius=1)
    closed = closing(opened, radius=1)
    cleaned = remove_small_components(closed, min_area=40)

    fig, axes = plt.subplots(1, 4, figsize=(10, 3))
    for ax, img, title in zip(
        axes,
        [raw, opened, closed, cleaned],
        ["Raw", "Opening", "Closing", "Components"],
    ):
        ax.imshow(img, cmap="gray")
        ax.set_title(title)
        ax.axis("off")

    path = OUT_DIR / "00_postprocessing_steps.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main() -> None:
    path = plot_postprocessing()
    print("Saved post-processing plot →", path)


if __name__ == "__main__":
    main()
