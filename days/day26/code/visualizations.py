"""Day 26 visualizations: synthetic change detection pipeline."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .change_detection import make_pair

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_change_pipeline() -> Path:
    t1, t2, change = make_pair()
    diff = np.abs(t2 - t1)
    pred = np.clip(diff / (diff.max() + 1e-6), 0, 1)

    fig, axes = plt.subplots(1, 4, figsize=(10, 3))
    for ax, img, title in zip(
        axes,
        [t1, t2, pred, change],
        ["T1", "T2", "Predicted change", "Ground truth"],
    ):
        ax.imshow(img, cmap="gray")
        ax.set_title(title)
        ax.axis("off")

    path = OUT_DIR / "00_change_detection_pipeline.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main() -> None:
    path = plot_change_pipeline()
    print("Saved change detection plot →", path)


if __name__ == "__main__":
    main()
