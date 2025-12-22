"""Day 22 visualizations: compare loss curves under imbalance."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .losses import binary_cross_entropy, dice_loss, focal_loss, iou_loss

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _loss_vs_foreground_prob(
    fg_ratio: float = 0.05, bg_prob: float = 0.05, steps: int = 200
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    probs = np.linspace(0.01, 0.99, steps)
    bce_vals = []
    dice_vals = []
    iou_vals = []
    focal_vals = []

    fg_pixels = int(1000 * fg_ratio)
    bg_pixels = 1000 - fg_pixels
    gt = np.concatenate([np.ones(fg_pixels), np.zeros(bg_pixels)])

    for p in probs:
        pred = np.concatenate(
            [np.full(fg_pixels, p), np.full(bg_pixels, bg_prob)]
        )
        bce_vals.append(binary_cross_entropy(pred, gt))
        dice_vals.append(dice_loss(pred, gt))
        iou_vals.append(iou_loss(pred, gt))
        focal_vals.append(focal_loss(pred, gt))

    return probs, np.array(bce_vals), np.array(dice_vals), np.array(iou_vals), np.array(focal_vals)


def plot_loss_curves() -> Path:
    probs, bce_vals, dice_vals, iou_vals, focal_vals = _loss_vs_foreground_prob()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(probs, bce_vals, label="BCE")
    ax.plot(probs, dice_vals, label="Dice")
    ax.plot(probs, iou_vals, label="IoU")
    ax.plot(probs, focal_vals, label="Focal")
    ax.set_title("Loss Behavior vs Foreground Confidence")
    ax.set_xlabel("Predicted foreground probability (foreground pixels)")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()

    path = OUT_DIR / "00_loss_curves.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main() -> None:
    path = plot_loss_curves()
    print("Saved loss curves →", path)


if __name__ == "__main__":
    main()
