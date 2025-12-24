"""Day 23 visualizations: metric curves across thresholds."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .metrics import boundary_f1, dice_score, iou_score, precision_recall_f1

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _make_circle_mask(size: int = 96, radius: int = 28) -> np.ndarray:
    yy, xx = np.ogrid[:size, :size]
    cy = size // 2
    cx = size // 2
    return ((yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2).astype(np.float32)


def _synthetic_prediction(gt: np.ndarray, noise: float = 0.2, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pred = gt + rng.normal(0, noise, size=gt.shape)
    pred = np.clip(pred, 0.0, 1.0)
    return pred.astype(np.float32)


def plot_metric_curves() -> Path:
    gt = _make_circle_mask()
    pred = _synthetic_prediction(gt)

    thresholds = np.linspace(0.05, 0.95, 40)
    ious = []
    dices = []
    f1s = []
    b_f1s = []

    for t in thresholds:
        ious.append(iou_score(pred, gt, threshold=t))
        dices.append(dice_score(pred, gt, threshold=t))
        _, _, f1 = precision_recall_f1(pred, gt, threshold=t)
        f1s.append(f1)
        b_f1s.append(boundary_f1(pred, gt, threshold=t, tolerance=2))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(thresholds, ious, label="IoU")
    ax.plot(thresholds, dices, label="Dice")
    ax.plot(thresholds, f1s, label="F1")
    ax.plot(thresholds, b_f1s, label="Boundary F1")
    ax.set_title("Segmentation Metrics vs Threshold")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()

    path = OUT_DIR / "00_metrics_vs_threshold.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main() -> None:
    path = plot_metric_curves()
    print("Saved metric curves →", path)


if __name__ == "__main__":
    main()
