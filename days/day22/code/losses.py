"""Day 22: loss functions for dense prediction (NumPy demo)."""
from __future__ import annotations

import numpy as np


def binary_cross_entropy(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    """Compute mean binary cross-entropy loss on probabilities."""

    pred = np.clip(pred, eps, 1 - eps)
    loss = -(gt * np.log(pred) + (1 - gt) * np.log(1 - pred))
    return float(loss.mean())


def dice_loss(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    """Compute Dice loss for probabilistic masks."""

    pred_f = pred.reshape(-1)
    gt_f = gt.reshape(-1)
    intersection = np.sum(pred_f * gt_f)
    union = np.sum(pred_f) + np.sum(gt_f)
    dice = (2 * intersection + eps) / (union + eps)
    return float(1 - dice)


def iou_loss(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    """Compute IoU (Jaccard) loss for probabilistic masks."""

    pred_f = pred.reshape(-1)
    gt_f = gt.reshape(-1)
    intersection = np.sum(pred_f * gt_f)
    union = np.sum(pred_f) + np.sum(gt_f) - intersection
    iou = (intersection + eps) / (union + eps)
    return float(1 - iou)


def focal_loss(pred: np.ndarray, gt: np.ndarray, gamma: float = 2.0, eps: float = 1e-6) -> float:
    """Compute focal loss for probabilistic masks."""

    pred = np.clip(pred, eps, 1 - eps)
    ce = -(gt * np.log(pred) + (1 - gt) * np.log(1 - pred))
    focal = ((1 - pred) ** gamma) * ce
    return float(focal.mean())


def make_synthetic_batch(
    batch: int = 2,
    height: int = 64,
    width: int = 64,
    fg_ratio: float = 0.08,
    seed: int | None = 7,
) -> tuple[np.ndarray, np.ndarray]:
    """Create synthetic probabilistic predictions and binary masks."""

    rng = np.random.default_rng(seed)
    gt = (rng.random((batch, height, width)) < fg_ratio).astype(np.float32)
    pred = rng.random((batch, height, width)).astype(np.float32)
    return pred, gt


def main() -> None:
    pred, gt = make_synthetic_batch()
    print("BCE:", binary_cross_entropy(pred, gt))
    print("Dice:", dice_loss(pred, gt))
    print("IoU:", iou_loss(pred, gt))
    print("Focal:", focal_loss(pred, gt))
    print("BCE + Dice:", binary_cross_entropy(pred, gt) + dice_loss(pred, gt))


if __name__ == "__main__":
    main()
