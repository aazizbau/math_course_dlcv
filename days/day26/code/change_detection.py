"""Day 26: change detection utilities (NumPy demo)."""
from __future__ import annotations

import numpy as np


def siamese_diff(f1: np.ndarray, f2: np.ndarray) -> np.ndarray:
    """Absolute difference between feature tensors."""

    return np.abs(f1 - f2)


def binary_cross_entropy(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    pred = np.clip(pred, eps, 1 - eps)
    loss = -(gt * np.log(pred) + (1 - gt) * np.log(1 - pred))
    return float(loss.mean())


def dice_loss(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    pred_f = pred.reshape(-1)
    gt_f = gt.reshape(-1)
    intersection = np.sum(pred_f * gt_f)
    union = np.sum(pred_f) + np.sum(gt_f)
    dice = (2 * intersection + eps) / (union + eps)
    return float(1 - dice)


def focal_loss(pred: np.ndarray, gt: np.ndarray, gamma: float = 2.0, eps: float = 1e-6) -> float:
    pred = np.clip(pred, eps, 1 - eps)
    ce = -(gt * np.log(pred) + (1 - gt) * np.log(1 - pred))
    focal = ((1 - pred) ** gamma) * ce
    return float(focal.mean())


def make_pair(height: int = 64, width: int = 64, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return synthetic T1, T2 images and a change mask."""

    rng = np.random.default_rng(seed)
    t1 = rng.random((height, width)).astype(np.float32)
    t2 = t1.copy()

    change = np.zeros((height, width), dtype=np.float32)
    rr, cc = np.ogrid[:height, :width]
    blob = (rr - 38) ** 2 + (cc - 36) ** 2 <= 9**2
    change[blob] = 1
    t2[blob] = np.clip(t2[blob] + 0.6, 0, 1)

    noise = rng.normal(0, 0.03, size=(height, width)).astype(np.float32)
    t2 = np.clip(t2 + noise, 0, 1)
    return t1, t2, change


def main() -> None:
    t1, t2, change = make_pair()
    diff = np.abs(t2 - t1)
    pred = np.clip(diff / diff.max(), 0, 1)

    print("BCE:", binary_cross_entropy(pred, change))
    print("Dice:", dice_loss(pred, change))
    print("Focal:", focal_loss(pred, change))
    print("BCE + Dice:", binary_cross_entropy(pred, change) + dice_loss(pred, change))


if __name__ == "__main__":
    main()
