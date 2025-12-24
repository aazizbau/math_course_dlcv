"""Day 23: segmentation evaluation metrics (NumPy demo)."""
from __future__ import annotations

import numpy as np


def _to_binary(mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (mask >= threshold).astype(np.uint8)


def confusion_counts(pred: np.ndarray, gt: np.ndarray, threshold: float = 0.5) -> tuple[int, int, int, int]:
    """Return TP, FP, FN, TN for binary masks."""

    pred_bin = _to_binary(pred, threshold)
    gt_bin = _to_binary(gt, 0.5)

    tp = int(np.sum((pred_bin == 1) & (gt_bin == 1)))
    fp = int(np.sum((pred_bin == 1) & (gt_bin == 0)))
    fn = int(np.sum((pred_bin == 0) & (gt_bin == 1)))
    tn = int(np.sum((pred_bin == 0) & (gt_bin == 0)))
    return tp, fp, fn, tn


def iou_score(pred: np.ndarray, gt: np.ndarray, threshold: float = 0.5, eps: float = 1e-6) -> float:
    tp, fp, fn, _ = confusion_counts(pred, gt, threshold)
    return float((tp + eps) / (tp + fp + fn + eps))


def dice_score(pred: np.ndarray, gt: np.ndarray, threshold: float = 0.5, eps: float = 1e-6) -> float:
    tp, fp, fn, _ = confusion_counts(pred, gt, threshold)
    return float((2 * tp + eps) / (2 * tp + fp + fn + eps))


def precision_recall_f1(pred: np.ndarray, gt: np.ndarray, threshold: float = 0.5, eps: float = 1e-6) -> tuple[float, float, float]:
    tp, fp, fn, _ = confusion_counts(pred, gt, threshold)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    f1 = (2 * precision * recall) / (precision + recall + eps)
    return float(precision), float(recall), float(f1)


def mean_iou(pred: np.ndarray, gt: np.ndarray, num_classes: int, eps: float = 1e-6) -> float:
    """Compute mean IoU for integer-labeled multi-class masks."""

    ious = []
    for c in range(num_classes):
        pred_c = pred == c
        gt_c = gt == c
        tp = np.sum(pred_c & gt_c)
        fp = np.sum(pred_c & ~gt_c)
        fn = np.sum(~pred_c & gt_c)
        iou = (tp + eps) / (tp + fp + fn + eps)
        ious.append(iou)
    return float(np.mean(ious))


def _erode(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    """Binary erosion with a square structuring element."""

    if radius <= 0:
        return mask.copy()
    pad = radius
    padded = np.pad(mask, pad, mode="constant")
    h, w = mask.shape
    out = np.zeros_like(mask)
    for i in range(h):
        for j in range(w):
            window = padded[i : i + 2 * radius + 1, j : j + 2 * radius + 1]
            out[i, j] = 1 if np.all(window == 1) else 0
    return out


def _dilate(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    """Binary dilation with a square structuring element."""

    if radius <= 0:
        return mask.copy()
    pad = radius
    padded = np.pad(mask, pad, mode="constant")
    h, w = mask.shape
    out = np.zeros_like(mask)
    for i in range(h):
        for j in range(w):
            window = padded[i : i + 2 * radius + 1, j : j + 2 * radius + 1]
            out[i, j] = 1 if np.any(window == 1) else 0
    return out


def boundary_f1(pred: np.ndarray, gt: np.ndarray, threshold: float = 0.5, tolerance: int = 2, eps: float = 1e-6) -> float:
    """Approximate boundary F1 using morphological edges and dilation tolerance."""

    pred_bin = _to_binary(pred, threshold)
    gt_bin = _to_binary(gt, 0.5)

    pred_edge = pred_bin ^ _erode(pred_bin, 1)
    gt_edge = gt_bin ^ _erode(gt_bin, 1)

    gt_dil = _dilate(gt_edge, tolerance)
    pred_dil = _dilate(pred_edge, tolerance)

    tp = np.sum((pred_edge == 1) & (gt_dil == 1))
    fp = np.sum((pred_edge == 1) & (gt_dil == 0))
    fn = np.sum((gt_edge == 1) & (pred_dil == 0))

    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    return float((2 * precision * recall) / (precision + recall + eps))


def main() -> None:
    rng = np.random.default_rng(42)
    gt = (rng.random((64, 64)) > 0.9).astype(np.float32)
    pred = rng.random((64, 64)).astype(np.float32)

    print("IoU:", iou_score(pred, gt))
    precision, recall, f1 = precision_recall_f1(pred, gt)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1:", f1)
    print("Dice:", dice_score(pred, gt))
    print("Boundary F1:", boundary_f1(pred, gt))


if __name__ == "__main__":
    main()
