"""Day 19: toy feature-map and filter activation demos (NumPy-based)."""
from __future__ import annotations

import numpy as np


def feature_map(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    h, w_in = x.shape
    k_h, k_w = w.shape
    out_h = h - k_h + 1
    out_w = w_in - k_w + 1
    y = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            y[i, j] = np.sum(x[i : i + k_h, j : j + k_w] * w)
    return y


def gradient_ascent_filter(w_shape: tuple[int, int], steps: int = 100, lr: float = 0.1) -> np.ndarray:
    w = np.random.randn(*w_shape)
    pattern = np.random.randn(*w_shape)
    for _ in range(steps):
        loss_grad = pattern
        w += lr * loss_grad
    return w


def main() -> None:
    img = np.random.rand(6, 6)
    kernel = np.array([[1, -1], [0.5, 0]])
    fm = feature_map(img, kernel)
    print("Feature map shape:", fm.shape)
    dream = gradient_ascent_filter((3, 3))
    print("Dream filter pattern:\n", dream)


if __name__ == "__main__":
    main()
