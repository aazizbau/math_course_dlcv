"""Day 16: dilated convolution utilities and receptive-field helpers."""
from __future__ import annotations

import numpy as np


def dilated_conv2d(x: np.ndarray, w: np.ndarray, dilation: int = 2) -> np.ndarray:
    h, w_in = x.shape
    k_h, k_w = w.shape
    eff_h = k_h + (k_h - 1) * (dilation - 1)
    eff_w = k_w + (k_w - 1) * (dilation - 1)
    out_h = h - eff_h + 1
    out_w = w_in - eff_w + 1
    out = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            region = x[i : i + eff_h : dilation, j : j + eff_w : dilation]
            out[i, j] = np.sum(region * w)
    return out


def effective_kernel(k: int, dilation: int) -> int:
    return k + (k - 1) * (dilation - 1)


def main() -> None:
    x = np.arange(1, 17).reshape(4, 4)
    w = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    print("Dilated conv (d=2):")
    print(dilated_conv2d(x, w, dilation=2))
    for d in range(1, 5):
        print(f"d={d}, effective kernel={effective_kernel(3, d)}")


if __name__ == "__main__":
    main()
