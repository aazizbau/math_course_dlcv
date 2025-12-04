"""Day 13 pooling helpers: max/avg pooling and strided conv demos."""
from __future__ import annotations

from typing import Tuple

import numpy as np


Array = np.ndarray


def pool2d(x: Array, kernel: int = 2, stride: int = 2, mode: str = "max") -> Array:
    """Apply 2D max/avg pooling on a 2D array."""
    h, w = x.shape
    out_h = (h - kernel) // stride + 1
    out_w = (w - kernel) // stride + 1
    out = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            window = x[i * stride : i * stride + kernel, j * stride : j * stride + kernel]
            if mode == "max":
                out[i, j] = np.max(window)
            elif mode == "avg":
                out[i, j] = np.mean(window)
            else:
                raise ValueError("mode must be 'max' or 'avg'")
    return out


def global_avg_pool(x: Array) -> float:
    return float(np.mean(x))


def strided_conv2d(x: Array, kernel: Array, stride: int = 2) -> Array:
    kh, kw = kernel.shape
    h, w = x.shape
    out_h = (h - kh) // stride + 1
    out_w = (w - kw) // stride + 1
    out = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            window = x[i * stride : i * stride + kh, j * stride : j * stride + kw]
            out[i, j] = np.sum(window * kernel)
    return out


def demo() -> None:
    img = np.array(
        [
            [1, 3, 2, 8],
            [4, 6, 5, 2],
            [7, 1, 0, 3],
            [2, 9, 4, 1],
        ]
    )
    print("Original:\n", img)
    print("MaxPool 2x2:\n", pool2d(img, mode="max"))
    print("AvgPool 2x2:\n", pool2d(img, mode="avg"))
    kernel = np.array([[1, -1], [-1, 1]])
    print("Strided conv (stride=2):\n", strided_conv2d(img, kernel, stride=2))


if __name__ == "__main__":
    demo()
