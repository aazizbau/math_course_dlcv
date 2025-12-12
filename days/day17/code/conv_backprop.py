"""Day 17: naive convolution forward/backward for educational demos."""
from __future__ import annotations

import numpy as np


def conv2d_forward(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    h, w_in = x.shape
    k_h, k_w = w.shape
    out_h = h - k_h + 1
    out_w = w_in - k_w + 1
    y = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            y[i, j] = np.sum(x[i : i + k_h, j : j + k_w] * w)
    return y


def conv2d_backward(x: np.ndarray, w: np.ndarray, dY: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    k_h, k_w = w.shape
    out_h, out_w = dY.shape

    dW = np.zeros_like(w)
    for u in range(k_h):
        for v in range(k_w):
            patch = x[u : u + out_h, v : v + out_w]
            dW[u, v] = np.sum(dY * patch)

    rot_w = np.rot90(w, 2)
    pad = k_h - 1
    padded = np.pad(dY, pad_width=pad)
    dX = np.zeros_like(x)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            dX[i, j] = np.sum(padded[i : i + k_h, j : j + k_w] * rot_w)
    return dW, dX


def main() -> None:
    x = np.random.randn(5, 5)
    w = np.random.randn(3, 3)
    y = conv2d_forward(x, w)
    dY = np.ones_like(y)
    dW, dX = conv2d_backward(x, w, dY)
    print("Forward output shape:", y.shape)
    print("dW shape:", dW.shape, "dX shape:", dX.shape)


if __name__ == "__main__":
    main()
