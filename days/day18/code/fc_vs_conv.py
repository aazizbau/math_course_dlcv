"""Day 18: basic FC vs conv gradient demos."""
from __future__ import annotations

import numpy as np


def fc_backward(x: np.ndarray, W: np.ndarray, dy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dW = np.outer(dy, x)
    dx = W.T @ dy
    return dW, dx


def conv_weight_grad(x: np.ndarray, w: np.ndarray, dy: np.ndarray) -> np.ndarray:
    k_h, k_w = w.shape
    out_h, out_w = dy.shape
    dW = np.zeros_like(w)
    for u in range(k_h):
        for v in range(k_w):
            patch = x[u : u + out_h, v : v + out_w]
            dW[u, v] = np.sum(dy * patch)
    return dW


def main() -> None:
    x = np.random.randn(5)
    W = np.random.randn(3, 5)
    dy_fc = np.random.randn(3)
    dW_fc, dx_fc = fc_backward(x, W, dy_fc)
    print("FC dW shape:", dW_fc.shape, "dx shape:", dx_fc.shape)

    img = np.random.randn(6, 6)
    kernel = np.random.randn(3, 3)
    dy_conv = np.random.randn(4, 4)
    dW_conv = conv_weight_grad(img, kernel, dy_conv)
    print("Conv dW:\n", dW_conv)


if __name__ == "__main__":
    main()
