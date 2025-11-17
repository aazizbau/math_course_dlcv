"""Day 5 computations: chain rule and backprop gradient demo."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


def forward(x: float, w1: float, w2: float) -> Tuple[float, float]:
    h = np.tanh(w1 * x)
    y = w2 * h
    return h, y


def backward(x: float, h: float, y: float, target: float, w1: float, w2: float) -> Tuple[float, float, float]:
    dL_dy = y - target
    dy_dw2 = h
    dy_dh = w2

    dh_dw1 = (1 - h**2) * x
    dh_dx = (1 - h**2) * w1

    dL_dw2 = dL_dy * dy_dw2
    dL_dw1 = dL_dy * dy_dh * dh_dw1
    dL_dx = dL_dy * dy_dh * dh_dx
    return dL_dw1, dL_dw2, dL_dx


@dataclass
class BackpropExample:
    x: float = 1.0
    target: float = 0.5
    w1: float = 1.2
    w2: float = -0.8

    def run(self) -> Tuple[float, float, float]:
        h, y = forward(self.x, self.w1, self.w2)
        grads = backward(self.x, h, y, self.target, self.w1, self.w2)
        return grads


def main() -> None:
    example = BackpropExample()
    grads = example.run()
    print("Gradients (dL/dw1, dL/dw2, dL/dx):", grads)


if __name__ == "__main__":
    main()
