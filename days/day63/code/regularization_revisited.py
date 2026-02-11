"""Day 63: regularization unification toy demos."""
from __future__ import annotations

import numpy as np


def model(x: float, w: float) -> float:
    return float(np.tanh(w * x))


def sensitivity(x: float, w: float, dw: float) -> float:
    return abs(model(x, w + dw) - model(x, w))


def main() -> None:
    x, w = 1.0, 5.0
    print("Original:", model(x, w))
    print("Small weight change:", model(x, w + 0.1))
    print("Large weight change:", model(x, w + 1.0))
    print("Sensitivity(dw=0.1):", sensitivity(x, w, 0.1))


if __name__ == "__main__":
    main()
