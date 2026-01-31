"""Day 54: margin and robust loss demo."""
from __future__ import annotations

import numpy as np


def hinge_loss(y: np.ndarray, f: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, 1.0 - y * f)


def huber_loss(r: np.ndarray, delta: float = 1.0) -> np.ndarray:
    return np.where(np.abs(r) <= delta, 0.5 * r**2, delta * (np.abs(r) - 0.5 * delta))


def main() -> None:
    y = np.array([1, -1, 1])
    f = np.array([2.0, 0.2, -0.5])
    print("Hinge losses:", hinge_loss(y, f))

    residuals = np.array([-2.0, -0.5, 0.5, 3.0])
    print("Huber losses:", huber_loss(residuals))


if __name__ == "__main__":
    main()
