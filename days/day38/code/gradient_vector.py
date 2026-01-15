"""Day 38: gradient vector and steepest descent (NumPy demo)."""
from __future__ import annotations

import numpy as np


def grad_f(x: float, y: float) -> np.ndarray:
    return np.array([2 * x, 2 * y])


def grad_quad(A: np.ndarray, x: np.ndarray) -> np.ndarray:
    return A @ x


def main() -> None:
    x, y = 1.0, -1.5
    g = grad_f(x, y)
    print("Gradient:", g)
    print("Steepest descent:", -g)

    A = np.array([[10.0, 0.0], [0.0, 1.0]])
    x_vec = np.array([1.0, 1.0])
    print("Quadratic gradient:", grad_quad(A, x_vec))


if __name__ == "__main__":
    main()
