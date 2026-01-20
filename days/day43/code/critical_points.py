"""Day 43: critical points, Hessian classification (NumPy demo)."""
from __future__ import annotations

import numpy as np


def f(x: np.ndarray) -> float:
    return x[0] ** 2 - x[1] ** 2


def grad_f(x: np.ndarray) -> np.ndarray:
    return np.array([2 * x[0], -2 * x[1]])


def hessian_f(_: np.ndarray) -> np.ndarray:
    return np.array([[2.0, 0.0], [0.0, -2.0]])


def classify_hessian(H: np.ndarray) -> str:
    eigvals = np.linalg.eigvalsh(H)
    if np.all(eigvals > 0):
        return "minimum"
    if np.all(eigvals < 0):
        return "maximum"
    if np.any(eigvals == 0):
        return "flat/degenerate"
    return "saddle"


def main() -> None:
    x0 = np.array([0.0, 0.0])
    grad = grad_f(x0)
    H = hessian_f(x0)
    print("Gradient at (0,0):", grad)
    print("Hessian:\n", H)
    print("Critical point type:", classify_hessian(H))


if __name__ == "__main__":
    main()
