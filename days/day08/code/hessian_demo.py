"""Day 8: Hessian, curvature, and Newton's method demo."""
from __future__ import annotations

import numpy as np


def loss(w: np.ndarray) -> float:
    x, y = w
    return 3 * x**2 + 0.8 * x * y + y**2


def grad(w: np.ndarray) -> np.ndarray:
    x, y = w
    return np.array([6 * x + 0.8 * y, 0.8 * x + 2 * y])


def hessian(_w: np.ndarray | None = None) -> np.ndarray:
    return np.array([[6.0, 0.8], [0.8, 2.0]])


def newton_step(w: np.ndarray) -> np.ndarray:
    H = hessian(None)
    g = grad(w)
    return w - np.linalg.inv(H) @ g


def demo() -> None:
    w = np.array([2.5, -2.0], dtype=float)
    print("Initial point:", w)
    for i in range(3):
        w = newton_step(w)
        print(f"Newton step {i+1} → {w}")


if __name__ == "__main__":
    demo()
