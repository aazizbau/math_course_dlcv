"""Day 44: SGD vs GD near a saddle (NumPy demo)."""
from __future__ import annotations

import numpy as np


def grad_saddle(x: np.ndarray) -> np.ndarray:
    return np.array([2 * x[0], -2 * x[1]])


def run_gd(x0: np.ndarray, lr: float = 0.1, steps: int = 50) -> np.ndarray:
    x = x0.copy()
    path = [x.copy()]
    for _ in range(steps):
        x = x - lr * grad_saddle(x)
        path.append(x.copy())
    return np.array(path)


def run_sgd(x0: np.ndarray, lr: float = 0.1, steps: int = 50, noise: float = 0.1) -> np.ndarray:
    rng = np.random.default_rng(0)
    x = x0.copy()
    path = [x.copy()]
    for _ in range(steps):
        grad = grad_saddle(x)
        x = x - lr * (grad + rng.normal(0, noise, size=2))
        path.append(x.copy())
    return np.array(path)


def main() -> None:
    x0 = np.array([1.0, 1.0])
    gd = run_gd(x0)
    sgd = run_sgd(x0)
    print("GD final:", gd[-1])
    print("SGD final:", sgd[-1])


if __name__ == "__main__":
    main()
