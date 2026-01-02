"""Day 34: condition number and stability demos (NumPy-based)."""
from __future__ import annotations

import numpy as np


def condition_number(A: np.ndarray) -> float:
    return float(np.linalg.cond(A))


def run_gd(A: np.ndarray, steps: int = 20, lr: float = 0.5) -> np.ndarray:
    x = np.array([1.0, 1.0])
    for _ in range(steps):
        x -= lr * A @ x
    return x


def main() -> None:
    A_good = np.eye(2)
    A_bad = np.array([[1.0, 0.0], [0.0, 1e-3]])

    print("Cond(A_good):", condition_number(A_good))
    print("Cond(A_bad):", condition_number(A_bad))
    print("GD good:", run_gd(A_good))
    print("GD bad:", run_gd(A_bad))


if __name__ == "__main__":
    main()
