"""Day 67: scaling laws and double-descent toy demo."""
from __future__ import annotations

import numpy as np


def conceptual_double_descent(capacity: np.ndarray) -> np.ndarray:
    return 1.0 / capacity + 0.02 * (capacity - 50.0) ** 2 / 1000.0


def power_law_loss(n: np.ndarray, alpha: float = 0.3, c: float = 1.0) -> np.ndarray:
    return c * n ** (-alpha)


def main() -> None:
    cap = np.linspace(1, 100, 200)
    err = conceptual_double_descent(cap)
    print('Min conceptual error:', float(err.min()))

    n = np.array([1e3, 1e4, 1e5, 1e6])
    print('Power-law losses:', power_law_loss(n))


if __name__ == '__main__':
    main()
