"""Day 51: regularization geometry demo."""
from __future__ import annotations

import numpy as np


def l1_penalty(theta: np.ndarray) -> float:
    return float(np.sum(np.abs(theta)))


def l2_penalty(theta: np.ndarray) -> float:
    return float(np.sum(theta**2))


def main() -> None:
    theta = np.array([1.0, -2.0, 0.5])
    print("L1 penalty:", l1_penalty(theta))
    print("L2 penalty:", l2_penalty(theta))


if __name__ == "__main__":
    main()
