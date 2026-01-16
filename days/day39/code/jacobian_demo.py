"""Day 39: Jacobian demos for vector-valued functions (NumPy-based)."""
from __future__ import annotations

import numpy as np


def f(v: np.ndarray) -> np.ndarray:
    x, y = v
    return np.array([x**2, x * y])


def jacobian_numeric(func, v: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    n = v.shape[0]
    out = func(v)
    J = np.zeros((out.size, n))
    for i in range(n):
        dv = np.zeros_like(v)
        dv[i] = eps
        J[:, i] = (func(v + dv) - out) / eps
    return J


def jacobian_analytic(v: np.ndarray) -> np.ndarray:
    x, y = v
    return np.array([[2 * x, 0.0], [y, x]])


def main() -> None:
    v = np.array([2.0, 3.0])
    J_num = jacobian_numeric(f, v)
    J_ana = jacobian_analytic(v)
    print("Numeric Jacobian:\n", J_num)
    print("Analytic Jacobian:\n", J_ana)


if __name__ == "__main__":
    main()
