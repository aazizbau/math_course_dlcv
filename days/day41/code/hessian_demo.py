"""Day 41: Hessian demos (NumPy-based)."""
from __future__ import annotations

import numpy as np


def f(v: np.ndarray) -> float:
    return v[0] ** 2 - v[1] ** 2


def hessian_numeric(func, x: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    n = len(x)
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dx_i = np.zeros(n)
            dx_j = np.zeros(n)
            dx_i[i] = eps
            dx_j[j] = eps
            H[i, j] = (
                func(x + dx_i + dx_j)
                - func(x + dx_i)
                - func(x + dx_j)
                + func(x)
            ) / (eps**2)
    return H


def main() -> None:
    x = np.array([1.0, 1.0])
    H = hessian_numeric(f, x)
    print("Hessian:\n", H)
    eigvals = np.linalg.eigvalsh(H)
    print("Eigenvalues:", eigvals)


if __name__ == "__main__":
    main()
