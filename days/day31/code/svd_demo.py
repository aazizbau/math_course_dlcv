"""Day 31: SVD geometry demo (NumPy-based)."""
from __future__ import annotations

import numpy as np


def svd_summary(A: np.ndarray) -> None:
    U, S, Vt = np.linalg.svd(A)
    print("U:\n", U)
    print("Singular values:\n", S)
    print("Vt:\n", Vt)


def low_rank_approx(A: np.ndarray, k: int = 1) -> np.ndarray:
    U, S, Vt = np.linalg.svd(A)
    return (U[:, :k] * S[:k]) @ Vt[:k, :]


def main() -> None:
    A = np.array([[3.0, 1.0], [1.0, 3.0], [0.0, 2.0]])
    svd_summary(A)
    approx = low_rank_approx(A, k=1)
    print("Low-rank (k=1) approx:\n", approx)


if __name__ == "__main__":
    main()
