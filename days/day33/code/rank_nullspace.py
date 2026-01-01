"""Day 33: rank and null space demos (NumPy-based)."""
from __future__ import annotations

import numpy as np


def rank_and_nullspace(A: np.ndarray, tol: float = 1e-6) -> tuple[int, np.ndarray]:
    """Return rank and a basis for the null space."""

    U, S, Vt = np.linalg.svd(A)
    rank = int(np.sum(S > tol))
    null_space = Vt[rank:].T
    return rank, null_space


def low_rank_projection(X: np.ndarray, k: int = 1) -> np.ndarray:
    """Project data onto top-k singular vectors."""

    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    return (U[:, :k] * S[:k]) @ Vt[:k, :]


def main() -> None:
    A = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [1.0, 1.0, 1.0]])
    rank, null_space = rank_and_nullspace(A)
    print("Rank:", rank)
    print("Null space basis:\n", null_space)

    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, size=(100, 3))
    X_proj = low_rank_projection(X, k=1)
    print("Projection variance:", np.var(X_proj, axis=0))


if __name__ == "__main__":
    main()
