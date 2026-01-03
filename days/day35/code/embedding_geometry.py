"""Day 35: embedding geometry demos (NumPy-based)."""
from __future__ import annotations

import numpy as np


def l2_distance(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.linalg.norm(x - y))


def cosine_similarity(x: np.ndarray, y: np.ndarray, eps: float = 1e-6) -> float:
    denom = np.linalg.norm(x) * np.linalg.norm(y) + eps
    return float(np.dot(x, y) / denom)


def detect_collapse(embeddings: np.ndarray) -> np.ndarray:
    cov = np.cov(embeddings, rowvar=False)
    eigvals = np.linalg.eigvalsh(cov)
    return eigvals


def main() -> None:
    x = np.array([1.0, 2.0])
    y = np.array([2.0, 4.0])
    print("L2 distance:", l2_distance(x, y))
    print("Cosine similarity:", cosine_similarity(x, y))

    rng = np.random.default_rng(0)
    embeddings = rng.normal(0, 1, size=(100, 64))
    eigvals = detect_collapse(embeddings)
    print("Smallest eigenvalues:", eigvals[:5])


if __name__ == "__main__":
    main()
