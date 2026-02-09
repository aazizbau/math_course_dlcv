"""Day 61: self-supervised learning demo (cosine similarity)."""
from __future__ import annotations

import numpy as np


def cosine_similarity(z1: np.ndarray, z2: np.ndarray) -> float:
    return float(np.dot(z1, z2) / (np.linalg.norm(z1) * np.linalg.norm(z2)))


def main() -> None:
    z1 = np.array([0.8, 0.1])
    z2 = np.array([0.75, 0.15])
    print("Similarity:", cosine_similarity(z1, z2))


if __name__ == "__main__":
    main()
