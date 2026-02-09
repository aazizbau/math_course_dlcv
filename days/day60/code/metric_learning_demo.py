"""Day 60: metric learning demo (contrastive + triplet)."""
from __future__ import annotations

import numpy as np


def contrastive_loss(z1: np.ndarray, z2: np.ndarray, y: int, margin: float = 1.0) -> float:
    dist = np.linalg.norm(z1 - z2)
    if y == 1:
        return float(dist**2)
    return float(max(0.0, margin - dist) ** 2)


def triplet_loss(a: np.ndarray, p: np.ndarray, n: np.ndarray, margin: float = 0.5) -> float:
    d_ap = np.linalg.norm(a - p)
    d_an = np.linalg.norm(a - n)
    return float(max(0.0, d_ap**2 - d_an**2 + margin))


def main() -> None:
    z1 = np.array([0.0, 0.0])
    z2 = np.array([0.5, 0.5])
    z3 = np.array([2.0, 2.0])

    print("Contrastive (pos):", contrastive_loss(z1, z2, 1))
    print("Contrastive (neg):", contrastive_loss(z1, z3, 0))
    print("Triplet:", triplet_loss(z1, z2, z3))


if __name__ == "__main__":
    main()
