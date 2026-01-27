"""Day 49: entropy, cross-entropy, and KL divergence demos."""
from __future__ import annotations

import numpy as np


def entropy(p: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1)
    return float(-np.sum(p * np.log(p)))


def cross_entropy(p: np.ndarray, q: np.ndarray) -> float:
    q = np.clip(q, 1e-12, 1)
    return float(-np.sum(p * np.log(q)))


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1)
    q = np.clip(q, 1e-12, 1)
    return float(np.sum(p * np.log(p / q)))


def main() -> None:
    p = np.array([1.0, 0.0, 0.0])
    q = np.array([0.7, 0.2, 0.1])
    h = entropy(p)
    ce = cross_entropy(p, q)
    kl = kl_divergence(p, q)

    print("Entropy:", h)
    print("Cross-Entropy:", ce)
    print("KL Divergence:", kl)


if __name__ == "__main__":
    main()
