"""Day 59: manifold intuition demo (synthetic data)."""
from __future__ import annotations

import numpy as np


def make_swiss_roll(n: int = 1000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = 1.5 * np.pi * (1 + 2 * rng.random(n))
    x = t * np.cos(t)
    y = 21 * rng.random(n)
    z = t * np.sin(t)
    return np.stack([x, y, z], axis=1)


def main() -> None:
    X = make_swiss_roll(500)
    print("Swiss roll shape:", X.shape)
    print("Mean:", X.mean(axis=0))


if __name__ == "__main__":
    main()
