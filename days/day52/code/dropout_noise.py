"""Day 52: dropout and noise intuition demo."""
from __future__ import annotations

import numpy as np


def apply_dropout(x: np.ndarray, p: float = 0.5, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mask = rng.random(size=x.shape) < p
    return x * mask / p


def main() -> None:
    x = np.ones(10)
    dropped = apply_dropout(x, p=0.7)
    print("Original:", x)
    print("After dropout:", dropped)


if __name__ == "__main__":
    main()
