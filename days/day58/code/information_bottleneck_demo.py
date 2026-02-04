"""Day 58: information bottleneck toy compression demo."""
from __future__ import annotations

import numpy as np


def compress(X: np.ndarray) -> np.ndarray:
    return X @ np.array([[1.0], [1.0]])


def main() -> None:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((500, 2))
    Y = (X[:, 0] + X[:, 1] > 0).astype(int)
    T = compress(X)
    print("Input shape:", X.shape)
    print("Compressed shape:", T.shape)
    print("Class balance:", float(Y.mean()))


if __name__ == "__main__":
    main()
