"""Day 64: distribution shift and simple diagnostics."""
from __future__ import annotations

import numpy as np


def sample_train_test(n: int = 1000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train = rng.normal(0.0, 1.0, n)
    test = rng.normal(2.0, 1.0, n)
    return train, test


def mean_shift(train: np.ndarray, test: np.ndarray) -> float:
    return float(test.mean() - train.mean())


def main() -> None:
    train, test = sample_train_test()
    print("Train mean/std:", float(train.mean()), float(train.std()))
    print("Test mean/std:", float(test.mean()), float(test.std()))
    print("Mean shift:", mean_shift(train, test))


if __name__ == "__main__":
    main()
