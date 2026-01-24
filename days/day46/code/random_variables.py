"""Day 46: random variables and distributions (NumPy demo)."""
from __future__ import annotations

import numpy as np


def sample_gaussian(n: int = 10000, mean: float = 0.0, std: float = 1.0) -> np.ndarray:
    return np.random.normal(mean, std, size=n)


def sample_bernoulli(n: int = 10000, p: float = 0.3) -> np.ndarray:
    return np.random.binomial(1, p, size=n)


def sample_categorical(n: int = 10000, probs: np.ndarray | None = None) -> np.ndarray:
    if probs is None:
        probs = np.array([0.2, 0.5, 0.3])
    return np.random.choice(len(probs), size=n, p=probs)


def running_mean(samples: np.ndarray) -> np.ndarray:
    return np.cumsum(samples) / (np.arange(len(samples)) + 1)


def main() -> None:
    gauss = sample_gaussian()
    bern = sample_bernoulli()
    cat = sample_categorical()

    print("Gaussian mean/std:", gauss.mean(), gauss.std())
    print("Bernoulli mean:", bern.mean())
    print("Categorical counts:", np.bincount(cat))
    print("Running mean tail:", running_mean(gauss)[-5:])


if __name__ == "__main__":
    main()
