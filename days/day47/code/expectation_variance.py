"""Day 47: expectation, variance, and averaging (NumPy demo)."""
from __future__ import annotations

import numpy as np


def sample_mean_variance(n: int = 1000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    samples = rng.normal(0, 1, size=n)
    return float(samples.mean()), float(samples.var())


def variance_of_mean(n: int = 1000, trials: int = 200) -> float:
    rng = np.random.default_rng(0)
    means = []
    for _ in range(trials):
        samples = rng.normal(0, 1, size=n)
        means.append(samples.mean())
    return float(np.var(means))


def main() -> None:
    mean, var = sample_mean_variance(10000)
    print("Mean:", mean)
    print("Variance:", var)
    for n in [1, 5, 20, 100, 500]:
        print("Var(mean) for n=", n, ":", variance_of_mean(n))


if __name__ == "__main__":
    main()
