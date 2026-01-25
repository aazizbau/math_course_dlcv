"""Day 48: Maximum Likelihood Estimation (MLE) demos."""
from __future__ import annotations

import numpy as np


def neg_log_likelihood_bernoulli(p: float, y: int) -> float:
    p = np.clip(p, 1e-8, 1 - 1e-8)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)))


def neg_log_likelihood_gaussian(y: float, mu: float, sigma: float = 1.0) -> float:
    return float(0.5 * ((y - mu) ** 2) / (sigma**2))


def main() -> None:
    p = 0.8
    y = 1
    print("Bernoulli NLL (y=1, p=0.8):", neg_log_likelihood_bernoulli(p, y))

    y_val = 2.0
    mu = 1.5
    print("Gaussian NLL (y=2.0, mu=1.5):", neg_log_likelihood_gaussian(y_val, mu))


if __name__ == "__main__":
    main()
