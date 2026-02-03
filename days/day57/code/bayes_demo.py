"""Day 57: Bayesian thinking demo (priors and MAP)."""
from __future__ import annotations

import numpy as np


def log_prior(theta: np.ndarray, sigma: float = 1.0) -> float:
    return float(-0.5 * np.sum(theta**2) / (sigma**2))


def log_likelihood(y: float, y_hat: float, sigma: float = 1.0) -> float:
    return float(-0.5 * ((y - y_hat) ** 2) / (sigma**2))


def main() -> None:
    theta = np.array([0.2, -0.5, 1.0])
    y, y_hat = 1.5, 1.2
    print("Log prior:", log_prior(theta))
    print("Log likelihood:", log_likelihood(y, y_hat))
    print("MAP objective (negative):", -(log_prior(theta) + log_likelihood(y, y_hat)))


if __name__ == "__main__":
    main()
