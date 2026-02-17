"""Day 68: optimization geometry toy demos."""
from __future__ import annotations

import numpy as np


def sharp_basin(x: np.ndarray) -> np.ndarray:
    return x**4 + 0.1 * x**2


def flat_basin(x: np.ndarray) -> np.ndarray:
    return 0.2 * x**4 + 0.1 * x**2


def sgd_step(theta: float, lr: float, noise_std: float, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    grad = 4 * theta**3 + 0.2 * theta
    noise = rng.normal(0.0, noise_std)
    return float(theta - lr * grad + noise)


def main() -> None:
    x = np.linspace(-3, 3, 5)
    print('Sharp sample:', sharp_basin(x))
    print('Flat sample:', flat_basin(x))
    print('One noisy SGD step from theta=1.5:', sgd_step(1.5, lr=0.01, noise_std=0.02))


if __name__ == '__main__':
    main()
