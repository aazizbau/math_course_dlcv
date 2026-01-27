"""Day 50: bias-variance tradeoff demo."""
from __future__ import annotations

import numpy as np


def fit_linear(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    a = np.vstack([x, np.ones_like(x)]).T
    coeffs, _, _, _ = np.linalg.lstsq(a, y, rcond=None)
    return a @ coeffs


def main() -> None:
    rng = np.random.default_rng(0)
    x = np.linspace(-1, 1, 100)
    true = x**3

    y_noisy = true + 0.2 * rng.standard_normal(len(x))
    linear_pred = fit_linear(x, y_noisy)

    print("Linear fit MSE:", float(np.mean((linear_pred - true) ** 2)))
    print("Noise variance:", float(np.var(y_noisy - true)))


if __name__ == "__main__":
    main()
