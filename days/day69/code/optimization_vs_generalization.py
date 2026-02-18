"""Day 69: optimization vs generalization toy demos."""
from __future__ import annotations

import numpy as np


def generalization_gap(train_loss: float, test_loss: float) -> float:
    return float(test_loss - train_loss)


def poly_overfit_curve(seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.linspace(-1, 1, 20)
    y = x**2 + rng.normal(0, 0.1, size=x.shape)
    coeffs = np.polyfit(x, y, 15)
    poly = np.poly1d(coeffs)
    x_test = np.linspace(-1, 1, 200)
    y_pred = poly(x_test)
    return x, y, x_test, y_pred


def main() -> None:
    tr, te = 0.01, 0.14
    print('Train loss:', tr)
    print('Test loss:', te)
    print('Generalization gap:', generalization_gap(tr, te))


if __name__ == '__main__':
    main()
