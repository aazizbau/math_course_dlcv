"""Day 66: causal thinking toy demo (shortcuts vs stable signal)."""
from __future__ import annotations

import numpy as np


def make_dataset(n: int = 1000, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x_true = rng.standard_normal(n)
    x_spurious = x_true + 0.1 * rng.standard_normal(n)
    y = (x_true > 0).astype(int)
    return x_true, x_spurious, y


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a, b)[0, 1])


def main() -> None:
    x_true, x_spur, y = make_dataset()
    print('Corr(x_true, y):', correlation(x_true, y))
    print('Corr(x_spurious, y):', correlation(x_spur, y))


if __name__ == '__main__':
    main()
