"""Day 45: initialization and variance propagation (NumPy demo)."""
from __future__ import annotations

import numpy as np


def propagate(n_layers: int, init_std: float, width: int = 512, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, size=(width,))
    for _ in range(n_layers):
        W = rng.normal(0, 1, size=(width, width)) * init_std
        x = W @ x
    return float(np.var(x))


def he_std(n_in: int) -> float:
    return np.sqrt(2.0 / n_in)


def xavier_std(n_in: int, n_out: int) -> float:
    return np.sqrt(2.0 / (n_in + n_out))


def main() -> None:
    for std in [0.001, 0.01, 0.03]:
        print(std, propagate(3, std))

    print("He std (512):", he_std(512))
    print("Xavier std (512->512):", xavier_std(512, 512))


if __name__ == "__main__":
    main()
