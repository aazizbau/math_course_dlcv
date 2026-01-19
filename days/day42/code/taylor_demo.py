"""Day 42: Taylor expansion demos (NumPy-based)."""
from __future__ import annotations

import numpy as np


def f(x: float) -> float:
    return x**3


def taylor_first(x0: float, x: np.ndarray) -> np.ndarray:
    f0 = f(x0)
    df = 3 * x0**2
    return f0 + df * (x - x0)


def taylor_second(x0: float, x: np.ndarray) -> np.ndarray:
    f0 = f(x0)
    df = 3 * x0**2
    d2f = 6 * x0
    return f0 + df * (x - x0) + 0.5 * d2f * (x - x0) ** 2


def main() -> None:
    x0 = 1.0
    xs = np.linspace(0.5, 1.5, 5)
    print("f(x):", [f(x) for x in xs])
    print("1st order:", taylor_first(x0, xs))
    print("2nd order:", taylor_second(x0, xs))


if __name__ == "__main__":
    main()
