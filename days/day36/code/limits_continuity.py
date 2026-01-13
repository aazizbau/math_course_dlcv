"""Day 36: limits, continuity, and numerical derivatives (NumPy demo)."""
from __future__ import annotations

import numpy as np


def derivative_approx(f, x: float, h: float) -> float:
    return float((f(x + h) - f(x)) / h)


def f_square(x: float) -> float:
    return x**2


def f_abs(x: float) -> float:
    return abs(x)


def main() -> None:
    x = 2.0
    h = 1e-5
    approx = derivative_approx(f_square, x, h)
    print("Approx derivative:", approx)
    print("True derivative:", 2 * x)

    for h in [1e-1, 1e-2, 1e-3, 1e-4]:
        print("|x| derivative at 0 with h=", h, derivative_approx(f_abs, 0.0, h))


if __name__ == "__main__":
    main()
