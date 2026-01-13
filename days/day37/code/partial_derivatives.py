"""Day 37: partial derivatives for multivariate functions (NumPy demo)."""
from __future__ import annotations

import numpy as np


def f(x: float, y: float) -> float:
    return x**2 + y**2


def partials_numeric(x: float, y: float, h: float = 1e-5) -> tuple[float, float]:
    df_dx = (f(x + h, y) - f(x, y)) / h
    df_dy = (f(x, y + h) - f(x, y)) / h
    return float(df_dx), float(df_dy)


def f2(x: float, y: float) -> float:
    return x**2 * y + y**3


def partials_analytic_f2(x: float, y: float) -> tuple[float, float]:
    return float(2 * x * y), float(x**2 + 3 * y**2)


def main() -> None:
    x, y = 1.0, -2.0
    df_dx, df_dy = partials_numeric(x, y)
    print("f(x,y)=x^2+y^2")
    print("∂f/∂x:", df_dx)
    print("∂f/∂y:", df_dy)

    x2, y2 = 0.5, -1.5
    num_dx = (f2(x2 + 1e-5, y2) - f2(x2, y2)) / 1e-5
    num_dy = (f2(x2, y2 + 1e-5) - f2(x2, y2)) / 1e-5
    ana_dx, ana_dy = partials_analytic_f2(x2, y2)
    print("f2(x,y)=x^2*y + y^3")
    print("Numeric ∂f2/∂x:", num_dx)
    print("Analytic ∂f2/∂x:", ana_dx)
    print("Numeric ∂f2/∂y:", num_dy)
    print("Analytic ∂f2/∂y:", ana_dy)


if __name__ == "__main__":
    main()
