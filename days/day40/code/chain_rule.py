"""Day 40: chain rule and computational graph demo (NumPy-based)."""
from __future__ import annotations


def forward_chain(x: float) -> tuple[float, float, float]:
    u = x**2
    v = u + 1
    f = v**3
    return u, v, f


def backward_chain(x: float, u: float, v: float) -> float:
    df_dv = 3 * v**2
    dv_du = 1
    du_dx = 2 * x
    return df_dv * dv_du * du_dx


def main() -> None:
    x = 2.0
    u, v, f = forward_chain(x)
    df_dx = backward_chain(x, u, v)
    print("u:", u)
    print("v:", v)
    print("f:", f)
    print("df/dx:", df_dx)


if __name__ == "__main__":
    main()
