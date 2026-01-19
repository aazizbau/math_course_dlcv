"""Day 42 visualizations: Taylor approximation vs true function."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .taylor_demo import f, taylor_first, taylor_second

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_taylor_approximations() -> Path:
    x0 = 1.0
    xs = np.linspace(0.5, 1.5, 200)
    ys = np.array([f(x) for x in xs])
    ys_lin = taylor_first(x0, xs)
    ys_quad = taylor_second(x0, xs)

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(xs, ys, label="f(x)")
    ax.plot(xs, ys_lin, "--", label="1st-order")
    ax.plot(xs, ys_quad, ":", label="2nd-order")
    ax.set_title("Local Taylor Approximation")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    path = OUT_DIR / "00_taylor_approx.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_error_vs_step() -> Path:
    x0 = 1.0
    hs = np.linspace(0.0, 0.8, 60)
    true = f(x0 + hs)
    lin = taylor_first(x0, x0 + hs)
    quad = taylor_second(x0, x0 + hs)
    err_lin = np.abs(true - lin)
    err_quad = np.abs(true - quad)

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(hs, err_lin, label="1st-order error")
    ax.plot(hs, err_quad, label="2nd-order error")
    ax.set_title("Approximation Error vs Step Size")
    ax.set_xlabel("h")
    ax.set_ylabel("|error|")
    ax.grid(True, alpha=0.3)
    ax.legend()

    path = OUT_DIR / "01_taylor_error.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main() -> None:
    print("Saved Taylor approximation →", plot_taylor_approximations())
    print("Saved Taylor error plot →", plot_error_vs_step())


if __name__ == "__main__":
    main()
