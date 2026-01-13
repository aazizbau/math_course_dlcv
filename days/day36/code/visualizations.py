"""Day 36 visualizations: limit behavior and derivative convergence."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_difference_quotient() -> Path:
    x0 = 2.0
    hs = np.logspace(-4, -1, 30)
    f = lambda x: x**2
    grads = [(f(x0 + h) - f(x0)) / h for h in hs]

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(hs, grads, marker="o")
    ax.axhline(2 * x0, color="#c44e52", linestyle="--", label="True gradient")
    ax.set_xscale("log")
    ax.set_title("Derivative Approximation Converges")
    ax.set_xlabel("h")
    ax.set_ylabel("difference quotient")
    ax.grid(True, alpha=0.3)
    ax.legend()

    path = OUT_DIR / "00_difference_quotient.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_abs_kink() -> Path:
    x = np.linspace(-1, 1, 200)
    y = np.abs(x)

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(x, y)
    ax.set_title("|x| is continuous but not differentiable at 0")
    ax.grid(True, alpha=0.3)

    path = OUT_DIR / "01_abs_kink.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main() -> None:
    print("Saved difference quotient →", plot_difference_quotient())
    print("Saved |x| kink →", plot_abs_kink())


if __name__ == "__main__":
    main()
