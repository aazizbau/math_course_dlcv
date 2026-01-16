"""Day 39 visualizations: Jacobian local linearization."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .jacobian_demo import f, jacobian_analytic

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_local_linearization() -> Path:
    x0 = np.array([1.0, 1.0])
    J = jacobian_analytic(x0)
    f0 = f(x0)

    rng = np.random.default_rng(0)
    perturb = rng.normal(0, 0.2, size=(200, 2))
    real = np.array([f(x0 + d) for d in perturb])
    linear = np.array([f0 + J @ d for d in perturb])

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(real[:, 0], real[:, 1], alpha=0.5, label="f(x0+dx)")
    ax.scatter(linear[:, 0], linear[:, 1], alpha=0.5, label="f(x0)+J dx")
    ax.set_title("Local Linear Approximation")
    ax.set_xlabel("output 1")
    ax.set_ylabel("output 2")
    ax.grid(True, alpha=0.3)
    ax.legend()

    path = OUT_DIR / "00_jacobian_linearization.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main() -> None:
    path = plot_local_linearization()
    print("Saved Jacobian linearization →", path)


if __name__ == "__main__":
    main()
