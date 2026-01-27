"""Day 50 visualizations: bias vs variance curves."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_bias_variance_examples() -> Path:
    rng = np.random.default_rng(1)
    x = np.linspace(-1, 1, 120)
    true = x**3

    high_bias = 0.5 * x
    high_variance = true + 0.3 * rng.standard_normal(len(x))

    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.plot(x, true, label="True function", linewidth=2)
    ax.plot(x, high_bias, label="High bias")
    ax.plot(x, high_variance, label="High variance", alpha=0.8)
    ax.set_title("Bias vs Variance Example")
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = OUT_DIR / "00_bias_variance_examples.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_train_test_curve() -> Path:
    capacity = np.arange(1, 15)
    train = np.exp(-0.25 * capacity) + 0.05
    test = 0.2 + 0.4 * np.exp(-0.2 * capacity) + 0.02 * (capacity - 6) ** 2 / 36

    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(capacity, train, label="Train error")
    ax.plot(capacity, test, label="Test error")
    ax.set_xlabel("Model capacity")
    ax.set_ylabel("Error")
    ax.set_title("Train vs Test Error")
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = OUT_DIR / "01_train_test_curve.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main() -> None:
    path1 = plot_bias_variance_examples()
    path2 = plot_train_test_curve()
    print("Saved plots →", path1, path2)


if __name__ == "__main__":
    main()
