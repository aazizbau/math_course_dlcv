"""Day 34 visualizations: conditioning and gradient descent paths."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _gd_path(A: np.ndarray, lr: float = 0.5, steps: int = 20) -> np.ndarray:
    x = np.array([1.0, 1.0])
    path = [x.copy()]
    for _ in range(steps):
        x = x - lr * A @ x
        path.append(x.copy())
    return np.array(path)


def plot_conditioning_paths() -> Path:
    A_good = np.eye(2)
    A_bad = np.array([[1.0, 0.0], [0.0, 1e-3]])

    path_good = _gd_path(A_good)
    path_bad = _gd_path(A_bad)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for ax, path, title in zip(
        axes,
        [path_good, path_bad],
        ["Well-conditioned", "Ill-conditioned"],
    ):
        ax.plot(path[:, 0], path[:, 1], marker="o")
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", adjustable="box")

    path = OUT_DIR / "00_conditioning_paths.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_singular_spread() -> Path:
    matrices = {
        "Good": np.eye(2),
        "Bad": np.array([[1.0, 0.0], [0.0, 1e-4]]),
    }
    labels = []
    conds = []
    for name, A in matrices.items():
        labels.append(name)
        conds.append(np.linalg.cond(A))

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(labels, conds, color="#c44e52")
    ax.set_title("Condition Numbers")
    ax.set_ylabel("κ(A)")
    ax.grid(True, axis="y", alpha=0.3)

    path = OUT_DIR / "01_condition_numbers.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main() -> None:
    print("Saved GD paths →", plot_conditioning_paths())
    print("Saved condition numbers →", plot_singular_spread())


if __name__ == "__main__":
    main()
