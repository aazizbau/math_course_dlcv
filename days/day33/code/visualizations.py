"""Day 33 visualizations: rank collapse in 3D."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .rank_nullspace import low_rank_projection

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_rank_collapse() -> Path:
    rng = np.random.default_rng(1)
    X = rng.normal(0, 1, size=(300, 3))
    X2 = low_rank_projection(X, k=2)
    X1 = low_rank_projection(X, k=1)

    fig = plt.figure(figsize=(9, 3))
    for idx, data, title in zip(
        [1, 2, 3],
        [X, X2, X1],
        ["Original (3D)", "Rank-2 Projection", "Rank-1 Projection"],
    ):
        ax = fig.add_subplot(1, 3, idx, projection="3d")
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], s=6, alpha=0.6)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

    path = OUT_DIR / "00_rank_collapse.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main() -> None:
    path = plot_rank_collapse()
    print("Saved rank collapse plot →", path)


if __name__ == "__main__":
    main()
