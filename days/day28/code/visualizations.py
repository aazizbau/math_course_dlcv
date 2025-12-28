"""Day 28 visualizations: embedding distance over time."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .embeddings_demo import embed_patch, embedding_distance

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_embedding_drift() -> Path:
    rng = np.random.default_rng(4)
    base = rng.random((16, 16, 4)).astype(np.float32)

    distances = []
    for step in range(12):
        patch = base.copy()
        patch[4:8, 6:10, :] += step * 0.03
        patch = np.clip(patch, 0, 1)
        distances.append(embedding_distance(embed_patch(base), embed_patch(patch)))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(len(distances)), distances, marker="o")
    ax.set_title("Embedding Distance vs Change Intensity")
    ax.set_xlabel("Change step")
    ax.set_ylabel("Embedding distance")
    ax.grid(True, alpha=0.3)

    path = OUT_DIR / "00_embedding_drift.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main() -> None:
    path = plot_embedding_drift()
    print("Saved embedding drift plot →", path)


if __name__ == "__main__":
    main()
