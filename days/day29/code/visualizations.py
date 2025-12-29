"""Day 29 visualizations: graph layout and message passing effect."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .gnn_demo import build_grid_graph, message_passing

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_graph_message_passing(size: int = 6) -> Path:
    graph = build_grid_graph(size)
    coords = np.array([(r, c) for r in range(size) for c in range(size)])
    before = graph.node_features[:, 0]
    after = message_passing(graph, steps=3)[:, 0]

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for ax, vals, title in zip(axes, [before, after], ["Before", "After"]):
        ax.scatter(coords[:, 1], coords[:, 0], c=vals, cmap="coolwarm", s=60)
        ax.set_title(f"{title} Message Passing")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")

    path = OUT_DIR / "00_message_passing.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main() -> None:
    path = plot_graph_message_passing()
    print("Saved GNN message passing plot →", path)


if __name__ == "__main__":
    main()
