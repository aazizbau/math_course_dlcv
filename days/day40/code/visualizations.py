"""Day 40 visualizations: computational graph and gradient flow."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_computational_graph() -> Path:
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.axis("off")

    boxes = {
        "x": (0.2, 0.4),
        "u": (1.8, 0.4),
        "v": (3.4, 0.4),
        "f": (5.0, 0.4),
    }
    labels = {"x": "x", "u": "u=x^2", "v": "v=u+1", "f": "f=v^3"}

    for key, (x, y) in boxes.items():
        ax.add_patch(Rectangle((x, y), 1.0, 0.6, color="#4c72b0", alpha=0.8))
        ax.text(x + 0.5, y + 0.3, labels[key], ha="center", va="center", color="white")

    for src, dst in [("x", "u"), ("u", "v"), ("v", "f")]:
        x0, y0 = boxes[src]
        x1, y1 = boxes[dst]
        ax.add_patch(
            FancyArrowPatch(
                (x0 + 1.0, y0 + 0.3),
                (x1, y1 + 0.3),
                arrowstyle="->",
                mutation_scale=12,
                color="#222222",
            )
        )

    path = OUT_DIR / "00_computational_graph.png"
    fig.subplots_adjust(left=0.03, right=0.97, top=0.88, bottom=0.08)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main() -> None:
    path = plot_computational_graph()
    print("Saved computational graph →", path)


if __name__ == "__main__":
    main()
