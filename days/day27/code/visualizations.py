"""Day 27 visualizations: fusion strategy schematic."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COLOR_OPT = "#4c72b0"
COLOR_SAR = "#55a868"
COLOR_DEM = "#c44e52"
COLOR_FUSE = "#8172b2"


def plot_fusion_schematic() -> Path:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")
    ax.set_title("Multi-Modal Fusion (Early vs Mid vs Late)")

    # Inputs
    ax.add_patch(Rectangle((0.2, 2.6), 0.8, 0.4, color=COLOR_OPT, alpha=0.8))
    ax.add_patch(Rectangle((0.2, 1.8), 0.8, 0.4, color=COLOR_SAR, alpha=0.8))
    ax.add_patch(Rectangle((0.2, 1.0), 0.8, 0.4, color=COLOR_DEM, alpha=0.8))
    ax.text(0.6, 3.05, "Optical", ha="center", va="bottom", fontsize=9)
    ax.text(0.6, 2.25, "SAR", ha="center", va="bottom", fontsize=9)
    ax.text(0.6, 1.45, "DEM", ha="center", va="bottom", fontsize=9)

    # Early fusion block
    ax.add_patch(Rectangle((2.0, 2.0), 1.0, 0.8, color=COLOR_FUSE, alpha=0.8))
    ax.text(2.5, 2.35, "Early\nFusion", ha="center", va="center", fontsize=9)

    # Mid-level fusion block
    ax.add_patch(Rectangle((4.0, 2.6), 0.9, 0.4, color=COLOR_OPT, alpha=0.6))
    ax.add_patch(Rectangle((4.0, 1.8), 0.9, 0.4, color=COLOR_SAR, alpha=0.6))
    ax.add_patch(Rectangle((4.0, 1.0), 0.9, 0.4, color=COLOR_DEM, alpha=0.6))
    ax.text(4.45, 0.6, "Encoders", ha="center", fontsize=9)

    ax.add_patch(Rectangle((5.3, 1.6), 1.0, 0.8, color=COLOR_FUSE, alpha=0.8))
    ax.text(5.8, 2.0, "Mid\nFusion", ha="center", va="center", fontsize=9)

    # Late fusion block
    ax.add_patch(Rectangle((7.0, 2.6), 0.8, 0.4, color=COLOR_OPT, alpha=0.4))
    ax.add_patch(Rectangle((7.0, 1.8), 0.8, 0.4, color=COLOR_SAR, alpha=0.4))
    ax.add_patch(Rectangle((7.0, 1.0), 0.8, 0.4, color=COLOR_DEM, alpha=0.4))
    ax.add_patch(Rectangle((8.2, 1.6), 0.8, 0.8, color=COLOR_FUSE, alpha=0.8))
    ax.text(8.6, 2.0, "Late\nFusion", ha="center", va="center", fontsize=9)

    # Arrows
    for y in (2.8, 2.0, 1.2):
        ax.add_patch(FancyArrowPatch((1.0, y), (2.0, 2.4), arrowstyle="->", mutation_scale=12))
    for y in (2.8, 2.0, 1.2):
        ax.add_patch(FancyArrowPatch((1.0, y), (4.0, y), arrowstyle="->", mutation_scale=12))
    for y in (2.8, 2.0, 1.2):
        ax.add_patch(FancyArrowPatch((4.9, y), (5.3, 2.0), arrowstyle="->", mutation_scale=12))
    for y in (2.8, 2.0, 1.2):
        ax.add_patch(FancyArrowPatch((1.0, y), (7.0, y), arrowstyle="->", mutation_scale=12))
    for y in (2.8, 2.0, 1.2):
        ax.add_patch(FancyArrowPatch((7.8, y), (8.2, 2.0), arrowstyle="->", mutation_scale=12))

    path = OUT_DIR / "00_fusion_schematic.png"
    fig.subplots_adjust(left=0.04, right=0.98, top=0.88, bottom=0.08)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main() -> None:
    path = plot_fusion_schematic()
    print("Saved fusion schematic →", path)


if __name__ == "__main__":
    main()
