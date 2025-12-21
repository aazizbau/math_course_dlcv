"""Day 21 visualizations: simple UNet/FPN schematic diagrams."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BOX_COLOR = "#4c72b0"
DECODE_COLOR = "#dd8452"
PIVOT_COLOR = "#55a868"


def _draw_unet(ax: plt.Axes) -> None:
    ax.set_title("UNet Encoder–Decoder")
    ax.axis("off")

    encoder_x = [0, 1.8, 3.6, 5.4]
    decoder_x = [9.8, 8.0, 6.2, 4.4]
    depths = [0, -1.2, -2.4, -3.6]

    # encoder blocks
    for x, y in zip(encoder_x, depths):
        ax.add_patch(Rectangle((x, y), 1.2, 0.8, color=BOX_COLOR, alpha=0.8))

    # decoder blocks
    for x, y in zip(decoder_x, depths):
        ax.add_patch(Rectangle((x, y), 1.2, 0.8, color=DECODE_COLOR, alpha=0.8))

    # arrows down encoder
    for idx in range(len(encoder_x) - 1):
        arrow = FancyArrowPatch(
            (encoder_x[idx] + 1.2, depths[idx] + 0.4),
            (encoder_x[idx + 1], depths[idx + 1] + 0.4),
            arrowstyle="->",
            mutation_scale=12,
            color="black",
        )
        ax.add_patch(arrow)

    # arrows up decoder
    for idx in range(len(decoder_x) - 1):
        arrow = FancyArrowPatch(
            (decoder_x[idx], depths[idx] + 0.4),
            (decoder_x[idx + 1] + 1.2, depths[idx + 1] + 0.4),
            arrowstyle="->",
            mutation_scale=12,
            color="black",
        )
        ax.add_patch(arrow)

    # skip connections
    for ex, dx, y in zip(encoder_x, reversed(decoder_x), depths):
        skip = FancyArrowPatch(
            (ex + 1.2, y + 0.7),
            (dx, y + 0.7),
            arrowstyle="-|>",
            mutation_scale=12,
            linestyle="--",
            color="#222222",
        )
        ax.add_patch(skip)

    ax.text(2.4, 0.2, "Encoder", fontsize=10, fontweight="bold")
    ax.text(7.2, 0.2, "Decoder", fontsize=10, fontweight="bold")


def _draw_fpn(ax: plt.Axes) -> None:
    ax.set_title("Feature Pyramid Network (FPN)")
    ax.axis("off")

    levels = ["C2", "C3", "C4", "C5"]
    y_positions = [0, -1.1, -2.2, -3.3]
    for level, y in zip(levels, y_positions):
        ax.add_patch(Rectangle((0, y), 1.3, 0.8, color=BOX_COLOR, alpha=0.85))
        ax.text(0.65, y + 0.4, level, ha="center", va="center", color="white")

    p_levels = ["P2", "P3", "P4", "P5"]
    for level, y in zip(p_levels, y_positions):
        ax.add_patch(Rectangle((3, y), 1.3, 0.8, color=PIVOT_COLOR, alpha=0.85))
        ax.text(3.65, y + 0.4, level, ha="center", va="center", color="white")

    # lateral connections (Conv + add)
    for y in y_positions:
        ax.add_patch(
            FancyArrowPatch(
                (1.3, y + 0.4),
                (3, y + 0.4),
                arrowstyle="->",
                mutation_scale=12,
                linestyle="-",
                color="#444444",
            )
        )

    # top-down upsample arrows
    for idx in range(len(y_positions) - 1, 0, -1):
        ax.add_patch(
            FancyArrowPatch(
                (3.65, y_positions[idx] + 0.8),
                (3.65, y_positions[idx - 1] + 0.6),
                arrowstyle="->",
                mutation_scale=12,
                linestyle=":",
                color="#222222",
            )
        )

    ax.text(0.65, 0.4, "Backbone maps", ha="center", fontsize=10, fontweight="bold")
    ax.text(3.65, 0.4, "Pyramid outputs", ha="center", fontsize=10, fontweight="bold")


def create_encoder_decoder_diagram() -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    _draw_unet(axes[0])
    _draw_fpn(axes[1])
    fig.tight_layout()
    path = OUT_DIR / "00_encoder_decoder_fpn.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main() -> None:
    path = create_encoder_decoder_diagram()
    print("Saved encoder–decoder diagram →", path)


if __name__ == "__main__":
    main()
