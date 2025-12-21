"""Day 21: summarize encoder–decoder style architectures."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class Blueprint:
    """High-level description of a dense-prediction architecture."""

    name: str
    description: str
    skip_strategy: str
    excels_at: str


BLUEPRINTS: tuple[Blueprint, ...] = (
    Blueprint(
        "UNet",
        "Symmetric encoder–decoder with spatial skip concatenations",
        "Concatenate shallow + deep feature maps",
        "Pixel-precise segmentation and diffusion backbones",
    ),
    Blueprint(
        "FPN",
        "Top-down pyramid that fuses multi-scale backbone features",
        "Add lateral features across resolutions",
        "Detection/instance segmentation needing multi-scale context",
    ),
    Blueprint(
        "SegNet",
        "Encoder–decoder that remembers pooling indices",
        "Unpool using saved max-pooling masks",
        "Semantic segmentation when memory is limited",
    ),
)


def summarize_blueprints() -> Sequence[Blueprint]:
    """Return the static blueprint metadata so notebooks can render tables."""

    return list(BLUEPRINTS)


def main() -> None:
    for bp in BLUEPRINTS:
        print(
            f"{bp.name}: {bp.description} | skip={bp.skip_strategy} | excels at {bp.excels_at}"
        )


if __name__ == "__main__":
    main()
