"""Day 20: summarize modern CNN architectures and parameter counts."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Architecture:
    name: str
    key_idea: str
    solves: str


ARCHS = [
    Architecture("VGG", "Stack many 3x3 convs to go deep", "Need for depth"),
    Architecture("ResNet", "Skip connections with residual blocks", "Vanishing gradients"),
    Architecture("EfficientNet", "Compound scaling of depth/width/resolution", "Inefficient scaling"),
    Architecture("ConvNeXt", "Transformer-inspired CNN redesign", "Outdated CNN design"),
]


def main() -> None:
    for arch in ARCHS:
        print(f"{arch.name}: {arch.key_idea} → solves {arch.solves}")


if __name__ == "__main__":
    main()
