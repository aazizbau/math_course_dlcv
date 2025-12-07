"""Day 14 utilities for receptive field computation and multi-scale demos."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np


LayerSpec = Tuple[int, int]  # (kernel_size, stride)


def compute_rf(layers: Iterable[LayerSpec]) -> List[int]:
    """Return receptive-field history given (kernel, stride) pairs."""
    rf = 1
    jump = 1
    history = [rf]
    for k, s in layers:
        rf = rf + (k - 1) * jump
        jump *= s
        history.append(rf)
    return history


def dilated_effective_kernel(k: int, dilation: int) -> int:
    return k + (k - 1) * (dilation - 1)


@dataclass
class RFSimulator:
    layers: int = 6
    kernel: int = 3
    stride: int = 1

    def run(self) -> List[int]:
        specs = [(self.kernel, self.stride) for _ in range(self.layers)]
        return compute_rf(specs)[1:]


def main() -> None:
    layers = [
        (3, 1),
        (3, 1),
        (2, 2),
        (3, 1),
        (3, 1),
    ]
    print("Receptive field progression:", compute_rf(layers))
    print("Dilated kernel (3x3, dilation=4)", dilated_effective_kernel(3, 4))


if __name__ == "__main__":
    main()
