"""Day 15 utilities for convolution geometry (padding/stride)."""
from __future__ import annotations

from typing import Tuple


def conv_output_size(h_in: int, k: int, p: int, s: int) -> int:
    return (h_in - k + 2 * p) // s + 1


def output_shape(h_in: int, w_in: int, k: Tuple[int, int], p: Tuple[int, int], s: Tuple[int, int]) -> Tuple[int, int]:
    h_out = conv_output_size(h_in, k[0], p[0], s[0])
    w_out = conv_output_size(w_in, k[1], p[1], s[1])
    return h_out, w_out


def main() -> None:
    print("3x3 kernel, no padding:", conv_output_size(32, 3, 0, 1))
    print("3x3 kernel, same padding:", conv_output_size(32, 3, 1, 1))
    print("3x3 kernel, stride 2:", conv_output_size(32, 3, 1, 2))


if __name__ == "__main__":
    main()
