"""Day 53: data augmentation invariance demo."""
from __future__ import annotations

import numpy as np


def make_square() -> np.ndarray:
    img = np.zeros((100, 100))
    img[30:70, 40:60] = 1
    return img


def translate(img: np.ndarray, shift: int = 10) -> np.ndarray:
    return np.roll(img, shift, axis=1)


def rotate(img: np.ndarray) -> np.ndarray:
    return np.rot90(img)


def main() -> None:
    img = make_square()
    print("Original sum:", float(img.sum()))
    print("Translated sum:", float(translate(img).sum()))
    print("Rotated sum:", float(rotate(img).sum()))


if __name__ == "__main__":
    main()
