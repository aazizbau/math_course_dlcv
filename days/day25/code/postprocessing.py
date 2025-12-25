"""Day 25: post-processing utilities (NumPy-based)."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Component:
    label: int
    area: int


def _pad(mask: np.ndarray, radius: int) -> np.ndarray:
    return np.pad(mask, radius, mode="constant")


def erode(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    if radius <= 0:
        return mask.copy()
    padded = _pad(mask, radius)
    h, w = mask.shape
    out = np.zeros_like(mask)
    for i in range(h):
        for j in range(w):
            window = padded[i : i + 2 * radius + 1, j : j + 2 * radius + 1]
            out[i, j] = 1 if np.all(window == 1) else 0
    return out


def dilate(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    if radius <= 0:
        return mask.copy()
    padded = _pad(mask, radius)
    h, w = mask.shape
    out = np.zeros_like(mask)
    for i in range(h):
        for j in range(w):
            window = padded[i : i + 2 * radius + 1, j : j + 2 * radius + 1]
            out[i, j] = 1 if np.any(window == 1) else 0
    return out


def opening(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    return dilate(erode(mask, radius), radius)


def closing(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    return erode(dilate(mask, radius), radius)


def connected_components(mask: np.ndarray) -> tuple[np.ndarray, list[Component]]:
    """Return labels and component stats for binary masks."""

    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    components: list[Component] = []
    current = 0

    for i in range(h):
        for j in range(w):
            if mask[i, j] == 0 or labels[i, j] != 0:
                continue
            current += 1
            queue = deque([(i, j)])
            labels[i, j] = current
            area = 0

            while queue:
                r, c = queue.popleft()
                area += 1
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w:
                            if mask[nr, nc] == 1 and labels[nr, nc] == 0:
                                labels[nr, nc] = current
                                queue.append((nr, nc))

            components.append(Component(current, area))

    return labels, components


def remove_small_components(mask: np.ndarray, min_area: int = 50) -> np.ndarray:
    labels, components = connected_components(mask)
    keep = {comp.label for comp in components if comp.area >= min_area}
    out = np.zeros_like(mask)
    for label in keep:
        out[labels == label] = 1
    return out


def main() -> None:
    rng = np.random.default_rng(0)
    noisy = np.zeros((64, 64), dtype=np.uint8)
    rr, cc = np.ogrid[:64, :64]
    circle = (rr - 28) ** 2 + (cc - 30) ** 2 <= 12**2
    noisy[circle] = 1
    noisy[rng.random(noisy.shape) > 0.985] = 1
    noisy[rng.random(noisy.shape) > 0.995] = 0
    opened = opening(noisy, radius=1)
    closed = closing(opened, radius=1)
    cleaned = remove_small_components(closed, min_area=30)

    print("Raw pixels:", noisy.sum())
    print("After opening:", opened.sum())
    print("After closing:", closed.sum())
    print("After removal:", cleaned.sum())


if __name__ == "__main__":
    main()
