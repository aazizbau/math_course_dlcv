"""Day 65: OOD detection toy demos (energy + distance)."""
from __future__ import annotations

import numpy as np


def energy(logits: np.ndarray) -> float:
    return float(-np.log(np.sum(np.exp(logits))))


def centroid_distance(x: np.ndarray, centroids: np.ndarray) -> float:
    dists = np.linalg.norm(centroids - x[None, :], axis=1)
    return float(np.min(dists))


def main() -> None:
    logits_in = np.array([3.0, 1.0, 0.2])
    logits_ood = np.array([0.1, 0.0, -0.1])
    print('Energy (in):', energy(logits_in))
    print('Energy (ood-like):', energy(logits_ood))

    centroids = np.array([[0.0, 0.0], [2.0, 2.0], [-2.0, 1.5]])
    xin = np.array([0.2, -0.1])
    xout = np.array([6.0, 5.0])
    print('Nearest-centroid distance (in):', centroid_distance(xin, centroids))
    print('Nearest-centroid distance (out):', centroid_distance(xout, centroids))


if __name__ == '__main__':
    main()
