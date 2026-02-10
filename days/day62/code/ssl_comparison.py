"""Day 62: contrastive vs non-contrastive SSL toy utilities."""
from __future__ import annotations

import numpy as np


def info_nce_score(pos_sim: float, neg_sims: np.ndarray, tau: float = 0.1) -> float:
    num = np.exp(pos_sim / tau)
    den = num + np.sum(np.exp(neg_sims / tau))
    return float(-np.log(num / den))


def collapse_stat(Z: np.ndarray) -> float:
    return float(Z.var(axis=0).mean())


def main() -> None:
    pos_sim = 0.82
    neg_sims = np.array([0.12, 0.05, -0.1, 0.2])
    print("InfoNCE (toy):", info_nce_score(pos_sim, neg_sims))

    Z_collapsed = np.ones((100, 128))
    print("Collapsed variance:", collapse_stat(Z_collapsed))


if __name__ == "__main__":
    main()
