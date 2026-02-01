"""Day 55: calibration metrics demo (ECE + reliability)."""
from __future__ import annotations

import numpy as np


def expected_calibration_error(conf: np.ndarray, acc: np.ndarray, bins: int = 5) -> float:
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for i in range(bins):
        mask = (conf >= edges[i]) & (conf < edges[i + 1])
        if not np.any(mask):
            continue
        bin_conf = conf[mask].mean()
        bin_acc = acc[mask].mean()
        ece += (mask.mean()) * abs(bin_acc - bin_conf)
    return float(ece)


def main() -> None:
    conf = np.array([0.2, 0.4, 0.6, 0.8, 0.9])
    acc = np.array([0.1, 0.35, 0.55, 0.75, 0.85])
    print("ECE:", expected_calibration_error(conf, acc, bins=5))


if __name__ == "__main__":
    main()
