"""Day 11 visualizations: normalization effects and distribution stabilization."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np

from days.day11.code.normalization import DriftSimulation

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def anim_batchnorm_distribution(frames: int = 20, fps: int = 4) -> Path:
    sim = DriftSimulation()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_xlim(-4, 4)
    ax.set_ylim(0, 0.6)
    ax.set_title("BatchNorm: Distribution Stabilization")
    ax.grid(True)

    line1, = ax.plot([], [], label="Before BN")
    line2, = ax.plot([], [], label="After BN")
    ax.legend()

    def update(frame: int):
        noisy, bn = sim.step(frame)
        hist1 = np.histogram(noisy, bins=40, density=True)
        hist2 = np.histogram(bn, bins=40, density=True)
        centers1 = (hist1[1][:-1] + hist1[1][1:]) / 2
        centers2 = (hist2[1][:-1] + hist2[1][1:]) / 2
        line1.set_data(centers1, hist1[0])
        line2.set_data(centers2, hist2[0])
        return line1, line2

    anim = animation.FuncAnimation(fig, update, frames=frames, interval=300)
    path = OUT_DIR / "01_batchnorm_distribution.gif"
    anim.save(path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    return path


def main() -> None:
    artifact = anim_batchnorm_distribution()
    print(f"Saved BatchNorm distribution animation → {artifact}")


if __name__ == "__main__":
    main()
