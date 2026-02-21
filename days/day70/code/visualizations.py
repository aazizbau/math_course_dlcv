"""Day 70 visualizations: four-pillars map and coupled dynamics concept."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).resolve().parent.parent / 'outputs'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_four_pillars() -> Path:
    labels = ['Geometry', 'Optimization', 'Information', 'Statistics']
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    r = np.ones_like(angles)

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, polar=True)
    ax.scatter(angles, r, s=120)

    for a, lbl in zip(angles, labels):
        ax.text(a, 1.1, lbl, ha='center', va='center')

    for i in range(len(angles)):
        a1, a2 = angles[i], angles[(i + 1) % len(angles)]
        ax.plot([a1, a2], [1, 1], color='tab:blue', alpha=0.7)

    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_title('Day 70: Unified Four-Pillar View', pad=18)

    path = OUT_DIR / '00_four_pillars.png'
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_systemic_interaction() -> Path:
    t = np.linspace(0, 1, 200)
    geometry = 0.4 + 0.5 * (1 - np.exp(-4 * t))
    optimization = 0.3 + 0.6 * np.exp(-2 * (t - 0.4) ** 2)
    information = 0.2 + 0.6 * (1 - np.exp(-5 * t))
    statistics = 0.25 + 0.5 * (1 - np.exp(-3 * t))

    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.plot(t, geometry, label='Geometry')
    ax.plot(t, optimization, label='Optimization')
    ax.plot(t, information, label='Information')
    ax.plot(t, statistics, label='Statistics')
    ax.set_title('Coupled Dynamics During Learning (Conceptual)')
    ax.set_xlabel('Training progression')
    ax.set_ylabel('Influence level')
    ax.legend(ncol=2)
    ax.grid(True, alpha=0.3)

    path = OUT_DIR / '01_system_interaction.png'
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main() -> None:
    p1 = plot_four_pillars()
    p2 = plot_systemic_interaction()
    print('Saved plots ->', p1, p2)


if __name__ == '__main__':
    main()
