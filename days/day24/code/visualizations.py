"""Day 24 visualizations: learning-rate schedules overview."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from .training_strategies import cosine_annealing, one_cycle, step_decay, warmup_linear

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_lr_schedules(steps: int = 60) -> Path:
    step = step_decay(1e-3, 0.1, 20, steps)
    cosine = cosine_annealing(1e-3, 1e-6, steps)
    onecycle = one_cycle(1e-3, 1e-5, steps)
    warmup = warmup_linear(1e-3, 10, steps)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([p.step for p in step], [p.lr for p in step], label="Step decay")
    ax.plot([p.step for p in cosine], [p.lr for p in cosine], label="Cosine")
    ax.plot([p.step for p in onecycle], [p.lr for p in onecycle], label="One-cycle")
    ax.plot([p.step for p in warmup], [p.lr for p in warmup], label="Warmup")
    ax.set_title("Learning-Rate Schedules")
    ax.set_xlabel("Step")
    ax.set_ylabel("LR")
    ax.grid(True, alpha=0.3)
    ax.legend()

    path = OUT_DIR / "00_lr_schedules.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main() -> None:
    path = plot_lr_schedules()
    print("Saved LR schedule plot →", path)


if __name__ == "__main__":
    main()
