"""Day 24: training strategies for dense prediction (NumPy demo)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class SchedulePoint:
    step: int
    lr: float


def step_decay(initial_lr: float, gamma: float, step_size: int, steps: int) -> list[SchedulePoint]:
    lrs = []
    for t in range(steps):
        lr = initial_lr * (gamma ** (t // step_size))
        lrs.append(SchedulePoint(t, lr))
    return lrs


def cosine_annealing(lr_max: float, lr_min: float, steps: int) -> list[SchedulePoint]:
    lrs = []
    for t in range(steps):
        cos_term = 0.5 * (1 + np.cos(np.pi * t / (steps - 1)))
        lr = lr_min + (lr_max - lr_min) * cos_term
        lrs.append(SchedulePoint(t, float(lr)))
    return lrs


def one_cycle(max_lr: float, min_lr: float, steps: int, pct_up: float = 0.3) -> list[SchedulePoint]:
    lrs = []
    up_steps = int(steps * pct_up)
    down_steps = steps - up_steps
    for t in range(up_steps):
        lr = min_lr + (max_lr - min_lr) * (t / max(1, up_steps - 1))
        lrs.append(SchedulePoint(t, float(lr)))
    for t in range(down_steps):
        lr = max_lr - (max_lr - min_lr) * (t / max(1, down_steps - 1))
        lrs.append(SchedulePoint(up_steps + t, float(lr)))
    return lrs


def warmup_linear(base_lr: float, warmup_steps: int, total_steps: int) -> list[SchedulePoint]:
    lrs = []
    for t in range(total_steps):
        if t < warmup_steps:
            lr = base_lr * (t + 1) / warmup_steps
        else:
            lr = base_lr
        lrs.append(SchedulePoint(t, float(lr)))
    return lrs


def summarize(schedule: Iterable[SchedulePoint], name: str) -> None:
    pts = list(schedule)
    print(f"{name} schedule: first={pts[0].lr:.6f}, mid={pts[len(pts)//2].lr:.6f}, last={pts[-1].lr:.6f}")


def main() -> None:
    steps = 60
    summarize(step_decay(1e-3, 0.1, 20, steps), "Step")
    summarize(cosine_annealing(1e-3, 1e-6, steps), "Cosine")
    summarize(one_cycle(1e-3, 1e-5, steps), "OneCycle")
    summarize(warmup_linear(1e-3, 10, steps), "Warmup")


if __name__ == "__main__":
    main()
