# 45-Day Math Course for Deep Learning & Computer Vision

A 45-day sprint that links geometric intuition with practical deep learning and computer vision demos. Each day lives in its own folder under `days/dayXX/` and is anchored by a Jupyter notebook mixing discussion, formulas, and runnable Python. Supporting scripts and generated media sit alongside the notebook so you can reproduce figures or run them headless.

## Daily Workflow

1. Create `days/dayXX/dayXX_topic.ipynb` with all narrative + math tables.
2. Keep reusable helpers in `days/dayXX/code/` (import them inside the notebook or run from CLI).
3. Save derived assets (GIFs, plots, datasets) in `days/dayXX/outputs/`—track only the ones you want on GitHub.
4. Document optional exercises or TODOs at the bottom of the notebook so you can iterate later.

## Repository Layout

```
days/
  day01/
    day01_arrows_and_machines.ipynb
    code/
      arrows_and_machines.py
      visualizations.py
    outputs/
      *.gif
  day02/
    day02_norms_angles.ipynb
    code/
      norms_and_angles.py
      visualizations.py
    outputs/
      *.gif / *.png
  day03/
    day03_gradient_descent.ipynb
    code/
      gradient_descent.py
      visualizations.py
    outputs/
      *.gif / *.png
  day04/
    day04_momentum.ipynb
    code/
      momentum_methods.py
      visualizations.py
    outputs/
      *.gif / *.png
  day05/
    day05_backprop.ipynb
    code/
      backprop_demo.py
      visualizations.py
    outputs/
      *.gif / *.png
  day06/
    day06_landscapes.ipynb
    code/
      landscapes.py
      visualizations.py
    outputs/
      *.gif / *.png
  day07/
    day07_jacobians.ipynb
    code/
      jacobian_demo.py
      visualizations.py
    outputs/
      *.gif / *.png
  day08/
    day08_hessian.ipynb
    code/
      hessian_demo.py
      visualizations.py
    outputs/
      *.gif / *.png
  day09/
    day09_gradients.ipynb
    code/
      gradient_pathologies.py
      visualizations.py
    outputs/
      *.gif / *.png
  day10/
    day10_activations.ipynb
    code/
      activations.py
      visualizations.py
    outputs/
      *.gif / *.png
  day11/
    day11_normalization.ipynb
    code/
      normalization.py
      visualizations.py
    outputs/
      *.gif / *.png
  day12/
    day12_initialization.ipynb
    code/
      initialization.py
      visualizations.py
    outputs/
      *.gif / *.png
README.md
```

Future days should mirror this shape, making it easy to navigate the course timeline. Completed notebooks so far:

- `days/day01/day01_arrows_and_machines.ipynb` — geometric storytelling of matrix machines.
- `days/day02/day02_norms_angles.ipynb` — vector norms, angles, cosine similarity, and normalization.
- `days/day03/day03_gradient_descent.ipynb` — gradient descent intuition, learning rate effects, and visualizations.
- `days/day04/day04_momentum.ipynb` — momentum, Nesterov, and inertia-driven optimization stories.
- `days/day05/day05_backprop.ipynb` — chain rule intuition, backprop demo, and gradient-flow visualization.
- `days/day06/day06_landscapes.ipynb` — convex vs non-convex surfaces, curvature, and landscape animations.
- `days/day07/day07_jacobians.ipynb` — Jacobian intuition, local linearization, and sensitivity visualizations.
- `days/day08/day08_hessian.ipynb` — Hessians, curvature intuition, and Newton-vs-GD comparisons.
- `days/day09/day09_gradients.ipynb` — vanishing/exploding gradients, simulations, and mitigation strategies.
- `days/day10/day10_activations.ipynb` — activation geometry, derivatives, and optimization effects.
- `days/day11/day11_normalization.ipynb` — BatchNorm vs LayerNorm, stability intuition, and distribution visualizations.
- `days/day12/day12_initialization.ipynb` — Xavier/He intuition and signal-balance simulations.

## Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy matplotlib
```

Launch notebooks with `jupyter lab` or `jupyter notebook` from the repo root so relative imports (e.g., `days.day01.code`) resolve correctly.

## Running Day 1 Code Headless

```bash
# Day 1
python -m days.day01.code.arrows_and_machines
python -m days.day01.code.visualizations   # GIFs → days/day01/outputs/

# Day 2
python -m days.day02.code.norms_and_angles
python -m days.day02.code.visualizations   # GIFs/PNGs → days/day02/outputs/

# Day 3
python -m days.day03.code.gradient_descent
python -m days.day03.code.visualizations   # GIFs/PNGs → days/day03/outputs/

# Day 4
python -m days.day04.code.momentum_methods
python -m days.day04.code.visualizations   # GIFs/PNGs → days/day04/outputs/

# Day 5
python -m days.day05.code.backprop_demo
python -m days.day05.code.visualizations   # GIFs/PNGs → days/day05/outputs/

# Day 6
python -m days.day06.code.landscapes
python -m days.day06.code.visualizations   # GIFs/PNGs → days/day06/outputs/

# Day 7
python -m days.day07.code.jacobian_demo
python -m days.day07.code.visualizations   # GIFs/PNGs → days/day07/outputs/

# Day 8
python -m days.day08.code.hessian_demo
python -m days.day08.code.visualizations   # GIFs/PNGs → days/day08/outputs/

# Day 9
python -m days.day09.code.gradient_pathologies
python -m days.day09.code.visualizations   # GIFs/PNGs → days/day09/outputs/

# Day 10
python -m days.day10.code.activations
python -m days.day10.code.visualizations   # GIFs/PNGs → days/day10/outputs/

# Day 11
python -m days.day11.code.normalization
python -m days.day11.code.visualizations   # GIFs/PNGs → days/day11/outputs/

# Day 12
python -m days.day12.code.initialization
python -m days.day12.code.visualizations   # GIFs/PNGs → days/day12/outputs/
```

Each notebook (`days/dayXX/*.ipynb`) walks through the same math with commentary and optional animation toggles.

## Git & GitHub Workflow

1. **Initialize (first time only)**
   ```bash
   git init
   git add .
   git commit -m "Day 1: Arrows & Machines"
   git branch -M main
   git remote add origin git@github.com:<your-user>/math_course_dlcv.git
   ```
2. **Sync changes** after editing notebooks or code
   ```bash
   git add days/day01 day01_arrows_and_machines.ipynb README.md
   git commit -m "Update Day 1 notebook"   # use a descriptive message
   git push -u origin main                  # first push
   # subsequent pushes
   git push
   ```
3. **Pull updates** before starting a new day
   ```bash
   git pull --rebase origin main
   ```
4. **New days**
   ```bash
   cp -R days/day01 days/day02   # or scaffold fresh folders
   git checkout -b day02
   ```

If you prefer HTTPS, replace the SSH remote URL accordingly. Keep large binaries out of Git (consider Git LFS) and avoid committing `.venv/` or `__pycache__/`—they are ignored via `.gitignore`.
