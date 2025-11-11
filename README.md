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
README.md
```

Future days should mirror this shape, making it easy to navigate the course timeline.

## Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy matplotlib
```

Launch notebooks with `jupyter lab` or `jupyter notebook` from the repo root so relative imports (e.g., `days.day01.code`) resolve correctly.

## Running Day 1 Code Headless

```bash
python days/day01/code/arrows_and_machines.py
python days/day01/code/visualizations.py  # writes GIFs to days/day01/outputs/
```

The notebook `days/day01/day01_arrows_and_machines.ipynb` walks through the same math with commentary plus links back to these scripts.

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
