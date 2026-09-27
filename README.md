[README.md](https://github.com/user-attachments/files/32709305/README.md)
# Wildfire PINN

Physics-informed neural network for wildfire spread forecasting. Master's project, San José State University (MSAI, Dec 2026).

Part of a team master's project — this repo contains my individual PINN work (the physics core); the full wildfire forecasting system (UNet perception, RL decisions, app) lives with the team.

## Current work

**2D level-set PINN.** The model learns the signed-distance field φ(x, y, t) of the fire front, constrained by the level-set PDE:

φ_t + R|∇φ| = 0

The spread-rate field R is a computed input, not a learned quantity — exactly one network solves for φ given R. Validated against the analytical expanding-circle solution, with diagnostics including interface extraction, front-normal checks, and a 3D spacetime isosurface. A uniform-wind analytical test (shifted-circle target) is the second validation check. Each experiment is a self-contained script with a run log.

**Real-data pipeline: Palisades Fire (Jan 7–31, 2025).** End-to-end pipeline from VIIRS satellite fire detections → 14 daily perimeters → signed-distance fields → a first-order spread-rate field R(x, y, t) (documented proxy, not full Rothermel). First training run completed 4,000 epochs with no divergence (total loss 0.166 → 0.119); held-out forecast metrics are pending.

## Stage 1 — 1D heat-equation baseline (validated)

PINN mapping `[x, t] → u`: two 50-unit tanh layers, TensorFlow/Keras, PDE residual via automatic differentiation. 1,000 collocation points, Adam (lr 1e-3), 10,000 epochs.

Key fix: a residual-only loss admits degenerate constant solutions, so the loss includes weighted initial/boundary terms — `Loss = PDE + w·(IC + BC)`.

Validated against the closed-form solution `u(x,t) = sin(πx)·e^(−απ²t)` (α = 0.01):

| IC/BC weight | Relative L2 error |
| --- | --- |
| 1× | 0.13% |
| 10× | 0.23% |
| 100× | 0.47% |

All runs land under 0.5% error; 100× over-constrains the IC/BC terms at the expense of the PDE residual.

## Repository layout

- `1d-baseline/` — 1D prototype notebook, IC/BC weight ablation, validation figures
- `2d-levelset/` — 2D level-set PINN: self-contained experiment scripts, run logs, figures
- `real-data/` — Palisades Fire pipeline and pipeline report

## Run it

- 1D baseline: open `1d-baseline/WildfirePINN.ipynb` — the baseline training cell, IC/BC weight ablation, and validation plots are all in the notebook.
- 2D level-set: each experiment is a standalone script, e.g. `python 2d-levelset/levelset_pinn.py` (see run logs in the same folder for hyperparameters and results).

## Data note

Large data files (satellite detections, NetCDF grids, model checkpoints) are backed up to Drive and available on request — they don't live in git.
