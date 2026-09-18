# Wildfire PINN — 1D Heat-Equation Prototype

Physics-informed neural network prototype for the 1D heat equation, a first step toward wildfire spread modeling. Master's project, San José State University. Part of a team master's project at San José State University — this repo contains my individual PINN prototype; the full wildfire forecasting system (UNet perception, RL decisions, app) lives with the team.


## Method

- PINN mapping `[x, t] → u`: two 50-unit tanh layers, TensorFlow/Keras, PDE residual via automatic differentiation
- 1,000 collocation points, Adam (lr 1e-3), 10,000 epochs
- Key fix: a residual-only loss admits degenerate constant solutions, so the loss includes weighted initial/boundary terms — `Loss = PDE + w·(IC + BC)`

## Results

Validated against the closed-form solution `u(x,t) = sin(πx)·e^(−απ²t)` (α = 0.01):

| IC/BC weight | Relative L2 error |
|---|---|
| 1× | 0.13% |
| 10× | 0.23% |
| 100× | 0.47% |

<img width="1189" height="348" alt="12e602f6-7e07-45bb-ac77-d27ae85804a5" src="https://github.com/user-attachments/assets/bd548d96-2b3b-402d-ab83-6c0180d9f6b2" />
<img width="548" height="332" alt="b2bcdd67-e523-4a22-86ce-22e4dde24f06" src="https://github.com/user-attachments/assets/c1760a0e-28a6-47e1-bc1a-d853ac4cc66a" />

All runs land under 0.5% error; 100× over-constrains the IC/BC terms at the expense of the PDE residual.

## Run it

Open `WildfirePINN.ipynb` — the baseline training cell, IC/BC weight ablation, and validation plots are all in the notebook.

## Next steps

Nonlinear reaction term (Fisher–KPP-style dynamics) or 2D extension, per advisor direction.
