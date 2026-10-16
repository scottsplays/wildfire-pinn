# Experiment plan

This directory is a diagnostic suite. It does not replace the frozen WildfirePINN v1 checkpoint, the Kincade trainer, or the R-sweep results.

## What already exists

The production network is `LevelSetPINN` in `wildfire_pinn/pinn.py`: a tanh MLP from `(tau, X, Y)` to one scalar. The Kincade model uses hidden width 128 and 4 hidden blocks. Training normalizes by the domain side `L` and the full time span `T`:

- `tau = t / T`
- `X = (x - xc) / L`, `Y = (y - yc) / L`
- the network output is `Phi = phi / L`
- the coded residual is `Phi_tau + (R_eff * T / L) |grad_XY Phi|`

That coded residual is `(T / L)` times the physical residual `phi_t + R_eff |grad_xy phi|`, in 1/s. `phi < 0` is burned. For Kincade, `L = 53550 m` and `T = 523680 s`.

`directed_residual` applies the wind factor. `R` from `compute_spread_rate` is

`C_REF * r_rel[fuel] * W(wind speed) * S(slope) * M(moisture)`

with `C_REF = 0.177 m/s`. Wind direction is not inside that field. The residual then uses one domain-mean blowing-toward unit vector per day:

`R_eff = clip(R * (1 + 0.6 * wind · n), min=0)`

`n = grad phi / |grad phi|` points out of the burned region. Initial and observation targets are signed-distance fields from `signed_distance` (negative inside). v1 trains on the persistent burned-region masks for passes 0–11, with validation 12–15 and test 16–19. The frozen v1 loss uses `w_pde = 0`.

`train_synthetic` and `levelset_pinn_poc.py` already solve a circle, but only in dimensionless coordinates (`[-1, 1]` or the unit square) and without a finite-difference solver. `levelset_pinn_poc.py` is TensorFlow and is not part of the local training path. There is no upwind / CFL level-set integrator in the repository. The new solver is `solver.py`. The new physical-unit circle is `circle.py`.

## Prior measured failure

The constant-R diagnostic (`experiments/r_sweep/diagnostic/`) trained the real architecture for 4,000 epochs on Kincade. Validation IoU was 0.388 at R = 0, 0.493 at R = 0.01 m/s, 0.273 at R = 0.1 m/s, 0.416 with `w_pde = 0`, and 0.953 for persistence of the pass-11 mask. Those checkpoints stay where they are.

## What this suite adds

| Piece | Role |
| --- | --- |
| `solver.py` | Godunov upwind integrator in meters and seconds. R may be a scalar or a grid. |
| `circle.py` | Exact expanding circle and front metrics. |
| `experiment1.py` | PINN vs numerical solver vs the exact circle. |
| `test_experiment1.py` | Derivative chain rule, units, CFL, R = 0, and front speed. |

Experiment 1 reuses `LevelSetPINN`, `isotropic_residual`, `zero_contours`, and the Kincade normalization. It uses a 64-wide, 4-block network, 1,500 epochs, and no eikonal term. The real 128-wide Kincade trainer is not modified.

## Later experiments, not started

2. **Spatial R.** `R(x, y) = R0 [1 + a sin(kx x) cos(ky y)]` with `a < 1` so R stays positive, plus a separate wind-aligned speed `R0 (1 + c wind · n)`. Generate the reference with `solver.py`, train the same PINN on that prescribed R, and compare fronts. No new architecture.

3. **Recover R.** Hide the Experiment 2 field. Method A estimates normal speed from successive signed-distance contours. Method B fits `softplus(f_theta)` with the level-set constraint, staged if a joint PINN is unstable. Compare to a constant-R baseline. A small PDE residual is not counted as recovering R.

4. **Kincade geometry.** Only if Experiments 1–2 show that the PINN and the solver agree on known R. Start from the pass-11 persistent mask, converted with the existing `signed_distance`. Propagate constant R = 0.01, the existing weather R, and a weather scale fit on the training interval only, all with the same numerical solver. Score validation and test separately against persistence and the frozen v1 checkpoint. Do not feed future perimeters in.

5. **Multi-fire feasibility.** Read the pipeline only. Do not download data. Required inputs are a perimeter time series, a meter grid, fuel, slope, wind speed, wind direction, and moisture. The current loader is built around one `FireCase` and a chronological split inside that fire. A later benchmark should keep fires intact: train fires separate from test fires, and compare growth rates rather than pooling pixels.

## Reproduce Experiment 1

From the repository root:

```
python -m unittest experiments.wildfirepinn_v2_diagnostics.test_experiment1
python -m experiments.wildfirepinn_v2_diagnostics.experiment1 --smoke
python -m experiments.wildfirepinn_v2_diagnostics.experiment1
```

Outputs go to `experiments/wildfirepinn_v2_diagnostics/experiment1/`. A run whose history already contains the requested number of epochs is skipped.
