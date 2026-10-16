# WildfirePINN v2 diagnostics

Experiment 1 is the expanding circle with a known solution. Experiments 2–5 have not been run.

Smoke runs are pipeline checks. The table below is the full 1,500-epoch comparison.

Domain side 4000 m, ignition (200, -150) m, r0 400 m, T 4000 s, grid 401 nodes (10.0 m). Network: LevelSetPINN hidden 64, layers 4. Loss: 10 L_ic + 1 L_pde. No eikonal term and no boundary loss. PDE collocation excludes a 50 m disk around the ignition point, where the signed-distance gradient is undefined.

Radius error is the mean contour radius minus r0 + R t. Displacement is the mean absolute radial miss of the zero contour. Stationary repeats the initial circle. PDE residual is the mean absolute physical residual in m/s, away from the ignition disk. A small residual is not by itself a correct front.

## Final time

| R m/s | method | radius error m | radial miss m | IoU | PDE residual m/s | IC RMSE m | train s |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | pinn | -155.35 | 155.35 | 0.371 | 0.0066 | 40.13 | 44.3 |
| 0 | numerical | -0.01 | 0.01 | 1.000 | — | — | — |
| 0 | stationary | -0.01 | 0.01 | 1.000 | — | — | — |
| 0.01 | pinn | -113.43 | 113.43 | 0.549 | 0.0063 | 40.51 | 43.0 |
| 0.01 | numerical | -0.22 | 0.22 | 0.999 | — | — | — |
| 0.01 | stationary | -40.01 | 40.01 | 0.825 | — | — | — |
| 0.1 | pinn | -29.62 | 29.62 | 0.929 | 0.0060 | 42.99 | 42.7 |
| 0.1 | numerical | -1.57 | 1.57 | 0.997 | — | — | — |
| 0.1 | stationary | -400.01 | 400.01 | 0.250 | — | — | — |

## Radius error by time

| R m/s | t s | true radius m | PINN error m | numerical error m | stationary error m | PINN IoU | numerical IoU |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 400.0 | -56.61 | -0.01 | -0.01 | 0.739 | 1.000 |
| 0 | 1333 | 400.0 | -67.71 | -0.01 | -0.01 | 0.689 | 1.000 |
| 0 | 2667 | 400.0 | -83.09 | -0.01 | -0.01 | 0.628 | 1.000 |
| 0 | 4000 | 400.0 | -155.35 | -0.01 | -0.01 | 0.371 | 1.000 |
| 0.01 | 0 | 400.0 | -56.08 | -0.01 | -0.01 | 0.741 | 1.000 |
| 0.01 | 1333 | 413.3 | -60.08 | -0.08 | -13.34 | 0.725 | 1.000 |
| 0.01 | 2667 | 426.7 | -66.23 | -0.15 | -26.67 | 0.711 | 1.000 |
| 0.01 | 4000 | 440.0 | -113.43 | -0.22 | -40.01 | 0.549 | 0.999 |
| 0.1 | 0 | 400.0 | -49.87 | -0.01 | -0.01 | 0.766 | 1.000 |
| 0.1 | 1333 | 533.3 | -28.46 | -0.65 | -133.34 | 0.894 | 0.997 |
| 0.1 | 2667 | 666.7 | -18.55 | -1.15 | -266.67 | 0.944 | 0.997 |
| 0.1 | 4000 | 800.0 | -29.62 | -1.57 | -400.01 | 0.929 | 0.997 |

Perimeters: `experiment1/figures/perimeters.png`.
Radius errors: `experiment1/figures/radius_error.png`.

Reproduce with `python -m experiments.wildfirepinn_v2_diagnostics.experiment1` from the repository root. A finished run is skipped when its history already has the requested epoch count.

## Reading

These numbers are from one seed, 1,500 epochs, and a 64-wide network. The numerical solver uses the same initial field, the same R, and the same 10 m grid.

The solver tracks the exact circle. Final radius error is 0.01 m at R = 0, 0.22 m at R = 0.01 m/s, and 1.57 m at R = 0.1 m/s. Final IoU is 1.000, 0.999, and 0.997.

The PINN does not. At R = 0 the front should stay at 400 m. The predicted radius is already 57 m small at t = 0, then 155 m small at 4,000 s, and IoU falls from 0.739 to 0.371. At R = 0.01 m/s the true circle grows only 40 m. The PINN finishes 113 m small, IoU 0.549, which is worse than leaving the initial circle in place (IoU 0.825). At R = 0.1 m/s the true circle grows 400 m. The PINN follows that expansion: final radius error 30 m and IoU 0.929, against 0.250 for a frozen circle and 0.997 for the solver.

The physical PDE residual is about 0.006 m/s for all three networks. The stationary failure and the fast-expansion success have the same residual, so a small residual is not evidence that the front is right. Initial-condition RMSE is about 40 m in every run, and the zero contour is biased inward from the first frame.

What this measures: under an exact constant-R circle, the Godunov solver solves the equation, and this PINN solves it only when the required motion is large compared with its initial radius bias. What this does not measure: spatially varying R, recovery of R, or the Kincade forecast. Those experiments have not been run.
