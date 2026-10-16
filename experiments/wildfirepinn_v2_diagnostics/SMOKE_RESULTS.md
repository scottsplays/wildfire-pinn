# WildfirePINN v2 diagnostics

Experiment 1 is the expanding circle with a known solution. Experiments 2–5 have not been run.

This file was produced by `--smoke` (2 epochs). It is not evidence about whether the PINN solves the level-set equation.

Domain side 4000 m, ignition (200, -150) m, r0 400 m, T 4000 s, grid 401 nodes (10.0 m). Network: LevelSetPINN hidden 64, layers 4. Loss: 10 L_ic + 1 L_pde. No eikonal term and no boundary loss. PDE collocation excludes a 50 m disk around the ignition point, where the signed-distance gradient is undefined.

Radius error is the mean contour radius minus r0 + R t. Displacement is the mean absolute radial miss of the zero contour. Stationary repeats the initial circle. PDE residual is the mean absolute physical residual in m/s, away from the ignition disk. A small residual is not by itself a correct front.

## Final time

| R m/s | method | radius error m | radial miss m | IoU | PDE residual m/s | IC RMSE m | train s |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | pinn | 2205.62 | 2205.62 | 0.000 | 0.2130 | 954.88 | 0.0 |
| 0 | numerical | -0.01 | 0.01 | 1.000 | — | — | — |
| 0 | stationary | -0.01 | 0.01 | 1.000 | — | — | — |
| 0.01 | pinn | 2162.92 | 2162.92 | 0.000 | 0.2149 | 954.33 | 0.1 |
| 0.01 | numerical | -0.22 | 0.22 | 0.999 | — | — | — |
| 0.01 | stationary | -40.01 | 40.01 | 0.825 | — | — | — |
| 0.1 | pinn | 1764.30 | 1764.30 | 0.000 | 0.2330 | 949.64 | 0.1 |
| 0.1 | numerical | -1.57 | 1.57 | 0.997 | — | — | — |
| 0.1 | stationary | -400.01 | 400.01 | 0.250 | — | — | — |

## Radius error by time

| R m/s | t s | true radius m | PINN error m | numerical error m | stationary error m | PINN IoU | numerical IoU |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 400.0 | 1162.56 | -0.01 | -0.01 | 0.000 | 1.000 |
| 0 | 1333 | 400.0 | 1806.83 | -0.01 | -0.01 | 0.000 | 1.000 |
| 0 | 2667 | 400.0 | 1917.63 | -0.01 | -0.01 | 0.000 | 1.000 |
| 0 | 4000 | 400.0 | 2205.62 | -0.01 | -0.01 | 0.000 | 1.000 |
| 0.01 | 0 | 400.0 | 1165.17 | -0.01 | -0.01 | 0.000 | 1.000 |
| 0.01 | 1333 | 413.3 | 1800.34 | -0.08 | -13.34 | 0.000 | 1.000 |
| 0.01 | 2667 | 426.7 | 1890.85 | -0.15 | -26.67 | 0.000 | 1.000 |
| 0.01 | 4000 | 440.0 | 2162.92 | -0.22 | -40.01 | 0.000 | 0.999 |
| 0.1 | 0 | 400.0 | 1185.76 | -0.01 | -0.01 | 0.000 | 1.000 |
| 0.1 | 1333 | 533.3 | 1735.58 | -0.65 | -133.34 | 0.000 | 0.997 |
| 0.1 | 2667 | 666.7 | 1650.24 | -1.15 | -266.67 | 0.000 | 0.997 |
| 0.1 | 4000 | 800.0 | 1764.30 | -1.57 | -400.01 | 0.000 | 0.997 |

Perimeters: `experiment1/figures/perimeters.png`.
Radius errors: `experiment1/figures/radius_error.png`.

Reproduce with `python -m experiments.wildfirepinn_v2_diagnostics.experiment1` from the repository root. A finished run is skipped when its history already has the requested epoch count.
