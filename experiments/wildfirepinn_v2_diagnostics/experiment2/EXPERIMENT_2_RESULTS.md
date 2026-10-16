# Experiment 2

Hard-IC PINN against a numerical reference. R is prescribed. Oracle future-front sampling is not used. The R-scaled ansatz is not used. Experiment 1, Experiment 1B, and the Kincade model were not modified.

Spatial R uses R0 = 0.05 m/s, amplitude 0.5, wavelength 1500 m, so R stays in [0.025, 0.075] m/s. Wind cases use a constant base rate 0.05 m/s and c = 0.6. Wind points in the direction the wind blows toward. The initial circle has radius 400 m. The irregular fire is the polar curve r(1 + 0.15 cos 3θ + 0.1 sin 5θ), converted to a signed-distance grid.

| scenario | init IoU | final IoU | symmetric m | p95 m | area error km² | volume residual m/s | front residual m/s | components | train s | solver s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2a | 1.000 | 0.897 | 32.41 | 81.79 | 0.051 | 0.0100 | 0.0096 | 1 | 41.8 | 0.08 |

## 2a

spatial R, circular initial fire, uniform samples.

| t s | IoU | symmetric m | p95 m | area error km² |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 1.000 | 0.00 | 0.00 | 0.000 |
| 1333 | 0.947 | 12.83 | 23.74 | 0.001 |
| 2667 | 0.916 | 23.22 | 54.07 | 0.026 |
| 4000 | 0.897 | 32.41 | 81.79 | 0.051 |

Radial growth beyond 400 m where R ≥ R0: PINN 200.3 m, numerical 220.4 m. Where R < R0: PINN 201.0 m, numerical 175.4 m.

Numerical R = 0 changes the field by at most 0.0000 m. Final burned components: 1. Contours: 1.

Grid refinement of the spatial-R numerical front against the 10 m run:

- 40m: {'solver_seconds': 0.015099999989615753, 'symmetric_to_10m_m': 3.1506376001421392}
- 20m: {'solver_seconds': 0.09726129999035038, 'symmetric_to_10m_m': 1.0081658067587556}
- 10m_solver_seconds: 2.016425600013463

Figures are in `experiment2/figures/`. Finished runs are skipped when the history already has the requested epoch count.

## Reading

Experiment 2A does not pass the spatial-speed check, so the wind, irregular-shape, and predicted-front trainings were not started. The unit tests for those pieces did run: the wind step matches the expansion-plus-advection identity on a planar front, a circle advances farther downwind than upwind, a mid-run wind change moves the later growth onto the new axis, c ≥ 1 is refused, and the irregular signed-distance contour lies within 1.5 cells of its polygon.

On 2A the hard initial condition is exact (IoU 1.000, contour distance 0). By 4,000 s the PINN IoU is 0.897, the symmetric contour distance is 32 m, and the 95th percentile is 82 m. The burned region stays one piece and the area error is only 0.05 km². The shape is the failure. Where the numerical front sees R ≥ 0.05 m/s it has grown 220 m; where R is lower it has grown 175 m. The PINN has grown 200 m on both sides.

Near that numerical front the PINN gradient magnitude is 1.00. Its time derivative is −0.055 m/s on both the fast side, where the local R averages 0.057 m/s, and the slow side, where R averages 0.040 m/s. The slow side therefore carries the larger residual, about 0.015 m/s against 0.005 m/s. The network settled on one front speed. The PDE loss was already flat by epoch 750 (1.7×10⁻⁴ to 1.6×10⁻⁴).

The 20 m numerical front used as the reference is 1.0 m from a 10 m run, and the 40 m front is 3.2 m from that 10 m run. The solver takes 0.08 s on the training grid and about 2 s on the 10 m grid. The PINN’s 32 m contour error is not a coarse-reference artifact. Numerical R = 0 leaves the field unchanged.

A known smooth R is not yet reproduced at the front, so these runs do not support moving on to estimating an unknown R.

A follow-up on the same 2A checkpoint separates sampling, representation, and the PDE gradient. The writeup is `experiment2/cause_isolation/CAUSE_ISOLATION.md`. The front is 3.5% of the uniform points, but high and low R each occupy half the batch and the residual already tracks R throughout the domain. Direct supervision fits a constant-rate circle to 4.5 m and leaves the spatially varying target at 32 m, with both sides still growing together. At that constant-speed solution the high-R and low-R parameter gradients cancel, and one Adam step changes the speed gap by about 5×10⁻⁶ m/s.
