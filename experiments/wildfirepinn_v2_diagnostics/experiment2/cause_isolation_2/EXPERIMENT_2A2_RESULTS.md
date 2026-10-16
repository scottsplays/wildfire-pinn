# Experiment 2A.2

Hard-IC models on the Experiment 2A spatial rate. The baseline is the saved 2A checkpoint. Training does not use future fronts or the numerical field as a target. The level-set equation and the R-scaled ansatz are unchanged. High and low regions are the fixed partition R ≥ R0 and R < R0.

Commands:

```
python -m unittest experiments.wildfirepinn_v2_diagnostics.test_experiment2a2
python -m experiments.wildfirepinn_v2_diagnostics.experiment2.cause_isolation_2 --experiment A
python -m experiments.wildfirepinn_v2_diagnostics.experiment2.cause_isolation_2 --experiment B
python -m experiments.wildfirepinn_v2_diagnostics.experiment2.cause_isolation_2 --experiment C
python -m experiments.wildfirepinn_v2_diagnostics.experiment2.cause_isolation_2 --experiment E
python -m experiments.wildfirepinn_v2_diagnostics.experiment2.cause_isolation_2 --experiment B --seed 1
python -m experiments.wildfirepinn_v2_diagnostics.experiment2.cause_isolation_2 --experiment B --seed 2
```

Numerical solver time on the 20 m grid: 0.076 s. That reference is shared. Fourier features use the R wavelength, its first harmonic, and the axis and diagonal directions (8 frequencies, 16 features).

| run | init IoU | final IoU | symmetric m | p95 m | area error km² | volume residual m/s | front residual m/s | high growth m | low growth m | gap m | gap error m | sign | train s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| A seed 0 | 1.000 | 0.897 | 32.41 | 81.79 | 0.051 | 0.0100 | 0.0117 | 200.3 | 201.0 | -0.7 | -45.7 | no | 41.8 |
| B seed 0 | 1.000 | 0.920 | 21.71 | 64.52 | 0.047 | 0.0160 | 0.0234 | 246.3 | 187.4 | 58.9 | 13.9 | yes | 43.0 |
| B seed 1 | 1.000 | 0.887 | 32.35 | 95.06 | 0.089 | 0.0171 | 0.0260 | 248.6 | 202.7 | 46.0 | 1.0 | yes | 53.7 |
| B seed 2 | 1.000 | 0.913 | 24.32 | 67.42 | 0.079 | 0.0165 | 0.0265 | 255.1 | 198.1 | 57.0 | 12.0 | yes | 49.2 |
| C seed 0 | 1.000 | 0.897 | 32.49 | 81.81 | 0.051 | 0.0100 | 0.0117 | 200.2 | 200.9 | -0.7 | -45.7 | no | 34.3 |
| E seed 0 | 1.000 | 0.915 | 24.36 | 67.66 | 0.065 | 0.0156 | 0.0231 | 257.4 | 188.0 | 69.4 | 24.4 | yes | 45.3 |

Numerical growth: high 220.4 m, low 175.4 m, gap 45.0 m.

## A seed 0

saved Experiment 2A hard-IC checkpoint.

| t s | IoU | symmetric m | p95 m | area error km² |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 1.000 | 0.00 | 0.00 | 0.000 |
| 1333 | 0.947 | 12.83 | 23.74 | 0.001 |
| 2667 | 0.916 | 23.22 | 54.07 | 0.026 |
| 4000 | 0.897 | 32.41 | 81.79 | 0.051 |

Trained uniform gradient norm 0.0200, high/low cosine -0.999, PCGrad norm / uniform norm 0.520, balanced-loss relative difference 0.0016, φ_t high -0.0512 m/s, low -0.0512 m/s, corr(φ_t, R) -0.014.

## B seed 0

Fourier features, full spatial R.

| t s | IoU | symmetric m | p95 m | area error km² |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 1.000 | 0.00 | 0.00 | 0.000 |
| 1333 | 0.964 | 8.41 | 21.21 | 0.011 |
| 2667 | 0.951 | 12.71 | 26.74 | 0.004 |
| 4000 | 0.920 | 21.71 | 64.52 | 0.047 |

Trained uniform gradient norm 0.0101, high/low cosine 0.079, PCGrad norm / uniform norm 2.008, balanced-loss relative difference 0.0013, φ_t high -0.0646 m/s, low -0.0461 m/s, corr(φ_t, R) -0.611.

## B seed 1

Fourier features, full spatial R.

| t s | IoU | symmetric m | p95 m | area error km² |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 1.000 | 0.00 | 0.00 | 0.000 |
| 1333 | 0.954 | 10.47 | 23.89 | -0.004 |
| 2667 | 0.932 | 18.31 | 46.47 | 0.022 |
| 4000 | 0.887 | 32.35 | 95.06 | 0.089 |

Trained uniform gradient norm 0.0122, high/low cosine -0.068, PCGrad norm / uniform norm 2.136, balanced-loss relative difference 0.0013, φ_t high -0.0634 m/s, low -0.0477 m/s, corr(φ_t, R) -0.520.

## B seed 2

Fourier features, full spatial R.

| t s | IoU | symmetric m | p95 m | area error km² |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 1.000 | 0.00 | 0.00 | 0.000 |
| 1333 | 0.970 | 7.70 | 20.38 | -0.000 |
| 2667 | 0.958 | 11.22 | 33.46 | 0.005 |
| 4000 | 0.913 | 24.32 | 67.42 | 0.079 |

Trained uniform gradient norm 0.0108, high/low cosine 0.082, PCGrad norm / uniform norm 2.009, balanced-loss relative difference 0.0013, φ_t high -0.0656 m/s, low -0.0472 m/s, corr(φ_t, R) -0.614.

## C seed 0

staged lambda from constant R to spatial R.

| t s | IoU | symmetric m | p95 m | area error km² |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 1.000 | 0.00 | 0.00 | 0.000 |
| 1333 | 0.947 | 12.87 | 23.92 | 0.000 |
| 2667 | 0.915 | 23.28 | 54.03 | 0.025 |
| 4000 | 0.897 | 32.49 | 81.81 | 0.051 |

Trained uniform gradient norm 0.0200, high/low cosine -0.999, PCGrad norm / uniform norm 0.515, balanced-loss relative difference 0.0016, φ_t high -0.0512 m/s, low -0.0512 m/s, corr(φ_t, R) -0.011.

## E seed 0

Fourier features plus staged lambda.

| t s | IoU | symmetric m | p95 m | area error km² |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 1.000 | 0.00 | 0.00 | 0.000 |
| 1333 | 0.958 | 9.35 | 23.41 | 0.015 |
| 2667 | 0.946 | 13.04 | 39.89 | 0.011 |
| 4000 | 0.915 | 24.36 | 67.66 | 0.065 |

Trained uniform gradient norm 0.0120, high/low cosine 0.125, PCGrad norm / uniform norm 2.010, balanced-loss relative difference 0.0021, φ_t high -0.0651 m/s, low -0.0458 m/s, corr(φ_t, R) -0.629.

Figures are in `cause_isolation_2/figures/`.

## Reading

Fourier features at the wavelength of R are what break the constant-speed solution. The staged curriculum on the original network does not. None of the runs meet the full numeric targets (final IoU at least 0.95, symmetric distance at most 20 m, and growth-gap error at most 10 m) at the same time.

Experiment A is the saved 2A checkpoint, not a new training. Initial IoU is 1. Final IoU is 0.897, the symmetric distance is 32.4 m, and the growth gap is −0.7 m against the numerical gap of 45.0 m. φ_t is −0.051 m/s on both sides of R0. The high-R and low-R parameter gradients have cosine −0.999.

Experiment B adds 8 spatial frequencies: the R wavelength, its first harmonic, and the axis and diagonal directions. The first layer grows from 12,801 to 13,825 parameters. On seed 0 the final IoU is 0.920, the symmetric distance is 21.7 m, and the 95th percentile is 64.5 m. Growth is 246.3 m where R ≥ R0 and 187.4 m where R is lower, so the gap is 58.9 m and the gap error is +13.9 m. The sign matches the solver. φ_t is −0.065 m/s and −0.046 m/s in those two regions, and its correlation with R is −0.61. The volume residual rises from 0.010 m/s to 0.016 m/s. The constant-speed fit had the smaller residual and the wrong shape.

Seeds 1 and 2 also get the positive gap. Their gaps are 46.0 m and 57.0 m. Seed 1's gap error is 1.0 m because both sides are about 28 m too large (248.6 m and 202.7 m), not because the front matches. Symmetric distances on the three seeds are 21.7 m, 32.4 m, and 24.3 m. Each Fourier run leaves a second burned component of 1–4 grid cells (0.0004–0.0016 km²) beside a main region of about 1.13 km².

Experiment C keeps the original network and raises λ from 0 to 1. Through the constant-R quarter the loss falls to 8×10⁻⁵ and the gap is a few meters. As soon as λ leaves 0, the high/low gradient cosine falls from about +1 to −1 and stays there. The final state matches A: IoU 0.897, gap −0.7 m, φ_t −0.051 m/s on both sides.

Experiment E uses Fourier features and the same schedule. It also gets the correct sign, with φ_t correlation −0.63, but the gap is 69.4 m (error +24.4 m) and the symmetric distance is 24.4 m. Staging does not improve on Fourier features alone. The final burned region is one piece.

Experiment D was not trained. On a fixed batch the average of the two regional means differs from the uniform mean by 0.16%, so that loss is the baseline loss. PCGrad was applied to the same gradients. At initialization the regional gradients agree (cosine 0.9999) and PCGrad returns their sum, about twice the uniform step. At the trained A solution the regional norms are 0.129 and 0.165, the uniform norm is 0.020, and the PCGrad norm is 0.010. The projection removes the opposing part and leaves a smaller step than Adam, not a new spatial-speed direction.

The gradient cancellation is a symptom of a network whose correction is one speed. On A and C the cosine locks at −1 while φ_t stays flat. On B and E the cosine stays near 0 and φ_t follows R. The numerical solver on this 20 m grid takes 0.076 s. The best Fourier training takes 43 s and is still 22 m from that front, with a 65 m tail. These runs do not support moving on to wind-dependent R or to estimating an unknown R. The open question is how to keep the Fourier speed contrast without the overshoot and the wrinkled front.
