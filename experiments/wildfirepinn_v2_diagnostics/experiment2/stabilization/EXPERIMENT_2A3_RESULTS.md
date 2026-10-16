# Experiment 2A.3

Fourier hard-IC models on the Experiment 2A spatial rate. F3 is the saved 2A.2 Fourier seed-0 checkpoint. Training does not use the numerical field or its future front. The level-set equation and the hard initial condition are unchanged.

Commands:

```
python -m unittest experiments.wildfirepinn_v2_diagnostics.test_experiment2a3
python -m experiments.wildfirepinn_v2_diagnostics.experiment2.stabilization --phase diagnose
python -m experiments.wildfirepinn_v2_diagnostics.experiment2.stabilization --phase F --config F1
python -m experiments.wildfirepinn_v2_diagnostics.experiment2.stabilization --phase F --config F2
python -m experiments.wildfirepinn_v2_diagnostics.experiment2.stabilization --phase F --config F4
python -m experiments.wildfirepinn_v2_diagnostics.experiment2.stabilization --phase G --base F1
python -m experiments.wildfirepinn_v2_diagnostics.experiment2.stabilization --phase H --base F1 --eikonal-weight 1e-3
python -m experiments.wildfirepinn_v2_diagnostics.experiment2.stabilization --phase H --base F1 --eikonal-weight 1e-2
python -m experiments.wildfirepinn_v2_diagnostics.experiment2.stabilization --phase F --config F1 --seed 1
python -m experiments.wildfirepinn_v2_diagnostics.experiment2.stabilization --phase F --config F1 --seed 2
```

F1 is the two diagonal modes of sin(kx) cos(ky). F2 adds the axis-aligned fundamentals. F4 is the axis-aligned fundamental and first harmonic, with no diagonals. High-frequency fraction is angular Fourier power in modes 6 and above, divided by all nonzero modes.

| run | init IoU | final IoU | symmetric m | p95 m | area km² | high m | low m | gap error m | sign | volume residual | front residual | HF fraction | components | other m² | params | train s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| F1 s0 | 1.000 | 0.962 | 12.45 | 26.50 | 0.023 | 229.3 | 180.8 | 3.5 | yes | 0.0073 | 0.0102 | 0.039 | 1 | 0.0 | 13057 | 47.5 |
| F1 s1 | 1.000 | 0.943 | 16.38 | 35.21 | -0.025 | 219.5 | 167.7 | 6.8 | yes | 0.0121 | 0.0159 | 0.107 | 1 | 0.0 | 13057 | 37.6 |
| F1 s2 | 1.000 | 0.942 | 17.89 | 39.33 | 0.028 | 223.9 | 185.0 | -6.1 | yes | 0.0118 | 0.0178 | 0.130 | 1 | 0.0 | 13057 | 39.8 |
| F1G s0 | 1.000 | 0.955 | 14.19 | 31.05 | 0.051 | 235.6 | 188.8 | 1.8 | yes | 0.0094 | 0.0068 | 0.029 | 1 | 0.0 | 13057 | 36.1 |
| F1H1em02 s0 | 1.000 | 0.956 | 13.58 | 29.18 | -0.001 | 213.0 | 176.9 | -8.8 | yes | 0.0115 | 0.0122 | 0.056 | 1 | 0.0 | 13057 | 54.7 |
| F1H1em03 s0 | 1.000 | 0.948 | 16.51 | 33.73 | 0.049 | 227.8 | 189.8 | -7.0 | yes | 0.0079 | 0.0095 | 0.039 | 1 | 0.0 | 13057 | 51.5 |
| F2 s0 | 1.000 | 0.938 | 17.93 | 55.38 | 0.039 | 243.7 | 181.5 | 17.2 | yes | 0.0122 | 0.0180 | 0.172 | 1 | 0.0 | 13313 | 47.0 |
| F3 s0 | 1.000 | 0.920 | 21.71 | 64.52 | 0.047 | 246.3 | 187.4 | 13.9 | yes | 0.0160 | 0.0233 | 0.165 | 2 | 1600.0 | 13825 | 43.0 |
| F4 s0 | 1.000 | 0.911 | 25.00 | 82.09 | 0.068 | 265.6 | 189.9 | 30.7 | yes | 0.0153 | 0.0246 | 0.159 | 1 | 0.0 | 13313 | 33.5 |

| run | IoU t=0 | IoU T/3 | IoU 2T/3 | IoU T | curvature 1/m | front grad | front Hessian 1/m |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| F1 s0 | 1.000 | 0.977 | 0.962 | 0.962 | 0.0051 | 1.110 | 0.0077 |
| F1 s1 | 1.000 | 0.973 | 0.958 | 0.943 | 0.0076 | 1.127 | 0.0135 |
| F1 s2 | 1.000 | 0.963 | 0.944 | 0.942 | 0.0093 | 1.174 | 0.0162 |
| F1G s0 | 1.000 | 0.976 | 0.962 | 0.955 | 0.0042 | 1.116 | 0.0062 |
| F1H1em02 s0 | 1.000 | 0.965 | 0.958 | 0.956 | 0.0043 | 1.021 | 0.0048 |
| F1H1em03 s0 | 1.000 | 0.969 | 0.953 | 0.948 | 0.0043 | 1.058 | 0.0055 |
| F2 s0 | 1.000 | 0.967 | 0.957 | 0.938 | 0.0189 | 1.251 | 0.0173 |
| F3 s0 | 1.000 | 0.964 | 0.951 | 0.920 | 0.0254 | 1.380 | 0.0501 |
| F4 s0 | 1.000 | 0.961 | 0.942 | 0.911 | 0.0257 | 1.291 | 0.0279 |

Numerical growth: high 220.4 m, low 175.4 m, gap 45.0 m. Numerical contour high-frequency fraction is 0.00117 and curvature RMS is 0.00191 1/m. Numerical-field Hessian near the front is 0.00151 1/m. Solver time on the 20 m grid is 0.076 s.

## Reading

The eight-frequency Fourier field oscillates because of its frequency bank. On the saved seed-0 checkpoint, high-frequency angular power is 0.165 against 0.00117 on the numerical contour, and contour-curvature RMS is 0.0254 1/m against 0.00191 1/m. The same roughness is in the represented field: volume |∇φ| has mean 1.12 and standard deviation 0.41, and the mean Hessian is 0.0255 1/m against 0.00151 1/m for the numerical field near its front. The predicted-front residual is 0.0233 m/s and the numerical-front residual is 0.0225 m/s. Signed radius error averages +13.4 m. Overshoot covers 59% of angles, with mean overshoot +31.3 m and mean undershoot −12.5 m. The fast side is 28.5 m too far and the slow side is 8.6 m too far. Predicted radial speed exceeds local R by 0.0056 m/s. A second burned component covers 1600 m², four cells, beside the main 1.127 km².

Keeping only the two diagonal modes of sin(kx) cos(ky), F1, is the change that improves the solution. Those modes are the exact Fourier content of the prescribed rate. F1 has 13,057 parameters, against 13,313 for F2 and F4 and 13,825 for the original bank. The smaller bank is the more accurate one. On seed 0, F1 reaches IoU 0.962, symmetric distance 12.45 m, 95th percentile 26.5 m, growth 229.3 m and 180.8 m, gap error +3.5 m, one burned component, high-frequency fraction 0.039, and curvature 0.0051 1/m. F2, which adds axis-aligned fundamentals, returns the high-frequency fraction to 0.172 and the gap error to +17.2 m. F4, which keeps the first harmonic and drops the diagonals, is the worst of the four: IoU 0.911, symmetric distance 25.0 m, gap error +30.7 m, high-frequency fraction 0.159. The harmonic wrinkles the front even when it cannot represent the rate.

Predicted-front sampling and the near-front Eikonal penalty were tested only on F1, and neither improved the perimeter. Front sampling lowered the near-front residual from 0.0102 m/s to 0.0068 m/s and the high-frequency fraction from 0.039 to 0.029, but both sides overshot: high-R growth 235.6 m against 220.4 m, low-R growth 188.8 m against 175.4 m. The gap error of +1.8 m is that shared overshoot, and the symmetric distance rose to 14.2 m. The gap stayed positive through training. Eikonal weights 10⁻³ and 10⁻² move |∇φ| toward 1. High-frequency energy stays at 0.039 and 0.056. At 10⁻² the area error is −0.001 km² while the gap error becomes −8.8 m, because the fast side undershoots. At 10⁻³ the final IoU is 0.948. Weight zero is the better fire forecast. No combined configuration was trained.

F1 seed 0 meets the provisional numeric targets. Seeds 1 and 2 miss the final-IoU target. Their final IoUs are 0.943 and 0.942, symmetric distances 16.4 m and 17.9 m, and gap errors +6.8 m and −6.1 m, all with the correct sign and one component. High-frequency fractions rise to 0.107 and 0.130. IoU at T/3 and 2T/3 is 0.973 and 0.958 for seed 1, and 0.963 and 0.944 for seed 2, so the miss accumulates in time. Across the original eight-frequency seeds the symmetric distance was 21.7, 32.4, and 24.3 m. F1 is better and less variable, and it is still seed-dependent.

A lower PDE residual is an unreliable guide to perimeter accuracy. F1 seed 0 has both the lowest volume residual in this table, 0.0073 m/s, and the best contour. The Eikonal run at 10⁻³ has volume residual 0.0079 m/s and a worse contour, 16.5 m. Front sampling has the lowest near-front residual, 0.0068 m/s, and a worse contour than uniform F1. The earlier constant-speed model had volume residual 0.010 m/s, a 32 m contour error, and no speed contrast. A small residual can be that constant-speed floor.

The stabilized direct PINN misses the provisional targets on two of three seeds. The remaining contour error is 12–18 m, the 95th percentile is 27–39 m, and high-frequency energy stays well above the numerical contour. The Godunov reference on this 20 m grid takes 0.076 s and stays about 1 m from a 10 m grid. A hybrid that puts a prescribed or learned rate into that numerical level-set solver is the better next architecture for propagation. Further direct level-set PINN work is justified only for the representation question already answered here: the two rate modes carry the speed contrast, and extra Fourier modes bring the wrinkles back.
