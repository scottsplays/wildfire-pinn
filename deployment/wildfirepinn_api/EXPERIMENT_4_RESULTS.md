# Experiment 4 results

Synthetic generalization of the Experiment 3 inverse solvers, then a local forecast API. No earlier experiment file or checkpoint was modified. The Gauss-inspired heat model was not started. Nothing was deployed to Google Cloud.

```text
python -m unittest deployment.wildfirepinn_api.tests.test_estimators deployment.wildfirepinn_api.tests.test_api deployment.wildfirepinn_api.tests.test_pinn
python -m deployment.wildfirepinn_api.generalization
```

The study uses the Experiment 2 domain, a 201-node grid (20 m), radius 400 m, and duration 4000 s. Training perimeters are t = 0, 1000, and 2000 s. Scores use t = 3000 and 4000 s. The true rate is used only after the fit, to score the forecast and the rate error. Fourier always uses the frozen 1500 m product family and the frozen Nelder–Mead starts. The start with the smaller training-contour error is kept. Held-out times are not used to choose it.

Case A is the original field R = 0.05 [1 + 0.5 sin(kx) cos(ky)] at 1500 m. Case B is the same product at 2200 m with a phase shift and amplitude 0.35. Case C is the nonperiodic field 0.03 + 0.035 / (1 + exp(-(x + 0.25 y) / 700)). Two ignitions are (200, -150) m and (-400, 300) m. Final-circle clearance stays above 1100 m.

Case D, an anisotropic rate, was not run. The Godunov solver advances phi_t + R(x, y) |grad phi| = 0. A normal-dependent speed is a different Hamiltonian. The wind-advection solver in `directional.py` was not part of these fits.

The direct PINN was not entered in this comparison. It does not estimate R, and it refuses every domain except its training circle. That check is the unit test below.

## Forecast scores

IoU, symmetric distance, 95th-percentile distance, and area error are at 4000 s unless noted. Gap error is the predicted high-minus-low radial-growth gap minus the reference gap. The split is the median of the true rate on the reference contour. Rate RMSE is meters per second. Annulus means cells burned by 2000 s but not at 0 s. Exterior means cells still unburned at 2000 s. Correlation is omitted when a field has no variation.

| Case | Ignition | Model | IoU 3000 | IoU 4000 | Sym m | P95 m | Area km2 | Gap error m | Annulus RMSE | Exterior RMSE | Fit s | Solve s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | primary | constant | 0.908 | 0.897 | 32.23 | 69.42 | +0.016 | -52.6 | 0.0105 | 0.0126 | 0.00 | 0.03 |
| A | primary | ray | 0.993 | 0.974 | 6.67 | 15.10 | +0.010 | -9.9 | 0.0010 | 0.0159 | 0.02 | 0.03 |
| A | primary | Fourier | 0.999 | 0.999 | 0.33 | 0.62 | +0.001 | +0.6 | 0.0001 | 0.0001 | 3.98 | 0.04 |
| A | second | constant | 0.905 | 0.889 | 34.99 | 79.77 | -0.002 | -64.8 | 0.0126 | 0.0125 | 0.00 | 0.03 |
| A | second | ray | 0.990 | 0.971 | 8.95 | 16.85 | -0.011 | -5.4 | 0.0008 | 0.0166 | 0.03 | 0.05 |
| A | second | Fourier | 1.000 | 1.000 | 0.11 | 0.22 | +0.000 | -0.2 | 0.0000 | 0.0000 | 5.68 | 0.05 |
| B | primary | constant | 0.916 | 0.900 | 31.31 | 65.89 | -0.003 | -60.9 | 0.0102 | 0.0089 | 0.00 | 0.04 |
| B | primary | ray | 0.996 | 0.994 | 1.84 | 5.77 | -0.003 | +0.2 | 0.0006 | 0.0128 | 0.03 | 0.04 |
| B | primary | Fourier | 0.918 | 0.898 | 31.79 | 70.17 | -0.081 | -38.4 | 0.0108 | 0.0130 | 3.72 | 0.04 |
| B | second | constant | 0.913 | 0.895 | 31.52 | 67.32 | -0.016 | -59.2 | 0.0099 | 0.0092 | 0.00 | 0.03 |
| B | second | ray | 0.992 | 0.990 | 3.13 | 7.46 | -0.011 | -0.7 | 0.0006 | 0.0124 | 0.03 | 0.04 |
| B | second | Fourier | 0.913 | 0.898 | 30.54 | 65.79 | -0.010 | -58.1 | 0.0098 | 0.0092 | 3.89 | 0.03 |
| C | primary | constant | 0.959 | 0.948 | 15.64 | 25.37 | -0.006 | -31.2 | 0.0040 | 0.0112 | 0.00 | 0.03 |
| C | primary | ray | 0.994 | 0.990 | 2.10 | 4.59 | -0.011 | -2.6 | 0.0005 | 0.0073 | 0.03 | 0.03 |
| C | primary | Fourier | 0.969 | 0.959 | 12.80 | 29.75 | -0.009 | -21.8 | 0.0031 | 0.0118 | 3.70 | 0.03 |
| C | second | constant | 0.960 | 0.952 | 14.77 | 25.53 | -0.009 | -29.4 | 0.0038 | 0.0119 | 0.00 | 0.03 |
| C | second | ray | 0.995 | 0.995 | 1.99 | 4.26 | -0.006 | -2.0 | 0.0005 | 0.0082 | 0.03 | 0.03 |
| C | second | Fourier | 0.961 | 0.953 | 14.42 | 25.83 | -0.005 | -28.4 | 0.0037 | 0.0119 | 3.57 | 0.03 |

Case A, primary ignition, reproduces Experiment 3: constant IoU 0.897 and 32.23 m, ray IoU 0.974 and 6.67 m, Fourier IoU 0.999 and 0.33 m with r0 = 0.0501 and amplitude 0.504. The training fit that selected that Fourier start was 0.14 m.

Constant-rate gaps at 4000 s are within 1 m of zero on every fire. The reference gaps are 29 to 65 m. A 0.3 m residual that happens to be positive is not a recovered asymmetry.

Ray gap errors are -9.9 to +0.2 m, and the sign matches the reference on all six fires. Fourier gap errors are under 1 m on case A and 22 to 58 m on cases B and C.

Annulus correlations for the ray model are 0.997 to 0.999. Exterior correlations are 0.02 to 0.06 on the two oscillatory fields and 0.96 to 0.97 on the smooth logistic. The annulus is 612 to 727 cells out of 40,401, about 1.5 to 1.8 percent of the grid. Agreement outside that annulus on case A is the Fourier formula filling the domain, not an independent measurement. The ray model does not recover the 1500 m wave outside the swept radii (exterior RMSE 0.016 m/s, correlation about 0.04).

No supported run failed. Fit time is about 0.03 s for the ray model and 4 to 6 s for Fourier. The forecast solve is about 0.03 to 0.05 s.

The figure is `results/figures/forecast_iou.png`.

## PINN checkpoint

`direct_pinn_experimental` loads F1 seed 0. SHA-256 is `cf1f2d6feab3d42045f4f034007ae71b29b46a133e1d7f2360a115771412d928`. The unit test compares its 4000 s field with `predict_phi` from Experiment 2 and with the published F1 seed-0 score (IoU 0.962, symmetric distance 12.45 m). The test passed. Requests that supply a perimeter, another grid, or another benchmark are rejected. Seeds 1 and 2 are not loaded.

## Reading

1. Yes, along the path the front has already traveled. A two-parameter family recovers its own coefficients when that family is the truth. The ray model recovers the radial speeds without being given the formula. A prescribed 1500 m product does not recover a 2200 m wave or a logistic field.

2. The ray rate improves the held-out perimeter against constant-rate propagation on all six fires. The largest ray symmetric distance is 8.95 m, against 15 to 35 m for the constant rate. Fourier improves the forecast only on case A.

3. The annulus is constrained: ray RMSE there is at most 0.001 m/s. The exterior of a short-wavelength field is not. Case C looks better outside the annulus because the true field is smooth, which is not evidence that a general 2D rate is identifiable from three perimeters.

4. The ray hybrid outperforms constant-rate numerical propagation on IoU, contour distance, and growth asymmetry for every fire in this study. Fourier outperforms the constant rate only when its prescribed family matches the truth.

5. No. There is no wind, no fuel, and no real-fire holdout. The API will accept a projected meter CRS and will not score it.

6. A callable research API now runs locally. Deploying it still needs an approved cloud project, a secret for the API key, a container registry, and a remote request that has not been made. See `DEPLOYMENT_READINESS.md`.

The default model is `ray_inverse_solver`. It is not labeled a PINN.
