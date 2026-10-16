# Experiment 3

A hybrid inverse problem on the Experiment 2A spatial rate. Training sees perimeters at 0, 1000, and 2000 s. Forecasts at 3000 and 4000 s are held out. The true rate field is used only to build those perimeters and to score R after training. The level-set equation is phi_t + R|grad phi| = 0, integrated by the Godunov solver.

The two-parameter diagnostic is told the 1500 m product form sin(kx) cos(ky). The ray model and the MLP are not. The MLP is trained by the discrete adjoint of the Godunov update, one step at a time, and matched to observed perimeters. Its forecast is then recomputed with the NumPy solver. A single deep autograd graph of the full rollout overflows the process stack, so that form is not used.

Commands:

```
python -m unittest experiments.wildfirepinn_v3.test_experiment3
python -m experiments.wildfirepinn_v3.experiment3 --phase all
```

| run | study | fit m | IoU 3000 | sym 3000 | IoU 4000 | sym 4000 | p95 4000 | area km² | gap error m | sign | R rmse annulus | R corr annulus | R rmse exterior | train s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| persistence s0 | main | — | 0.835 | 45.72 | 0.713 | 90.42 | 119.55 | -0.310 | -10.0 | yes | — | — | — | 0.0 |
| constant s0 | main | 13.57 | 0.908 | 25.42 | 0.897 | 32.23 | 69.42 | 0.016 | -45.9 | no | 0.0105 | — | 0.0126 | 0.0 |
| oracle s0 | main | — | 1.000 | 0.00 | 1.000 | 0.00 | 0.00 | 0.000 | 0.0 | yes | 0.0000 | 1.000 | 0.0000 | 0.0 |
| ray s0 | main | 1.00 | 0.993 | 2.14 | 0.974 | 6.67 | 15.11 | 0.010 | -11.4 | yes | 0.0010 | 0.997 | 0.0159 | 0.0 |
| fourier s0 | main | 13.31 | 0.911 | 24.81 | 0.901 | 31.34 | 76.61 | 0.044 | -45.9 | no | 0.0102 | 1.000 | 0.0125 | 0.8 |
| fourier s1 | main | 0.14 | 0.999 | 0.25 | 0.999 | 0.33 | 0.62 | 0.001 | -0.1 | yes | 0.0001 | 1.000 | 0.0001 | 1.8 |
| fourier s2 | main | 3.10 | 0.973 | 5.92 | 0.972 | 7.73 | 12.06 | -0.031 | -2.1 | yes | 0.0024 | 1.000 | 0.0024 | 1.4 |
| x-only s0 | main | 13.32 | 0.911 | 24.84 | 0.901 | 31.38 | 76.95 | 0.044 | -46.0 | no | 0.0103 | 0.427 | 0.0125 | 0.8 |
| x-only s1 | main | 11.29 | 0.919 | 23.17 | 0.899 | 31.79 | 104.18 | 0.047 | -54.4 | no | 0.0096 | 0.427 | 0.0139 | 1.4 |
| fourier s0 | sparse | 17.13 | 0.912 | 24.30 | 0.903 | 30.66 | 75.00 | 0.041 | -45.0 | no | 0.0101 | 1.000 | 0.0123 | 0.9 |
| fourier s1 | sparse | 0.17 | 0.999 | 0.25 | 0.998 | 0.34 | 0.83 | 0.002 | -0.3 | yes | 0.0001 | 1.000 | 0.0001 | 1.0 |
| ray s0 | sparse | 0.89 | 0.990 | 3.37 | 0.966 | 9.08 | 19.23 | 0.016 | -13.9 | yes | 0.0020 | 0.982 | 0.0160 | 0.0 |
| fourier s0 | dense | 1.02 | 0.993 | 2.53 | 0.989 | 3.44 | 5.77 | 0.012 | -2.0 | yes | 0.0009 | 1.000 | 0.0010 | 1.3 |
| fourier s1 | dense | 0.12 | 0.999 | 0.25 | 0.999 | 0.33 | 0.62 | 0.001 | -0.1 | yes | 0.0001 | 1.000 | 0.0001 | 1.7 |
| ray s0 | dense | 0.85 | 0.994 | 1.73 | 0.978 | 5.71 | 13.23 | 0.006 | -9.9 | yes | 0.0009 | 0.998 | 0.0158 | 0.0 |
| fourier s0 | perturb20 | 5.00 | 0.955 | 11.89 | 0.958 | 12.00 | 27.02 | 0.016 | -6.9 | yes | 0.0014 | 1.000 | 0.0017 | 1.0 |
| fourier s0 | perturb40 | 5.56 | 0.919 | 22.42 | 0.925 | 21.83 | 37.84 | -0.005 | -6.7 | yes | 0.0013 | 1.000 | 0.0014 | 1.0 |
| fourier s0 | second_ignition | 0.53 | 0.997 | 1.01 | 0.995 | 1.32 | 2.82 | -0.003 | -2.1 | yes | 0.0005 | 1.000 | 0.0004 | 1.1 |
| mlp s0 | main | 7.50 | 0.949 | 15.00 | 0.918 | 24.95 | 62.98 | 0.033 | -48.0 | no | 0.0065 | 0.775 | 0.0175 | 17.7 |
| mlp s1 | main | 6.63 | 0.938 | 17.55 | 0.906 | 28.87 | 51.54 | 0.058 | -45.6 | no | 0.0060 | 0.818 | 0.0172 | 17.9 |
| mlp s2 | main | 8.36 | 0.945 | 15.37 | 0.924 | 23.15 | 69.81 | 0.018 | -48.8 | no | 0.0071 | 0.744 | 0.0175 | 18.2 |
| mlp s0 | perturb40 | 8.49 | 0.883 | 33.47 | 0.875 | 39.50 | 78.95 | 0.049 | -52.3 | no | 0.0064 | 0.781 | 0.0164 | 18.3 |
| mlp s0 | second_ignition | 8.19 | 0.934 | 19.64 | 0.907 | 30.71 | 64.97 | -0.003 | -45.7 | yes | 0.0070 | 0.835 | 0.0315 | 19.2 |

Reference solver time for the hidden field on the 20 m grid: 0.041 s.

## Reading

The perimeters determine the rate along the path the front actually traveled. They do not, by themselves, determine the rate on the rest of the domain. The swept annulus is 676 cells, 1.7% of the 201 by 201 grid. The initial disk is 1,252 cells. The exterior, which the forecast must enter, is 38,473 cells. Along the training path, the average of the 0–1000 s and 1000–2000 s radial speeds correlates 0.994 with the true rate at the late mid-radius. The same speeds correlate 0.70 with the true rate 150 m farther out. The early and late speeds agree with each other (correlation 0.978, rms difference 0.003 m/s), so one fire does not show a large radial change of R inside the annulus.

A two-parameter family that is told the form R0 [1 + a sin(kx) cos(ky)], with wavelength 1500 m, can recover the hidden coefficients from the training perimeters. The successful initialization reached R0 = 0.0501 and a = 0.504, against the hidden values 0.05 and 0.5. Its training contour error is 0.14 m. At 4000 s the forecast has IoU 0.999, symmetric distance 0.33 m, 95th percentile 0.62 m, and growth 220.7 m against 175.8 m (gap error −0.1 m, correct sign). A second initialization reached a = 0.488 and a 7.7 m forecast. The initialization at a = 0 stayed at a = 0.001 after 50 Nelder–Mead evaluations and forecast like a constant rate: symmetric distance 31.3 m and gap error −46 m. Recovery is real, and it is initialization-dependent at this budget. Because the form is global, a successful fit also sets R outside the annulus (exterior rmse 0.0001 m/s). That exterior agreement is the parametric assumption, not an independent measurement.

The same family is recoverable from other observation designs when the search does not stay at a = 0. Endpoints at 0 and 2000 s, started from (R0, a) = (0.02, 0.3), reach a = 0.495 and a 0.34 m forecast. Five training times reach a = 0.504. A second ignition at (−400, 300) m reaches R0 = 0.0499 and a = 0.485, with a 1.3 m forecast. A 20 m radial perturbation of the training perimeters leaves a 12.0 m forecast, IoU 0.958, and the correct growth sign. A 40 m perturbation leaves a 21.8 m forecast. An x-only family, R0 [1 + b sin(kx)], cannot represent the product and its best training fit is 11.3 m, with the growth gap on the wrong side.

The ray model uses no formula. It paints each ray from the ignition with the observed radial speed and holds that speed outside the measured radii. Training fit is 1.0 m. The 4000 s forecast has IoU 0.974, symmetric distance 6.67 m, 95th percentile 15.1 m, and growth 215.8 m against 182.2 m (gap 33.6 m, truth 45.0 m, correct sign). Annulus rmse is 0.0010 m/s. Exterior rmse is 0.0159 m/s. The forecast is accurate because this rate varies slowly across the next 2000 s of travel; the exterior field itself is not identified.

Constant-rate propagation, with the one speed 0.0482 m/s taken from the training perimeters, has 4000 s IoU 0.897, symmetric distance 32.2 m, and growth 190.4 m against 191.3 m. The gap is −0.9 m. Persistence of the 2000 s perimeter is 90.4 m away at 4000 s. The oracle, which is the hidden rate in the same solver, has error 0 on this grid and takes 0.041 s.

The small MLP does not recover the propagating asymmetry. Three seeds, 30 epochs, and the discrete adjoint of phi_t + R|grad phi| = 0 give 4000 s IoU 0.918, 0.906, and 0.924, and symmetric distances 25.0, 28.9, and 23.2 m. Seed 0 grows 193.9 m on the fast side and 196.9 m on the slow side (gap −3.0 m). Annulus correlation with the true rate is 0.74 to 0.82, and the exterior correlation is about 0. The training contour loss fell from 0.106 to about 0.04 and was still decreasing at epoch 30. That training fit does not become the correct forecast. These runs do not support a claim that the network has learned the spatial rate.

The engineering targets are met by the successful two-parameter fits and, on IoU and symmetric distance, by the ray model. The ray model's gap error is −11.4 m, just outside the 10 m target. The MLP misses the targets on every seed. The zero-amplitude parametric start also misses them. Nothing here is a real-fire model: the rate has no wind or fuel, the parametric success uses the true functional family, and the free network does not carry the asymmetry.

An API would need a perimeter input, the Godunov propagator, and a rate estimate whose unsupported region is labeled. The ray model can ship with the annulus mask. The two-parameter fit can ship only if the family is an explicit, tested assumption, together with the spread across initializations. The MLP checkpoint should not be the served model. Also still required: wind and fuel covariates, a held-out real-fire test, and a timing check on the Kincade grid. The solver on this 20 m grid takes 0.041 s. No service was deployed in this experiment.

