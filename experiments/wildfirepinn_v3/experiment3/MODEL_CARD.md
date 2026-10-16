# Model card — Experiment 3 inverse rate

## What this is

A synthetic inverse problem. Observed fire perimeters at 0, 1000, and 2000 s are used to estimate a positive rate of spread R(x, y). The forecast is the Godunov solution of

phi_t + R |grad phi| = 0

at 3000 and 4000 s. Those two times are not used in fitting.

## Which estimator to cite

The two-parameter fit that reached R0 = 0.0501 m/s and amplitude 0.504, and the ray-wise speed model.

The two-parameter fit is told the family R0 [1 + a sin(2 pi x / 1500) cos(2 pi y / 1500)] and the bounds 0.005 to 0.15 m/s and |a| <= 0.9. It is optimized by Nelder–Mead on the training contour distance, with at most 50 evaluations. One of three initializations stayed at a constant rate. The reported success is the initialization that did not.

The ray model sets R along each ray from the ignition equal to the observed radial speed and extrapolates that speed outside the measured radii. It has no trainable seed.

The MLP (width 32, two hidden tanh layers, R = 0.15 sigmoid(net(x/L, y/L)), 30 epochs, Adam learning rate 0.05) is a negative result. Its 4000 s growth gap has the wrong sign on all three seeds. Checkpoints are in `runs/rate_mlp_seed*.pt` for inspection. They are not a forecast model.

## Data

Synthetic circle on [-2000, 2000]^2 m, 201 nodes, ignition (200, -150) m, initial radius 400 m. The hidden rate is the Experiment 2A field, used to generate perimeters and, after training, to score R. Training code does not receive that array.

## Metrics that matter

Perimeter scores at 4000 s, the high-R versus low-R growth gap, and rate error split into the swept annulus and the exterior. A high correlation can coexist with a wrong amplitude or with a wrong forecast, so correlation is not the selection metric.

## Limits

- About 1.7% of the grid is the swept annulus. Exterior R is identified only when a global formula is assumed.
- The MLP adjoint is the exact derivative of the discrete Godunov step. A single deep autograd graph overflowed the process stack and was not used.
- No wind, no fuel, no real perimeter, no operational use.
- Not an API. Deployment would require a labeled unsupported region, a real-fire test, and wind and fuel inputs.

## Runtime

Reference solve on this grid: 0.041 s. Parametric fits: about 1–2 s. MLP training: about 18 s per seed on CPU.
