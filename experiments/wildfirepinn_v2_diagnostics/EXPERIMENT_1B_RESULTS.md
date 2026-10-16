# Experiment 1B

Hard initial-condition ablations on the same circle as Experiment 1. Experiment 1 checkpoints were not retrained. Experiments 2–5 were not run.

Runs below use 1,500 epochs and seed 0.

Audit of Experiment 1, measured on its checkpoints: the exact-field contour extractor is off by 0.008 m, so the reported radius error is not a contour-extraction bug. Normalization, the sign of phi, and the conversion from the coded residual to m/s match the chain rule. On the trained R = 0 network the gradient magnitude on the true front is about 0.62 rather than 1, and phi is biased about +33 m in the 50 m band around that front. A positive bias moves the zero contour inward. About 3% of uniform collocation points fall within 100 m of the moving front. The volume-averaged residual stays near 0.006 m/s while the front residual is larger, and at R = 0 the front time derivative changes from about −0.010 m/s to +0.028 m/s, which continues the inward drift.

B forces phi(x, y, 0) = phi0 and trains only the PDE. C uses the same ansatz with half of the PDE samples inside a 100 m band of the analytical front. That band is an oracle: the true future front is known. E multiplies the learned correction by R. Its R = 0 result is exact by algebra, before any learning, and is not evidence that the network solved the PDE.

| config | R | init IoU | final IoU | init radius m | final radius m | final radial miss m | IC RMSE m | volume residual m/s | front residual m/s | inferred speed m/s | train s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 0 | 0.739 | 0.371 | -56.61 | -155.35 | 155.35 | 40.13 | 0.0066 | 0.0100 | -0.0247 | 44.3 |
| A | 0.01 | 0.741 | 0.549 | -56.08 | -113.43 | 113.43 | 40.51 | 0.0063 | 0.0079 | -0.0043 | 43.0 |
| A | 0.1 | 0.766 | 0.929 | -49.87 | -29.62 | 29.62 | 42.99 | 0.0060 | 0.0070 | 0.1051 | 42.7 |
| B | 0 | 1.000 | 0.983 | -0.01 | 2.92 | 2.92 | 0.00 | 0.0018 | 0.0022 | 0.0007 | 39.3 |
| B | 0.01 | 1.000 | 0.986 | -0.01 | 2.74 | 2.74 | 0.00 | 0.0018 | 0.0022 | 0.0107 | 39.5 |
| B | 0.1 | 1.000 | 0.994 | -0.01 | 0.95 | 1.97 | 0.00 | 0.0020 | 0.0022 | 0.1002 | 41.6 |
| C | 0 | 1.000 | 0.997 | -0.01 | 0.13 | 0.59 | 0.00 | 0.0020 | 0.0012 | 0.0000 | 48.0 |
| C | 0.01 | 1.000 | 0.997 | -0.01 | 0.06 | 0.61 | 0.00 | 0.0020 | 0.0012 | 0.0100 | 40.4 |
| C | 0.1 | 1.000 | 0.997 | -0.01 | -0.40 | 1.21 | 0.00 | 0.0020 | 0.0013 | 0.0999 | 38.4 |
| E | 0 | 1.000 | 1.000 | -0.01 | -0.01 | 0.01 | 0.00 | 0.0000 | 0.0000 | 0.0000 | 36.7 |
| E | 0.01 | 1.000 | 1.000 | -0.01 | -0.06 | 0.06 | 0.00 | 0.0000 | 0.0000 | 0.0100 | 37.9 |
| E | 0.1 | 1.000 | 0.994 | -0.01 | -2.99 | 2.99 | 0.00 | 0.0007 | 0.0008 | 0.0993 | 37.7 |

Prescribed speeds are 0, 0.01, and 0.1 m/s. Inferred speed is the change in predicted radius divided by T. The numerical solver's final radius error on this grid is about 0.01 m, 0.22 m, and 1.57 m for those three speeds.

Targets were initial IoU at least 0.99, final IoU at least 0.99, final mean radial error under 5 m, and no drift at R = 0. A configuration that misses them is reported as a miss.

Figures: `experiment1b/figures/perimeters.png` and `experiment1b/figures/final_radius_error.png`.

Reproduce with `python -m experiments.wildfirepinn_v2_diagnostics.experiment1b --configs B` from the repository root. Add `C` or `E` to train those ablations. Finished histories are skipped.

## Reading

Configuration D was not run. B's loss is already only the PDE residual, so a larger PDE weight mostly rescales the Adam step. The remaining B error was a few meters, and C shows that where the samples sit accounts for that remainder.

Targets, seed 0, 1,500 epochs:

| config | initial IoU | final IoU, all three R | final radial miss | R = 0 drift |
| --- | --- | --- | --- | --- |
| A soft IC | 0.74–0.77, miss | 0.371 / 0.549 / 0.929, miss | 155 / 113 / 30 m, miss | radius error grows from −57 m to −155 m |
| B hard IC | 1.000 | 0.983 / 0.986 / 0.994, R = 0 and 0.01 miss 0.99 | 2.92 / 2.74 / 1.97 m, all under 5 m | +2.92 m in 4,000 s, inferred speed 0.0007 m/s |
| C hard IC + oracle front | 1.000 | 0.997 / 0.997 / 0.997 | 0.59 / 0.61 / 1.21 m | +0.13 m, inferred speed 0.00004 m/s |
| E R-scaled | 1.000 | 1.000 / 1.000 / 0.994 | 0.01 / 0.06 / 2.99 m | exact, and the PDE loss is 0 at epoch 1 |

B and C both make the initial field exact: IC RMSE is 0.00 m and the initial radius error is the contour extractor's −0.008 m. The soft model never does this. Its front gradient magnitude is about 0.62, so the zero contour sits about 50 m inside the true circle before any propagation.

With that geometry fixed, uniform PDE samples already keep the final radius within 3 m and the inferred speeds within 0.0007 m/s of the prescribed values. Oracle front sampling, which uses the true future front and is not a forecast method, removes the last 3 m and clears the 0.99 IoU line. The PDE itself was not the broken piece once the initial field was exact.

E meets the numeric targets, including a perfect stationary fire. That stationary score is algebraic: the correction is multiplied by R, the loss is zero on the first step, and the weights receive no useful gradient. At R = 0.1 m/s, where the factor does not remove the problem, E's final radius error is −3.0 m, similar to B and larger than C. E is evidence about the parameterization, not about learning the stationary PDE.

The numerical solver remains the reference. Its final radius errors are 0.01 m, 0.22 m, and 1.57 m. Hard IC brings the PINN into that range. It does not show up in a small volume residual: A and B both have volume residuals of a few thousandths of a meter per second, and only B tracks the front.

The smallest change supported by these runs is to build the level set as phi0 plus (t/T) times a network correction, with derivatives taken through phi0, and to drop the soft initial-condition penalty. phi0 on this benchmark is the exact circle. On a real fire it would be the signed distance of the initial observed mask. Oracle front sampling is not part of that change. The R-scaled factor is not part of it either, because its stationary success does not come from the PDE loss.
