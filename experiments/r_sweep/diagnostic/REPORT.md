# Constant-R diagnostic

Four trainings, seed 0, 4,000 epochs, persistent Kincade targets, passes 0–11 / 12–15 / 16–19. Architecture and preprocessing match the existing level-set network. The other 47 sweep runs were not started.

Constant R uses w_pde = 1. The control uses the same batches with w_pde = 0, so the PDE does not update the weights. Persistence repeats the pass-11 mask.

## Results

Validation means over passes 12–15. Seed 0. Contours exist on every scored pass. PDE residual is the mean absolute physical residual on a fixed 4,096-point probe. For the three constant-R runs the probe uses that run's R. For the control it uses the weather rate, which was not in the loss. Area error is predicted minus observed. New-area recall is the fraction of cells that burn after pass 11 and are predicted burned. Predicted growth counts every cell the model marks burned outside the pass-11 mask. Front displacement is the symmetric distance from the pass-11 boundary. Observed validation growth is 11.7 km² and the observed front moves 276 m.

| run | val IoU | signed area km² | PDE residual m/s | IC IoU | new-area recall | predicted growth km² | predicted front m |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| R = 0 | 0.388 | -134.7 | 0.0036 | 0.685 | 0.000 | 12.8 | 1722 |
| R = 0.01 | 0.493 | +3.6 | 0.0044 | 0.563 | 0.048 | 87.8 | 1847 |
| R = 0.1 | 0.273 | +467.1 | 0.0075 | 0.640 | 0.583 | 521.8 | 5021 |
| w_pde = 0 | 0.416 | +141.6 | 0.0674 | 0.702 | 0.433 | 213.4 | 2292 |
| persistence | 0.953 | -11.7 | — | — | 0.000 | 0.0 | 0 |

Full columns are in `results.csv`. Per-pass rows are in `horizons.csv`.

## Reading

Changing R changes the validation forecast on this seed. R = 0.01 m/s reaches val IoU 0.493, above the no-physics control at 0.416 and above R = 0 at 0.388. R = 0.1 m/s is worse, at 0.273, because the front runs away. That spread is real. It does not close the gap to persistence, which scores 0.953 by repeating the pass-11 mask.

Initial-shape reconstruction is not the main failure. Every network has a zero contour at pass 0, and initial IoU is 0.563–0.702. The control, with the PDE weight at zero, fits the initial fire best (0.702) and then the later perimeter expands. The perimeters at pass 0 are smooth ovals around an irregular observed outline, so the network simplifies the shape, but that error is smaller than the later forecast error.

PDE enforcement is also not the failure for the constant-R runs. Probe residuals are 0.004–0.007 m/s and the near-zero spatial-gradient fraction is 0. R = 0 produces a nearly stationary field: it misses the whole 11.7 km² of new burn and under-covers the validation fire by 135 km². R = 0.1 satisfies a fast eikonal and predicts 522 km² of growth against 11.7 km² observed, moving the front about 5 km against an observed move of 276 m. R = 0.01 is closest in total area (+3.6 km²) but the new burn is in the wrong place: it predicts 88 km² of growth and recalls 5% of the cells that actually burn after pass 11. Its front stays about 1.8 km from the pass-11 boundary.

The validation fire is almost persistent. After pass 11 it grows by about 12 km² and the boundary moves about 280 m, so a motionless mask is a strong forecast. A single constant R, enforced over the full box, does not reproduce that small irregular growth. The limit visible in these four runs is that combination of the constant-speed PDE and this dataset, not a failure to fit the initial mask and not a failure to drive the residual down.

Test metrics were stored and were not used to choose among these four runs. The other 47 sweep runs were not started. Checkpoints and trainer state are under `experiments/r_sweep/diagnostic/runs/`. Running `python run_r_sweep.py --diagnostic` again skips any run whose history already has 4,000 epochs.

Side-by-side perimeters: `experiments/r_sweep/diagnostic/figures/perimeters.png`.
