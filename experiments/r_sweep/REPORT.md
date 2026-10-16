# Propagation-speed sweep

## Status

A two-epoch smoke test ran for one seed and one setting of each mode. The 4,000-epoch sweep did not run. The numbers in this file check that training, scoring, and figure code finish. They are not evidence about whether a different R improves the forecast.

## Experimental setup

Kincade 2019, persistent burned-region targets, chronological passes 0–11 train, 12–15 validation, 16–19 test. Network 4×128 tanh, Adam learning rate 0.001, weights w_ic=10.0, w_pde=1.0, w_obs=5.0, w_eik=0. Thirds sampling. Seeds control initialization and the batch stream. Future perimeters are not model inputs. Weather at later days is the same field the original residual samples.

The frozen v1 checkpoint uses w_pde=0, so changing R cannot change that file. This sweep turns the PDE on at the class-default weight 1 and varies only the pre-wind rate. Mode C is weather R with scale 1.

## Original R and units

Weather R is the stored spread-rate volume in m/s. Wind direction is not inside that volume. `directed_residual` forms R_eff = clip(R * (1 + 0.6 * alignment), min=0), with alignment in [-1, 1], so R_eff is between 0.4 R and 1.6 R. The network outputs dimensionless Phi. The coded residual is Phi_tau + (R_eff * T/L) |grad_XY Phi|. Dividing by T/L (9.7793 s/m) returns dphi/dt + R_eff |grad phi| in m/s. A unit check on a known gradient matched to 0.099564 m/s.

Collocation points are uniform in the model box and in tau over the whole fire, including times after the training passes. That matches `train_real`. The residual is not restricted to a front band. A zero spatial gradient is guarded by a 1e-12 floor inside the square root; the smoke and full runs record the fraction of probe points with |grad Phi| < 0.001.

`train_real` builds the network before `set_seed`, so its seed does not control Xavier initialization. This sweep seeds before the network is built. Mode C is the same loss and the same weather R, not a bitwise replay of an old process.

Domain-median weather R is 0.0344 m/s.

## Results

```
mode,r_m_s,r_scale,seed,epochs,w_pde,smoke,train_seconds,train_loss,train_pde_loss,train_ic_loss,train_obs_loss,val_iou,val_dice,val_area_mae_km2,val_area_signed_km2,val_sym_dist_m,val_contour_fraction,test_iou,test_dice,test_area_mae_km2,test_area_signed_km2,test_sym_dist_m,test_contour_fraction,pde_residual_m_s,near_zero_grad_fraction,selected_on_validation
constant,0.05,,0.0,2,1.0,True,0.5379485999874305,1.3084065914154053,0.1937102973461151,0.05023851618170738,0.12246222048997879,0.1710842384094204,0.29217423285521166,1179.1514249999998,1179.1514249999998,10350.017882342576,1.0,0.15579718431223105,0.2695679641948592,1375.1774999999998,1375.1774999999998,13355.272646663814,1.0,0.03041519783437252,0.0,False
weather_scale,,0.5,0.0,2,1.0,True,0.8068113000190351,1.3804821968078613,0.27531903982162476,0.051766570657491684,0.11749949306249619,0.17858045930213523,0.3030416856410641,1093.4777250000002,1093.4777250000002,8872.588840209854,1.0,0.16786527837294918,0.28745275773136764,1257.7740749999998,1257.7740749999998,11238.68970505967,1.0,0.03024468570947647,0.0,True
baseline,,1.0,0.0,2,1.0,True,0.7178657999902498,1.663567304611206,0.573734700679779,0.05888931825757027,0.10018786787986755,0.09922560660815694,0.18053582374389182,2261.2608,2261.2608,,0.0,0.10098896489416623,0.18345098217122238,2256.83415,2256.83415,,0.0,0.060072802007198334,0.0,False
persistence,,,,0,1.0,True,0.0,,,,,0.9533006392488322,0.9760679408468321,11.655900000000003,-11.655900000000003,276.3963424805699,1.0,0.9365859055439989,0.9672486599011202,16.082550000000005,-16.082550000000005,284.3560988643014,1.0,,,False
```

## Best validation configuration

Selected on validation only because it kept a zero contour: weather-R scale 0.50. Validation IoU 0.179, validation area MAE 1093 km², contour fraction 1.00. Those scores are from two epochs. They do not identify a useful R.

## Held-out test of that configuration

Test IoU 0.1679, test area MAE 1257.77 km², test contour fraction 1.00. This is the test score of the validation winner. It is not the best test score in the table.

## Persistence

Persistence repeats the pass-11 persistent mask with no motion. Validation IoU 0.9533, test IoU 0.9366.

## What this can and cannot say

A lower PDE residual at a new R shows that the magnitude was more compatible with the masks. A higher validation IoU, with contours still present and area error not worse, is evidence that the magnitude also helped the forecast. If the residual falls and IoU does not, the limitation is not fixed by rescaling R. That pattern points to the PDE form, the sparse VIIRS targets, or the network, which this sweep does not separate from each other.

Smoke figures are in `experiments/r_sweep/smoke/figures/`. The full command writes `experiments/r_sweep/results.csv` and `experiments/r_sweep/figures/`.

## Next command

```powershell
python run_r_sweep.py
```

Fifty-one trainings of 4,000 epochs. From the smoke timing, a rough CPU estimate is about 19.5 hours. Completed runs are skipped on a restart.
