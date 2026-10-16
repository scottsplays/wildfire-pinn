# WildfirePINN v1

Frozen Kincade checkpoint. This card describes the selected model only. Experimental losses that were not used to train this file are documented in `PHYSICS_ABLATION.md`.

## Identity

| | |
|---|---|
| Name | WildfirePINN v1 |
| What it is | Observation-only level-set network |
| Checkpoint | `outputs/kincade_final_candidates/observation/levelset_pinn.pt` |
| SHA-256 | `93de4791f6017664174b6dec0bc9434bd2bbe5fffd52dc63d795168f4890e996` |
| Dataset | 2019 Kincade Fire, VIIRS SNPP, EPSG:3310, 90 m |
| Selected on | Validation passes 12–15 |
| Test used for selection | No |

The checkpoint file is local. The hash is the immutability check. `run_wildfirepinn_v1.py` recomputes it and refuses to continue if the bytes change during loading.

## Inputs

The network takes three scalars, already normalized:

- τ = t / T, with T = 523,680 s (6.061 days), τ = 0 at the first overpass
- X = (x − x_c) / L
- Y = (y − y_c) / L

L = 53,550 m. The center (x_c, y_c) is the center of the model grid. Weather, fuel, slope, moisture, and the spread rate R are not inputs to v1.

## Output

The network returns Φ. The field used for maps and metrics is φ = Φ · L, in meters.

- φ < 0: predicted burned region
- φ = 0: predicted fire perimeter
- φ > 0: outside the predicted burned region

## Architecture

| | |
|---|---|
| Type | Multilayer perceptron |
| Hidden layers | 4 |
| Hidden width | 128 |
| Activation | tanh |
| Normalization | Input coordinates only, as above. No batch normalization. |
| Initialization | Xavier normal. Hidden layers use the tanh gain. The output layer uses gain 1. Biases start at 0. |
| Parameter count | 50,177 |

The layers are `Linear(3, 128)`, tanh, three blocks of `Linear(128, 128)` and tanh, then `Linear(128, 1)`.

## Data split

Chronological. Nothing is shuffled.

| Split | Passes | Role |
|---|---|---|
| Train | 0–11 | Fit the signed distance |
| Validation | 12–15 | Model selection |
| Test | 16–19 | Untouched report |

The observation target on the training passes is the persistent burned region: the cumulative union of accepted VIIRS pixel disks. Trimmed spatial outliers are excluded. φ is negative inside that region. Validation and test are scored against the same kind of region. They are not hotspot-hull scores.

## Training of this checkpoint

| | |
|---|---|
| Optimizer | Adam |
| Learning rate | 0.001 |
| Epochs | 4,000 |
| Seed | 0 |
| Observation sampling | Thirds: inside, on the boundary, and outside, averaged so the exterior does not dominate |
| Initial condition | Pass 0, 4,096 points, weight 10 |
| Later observations | Passes 1–11, 1,024 points each, averaged, weight 5 |
| PDE weight | 0 |
| Eikonal weight | 0 |

The objective that updates the weights is

```
L = 10 * L_ic + 5 * L_obs
```

`L_ic` and `L_obs` are mean squared errors of φ against the persistent signed-distance target. The trainer still evaluates a weather-rate PDE residual so that its random stream matches the other Kincade runs, then multiplies that term by zero. It does not change this checkpoint.

## Held-out behavior

Recomputed by `run_wildfirepinn_v1.py` from the frozen file. Means are over passes. Area error is the mean absolute difference between predicted and persistent raster area. Boundary distance is the symmetric mean distance between the predicted φ = 0 line and the persistent boundary. A φ = 0 contour exists on all 20 passes.

| Split | IoU | Dice | Boundary distance | Absolute area error |
|---|---|---|---|---|
| Train | 0.686 | 0.812 | 805 m | 9.5 km² |
| Validation | 0.404 | 0.564 | 1,962 m | 67.0 km² |
| Test | 0.129 | 0.225 | 3,919 m | 38.0 km² |

Validation was the selection split. Test was not. On validation the predicted area is larger than the target on every pass (mean signed error +67.0 km²). On test the region continues south of the persistent fire, and overlap falls from IoU 0.18 to 0.08 across passes 16–19.
