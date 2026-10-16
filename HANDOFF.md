# WildfirePINN v1 handoff

This is the frozen Kincade v1 checkpoint packaged so a teammate can load it and run inference. Nothing in this handoff retrains the network, changes the split, or selects a different model.

v1 is an observation-only neural level-set baseline. Its loss weight on the level-set PDE is `w_pde = 0`. It is not an operational wildfire forecasting product. Test IoU is about 0.13. Do not deploy it as a perimeter forecast.

Calling this checkpoint a PINN is misleading. The useful description is: a level-set neural baseline developed inside the WildfirePINN project.

## What a teammate needs

Copy the repository **and** the local data and checkpoint. A git clone by itself is not enough. `data/`, `outputs/`, and `*.pt` are gitignored.

| File | Role |
| --- | --- |
| `wildfire_pinn/` | Model, normalization, and data loaders |
| `infer_wildfirepinn_v1.py` | Smoke test and pixel confusion counts |
| `requirements.txt` | Install list |
| `data/kincade_stack.nc` | Grid and weather stack |
| `data/kincade_R.nc` | Spread-rate field stored with the case |
| `data/kincade_perimeters.geojson` | Twenty overpass times and the grid extent |
| `data/kincade_firms.csv` | VIIRS detections used to rebuild the persistent burned region |
| `outputs/kincade_final_candidates/observation/levelset_pinn.pt` | Frozen weights. SHA-256 `93de4791f6017664174b6dec0bc9434bd2bbe5fffd52dc63d795168f4890e996` |

Do not copy the large rasters into `outputs/wildfirepinn_v1/handoff/`. That folder holds a checkpoint copy, `config.json`, the model card, the frozen metrics, and the confusion table. The inference script still loads the canonical `.pt` path above.

`firms_key.txt` is not required. Inference does not call the FIRMS API.

Optional context, not required to run:

- `MODEL_CARD.md` — architecture, loss, and primary scores
- `PHYSICS_ABLATION.md` — why the weather PDE was not kept
- `outputs/wildfirepinn_v1/final_metrics.json` — IoU, Dice, symmetric distance, area error
- `run_wildfirepinn_v1.py` — full 20-pass maps, also without training

## Install

From the repository root, with Python 3.12:

```powershell
pip install -r requirements.txt
```

The evaluation machine used PyTorch 2.12, NumPy, pandas, matplotlib, xarray, netCDF4, Shapely, GeoPandas, and SciPy. CPU is enough.

## Load the checkpoint

The network is built from the architecture stored in the file, then the weights are loaded. Hidden width is 128 and there are 4 hidden tanh layers (50,177 parameters). Inputs are only normalized time and position. Weather is not an input.

Normalization, matching `predict_phi_meters`:

- `tau = (t - t0) / T`, with `tau = 0` at 2019-10-24 09:05 and `tau = 1` at 2019-10-30 10:33
- `X = (x - xc) / L`, `Y = (y - yc) / L`
- the network outputs `Phi`, and `phi = Phi * L` in meters

`phi < 0` is predicted burned, `phi = 0` is the predicted perimeter, and `phi > 0` is outside the fire. The training target was the persistent burned region: the cumulative union of accepted VIIRS pixel disks. Score that mask (`contains_xy` of the persistent polygon). Do not score the hotspot convex hulls.

The saved config has `w_pde = 0` and `observation_target = persistent_burn`. The class default `w_pde = 1` in `RealConfig` is not the weight this checkpoint was trained with.

## Run inference

From the repository root:

```powershell
python infer_wildfirepinn_v1.py
```

The script checks the SHA-256, runs pass 0 (2019-10-24 09:05), then counts true positives, false positives, true negatives, and false negatives on every pass. It writes:

- `outputs/wildfirepinn_v1/confusion_matrix.csv`
- `outputs/wildfirepinn_v1/confusion_matrix_per_pass.csv`
- `outputs/wildfirepinn_v1/handoff/`

It does not modify the canonical checkpoint.

A private Cloud Run API for this checkpoint is specified in `DEPLOYMENT.md`. That service is not deployed. Prediction stays behind Google Cloud IAM.

## Metrics

Primary scores stay IoU, Dice, symmetric boundary distance, and area error. Those live in `outputs/wildfirepinn_v1/final_metrics.json`.

| Split | IoU | Dice | Symmetric distance (m) | Area error (km²) |
| --- | ---: | ---: | ---: | ---: |
| Train (passes 0–11, not used to select) | 0.686 | 0.812 | 805 | 9.5 |
| Validation (passes 12–15, selection split) | 0.404 | 0.564 | 1962 | 67.0 |
| Test (passes 16–19, untouched) | 0.129 | 0.225 | 3919 | 38.0 |

The confusion table adds precision, recall (sensitivity), specificity, and accuracy. **Accuracy is not a primary metric.** Most of the 90 m grid is unburned, so true negatives make accuracy look strong even when the fire footprint is wrong. On the untouched test the burned region is a small fraction of the domain and IoU is 0.129.

Binary F1 from TP, FP, and FN is `2 TP / (2 TP + FP + FN)`. For one pass that is the same quantity as Dice. The script checks each pass against the Dice already stored in `per_pass_metrics.csv`. The mean of those per-pass values is the Dice in the table above. A single F1 computed from counts pooled across passes is a different average and is labeled separately in the CSV.

These counts were not used to retune a threshold, a weight, or the checkpoint.

## Limitation

The frozen model memorizes the observed persistent masks on the training passes. It does not enforce `phi_t + R |grad phi| = 0`. Weather-derived spread rates disagreed with those masks, and an observed-speed experiment that improved validation did not improve the untouched test. Later passes drift south of the observed Kincade perimeter. Treat v1 as a documented baseline for the master's project, not as a forecast to hand to an incident team.
