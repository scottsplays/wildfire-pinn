# Wildfire PINN

Level-set experiments for wildfire spread. Master's project, San José State University (MSAI, Dec 2026).

Part of a team master's project. This repository is the physics core. The rest of the forecasting system (perception, decisions, app) lives with the team.

The frozen Kincade deliverable is WildfirePINN v1, in the next section. `MODEL_CARD.md` is the authority for that checkpoint. Later sections describe the Palisades package trainer and the earlier PDE experiments.

## WildfirePINN v1

The frozen Kincade model is an observation-only level-set network. It reads normalized easting, northing, and time, and it outputs a signed distance φ. φ < 0 is the predicted burned region and φ = 0 is the predicted perimeter. Weather and the level-set PDE are not inputs to this checkpoint and are not part of its loss.

The case is the 2019 Kincade Fire. VIIRS active-fire detections become a persistent burned region: the cumulative union of accepted pixel disks on a 90 m California Albers grid. Twenty overpasses stay in time order and are cut 12 train, 4 validation, and 4 test (passes 0–11, 12–15, and 16–19).

v1 is the existing checkpoint `outputs/kincade_final_candidates/observation/levelset_pinn.pt` (SHA-256 `93de4791f6017664174b6dec0bc9434bd2bbe5fffd52dc63d795168f4890e996`). It was selected on validation. Test was not used to choose it. On the persistent targets, mean IoU is 0.686 on train, 0.404 on validation, and 0.129 on the untouched test. A zero contour is present on every pass. By the test window the predicted region has walked south of the observed fire.

The weather-derived propagation constraint was studied as an ablation. It conflicts with a perimeter that is stationary along most of its length, and an observed-speed variant that helped validation did not help the untouched test. v1 keeps the level-set field. The weather-derived PDE weight in the frozen loss is zero.

Reproduce the frozen evaluation, without training:

```powershell
python run_wildfirepinn_v1.py
```

Figures and tables go to `outputs/wildfirepinn_v1/`. The model card is `MODEL_CARD.md`. The ablation write-up is `PHYSICS_ABLATION.md`. Teammate setup and the one-pass inference command are in `HANDOFF.md`. A private HTTPS API for this same checkpoint is described in `DEPLOYMENT.md`. It is not deployed.

## Model

For Kincade, the shipped v1 checkpoint above does not use the PDE. The residual below is the constraint that was implemented and then set aside after the ablation. The network learns the signed-distance field φ(x, y, t) of the fire front. In the physics experiments, the spread-rate field R is an input to the residual, not a second network:

```
φ_t + R |∇φ| = 0
```

For the Palisades case the rate is a first-order proxy, not a full Rothermel model:

```
R = C_REF * r_rel[fuel] * W(wind speed) * S(slope) * M(moisture)
```

Wind direction is applied in the residual as a downwind factor on top of that scalar field. C_REF = 0.177 m/s is the calibration knob.

Training uses dimensionless coordinates Φ = φ / L, X = (x − xc) / L, Y = (y − yc) / L, τ = t / T, which turns the PDE into

```
Φ_τ + (R_eff * T / L) |∇Φ| = 0
```

For the frozen checkpoint, that residual is multiplied by zero. Observation loss fits the persistent burned region on chronological passes 0–11: the initial condition on pass 0, and later observations on passes 1–11. Passes 12–15 are validation. Passes 16–19 are the untouched test. The target on every split is the cumulative union of accepted VIIRS pixel disks. Reported scores are IoU, Dice, symmetric boundary distance, and area error. Loss weights and the exact sampling counts are in `MODEL_CARD.md`.

The Palisades package trainer is a separate path. It uses the first `k_train` overpasses, default 10, and scores later perimeters by the mean distance from the predicted φ = 0 contour to the observed hotspot hull.

## Two experiments

**Synthetic circle.** Constant R, circular ignition, analytical front `radius = r0 + R t`. Loss is the PDE residual, the initial condition, and an eikonal penalty that keeps φ a signed distance. This is the proof that the level-set residual is implemented correctly.

**Palisades Fire, January 2025.** VIIRS detections become one concave hull per satellite overpass. LANDFIRE elevation, slope, and FBFM40 fuel, plus gridMET wind and 100-hour fuel moisture, are reprojected to a 90 m California Albers grid (EPSG:3310). The spread-rate field is written from that stack, then the PINN is trained.

The FIRMS archive request covers 7–31 January 2025. In the Colab session the sensor returned detections only through 12 January (14 overpasses). Empty later windows are skipped rather than concatenated into the table.

## Run

```powershell
pip install -r requirements.txt
python main.py synthetic --epochs 2001
python main.py demo
```

`demo` trains the Palisades loss on a made-up fire so the code path can be checked without a NASA key or any downloads.

The real case needs a [FIRMS map key](https://firms.modaps.eosdis.nasa.gov/api/area/). Paste it on its own line in `firms_key.txt` at the repository root (that file is gitignored), then:

```powershell
python main.py pipeline --epochs 4000
```

`$env:FIRMS_MAP_KEY` and `--firms-key` still override the file.

`pipeline` downloads, builds perimeters, builds the stack, writes R, trains, and writes the held-out front error. A 4,000-epoch run is the notebook setting and is slow on CPU; `--epochs` shortens a trial. Derived files already on disk are reused. `--rebuild` recomputes them, and `--retrain` ignores a saved checkpoint.

Individual stages:

```powershell
python main.py download
python main.py perimeters
python main.py stack
python main.py spread-rate
python main.py train --epochs 4000
python main.py evaluate
python main.py check
```

For that package trainer, figures, checkpoints, and logs go to `outputs/<name>/`. The file it writes is `outputs/<name>/fronts.geojson`: one WGS84 feature per overpass for the observed hotspot hull and for the predicted φ=0 line. A time with no zero crossing is still in the file, with null geometry, and `front_error_m` on the scored piece matches `metrics.json`. Those hull scores are the package-trainer metric. v1 figures and tables are in `outputs/wildfirepinn_v1/`. Downloaded and derived data go to `data/` and are not committed.

## Another fire

Named datasets are `palisades` (the default) and `kincade` (Sonoma County, 23 October–6 November 2019). Palisades trains on the first `k_train` overpasses and holds the rest out. The Kincade dataset record keeps overpasses in time order and cuts them 60% train, 20% validation, and 20% test. On the 20-pass Kincade case that cut is passes 0–11, 12–15, and 16–19, the same indices as v1. A custom window is any other `--name` plus a box.

The command below starts a new package-trainer run. The frozen weights stay at `outputs/kincade_final_candidates/observation/levelset_pinn.pt`. Load those with `python infer_wildfirepinn_v1.py`. The package trainer's default target is the hotspot hull, and its class-default PDE weight is 1. v1's target is the persistent burned region, and the weight saved in the checkpoint is 0. Both facts are in `MODEL_CARD.md`.

```powershell
python main.py --name kincade pipeline --epochs 4000
```

`--crs` is the meter grid. `EPSG:3310` (California Albers) is right for a California fire and wrong outside California; a fire elsewhere needs a regional meter CRS such as `EPSG:5070`. The spread-rate formula is the same proxy in every box. It is not refit per fire, and a fuel code missing from the table stops the run.

## What was broken in the Colab notebook

The notebook is kept as the original session. The package is the version that runs locally. These are the failures it was hitting:

- The synthetic cell constructed `synth_model` and then loaded weights into a different object, `model`. That object was still the 128-unit Palisades network, so loading the 64-unit checkpoint raised a state-dict size error.
- Training, the stack, and the signed-distance cache used `/content/...` paths. After a runtime restart those files were gone and training raised `FileNotFoundError` on `palisades_stack.nc`.
- Cells were out of order: perimeters were read before they were built, and the stack was saved before the reprojection cell.
- Empty FIRMS responses were concatenated with the real detections.
- LANDFIRE nodata (`-32768`) was left in the slope raster, which makes the slope factor explode.
- Interface helpers built CPU tensors and called a model that might be on GPU.
- The training cell used `nn` without importing it, so it only ran if an earlier cell had succeeded in that same runtime.

Checkpoints now store the network width and depth. A file from the other experiment is refused with an instruction to pass `--retrain`, instead of a raw PyTorch size error.

## Layout

- `MODEL_CARD.md` — frozen Kincade v1 checkpoint
- `HANDOFF.md` — files, install, and the inference command
- `infer_wildfirepinn_v1.py` — load the frozen checkpoint; one-pass smoke test, then pixel counts
- `main.py` — command line for the package trainer
- `wildfire_pinn/` — data stages, level-set model, training, figures
- `wildfire_pinn/test_core.py` — physics, spread-rate, and a two-epoch training check
- `WildfirePINN.ipynb` — original Colab session
- `levelset_pinn_poc.py` — earlier TensorFlow circle prototype

## Data note

Satellite rasters, NetCDF grids, and checkpoints are local to `data/` and `outputs/`. They are not stored in git.
