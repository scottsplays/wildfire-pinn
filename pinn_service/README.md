# WildfirePINN inference service (`model_type: "pinn"`, port 8082)

This is the PINN's black-box service for the Fight Fire With AI backend, built to the sponsor's integration guide (rev 2, Oct 16 2026). The ML adapter on :8001 routes `model_type="pinn"` here. The service fetches its own inputs (native ingestion) and returns one GeoJSON FeatureCollection of ~1 km cells with `fire_probability` for horizons 0, 1, 3 and 6 h.

> **Research prototype, not an operational forecast.** Version `0.2.2-physics-baseline` uses the physics-only spread rate. The learned PINN correction (plan weeks 3–5) replaces the rate later without changing this contract. Every response says so in `metadata.forecast_method`.

## Run it locally (Windows PowerShell shown; bash is the same)

```powershell
pip install -r pinn_service/requirements.txt rasterio
python -m pinn_service.build_static_layers            # once: LANDFIRE fuel + slope -> pinn_service/static/bay_area_static.npz
$env:FIRMS_MAP_KEY = (Get-Content firms_key.txt)      # or leave firms_key.txt in the repo root
python -m pinn_service.native_ingestion --smoke       # checks FIRMS, Open-Meteo, gridMET and the static file
python pinn_service/serve.py                          # http://localhost:8082
```

```powershell
Invoke-RestMethod http://localhost:8082/health
curl.exe -s -X POST http://localhost:8082/predict -H "Content-Type: application/json" `
  -d '{\"bbox\":[-122.9,36.9,-121.5,38.3],\"time_horizon\":6,\"model_type\":\"pinn\",\"date\":\"2026-10-16\"}' -o response.json
python pinn_service/validate_contract.py response.json
```

(Windows PowerShell 5.1 needs the `http://` prefix and strips unescaped `"` from arguments to `curl.exe`.)

There is no network or key in replay mode: `python pinn_service/make_test_features.py`, then `$env:INGESTION="npz"` and run with `"date":"2026-10-16"`. That date is a synthetic fire in the Santa Cruz Mountains.

Tests (repo root, no network needed): `python -m unittest pinn_service.tests.test_service -v`
Check any response file against the contract: `python pinn_service/validate_contract.py response.json`
Docker: see the header of `Dockerfile` (pass the FIRMS key with `-e`, never bake it in).

## What happens on a request

1. **Validate.** The bbox is lon/lat, `time_horizon` is 0–6 and `model_type` must be `pinn`. `date` is either a date or an ISO timestamp; a bare date means **21:00 UTC** (sponsor FAQ Q2), and `additional_params.hour` overrides the hour. A missing `date` means now, and `time_horizon` above 6 (the team adapter defaults to 24) is served as 0/1/3/6. Vertex `{"instances": [...]}` bodies get `{"predictions": [...]}` back.
2. **Find the fire.** FIRMS VIIRS detections (S-NPP, NOAA-20, NOAA-21; nominal and high confidence) from the last 24 h, searched in the bbox **plus a 5 km buffer**. The backend may send a single 1 km cell, and a spread forecast needs the whole fire around it. Detections within 2 km form one fire. Groups with less than 25 % burnable LANDFIRE fuel within 500 m are dropped and listed in `metadata.detections_rejected`: refinery flares (Richmond, Martinez), power plants and crop burns show up in FIRMS every day. No wildland fire means an empty FeatureCollection with HTTP 200.
3. **Inputs per fire.** LANDFIRE fuel (FBFM40) and slope come from the bundled static file. Hourly 10 m wind, RH and temperature come from Open-Meteo at the fire centre. 100-h fuel moisture comes from gridMET; if gridMET is down, it falls back to an estimate from the last 24 h of RH/T. The spread rate R per hour uses the same formula as `wildfire_pinn/data_pipeline.py`.
4. **Ensemble forecast.** Each fire gets a 90 m grid (fire + 6 km margin, widened automatically if the front reaches the edge). Ten members advance `phi_t + R(1 + 0.6 n·w)₊|∇phi| = 0` hour by hour with that hour's wind. Member 0 is unperturbed; the others scale R and rotate the wind randomly. Seeds are fixed per reference time, so repeat calls give identical answers.
5. **Output.** A cell's `fire_probability` is the share of members that burn any part of it; horizon 0 is the detected fire (probability = detection confidence). Polygons are clamped to the request bbox, and `metadata.cells_outside_bbox_dropped` counts what was cut. Per contract rev 2, `members` is 1. The internal ensemble is reported as `ensemble_votes` / `ensemble_size`. `risk_label` is `ACTIVE_FIRE` at horizon 0, otherwise `HIGH_RISK` (≥ 0.6), `MODERATE_RISK` (≥ 0.3) or `LOW_RISK`.

## `additional_params`

| Key | Default | Meaning |
| --- | --- | --- |
| `confidence_threshold` | 0.0 | Leave out cells below this probability (cells at 0 are never sent) |
| `hour` | from `date` | UTC hour of the reference time |
| `lookback_hours` | 24 | FIRMS detections this far back form the current fire |
| `search_buffer_m` | 5000 | Look for the fire this far outside the bbox |
| `min_burnable_share` | 0.25 | Drop detection groups with less burnable fuel than this within 500 m (0 = keep all) |
| `members` | 10 | Internal ensemble size, 1–32 |
| `resolution_m` / `margin_m` | 90 / 6000 | Forecast grid spacing / initial window margin |
| `rate_log_sigma` / `wind_dir_sigma_deg` | 0.5 / 20 | Ensemble spread of R (log scale) and of wind direction |
| `high_risk` / `moderate_risk` | 0.6 / 0.3 | `risk_label` thresholds |
| `seed` | from reference time | Fix for reproducible experiments |

## Environment

| Variable | Default | Use |
| --- | --- | --- |
| `FIRMS_MAP_KEY` | `firms_key.txt` | NASA FIRMS map key (secret) |
| `INGESTION` | `native` | `npz` = offline replay from `$FEATURE_DIR/YYYY-MM-DD.npz` |
| `STATIC_LAYERS` | `pinn_service/static/bay_area_static.npz` | Fuel/slope file from `build_static_layers.py` |
| `GRID_SPEC` | unset → provisional CA lattice | Shared lattice: CSV `grid_id,min_lon,min_lat,max_lon,max_lat`, or JSON |
| `AIP_HTTP_PORT`, `AIP_HEALTH_ROUTE`, `AIP_PREDICT_ROUTE` | 8082, `/health`, `/predict` | Set by Vertex AI |

## Known gaps (tracked in the plan doc)

- **Lattice.** Cells are 0.01° × 0.01° on round 0.01° lines, the same cell geometry as the adapter's U-Net fixtures, so the two models' cells coincide. The grid_id numbering (row-major from (−124.5, 32.5), starting at 1) is provisional until the U-Net's numbering is shared; switch with `GRID_SPEC`.
- **Clipping.** Clamping to a 1 km request bbox throws away most of a 6 h spread forecast (see `cells_outside_bbox_dropped`). This needs a decision with the backend.
- **Weather.** Weather is taken at one point per fire, which is fine for windows of tens of km. Open-Meteo is a free public API; for production, check its terms and rate limits or move to NOAA HRRR.
- **Static coverage.** The static layers cover the Bay Area (+ ~25 km). Fires elsewhere get no fuel data and a warning. Rebuild with a larger `--bbox` for wider coverage.
- **Validation.** The physics-only rate is uncalibrated and not yet evaluated on real fires. Treat probabilities as a spread envelope, not a calibrated likelihood.
