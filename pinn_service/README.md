# WildfirePINN inference service (`model_type: "pinn"`)

This is the ML inference engine for the PINN, built to the sponsor's integration guide (Oct 2026). It runs on port 8080, takes the same requests locally and on Vertex AI, and returns one GeoJSON FeatureCollection of 1 km Bay Area cells with `fire_probability` for horizons 0, 1, 3 and 6 h.

> **Research prototype, not an operational forecast.** Version `0.1.0-physics-baseline` uses the physics-only spread rate. The learned PINN correction (plan weeks 3–5) replaces the rate later without changing this contract. Every response says so in `metadata.forecast_method`.

## Run it locally

```bash
pip install -r pinn_service/requirements.txt
python pinn_service/make_test_features.py          # synthetic fire day 2026-10-16 + quiet day 2026-10-17
python pinn_service/serve.py --port 8080           # or: cd pinn_service && python serve.py --port 8080
```

```bash
curl localhost:8080/health
curl -X POST localhost:8080/predict -H "Content-Type: application/json" \
  -d '{"bbox":[-122.9,36.9,-121.5,38.3],"time_horizon":6,"model_type":"pinn","date":"2026-10-16"}' > response.json
python pinn_service/validate_contract.py response.json
```

PowerShell: `Invoke-RestMethod localhost:8080/predict -Method Post -ContentType "application/json" -Body '{"bbox":[-122.9,36.9,-121.5,38.3],"time_horizon":6,"model_type":"pinn","date":"2026-10-16"}'`

Docker (from the repo root): `docker build -f pinn_service/Dockerfile -t wildfirepinn-inference:0.1.0 .` then `docker run --rm -p 8080:8080 wildfirepinn-inference:0.1.0`.

Tests (from the repo root): `python -m unittest pinn_service.tests.test_service -v`

## What happens on a request

1. The request is validated: bbox in lon/lat, `time_horizon` 0–6, `model_type` must be `pinn`, ISO date. Vertex `{"instances": [...]}` bodies get `{"predictions": [...]}` back.
2. The daily feature file is loaded (`feature_spec.md`). On a day with no `active_fire` pixel in the bbox, the response is an empty FeatureCollection (HTTP 200) with `fires_detected: 0`.
3. Each fire gets its own 90 m grid (fire + 6 km margin, widened automatically if the front reaches the edge). The spread rate R comes from fuel, slope, wind and 100-h moisture, using the same formula as `wildfire_pinn/data_pipeline.py`.
4. An ensemble of N members (default 10) runs the level-set equation `phi_t + R(1 + 0.6 n·w)₊|∇phi| = 0` forward to 1, 3 and 6 h. Member 0 is unperturbed; the others scale R and rotate the wind randomly. The seed is fixed per date and bbox, so repeat calls give identical answers.
5. A 1 km cell's `fire_probability` is the share of members that burn any part of it, and `members` = N. Horizon 0 is the observed fire (probability = detection confidence). Polygons are clamped to the bbox.

## `additional_params`

| Key | Default | Meaning |
| --- | --- | --- |
| `confidence_threshold` | 0.0 | Leave out cells below this probability (cells at 0 are never sent) |
| `members` | 10 | Ensemble size, 1–32 |
| `fire_threshold` | 0.5 | `active_fire` value counted as burning |
| `resolution_m` | 90 | Forecast grid spacing (coarsened automatically for very large fires) |
| `margin_m` | 6000 | Initial window margin around each fire |
| `rate_log_sigma` | 0.5 | Spread of the R multiplier across members (log scale) |
| `wind_dir_sigma_deg` | 20 | Spread of the wind direction across members |
| `seed` | from date + bbox | Fix for reproducible experiments |

## Environment

| Variable | Default | Use |
| --- | --- | --- |
| `FEATURE_SOURCE` | `local` | `gcs` in production |
| `FEATURE_DIR` | `pinn_service/test_features` | Local `.npz` folder |
| `FEATURE_GCS_TEMPLATE` | `gs://fightfirewai-features/daily/{date}.npz` | Production path |
| `STATIC_LAYERS` | unset | `.npz` with fine-resolution `fuel_fbfm40` / `slope_deg` |
| `GRID_SPEC` | unset → provisional grid | Backend `BayAreaGrid` export: CSV `grid_id,min_lon,min_lat,max_lon,max_lat`, or JSON |
| `AIP_HTTP_PORT`, `AIP_HEALTH_ROUTE`, `AIP_PREDICT_ROUTE` | 8080, `/health`, `/predict` | Set by Vertex AI |

## Known gaps (tracked in the plan doc)

- **grid_id values are provisional** (`metadata.grid_ids_provisional: true`) until the backend exports `BayAreaGrid`; then set `GRID_SPEC`.
- The physics-only rate is uncalibrated and has no real-fire evaluation yet. Treat the probabilities as a spread envelope, not a calibrated likelihood.
- The local flat-earth projection is accurate to well under one 90 m cell across a fire window (tens of km). It is not meant for state-wide windows.
- The sponsor's example feature channels have no fuel or moisture. Without them the service returns 422 and names the missing channels.
