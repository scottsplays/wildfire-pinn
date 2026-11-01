# ml_pinn_service: PINN fire-spread service (`model_type: "pinn"`, port 8082)

This is the PINN's inference service for the ML adapter. It follows the sponsor's local-integration guide (rev 2) and `docs/vertex-ai-endpoint-contract.md`. The Vertex AI endpoint name is `pinn-fire-spread-v1` (`docs/ml-team-interface-contract.md`).

The folder is self-contained. Nothing outside it was changed.

> **Status: `0.3.1-calibrated`, research prototype (physics-informed machine learning).** A supervised model (ridge regression trained on real California fires, `pinn_service/static/learned_alpha.json`) (trained on 218 fires) predicts the spread rate from the fire's growth between the last two VIIRS satellite passes, wind, moisture, fuel, slope, size and time of day. A level-set fire-spread model with an Anderson wind ellipse turns that rate into the forecast front, as a 10-member ensemble. Quiet edges are held. On 19 held-out real fires it doubled the 24 h new-area IoU of the v0.2 physics-only model (0.159 vs 0.072) and fixed its 16x over-prediction of growth. The v0.2 model is still available with `additional_params.method = "physics"`. Details and limits: `pinn_service/README.md`. Not an operational forecast.
> Owner: Scott Kennedy. Research code and history: https://github.com/scottsplays/wildfire-pinn

## Run it

Run every command from inside this folder:

```powershell
cd ml_pinn_service
pip install -r requirements.txt
# your own free NASA FIRMS map key: https://firms.modaps.eosdis.nasa.gov/api/map_key/
$env:FIRMS_MAP_KEY = "<your key>"            # or put it in ml_pinn_service/firms_key.txt (gitignored)
python -m pinn_service.native_ingestion --smoke   # optional: checks FIRMS, Open-Meteo, gridMET, fuel layers
python serve.py --port 8082
```

```powershell
Invoke-RestMethod http://localhost:8082/health
curl.exe -s -X POST http://localhost:8082/predict -H "Content-Type: application/json" `
  -d '{\"bbox\":[-122.9,36.9,-121.5,38.3],\"time_horizon\":6,\"model_type\":\"pinn\",\"date\":\"2026-10-16\"}' -o response.json
python -m pinn_service.validate_contract response.json
```

Accuracy checks: `POST /verify` (or `/predict` with `"task": "verify"` on Vertex AI) scores a past forecast against the satellite passes that came after it (new-area IoU, F1, precision, recall). Request format and suggested backend wiring: `pinn_service/SCORECARD.md`.

Without a key or network, use replay mode, which serves a synthetic Santa Cruz Mountains fire: `python -m pinn_service.make_test_features`, then `$env:INGESTION="npz"`, `python serve.py`, and request `"date":"2026-10-16"`.

Tests (no network needed): `python -m unittest pinn_service.tests.test_service`. Docker: see the header of `Dockerfile`.

## For the ML adapter

- `POST http://localhost:8082/predict` with the adapter's `PredictionRequest` as-is. Extra fields such as `mock` are ignored, `time_horizon` 24 is served as horizons 0/1/3/6, and `date: null` means now. A Vertex `{"instances": [...]}` body is also accepted and returns `{"predictions": [...]}`.
- The response is one FeatureCollection with every horizon. Each cell is a closed 0.01° × 0.01° Polygon in `[lon, lat]`, on the same cell edges as `ml_adapter/fixtures/frontend_team_geojson_files`. Each cell carries:
  - `grid_id`, `fire_probability` (0–1), `horizon_hours`, `source_model: "pinn"`, `members: 1`;
  - `risk_label`: `ACTIVE_FIRE` at 0 h, otherwise `HIGH_RISK`, `MODERATE_RISK` or `LOW_RISK`;
  - `ensemble_votes` / `ensemble_size` from the internal ensemble.
- A day with no wildland fire returns HTTP 200 with an empty `features` list. Errors return `{"detail": ...}` with 4xx/5xx.
- `metadata.detections_rejected` lists FIRMS hot spots that were dropped as non-wildland: refinery flares in Richmond and Martinez and crop burns appear every day. The scout may want the same filter.
- `python -m pinn_service.validate_contract <file>` checks any model's response against the contract rules, so it works on U-Net output too.

## Open questions for the team

1. **Clipping.** Output is clamped to the request bbox per the contract, but a 6 h forecast spreads km beyond a 1 km scout box. `metadata.cells_outside_bbox_dropped` counts what was cut. Can the scout send fire + ~10 km, or accept cells outside the box?
2. **grid_id numbering.** Cell geometry already matches the adapter fixtures. The numbering (row-major from (−124.5, 32.5), starting at 1) is provisional. What numbering does the U-Net use?
3. **risk_label vocabulary.** What does the U-Net use, so the labels match?
4. **Scout coverage.** Is it the Bay Area only, or all of California? The bundled fuel/slope layers cover the Bay Area + ~25 km.

More detail: `pinn_service/README.md` (behaviour, parameters, environment variables) and `pinn_service/feature_spec.md` (data sources).
