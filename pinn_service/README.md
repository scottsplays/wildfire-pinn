# WildfirePINN inference service (`model_type: "pinn"`, port 8082)

This is the PINN's black-box service for the Fight Fire With AI backend, built to the sponsor's integration guide (rev 2, Oct 16 2026). The ML adapter on :8001 routes `model_type="pinn"` here. The service fetches its own inputs (native ingestion) and returns one GeoJSON FeatureCollection of ~1 km cells with `fire_probability` for horizons 0, 1, 3 and 6 h.

> **Research prototype, not an operational forecast.** Version `0.3.1-calibrated` is a physics-informed, data-calibrated level-set model: the fire's own growth between two VIIRS overpasses calibrates the spread rate, a learned model adjusts it, and quiet edges are held. On 19 held-out *test* fires (scored once) it more than doubled the 24 h new-area IoU of the v0.2 physics-only rate (0.159 vs 0.072) and predicted growth amounts close to what was observed (0.84×, against 16.5× for physics). It did not beat simple geometric baselines (see "Validation"). The v0.2 method is still available with `additional_params.method = "physics"`. Every response states the method in `metadata.forecast_method`.

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
4. **Calibrated forecast (default, `method="calibrated"`).** Detections from the last `history_hours` (96) are grouped into VIIRS overpasses. Each overpass gives a cumulative burned mask (212 m pixel disks, one-pixel gaps closed, holes filled) on a 180 m grid. The latest overpass t_b is the forecast start. The latest earlier overpass, at least 3 h before and with less burned area, gives the recent growth. Edges that grew in that interval are *active*. Cells nearer to a quiet edge than to an active one (within 3 km) cannot burn. The spread-rate multiplier α that reproduces the observed growth (area-matched) goes, with wind, moisture, fuel mix, slope, size, growth rate and time of day, into a ridge regression (`static/learned_alpha.json`). That gives α for the next hours. Ten members then advance `phi_t + α·R·(1−e)/(1−e n·w)|∇phi| = 0` from t_b, hour by hour. R uses that hour's wind and Simard equilibrium moisture, and e is the Anderson (1983) wind ellipse. Members scale α by exp(N(0, σ)), where σ is the model's out-of-fire error (≈1.1 in log α), and they turn the wind. With only one overpass (a new fire) or no growth, α falls back to the typical value 0.26 and every edge is active. `metadata.fires[*]` reports `calibration_mode` (`learned`, `last-interval` or `fallback`), α, the calibration interval and the observed growth rate. Burned cells older than `lookback_hours` are not reported.
5. **Nowcast (`method="nowcast"`, v0.4).** This uses the same burned masks, overpasses and active fronts as step 4, but no spread model. The fire keeps growing at the rate seen between its last two overpasses (km²/h × hours since t_b), outward from its active edges, nearest cells first. Ten members scale that amount by exp(σ z), with σ ≈ 1.4 in log amount, measured on the validation fires. A cell's share of members is then mapped through a calibration curve fitted on the validation fires (`static/nowcast.json`), so new-growth probabilities top out near 0.4. That is honest at 12–24 h: even a perfect amount forecast places only about a quarter of the new burn correctly. A fire seen by only one overpass falls back to the calibrated forecast. On the validation fires this rule matched the best physics-informed variants (24 h new-area IoU 0.196). On the held-out test fires it scored 0.189 against v0.3's 0.159 (not significant at 24 h, significant at 12 h).
6. **Physics-only forecast (`method="physics"`, v0.2).** Each fire gets a 90 m grid (fire + 6 km margin, widened automatically if the front reaches the edge). Ten members advance `phi_t + R(1 + 0.6 n·w)₊|∇phi| = 0` hour by hour with that hour's wind. Member 0 is unperturbed; the others scale R and rotate the wind randomly. Seeds are fixed per reference time, so repeat calls give identical answers.
7. **Output.** A cell's `fire_probability` is the share of members that burn any part of it; horizon 0 is the detected fire (probability = detection confidence). Polygons are clamped to the request bbox, and `metadata.cells_outside_bbox_dropped` counts what was cut. Per contract rev 2, `members` is 1. The internal ensemble is reported as `ensemble_votes` / `ensemble_size`. `risk_label` is `ACTIVE_FIRE` at horizon 0, otherwise `HIGH_RISK` (≥ 0.6), `MODERATE_RISK` (≥ 0.3) or `LOW_RISK`.

## `additional_params`

| Key | Default | Meaning |
| --- | --- | --- |
| `confidence_threshold` | 0.0 | Leave out cells below this probability (cells at 0 are never sent) |
| `hour` | from `date` | UTC hour of the reference time |
| `method` | `calibrated` | `calibrated` (v0.3 physics-informed ML), `nowcast` (v0.4 active-edge persistence, calibrated probabilities) or `physics` (v0.2 uncalibrated baseline) |
| `lookback_hours` | 24 | FIRMS detections this far back form the current fire |
| `history_hours` | 96 | Calibrated: detections this far back give the fire's recent growth (FIRMS API limit 5 days) |
| `search_buffer_m` | 5000 | Look for the fire this far outside the bbox |
| `min_burnable_share` | 0.25 | Drop detection groups with less burnable fuel than this within 500 m (0 = keep all) |
| `members` | 10 | Internal ensemble size, 1–32 |
| `resolution_m` / `margin_m` | 180 calibrated, 90 physics / 6000 | Forecast grid spacing / initial window margin |
| `rate_log_sigma` / `wind_dir_sigma_deg` | model RMSE (calibrated), 0.5 (physics) / 20 | Ensemble spread of the rate (log scale) and of wind direction |
| `high_risk` / `moderate_risk` | 0.6 / 0.3 | `risk_label` thresholds |
| `seed` | from reference time | Fix for reproducible experiments |

## Environment

| Variable | Default | Use |
| --- | --- | --- |
| `FIRMS_MAP_KEY` | `firms_key.txt` | NASA FIRMS map key (secret) |
| `INGESTION` | `native` | `npz` = offline replay from `$FEATURE_DIR/YYYY-MM-DD.npz` |
| `STATIC_LAYERS` | `pinn_service/static/bay_area_static.npz` | Fuel/slope file from `build_static_layers.py` |
| `LEARNED_ALPHA` | `pinn_service/static/learned_alpha.json` | Learned α model written by `pinn_v2/realfire_final.py`; if it is missing the service uses the last-interval α and says so |
| `GRID_SPEC` | unset → provisional CA lattice | Shared lattice: CSV `grid_id,min_lon,min_lat,max_lon,max_lat`, or JSON |
| `AIP_HTTP_PORT`, `AIP_HEALTH_ROUTE`, `AIP_PREDICT_ROUTE` | 8082, `/health`, `/predict` | Set by Vertex AI |

## Known gaps (tracked in the plan doc)

- **Lattice.** Cells are 0.01° × 0.01° on round 0.01° lines, the same cell geometry as the adapter's U-Net fixtures, so the two models' cells coincide. The grid_id numbering (row-major from (−124.5, 32.5), starting at 1) is provisional until the U-Net's numbering is shared; switch with `GRID_SPEC`.
- **Clipping.** Clamping to a 1 km request bbox throws away most of a 6 h spread forecast (see `cells_outside_bbox_dropped`). This needs a decision with the backend.
- **Weather.** Weather is taken at one point per fire, which is fine for windows of tens of km. Open-Meteo is a free public API; for production, check its terms and rate limits or move to NOAA HRRR.
- **Static coverage.** The static layers cover the Bay Area (+ ~25 km). Fires elsewhere get no fuel data and a warning. Rebuild with a larger `--bbox` for wider coverage.
- **Skill.** On test fires v0.3 beats v0.2 but not a simple active-front buffer (see "Validation").
- **Validation.** Leads of 12 h and 24 h were tested, on VIIRS-derived burned areas of real fires with reanalysis weather. Live forecasts use forecast weather, which is less accurate. The 1, 3 and 6 h horizons use the same model but were not scored separately. Probabilities are ensemble shares, not calibrated likelihoods.

## Validation (summary)

The data are real California fires from the CAL FIRE FRAP catalogue (2012–2025). Burned areas come from VIIRS overpasses, and the train/validation/test split was made by fire event. The method was chosen on the validation fires. It was then frozen and scored once on the test fires (`pinn_v2/realfire_final.py --final`, 18 Oct 2026). The score is new-area IoU, the overlap of predicted and observed *new* burned area, averaged over fires.

| Method | Validation 24 h (26 fires) | **Test 24 h (19 fires)** | Test 12 h | Test growth ratio, 24 h |
| --- | --- | --- | --- | --- |
| v0.2 physics only | 0.074 | **0.072** | 0.074 | 16.5× |
| uniform buffer at last growth rate | 0.152 | **0.146** | 0.130 | 2.3× |
| buffer restricted to active fronts | — | **0.189** | 0.169 | 1.7× |
| **v0.3 calibrated (this service)** | 0.205 | **0.159** | 0.129 | 0.84× |

Test differences, paired over fires, with 95 % bootstrap CI:
- v0.3 vs physics: +0.088 [+0.067, +0.107] at 24 h (better on 95 % of fires) and +0.056 [+0.030, +0.084] at 12 h.
- v0.3 vs the uniform buffer: +0.017 [−0.006, +0.042] at 24 h. This is not significant.
- v0.3 vs the active-front buffer: −0.023 [−0.046, +0.002] at 24 h and −0.036 [−0.063, −0.009] at 12 h.

What the test shows:
- **Calibration is what helps.** Fitting the rate to the fire's own recent growth, and holding quiet edges, fixes the physics model's 16× over-prediction.
- **Physics did not place the new growth better than geometry.** Given the true amount of growth, the wind-ellipse and fuel shape scored lower than uniform growth on the active fronts (0.210 vs 0.230 at 24 h, CI below zero). At 12–24 h and VIIRS resolution, with weather taken at a single point, the physics shape adds no skill over that geometric baseline.
- **The validation result shrank on the test set** (0.205 → 0.159). The test fires are mostly the August 2020 lightning siege, with many fires burning at once.

### Fresh test: 2026 fires (scored once, 19 Oct 2026)

The 2026 fires are 10 California wildfires from this season (January to October), found directly in VIIRS data; farmland and industrial heat are excluded. No method was developed on them. Results at 24 h:

| Method | New-area IoU | Total IoU | Growth ratio |
| --- | --- | --- | --- |
| v0.2 physics only | 0.049 | 0.227 | 27× |
| uniform buffer | 0.160 | 0.669 | 1.6× |
| v0.3, 30-fire model | 0.158 | 0.654 | 1.9× |
| **v0.3.1, 218-fire model (deployed)** | **0.164** | **0.719** | **0.82×** |
| v0.4 nowcast (`method="nowcast"`) | 0.171 | 0.704 | 1.14× |

- v0.3 beats v0.2 on 90 % of fires: +0.109 [+0.061, +0.158].
- v0.4 and v0.3 are statistically tied: +0.013 [−0.033, +0.059].
- Following the rule set before the test, v0.4 would become the default only if it clearly won, so v0.3 stays the default.
- The 218-fire model was at least as good as the 30-fire one and much better on growth amount, so it replaced it as v0.3.1. The 30-fire model is kept as `static/learned_alpha_v030_30fires.json`.
- The nowcast's calibrated probabilities carried over to the new fires: Brier skill +0.086 over the base rate, against −0.49 without calibration.

**When physics helps** (validation fires, split by weather change with thresholds fixed in advance, `pinn_v2/physics_when.py`):
- When wind or moisture changed between the observed interval and the forecast interval, the physics-informed amount beat persistence at 24 h: +0.022 [+0.002, +0.041] for the physics-calibrated amount placed on active edges, and +0.015 for the full level set.
- In steady weather persistence was slightly better: −0.006.
- So physics earns its place where the fire's recent past says nothing about the new conditions.

v0.3 is deployed because it replaces the clearly worse v0.2 rate. It was the method frozen before the test, and the test set is not used to pick a different one. A switch to the active-front buffer, or a hybrid of the two, would need a new held-out evaluation, for example fires from the 2026 season.
