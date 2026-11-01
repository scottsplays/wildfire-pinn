# Forecast scorecard: how accurate were the PINN forecasts?

The service can check its own forecasts against what the VIIRS satellites saw afterwards. Give it a
past forecast time and a fire's bbox; it re-runs the forecast exactly as it would have been issued
then (only satellite data from before that moment), finds the satellite passes that came later, and
scores predicted vs. observed **new** burned area on the same 1 km grid the dashboard draws.

The service stays stateless: the backend decides which forecasts to check and stores the results.

## Request

Local / Cloud Run: `POST /verify`. On Vertex AI (which only routes `/predict`): `POST /predict` with
the same object plus `"task": "verify"`, inside `{"instances": [...]}` as usual.

```json
{
  "bbox": [-121.8, 36.2, -121.6, 36.3],
  "date": "2026-08-10T12:00:00Z",
  "leads_hours": [12, 24],
  "method": "calibrated"
}
```

| field | default | meaning |
|---|---|---|
| `bbox` | required | the fire's area, `[min_lon, min_lat, max_lon, max_lat]` (as for `/predict`) |
| `date` | required | when the forecast was issued, ISO-8601 UTC |
| `leads_hours` | `[12, 24]` | lead times to check (1-48 h) |
| `method` | `calibrated` | `calibrated` (v0.3.1, default model) or `nowcast` (v0.4) |
| `threshold` | model's best guess | probability at which a cell counts as "predicted to burn": 0.5 for calibrated, about 0.20 for nowcast (its probabilities are calibrated, so they rarely pass 0.4) |
| `tolerance_hours` | `3` | a satellite pass up to this long after `date + lead` still verifies the lead |
| `include_cells` | `false` | also return `cells`: the `grid_id`s of hit, false-alarm and missed cells and of the fire at issue time, for drawing the check on the map |
| `additional_params` | `{}` | passed to the forecast, as in `/predict` |

`POST /verify` also accepts a list of up to 50 such objects and then returns
`{"results": [...], "summary": [...]}` with averages per method and lead.

## Response (one lead shown)

```json
{
  "type": "pinn_scorecard",
  "model_version": "0.3.1-calibrated",
  "method": "calibrated",
  "issued_at": "2026-08-10T12:00:00Z",
  "threshold": 0.5,
  "fires_detected": 1,
  "results": [
    {
      "lead_hours": 24,
      "status": "scored",
      "valid_at": "2026-08-11T10:13:00Z",
      "forecast_horizon_hours": 22,
      "burned_km2_at_issue": 31.4,
      "model": {
        "new_area_iou": 0.21, "f1": 0.347, "precision": 0.31, "recall": 0.40, "growth_ratio": 1.29,
        "predicted_new_km2": 9.1, "observed_new_km2": 7.1, "hit_km2": 2.8,
        "false_alarm_km2": 6.3, "missed_km2": 4.3, "brier": 0.021, "base_rate": 0.018
      },
      "baseline_dilation": {"new_area_iou": 0.18, "f1": 0.305, "...": "..."}
    }
  ]
}
```

* **new_area_iou**: overlap of predicted and observed new burned area (the report's main metric);
  **f1, precision, recall**: the same counts in the U-Net's terms (F1 = 2 IoU / (1 + IoU));
  **growth_ratio**: predicted ÷ observed new area (1 = right amount).
* **baseline_dilation**: the current fire grown by one 1 km cell, scored the same way: the simple
  rule a useful model should beat (the U-Net chapter uses the same "one-pixel dilation" baseline).
* **status**: `scored`; `no_growth` (the fire did not grow, so IoU is undefined; `false_alarm_km2`
  still says how much growth was wrongly predicted); `pending` (the verifying pass has not happened
  yet: ask again later); `no_observation` (no VIIRS pass in the window, e.g. cloud or smoke);
  `no_fire` (no wildland fire at `date`).

A verify call re-runs one forecast, so it takes about as long as `/predict` (2-15 s). FIRMS serves
5 days of detections per request, so check a forecast within about 3 days of its lead time.

## Wiring it into the backend (suggested)

1. **Celery Beat task, twice a day** (e.g. 01:15 and 13:15 UTC, after the night and afternoon
   VIIRS passes). For each active incident, take the PINN forecast issued about 24-27 h earlier
   (from the prediction archive), and call verify with the incident's bbox, that issue time and
   `leads_hours: [12, 24]`. Use the same Vertex endpoint as `/predict`, with `"task": "verify"`.
2. **Store** each result (one row per incident × issue time × lead: status, IoU, F1, precision,
   recall, growth ratio, baseline IoU, `model_version`) in a new table or a JSON column. Skip
   `pending` results and retry them on the next run.
3. **Dashboard "Model accuracy" panel**: a table of recent scored forecasts, and the running means
   per model version and lead (only `scored` rows), shown next to the dilation baseline, e.g.
   "PINN 24 h: new-area IoU 0.17 (baseline 0.15), F1 0.29, 23 forecasts on 4 fires".

Off-season there may be no fires to check. The replay results below give the panel a starting point.

## Replay mode: score the model on past fires (offline)

Runs the deployed service code on saved fire folders (`data/fires/<id>/`: detections, weather, fuel)
as if each fire were happening live, issuing a forecast every day at 00:00 and 12:00 UTC:

```powershell
python -m pinn_service.scorecard --replay data/fires --fires 20260804_c38p11n120p73w 20260809_c36p24n121p71w
python -m pinn_service.scorecard --replay data/fires --catalogue data/fire_catalogue_2026.csv
```

Writes `outputs/scorecard/replay_forecasts.jsonl` (one verify result per line; an interrupted run
resumes where it stopped) and `replay_summary.txt` / `.json` (each fire counts once, with 95%
bootstrap intervals over fires).

Replay of the deployed service on the ten 2026 test fires (1 Nov 2026; 452 forecast runs):

| model | lead | fires | forecasts | new-area IoU (95% CI) | F1 | precision | recall | growth | dilation baseline IoU |
|---|---|---|---|---|---|---|---|---|---|
| v0.3.1 calibrated | 12 h | 10 | 128 | 0.154 [0.112, 0.196] | 0.238 | 0.297 | 0.366 | 1.33 | 0.131 |
| v0.3.1 calibrated | 24 h | 10 | 155 | 0.162 [0.122, 0.205] | 0.254 | 0.290 | 0.397 | 1.60 | 0.157 |
| v0.4 nowcast | 12 h | 10 | 128 | 0.164 [0.120, 0.209] | 0.247 | 0.308 | 0.349 | 1.00 | 0.131 |
| v0.4 nowcast | 24 h | 10 | 155 | 0.172 [0.129, 0.212] | 0.268 | 0.299 | 0.399 | 1.00 | 0.157 |

Both models tie the one-cell dilation baseline on IoU (no difference is significant), but they are
more precise (0.29-0.31 vs 0.15-0.20): the baseline catches more new fire by predicting about five
times too much area. The same ranking as the report's 180 m test (v0.3.1 0.164, v0.4 0.171).

These numbers are on the 1 km dashboard grid, so they differ from the report's 180 m test numbers:
1 km cells make both the forecast and the observation blockier, and the scorecard issues forecasts
on a fixed clock instead of right after each pass.
