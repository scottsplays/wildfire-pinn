# PINN inputs: sources and file formats

Since sponsor guide rev 2 the backend sends only `bbox`, `date` and `time_horizon`, and the service fetches everything else itself (`native_ingestion.py`). This file records where each input comes from, and the two `.npz` formats the service still reads.

## Live sources (INGESTION=native)

| Input | Source | Resolution / cadence | Notes |
| --- | --- | --- | --- |
| Current fire (horizon 0) | NASA FIRMS area API: `VIIRS_SNPP_NRT`, `VIIRS_NOAA20_NRT`, `VIIRS_NOAA21_NRT` (+ `_SP` archives for dates > 7 days old) | 375 m pixels, several passes/day | Nominal + high confidence only; each detection paints a disk of its pixel's area (scan × track); needs a map key |
| Wind speed + direction, RH, temperature | Open-Meteo forecast API (archive API for older dates) | hourly, ~1–11 km model grid | 10 m wind in m/s; one point per fire |
| 100-h dead fuel moisture | gridMET THREDDS point query | daily, ~4 km | Falls back to the Simard equilibrium-moisture estimate from 24 h RH/T |
| Fuel model (FBFM40), slope | LANDFIRE LF2023 / LF2020, bundled once | ~90 m | `python -m pinn_service.build_static_layers` |

## Static layer file (`STATIC_LAYERS`)

`np.savez_compressed` with:

- `lat`, `lon`: 1-D cell centres (EPSG:4326), evenly spaced, either order.
- `fuel_fbfm40`: int16, shape `(len(lat), len(lon))`. LANDFIRE FBFM40 codes; 0 / 91–99 = non-burnable.
- `slope_deg`: float32, same shape, degrees.

## Replay file (`INGESTION=npz`, `$FEATURE_DIR/YYYY-MM-DD.npz`)

For tests, offline demos and historical cases, one file per day can stand in for all live sources. Arrays are read by name: `lat`, `lon` plus 2-D channels. A stacked layout (`features` with shape (H, W, C) plus `channels` names) is also accepted.

| Name (aliases) | Units | Required |
| --- | --- | --- |
| `active_fire` (`fire_confidence`, `fire_mask`, `fire`) | 0–1 | yes |
| `fuel_fbfm40` (`fbfm40`, `fuel_model`) | FBFM40 code | yes, or from `STATIC_LAYERS` |
| `slope_deg` (`slope`), or `elevation_m` to derive it | degrees / m | yes, or from `STATIC_LAYERS` |
| `fm100_pct` (`fm100`, `fuel_moisture_100h`) | % | yes |
| `wind_u` + `wind_v` (blowing toward east/north), or `wind_speed` + `wind_dir` (meteorological, from) | m/s, degrees | yes |

`python pinn_service/make_test_features.py` writes synthetic examples of both formats into `pinn_service/test_features/`.
