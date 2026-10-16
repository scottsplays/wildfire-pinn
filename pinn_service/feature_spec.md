# Feature file spec: what the PINN reads

**Status:** proposal from the PINN side (Oct 16, 2026), to merge with the U-Net's spec into one daily file.
**File:** `gs://fightfirewai-features/daily/YYYY-MM-DD.npz` (production) or `$FEATURE_DIR/YYYY-MM-DD.npz` (local).

## Layout

The file is a NumPy `.npz`, written with `np.savez_compressed(path, **arrays)`. The service reads arrays **by name**, so channel order never matters. Two layouts are accepted:

1. **Named arrays (preferred).** `lat` and `lon` are 1-D cell-centre coordinates in EPSG:4326. Every channel is a 2-D array of shape `(len(lat), len(lon))`.
2. **Stacked.** `lat`, `lon`, plus `features` with shape `(H, W, C)` and `channels` (length-C string array of names). This suits a U-Net tensor. The names must match the table below.

`lat` and `lon` must be evenly spaced. Either may be ascending or descending. The grid should cover the Bay Area box `[-122.9, 36.9, -121.5, 38.3]`. Any resolution works: the service resamples to its own 90 m forecast grid. Finer is better for fuel and slope (see static layers below).

No normalisation, please. The PINN's spread-rate formula needs physical units. If the U-Net needs normalised inputs, normalise inside the U-Net's loader, or store both.

## Channels

| Name (aliases accepted) | Units | Needed by PINN | Notes |
| --- | --- | --- | --- |
| `active_fire` (`fire_confidence`, `fire_mask`, `fire`) | 0–1 | **Yes** | Horizon-0 fire. 1 = burning, or a detection confidence. Pixels ≥ `fire_threshold` (default 0.5) count as burning. From VIIRS/GOES, or the U-Net's output |
| `fuel_fbfm40` (`fbfm40`, `fuel_model`) | LANDFIRE FBFM40 code | **Yes** | Integer codes (101–204, 91–99 non-burnable). Unknown codes are treated as non-burnable and reported |
| `slope_deg` (`slope`) | degrees | **Yes**, or `elevation_m` | Derived from `elevation_m` if absent |
| `elevation_m` (`elevation`, `elev`) | m | Optional | Used only to derive slope |
| `fm100_pct` (`fm100`, `fuel_moisture_100h`) | % | **Yes** | gridMET 100-h dead fuel moisture |
| `wind_u`, `wind_v` (`u10`, `v10`) | m/s | **Yes**, or speed + direction | Direction the wind blows **toward**: u = east, v = north |
| `wind_speed`, `wind_dir` | m/s, degrees | Alternative to u/v | `wind_dir` is meteorological (blowing **from**, clockwise from north) |
| `ndvi`, `evi`, `temp_c`, `humidity_pct`, … | – | No | Ignored by the PINN; U-Net channels can live in the same file |

## Static layers (recommended)

Fuel and slope do not change day to day, and at 1 km they are too coarse for fire spread. Set `STATIC_LAYERS=/path/bay_area_static.npz` (same layout: `lat`, `lon`, `fuel_fbfm40`, `slope_deg`, ideally at 30–90 m from LANDFIRE). The service then takes fuel and slope from that file at its own resolution, and the daily file only needs `active_fire`, `fm100_pct` and wind.

## Example (Python)

```python
np.savez_compressed(
    "2026-10-16.npz",
    lat=lat, lon=lon,                      # 1-D float64
    active_fire=fire.astype("float32"),    # (H, W)
    fuel_fbfm40=fuel.astype("int16"),
    slope_deg=slope.astype("float32"),
    fm100_pct=fm100.astype("float32"),
    wind_u=u.astype("float32"), wind_v=v.astype("float32"),
)
```

`python pinn_service/make_test_features.py` writes two **synthetic** files in this layout (a fire day and a no-fire day) for local integration tests.
