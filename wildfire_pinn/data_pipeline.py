"""Palisades observations: detections, perimeters, terrain, weather, spread rate.

Stage order is fixed because the Colab notebook ran these steps out of order
and then trained against a stack path that a fresh runtime did not have:

    FIRMS detections -> perimeters -> LANDFIRE + gridMET -> stack -> R(x, y, t)
"""

from __future__ import annotations

import io
import json
import logging
import math
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from shapely.ops import unary_union

from wildfire_pinn.config import ProjectConfig

log = logging.getLogger("wildfire_pinn")

FIRMS_URL = (
    "https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
    "{key}/{product}/{bbox}/{days}/{start}"
)
GRIDMET_BASE = "http://thredds.northwestknowledge.net:8080/thredds/ncss/MET"

# (URL path, CF name inside the file, name stored in the stack)
GRIDMET_VARS = {
    "vs": ("wind_speed", "wind_speed"),
    "th": ("wind_from_direction", "wind_dir"),
    "tmmx": ("air_temperature", "air_temperature"),
    "rmin": ("relative_humidity", "relative_humidity"),
    "fm100": ("dead_fuel_moisture_100hr", "fuel_moisture"),
}

LANDFIRE_SERVICES = {
    "fuel": "Landfire_LF2023/LF2023_FBFM40_CONUS",
    "elevation": "Landfire_Topo/LF2020_Elev_CONUS",
    "slope": "Landfire_Topo/LF2020_SlpD_CONUS",
}

# Relative spread-rate tiers from the published ROS adjective bands
# (RMRS-GTR-153 Table 5, via NWCG S-290; dead fuel moisture D2L2,
# 5 mi/h midflame wind, zero slope). C_REF is the single calibration
# knob: 0.177 m/s = 31.6 chains/h, the geometric midpoint of the High band.
# High -> 1.0, moderate -> 0.2, low -> 0.07. Palisades grass and shrub
# codes stay on the high tier they were built with.
GRASS_SHRUB = (101, 102, 103, 121, 122, 123, 141, 142, 144, 145, 147)
TIMBER_UNDERSTORY = (161, 163, 165)
TIMBER_LITTER = (181, 182, 183, 184, 185, 186, 188, 189)
NONBURNABLE = frozenset({0, 91, 93, 98, 99})

# Moisture of extinction (%), Scott & Burgan (2005) via firestarter-io/fuelmodels.
# 143, 146, 162, 187, 201, and 202 are the Kincade codes that were absent
# from the Palisades window. Mx is the published extinction moisture.
# r_rel follows the LANDFIRE FBFM40 spread class for that model:
# 146 SH6 high; 162 TU2, 201 SB1, and 202 SB2 moderate; 143 SH3 and 187 TL7 low.
MX = {
    101: 15, 102: 15, 103: 30, 121: 15, 122: 15, 123: 15,
    141: 15, 142: 15, 143: 40, 144: 30, 145: 15, 146: 30, 147: 15,
    161: 20, 162: 30, 163: 30, 165: 25,
    181: 30, 182: 25, 183: 20, 184: 25, 185: 25, 186: 25, 187: 25, 188: 35, 189: 35,
    201: 25, 202: 25,
}
R_REL = {code: 1.0 for code in GRASS_SHRUB}
R_REL.update({code: 0.2 for code in TIMBER_UNDERSTORY})
R_REL.update({code: 0.07 for code in TIMBER_LITTER})
R_REL.update({146: 1.0, 162: 0.2, 201: 0.2, 202: 0.2, 143: 0.07, 187: 0.07})

C_REF = 0.177
C_W = 0.5
U_REF = 2.237  # m/s, 5 mi/h, the wind speed the relative tiers were tabulated at
C_S = 1.0

if set(R_REL) - set(MX):
    raise RuntimeError(f"Mx missing for fuel codes {sorted(set(R_REL) - set(MX))}")

STACK_ROLES = {
    "elevation_m": "terrain context; slope is stored on its own",
    "slope_deg": "slope factor S in R",
    "fuel_model": "FBFM40 code; lookup for the relative spread rate",
    "wind_speed": "wind-speed factor W in R",
    "wind_dir": "meteorological wind-from direction (degrees); the PDE uses the downwind vector",
    "fuel_moisture": "100-hour dead fuel moisture (%); moisture factor M in R",
    "air_temperature": "stored for diagnostics; not used by the PDE",
    "relative_humidity": "stored for diagnostics; not used by the PDE",
}


def firms_chunks(start: str, end: str, span: int = 5) -> list[tuple[str, int]]:
    """FIRMS area queries are capped at 5 days. Chunks cover [start, end] without gaps."""
    start_d = date.fromisoformat(start)
    end_d = date.fromisoformat(end)
    if end_d < start_d:
        raise ValueError(f"date_end {end} is before date_start {start}")
    chunks = []
    cursor = start_d
    while cursor <= end_d:
        days = min(span, (end_d - cursor).days + 1)
        chunks.append((cursor.isoformat(), days))
        cursor += timedelta(days=days)
    return chunks


def request_bytes(url: str, params: dict | None = None, timeout: int = 120, attempts: int = 3) -> bytes:
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            response = requests.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            return response.content
        except requests.RequestException as exc:
            last = exc
            log.warning("request failed (%s/%s): %s", attempt + 1, attempts, exc)
    raise RuntimeError(f"Download failed after {attempts} attempts: {url}") from last


def parse_firms_csv(text: str) -> pd.DataFrame:
    """Parse one FIRMS area response. An empty window is an empty frame, not an error."""
    stripped = text.lstrip()
    if stripped.startswith(("Invalid", "<", "{")):
        raise RuntimeError("FIRMS rejected the request: " + stripped[:300])
    frame = pd.read_csv(io.StringIO(text))
    if frame.empty or "latitude" not in frame.columns:
        return pd.DataFrame(columns=["latitude", "longitude", "acq_date", "acq_time", "confidence"])
    return frame


def download_firms(cfg: ProjectConfig, api_key: str | None) -> Path:
    key = (api_key or "").strip()
    if not key:
        raise RuntimeError(
            "A NASA FIRMS map key is required to download detections. "
            "Request one at https://firms.modaps.eosdis.nasa.gov/api/area/ "
            "and paste it into firms_key.txt, set FIRMS_MAP_KEY, or pass --firms-key. "
            f"If the CSV is already downloaded, place it at {cfg.firms_csv}."
        )
    west, south, east, north = cfg.bbox
    bbox = f"{west},{south},{east},{north}"
    frames = []
    for start, days in firms_chunks(cfg.date_start, cfg.date_end):
        url = FIRMS_URL.format(
            key=key, product=cfg.firms_product, bbox=bbox, days=days, start=start
        )
        text = request_bytes(url, timeout=120).decode("utf-8", errors="replace")
        frame = parse_firms_csv(text)
        log.info("FIRMS %s (%d days) -> %d detections", start, days, len(frame))
        if frame.empty:
            log.warning("no detections in %s; that window is skipped", start)
            continue
        frames.append(frame)
    if not frames:
        raise RuntimeError(
            f"FIRMS returned no detections from {cfg.date_start} to {cfg.date_end} "
            f"in {cfg.bbox} for product {cfg.firms_product}."
        )
    detections = pd.concat(frames, ignore_index=True)
    subset = [c for c in ("latitude", "longitude", "acq_date", "acq_time") if c in detections.columns]
    detections = detections.drop_duplicates(subset=subset)
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    detections.to_csv(cfg.firms_csv, index=False)
    log.info(
        "wrote %s (%d unique detections, %s -> %s)",
        cfg.firms_csv, len(detections),
        detections["acq_date"].min(), detections["acq_date"].max(),
    )
    return cfg.firms_csv


def normalize_acq_time(values) -> pd.Series:
    text = pd.Series(values).astype(str).str.strip()
    text = text.str.replace(r"\.0$", "", regex=True)
    text = text.str.replace(r"\D", "", regex=True)
    return text.str.zfill(4)


def overpass_timestamps(acq_date, acq_time) -> pd.Series:
    dates = pd.to_datetime(acq_date)
    clock = normalize_acq_time(acq_time)
    hours = clock.str[:2].astype(int)
    minutes = clock.str[2:].astype(int)
    if ((hours < 0) | (hours > 23) | (minutes < 0) | (minutes > 59)).any():
        raise RuntimeError("acq_time values are not valid HHMM times")
    return dates + pd.to_timedelta(hours, unit="h") + pd.to_timedelta(minutes, unit="m")


def filter_detections(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep nominal and high confidence VIIRS detections and tag each overpass."""
    required = {"latitude", "longitude", "acq_date", "acq_time", "confidence"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"FIRMS table is missing columns: {sorted(missing)}")
    out = frame.copy()
    out["acq_date"] = pd.to_datetime(out["acq_date"])
    out["confidence"] = out["confidence"].astype(str).str.lower().str.strip()
    log.info("confidence counts: %s", out["confidence"].value_counts().to_dict())
    kept = out[out["confidence"].isin(["n", "h"])].copy()
    log.info("kept %d / %d detections (nominal + high)", len(kept), len(out))
    if kept.empty:
        raise RuntimeError("No nominal or high confidence detections remain.")
    kept["overpass"] = (
        kept["acq_date"].dt.strftime("%Y-%m-%d") + "_" + normalize_acq_time(kept["acq_time"])
    )
    return kept


def trim_outliers(lonlat: np.ndarray, min_points: int = 4, crs: str = "EPSG:3310") -> np.ndarray:
    """Drop detections whose 3rd-nearest neighbor is farther than median + 4*MAD.

    Distances are measured in meters in ``crs``. The default is California
    Albers. If the cut would leave fewer than ``min_points``, the overpass
    is kept intact.
    """
    import geopandas as gpd
    from scipy.spatial.distance import cdist

    lonlat = np.asarray(lonlat, dtype=np.float64)
    count = len(lonlat)
    if count <= min_points + 1:
        return np.ones(count, dtype=bool)
    projected = gpd.GeoSeries(
        gpd.points_from_xy(lonlat[:, 0], lonlat[:, 1]), crs="EPSG:4326"
    ).to_crs(crs)
    xy = np.column_stack([projected.x.to_numpy(), projected.y.to_numpy()])
    third = np.sort(cdist(xy, xy), axis=1)[:, 3]
    median = np.median(third)
    mad = np.median(np.abs(third - median)) + 1e-9
    keep = third <= median + 4.0 * mad
    if int(keep.sum()) < min_points:
        return np.ones(count, dtype=bool)
    return keep


def as_polygonal(geom):
    """Keep polygon parts. Alpha shapes sometimes return a GeometryCollection."""
    if geom is None or geom.is_empty:
        return geom
    if geom.geom_type in ("Polygon", "MultiPolygon"):
        return unary_union(geom)
    if geom.geom_type == "GeometryCollection":
        parts = [part for part in geom.geoms if part.geom_type in ("Polygon", "MultiPolygon")]
        if not parts:
            return unary_union([])
        return unary_union(parts)
    return geom


def build_perimeters(
    detections: pd.DataFrame,
    out_path: Path,
    alpha: float = 100.0,
    min_points: int = 4,
    crs: str = "EPSG:3310",
):
    """One concave hull per satellite overpass, saved as EPSG:4326 GeoJSON."""
    import alphashape
    import geopandas as gpd

    if "overpass" not in detections.columns:
        detections = filter_detections(detections)
    rows = []
    skipped = []
    for overpass, group in detections.groupby("overpass", sort=True):
        points = group[["longitude", "latitude"]].to_numpy(np.float64)
        kept = points[trim_outliers(points, min_points=min_points, crs=crs)]
        if len(kept) < min_points:
            skipped.append((overpass, len(kept), "too few points after trimming"))
            continue
        try:
            shape = alphashape.alphashape(kept, alpha)
        except Exception as exc:
            skipped.append((overpass, len(kept), f"alphashape failed: {exc}"))
            continue
        shape = as_polygonal(shape)
        if shape is None or shape.is_empty or shape.geom_type not in ("Polygon", "MultiPolygon"):
            kind = getattr(shape, "geom_type", "empty")
            skipped.append((overpass, len(kept), f"no polygon ({kind})"))
            continue
        date_str, time_str = str(overpass).split("_", 1)
        rows.append({
            "overpass": overpass,
            "acq_date": date_str,
            "acq_time": time_str,
            "n_points": int(len(kept)),
            "n_raw": int(len(points)),
            "geometry": shape,
        })
    log.info("perimeters built: %d | skipped: %d", len(rows), len(skipped))
    for item in skipped:
        log.info("skipped %s", item)
    if not rows:
        raise RuntimeError("No perimeters could be built from the detections.")
    gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(out_path, driver="GeoJSON")
    log.info("wrote %s", out_path)
    return gdf


def _export_size(bbox: tuple[float, float, float, float], meters: float = 30.0) -> tuple[int, int]:
    """Pixel size requested from LANDFIRE.

    ceil(span * 111.32 km/deg / 30 m) is 1039 by 520 for the Palisades bbox,
    which is the export size the notebook requested.
    """
    west, south, east, north = bbox
    meters_per_degree = 111_320.0
    width = max(2, math.ceil((east - west) * meters_per_degree / meters - 1e-9))
    height = max(2, math.ceil((north - south) * meters_per_degree / meters - 1e-9))
    return width, height


def _write_download(content: bytes, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(content)
    temporary.replace(path)


def download_landfire(cfg: ProjectConfig) -> dict[str, Path]:
    west, south, east, north = cfg.bbox
    width, height = _export_size(cfg.bbox)
    names = {
        "fuel": cfg.fuel_tif,
        "elevation": cfg.elev_tif,
        "slope": cfg.slope_tif,
    }
    saved = {}
    for kind, service in LANDFIRE_SERVICES.items():
        base = (
            "https://lfps.usgs.gov/arcgis/rest/services/"
            f"{service}/ImageServer/exportImage"
        )
        params = {
            "bbox": f"{west},{south},{east},{north}",
            "bboxSR": 4326,
            "imageSR": 4326,
            "size": f"{width},{height}",
            "format": "tiff",
            "f": "json",
        }
        if kind == "fuel":
            params["pixelType"] = "U8"
            params["noData"] = 0
        raw = request_bytes(base, params=params, timeout=120)
        try:
            meta = json.loads(raw.decode("utf-8", errors="replace"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"LANDFIRE {service} did not return JSON: {raw[:200]!r}"
            ) from exc
        href = meta.get("href") if isinstance(meta, dict) else None
        if not href:
            raise RuntimeError(f"LANDFIRE {service} did not return an image: {str(meta)[:300]}")
        payload = request_bytes(str(href), timeout=300)
        if payload[:2] not in (b"II", b"MM"):
            raise RuntimeError(f"LANDFIRE file for {kind} is not a TIFF: {payload[:80]!r}")
        _write_download(payload, names[kind])
        log.info("wrote %s (%d bytes)", names[kind], len(payload))
        saved[kind] = names[kind]
    return saved


def _time_name(dataset) -> str:
    for name in ("day", "time"):
        if name in dataset.dims or name in dataset.coords:
            return name
    raise RuntimeError(f"No day/time coordinate in {list(dataset.dims)}")


def _reference_timestamp(dataset) -> pd.Timestamp:
    name = _time_name(dataset)
    first = dataset[name].values[0]
    if hasattr(first, "isoformat"):
        try:
            return pd.Timestamp(first.isoformat())
        except (TypeError, ValueError):
            pass
    return pd.Timestamp(first)


def download_gridmet(cfg: ProjectConfig) -> Path:
    import xarray as xr

    west, south, east, north = cfg.bbox
    pieces = []
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    for short, (standard_name, stored_name) in GRIDMET_VARS.items():
        year = pd.Timestamp(cfg.date_start).year
        if pd.Timestamp(cfg.date_end).year != year:
            raise RuntimeError(
                f"gridMET is stored one year per file. {cfg.date_start} to {cfg.date_end} crosses a year boundary."
            )
        url = (
            f"{GRIDMET_BASE}/{short}/{short}_{year}.nc?var={standard_name}"
            f"&north={north}&south={south}&west={west}&east={east}"
            f"&time_start={cfg.date_start}T00:00:00Z"
            f"&time_end={cfg.date_end}T00:00:00Z&accept=netCDF"
        )
        raw = request_bytes(url, timeout=180)
        if not (raw.startswith(b"CDF") or raw.startswith(b"\x89HDF")):
            raise RuntimeError(f"gridMET did not return NetCDF for {short}: {raw[:200]!r}")
        part_path = cfg.data_dir / f"{cfg.name}_gridmet_{short}.nc"
        _write_download(raw, part_path)
        with xr.open_dataset(part_path) as source:
            loaded = source.load()
        if standard_name in loaded.data_vars:
            variable = standard_name
        elif short in loaded.data_vars:
            variable = short
        else:
            numeric = [name for name in loaded.data_vars if loaded[name].ndim >= 1]
            if len(numeric) != 1:
                raise RuntimeError(
                    f"Could not find {standard_name} in {part_path.name}; "
                    f"variables are {list(loaded.data_vars)}"
                )
            variable = numeric[0]
        pieces.append(loaded[variable].rename(stored_name))
        log.info("gridMET %s %s", stored_name, dict(pieces[-1].sizes))
    merged = xr.merge(pieces)
    if "time" in merged.dims and "day" not in merged.dims:
        merged = merged.rename({"time": "day"})
    started = _reference_timestamp(merged).normalize()
    expected = pd.Timestamp(cfg.date_start)
    if started != expected:
        log.warning("gridMET starts on %s; the request asked for %s", started.date(), expected.date())
    save_dataset(merged, cfg.gridmet_nc)
    log.info("wrote %s", cfg.gridmet_nc)
    return cfg.gridmet_nc


def save_dataset(dataset, path: Path) -> None:
    """Write NetCDF via a temporary file so a failed write cannot truncate a good one.

    Windows will not replace a NetCDF file that is still open. Callers load and
    close datasets before writing.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoding = {}
    for name, variable in dataset.data_vars.items():
        if variable.ndim > 0 and np.issubdtype(variable.dtype, np.number):
            encoding[name] = {"zlib": True, "complevel": 4}
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    dataset.to_netcdf(temporary, engine="netcdf4", encoding=encoding)
    temporary.replace(path)


def load_dataset(path: Path):
    import xarray as xr

    with xr.open_dataset(path) as source:
        return source.load()


def _squeeze_band(data_array):
    if "band" in data_array.dims:
        data_array = data_array.isel(band=0, drop=True)
    data_array = data_array.squeeze(drop=True)
    if "y" in data_array.dims and "x" in data_array.dims:
        data_array = data_array.transpose("y", "x")
    return data_array


def _array_values(data_array) -> np.ndarray:
    values = np.asarray(data_array.values)
    if np.ma.isMaskedArray(values):
        values = values.filled(np.nan)
    return np.array(values, dtype=np.float64, copy=True)


def _mask_range(data_array, low: float, high: float):
    values = _array_values(data_array)
    invalid = ~np.isfinite(values) | (values < low) | (values > high)
    values[invalid] = np.nan
    return data_array.copy(data=values)


def _open_raster(path: Path):
    import rioxarray  # noqa: F401  registers the .rio accessor

    data = rioxarray.open_rasterio(path)
    if data.rio.crs is None:
        data = data.rio.write_crs("EPSG:4326")
    loaded = data.load()
    close = getattr(data, "close", None)
    if close is not None:
        close()
    return loaded


def _lon_lat_dims(dataset) -> tuple[str, str]:
    dims = set(dataset.dims)
    for x_name, y_name in (("lon", "lat"), ("longitude", "latitude"), ("x", "y")):
        if x_name in dims and y_name in dims:
            return x_name, y_name
    raise RuntimeError(f"gridMET file has no lon/lat dimensions: {list(dataset.dims)}")


def build_stack(cfg: ProjectConfig) -> Path:
    """Reproject terrain, fuel, and weather onto one meter grid in ``cfg.target_crs``."""
    import rioxarray  # noqa: F401
    import xarray as xr
    from rasterio.enums import Resampling

    missing = [path for path in (cfg.elev_tif, cfg.slope_tif, cfg.fuel_tif, cfg.gridmet_nc) if not path.exists()]
    if missing:
        raise RuntimeError(
            "Cannot build the stack; missing:\n  " + "\n  ".join(str(path) for path in missing)
        )

    log.info("reprojecting elevation")
    elevation = _open_raster(cfg.elev_tif)
    elevation = elevation.rio.reproject(cfg.target_crs, resolution=cfg.resolution_m)
    elevation = _mask_range(_squeeze_band(elevation), low=-1000, high=10000).rename("elevation_m")

    log.info("reprojecting slope")
    slope = _open_raster(cfg.slope_tif).rio.reproject_match(elevation)
    slope = _mask_range(_squeeze_band(slope), low=0, high=90).rename("slope_deg")

    log.info("reprojecting fuel model (nearest neighbor; codes are categories)")
    fuel = _open_raster(cfg.fuel_tif).rio.reproject_match(elevation, resampling=Resampling.nearest)
    fuel = _squeeze_band(fuel).rename("fuel_model")
    fuel_values = _array_values(fuel)
    fuel_values = np.where(np.isfinite(fuel_values), np.rint(fuel_values), 0.0)
    fuel = fuel.copy(data=fuel_values)
    present = np.unique(fuel_values.astype(int))
    log.info("fuel codes present: %s", present.tolist())

    weather = load_dataset(cfg.gridmet_nc)
    if "time" in weather.dims and "day" not in weather.dims:
        weather = weather.rename({"time": "day"})
    x_dim, y_dim = _lon_lat_dims(weather)
    weather = weather.rio.set_spatial_dims(x_dim=x_dim, y_dim=y_dim)
    weather = weather.rio.write_crs("EPSG:4326")
    matched = weather.rio.reproject_match(elevation)
    keep = {}
    for name in (
        "wind_speed", "wind_dir", "fuel_moisture", "air_temperature", "relative_humidity"
    ):
        if name not in matched:
            raise RuntimeError(
                f"Reprojected gridMET is missing {name}. Variables: {list(matched.data_vars)}"
            )
        keep[name] = matched[name]

    stack = xr.Dataset({
        "elevation_m": elevation,
        "slope_deg": slope,
        "fuel_model": fuel,
        **keep,
    })
    stack.attrs["reference_date"] = cfg.date_start
    stack.attrs["crs"] = cfg.target_crs
    stack.attrs["resolution_m"] = cfg.resolution_m
    stack.attrs["bbox_wgs84"] = ",".join(str(value) for value in cfg.bbox)
    save_dataset(stack, cfg.stack_nc)
    log.info("wrote %s %s", cfg.stack_nc, dict(stack.sizes))
    return cfg.stack_nc


def compute_spread_rate(
    fuel,
    slope_deg,
    wind_speed,
    moisture,
) -> tuple[np.ndarray, dict]:
    """First-order spread rate R(x, y, t) in m/s.

    R = C_REF * r_rel[fuel] * W(wind speed) * S(slope) * M(moisture).

    This is the documented proxy from the notebook, not a full Rothermel
    implementation. Wind direction is not part of this scalar field; the PDE
    residual applies it. Fuel codes that are not in the tables raise, rather
    than silently using a default rate.
    """
    fuel = np.asarray(fuel, dtype=np.float64)
    slope_deg = np.asarray(slope_deg, dtype=np.float64)
    wind_speed = np.asarray(wind_speed, dtype=np.float64)
    moisture = np.asarray(moisture, dtype=np.float64)
    if fuel.ndim != 2:
        raise ValueError(f"fuel must have shape (y, x); got {fuel.shape}")
    if wind_speed.ndim == 2:
        wind_speed = wind_speed[None, ...]
    if moisture.ndim == 2:
        moisture = moisture[None, ...]
    if wind_speed.shape[1:] != fuel.shape or moisture.shape != wind_speed.shape:
        raise ValueError(
            f"shape mismatch: fuel {fuel.shape}, wind {wind_speed.shape}, moisture {moisture.shape}"
        )
    if slope_deg.shape != fuel.shape:
        raise ValueError(f"slope shape {slope_deg.shape} does not match fuel {fuel.shape}")

    fuel_codes = np.where(np.isfinite(fuel), np.rint(fuel), 0).astype(int)
    fuel_codes = np.clip(fuel_codes, 0, None)
    present = set(np.unique(fuel_codes).tolist()) - set(NONBURNABLE)
    unknown = sorted(present - set(R_REL))
    if unknown:
        raise RuntimeError(
            f"Fuel codes with no parameters: {unknown}. "
            "Add an explicit r_rel and Mx; there is no default rate."
        )

    terrain_bad = ~np.isfinite(slope_deg) | (slope_deg < 0) | (slope_deg > 90)
    burnable = ~np.isin(fuel_codes, list(NONBURNABLE)) & ~terrain_bad
    if not burnable.any():
        raise RuntimeError("No burnable fuel cells in the domain.")

    lut_n = max(200, int(fuel_codes.max()) + 1)
    lut_rel = np.zeros(lut_n, dtype=np.float64)
    lut_mx = np.full(lut_n, 30.0, dtype=np.float64)
    for code, value in R_REL.items():
        lut_rel[code] = value
    for code, value in MX.items():
        lut_mx[code] = value
    relative = np.where(burnable, lut_rel[fuel_codes], 0.0)
    extinction = np.where(burnable, lut_mx[fuel_codes], 30.0)

    wind = np.where(np.isfinite(wind_speed), wind_speed, 0.0)
    moist = np.where(np.isfinite(moisture), moisture, extinction[None, ...])
    slope_filled = np.where(terrain_bad, 0.0, slope_deg)

    wind_factor = (1.0 + C_W * wind) / (1.0 + C_W * U_REF)
    slope_factor = 1.0 + C_S * np.tan(np.deg2rad(slope_filled))
    moisture_factor = np.clip(1.0 - moist / extinction[None, ...], 0.0, 1.0)
    rate = C_REF * relative[None, ...] * wind_factor * slope_factor[None, ...] * moisture_factor
    rate = np.where(burnable[None, ...], rate, 0.0)
    rate = np.nan_to_num(rate, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    daily = []
    for day in range(rate.shape[0]):
        values = rate[day][burnable]
        daily.append(
            f"day {day}: median R={np.median(values):.3f} m/s | "
            f"mean={values.mean():.3f} | max={values.max():.3f}"
        )
    stats = {
        "burnable_fraction": float(burnable.mean()),
        "daily": daily,
        "c_ref": C_REF,
    }
    return rate, stats


def build_spread_rate(cfg: ProjectConfig) -> Path:
    import xarray as xr

    if not cfg.stack_nc.exists():
        raise RuntimeError(f"Missing {cfg.stack_nc}. Build the stack first.")
    stack = load_dataset(cfg.stack_nc)
    rate, stats = compute_spread_rate(
        stack["fuel_model"].values,
        stack["slope_deg"].values,
        stack["wind_speed"].values,
        stack["fuel_moisture"].values,
    )
    dataset = xr.Dataset(
        {"R": (("day", "y", "x"), rate, {"units": "m/s"})},
        coords={
            "day": np.arange(rate.shape[0]),
            "y": stack["y"].values,
            "x": stack["x"].values,
        },
    )
    dataset.attrs["reference_date"] = str(stack.attrs.get("reference_date", cfg.date_start))
    dataset.attrs["description"] = (
        "R = C_REF * r_rel[fuel] * W(wind speed) * S(slope) * M(moisture). "
        "First-order proxy; full Rothermel is deferred. "
        "Wind direction is applied in the level-set residual, not in this field."
    )
    dataset.attrs["C_REF_m_s"] = C_REF
    dataset.attrs["C_W"] = C_W
    dataset.attrs["U_REF_m_s"] = U_REF
    dataset.attrs["C_S"] = C_S
    save_dataset(dataset, cfg.rate_nc)
    log.info("burnable fraction: %.3f", stats["burnable_fraction"])
    for line in stats["daily"]:
        log.info(line)
    log.info(
        "sanity check: the Jan 8 run moved the front on the order of a kilometer "
        "in about 12 hours, which is roughly 0.05-0.2 m/s. "
        "If the median is far outside that band, C_REF is the calibration knob."
    )
    log.info("wrote %s %s", cfg.rate_nc, rate.shape)
    return cfg.rate_nc


def describe_stack(path: Path) -> str:
    stack = load_dataset(path)
    lines = [f"{'variable':28s}{'shape':22s}role"]
    for name in stack.data_vars:
        role = STACK_ROLES.get(name, "metadata" if stack[name].ndim == 0 else "unspecified")
        lines.append(f"{name:28s}{str(tuple(stack[name].shape)):22s}{role}")
    mebibytes = sum(stack[name].nbytes for name in stack.data_vars) / 1e6
    lines.append(f"dims: {dict(stack.sizes)} | {mebibytes:.1f} MB")
    text = "\n".join(lines)
    log.info("\n%s", text)
    return text
