"""Build the bundled static layers (LANDFIRE FBFM40 fuel + slope) once.

    python -m pinn_service.build_static_layers                       # Bay Area box, ~90 m
    python -m pinn_service.build_static_layers --bbox -123.6 36.4 -121.0 39.2 --metres 90 --out pinn_service/static/norcal.npz

Runs on a machine with internet and rasterio (the training environment has it via
rioxarray). Uses the same LANDFIRE ImageServer services as
``wildfire_pinn/data_pipeline.py::download_landfire``. Output: an .npz with 1-D
``lat``/``lon`` cell centres and 2-D ``fuel_fbfm40`` (int16) and ``slope_deg``
(float32), which the service loads at start-up (``STATIC_LAYERS``).

The default box is the sponsor's Bay Area box plus a ~25 km margin, so fires near
the edge still have fuel data around them.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

SERVICES = {
    "fuel_fbfm40": "Landfire_LF2023/LF2023_FBFM40_CONUS",
    "slope_deg": "Landfire_Topo/LF2020_SlpD_CONUS",
}
BASE = "https://lfps.usgs.gov/arcgis/rest/services/{service}/ImageServer/exportImage"
DEFAULT_BBOX = (-123.2, 36.65, -121.2, 38.55)


def _get(url: str, timeout: float = 300) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "WildfirePINN"}), timeout=timeout) as r:
        return r.read()


def export(service: str, bbox, width: int, height: int, fuel: bool):
    import rasterio

    params = {
        "bbox": ",".join(str(v) for v in bbox), "bboxSR": 4326, "imageSR": 4326,
        "size": f"{width},{height}", "format": "tiff", "f": "json",
        "interpolation": "RSP_NearestNeighbor",
    }
    if fuel:
        params |= {"pixelType": "U8", "noData": 0}  # as in data_pipeline.download_landfire; codes <= 204
    meta = json.loads(_get(BASE.format(service=service) + "?" + urllib.parse.urlencode(params)))
    href = meta.get("href")
    if not href:
        raise RuntimeError(f"LANDFIRE {service}: {str(meta)[:300]}")
    payload = _get(href)
    with rasterio.open(io.BytesIO(payload)) as src:
        band = src.read(1).astype(np.float64)
        nodata = src.nodata
        t = src.transform
        lon = t.c + t.a * (np.arange(src.width) + 0.5)
        lat = t.f + t.e * (np.arange(src.height) + 0.5)
    if nodata is not None:
        band = np.where(band == nodata, np.nan, band)
    band = np.where(band < -1000, np.nan, band)  # LANDFIRE -9999 / -32768 fill values
    return lon, lat, band


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bbox", nargs=4, type=float, default=DEFAULT_BBOX)
    parser.add_argument("--metres", type=float, default=90.0)
    parser.add_argument("--out", default=str(Path(__file__).resolve().parent / "static" / "bay_area_static.npz"))
    args = parser.parse_args()
    west, south, east, north = args.bbox
    lat_mid = math.radians((south + north) / 2)
    width = math.ceil((east - west) * 111_320 * math.cos(lat_mid) / args.metres)
    height = math.ceil((north - south) * 110_950 / args.metres)
    print(f"requesting {width} x {height} pixels (~{args.metres:g} m) for {args.bbox}")
    arrays = {}
    lon = lat = None
    for name, service in SERVICES.items():
        lon_n, lat_n, band = export(service, args.bbox, width, height, fuel=name == "fuel_fbfm40")
        if lon is None:
            lon, lat = lon_n, lat_n
        elif band.shape != (len(lat), len(lon)):
            raise RuntimeError(f"{name} came back {band.shape}, expected {(len(lat), len(lon))}")
        arrays[name] = band
        print(f"{name}: {band.shape}, {np.isfinite(band).mean():.1%} valid")
    fuel = np.nan_to_num(arrays["fuel_fbfm40"], nan=0).astype(np.int16)
    slope = np.nan_to_num(arrays["slope_deg"], nan=0).astype(np.float32)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, lat=lat, lon=lon, fuel_fbfm40=fuel, slope_deg=slope,
                        source=np.array("LANDFIRE LF2023 FBFM40 + LF2020 slope (degrees)"))
    codes, counts = np.unique(fuel, return_counts=True)
    top = ", ".join(f"{c}:{n / fuel.size:.0%}" for c, n in sorted(zip(codes, counts), key=lambda x: -x[1])[:8])
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB); most common fuel codes {top}")


if __name__ == "__main__":
    main()
