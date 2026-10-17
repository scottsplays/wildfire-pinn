"""Check a /predict response against the sponsor's non-negotiable rules.

    python pinn_service/validate_contract.py response.json --bbox -122.9 36.9 -121.5 38.3

Usable by the backend team as well: it only needs the JSON.
"""

from __future__ import annotations

import argparse
import json
import sys

ALLOWED_MODELS = {"unet", "pinn", "rl", "ensemble"}
ALLOWED_HORIZONS = {0, 1, 3, 6}


def validate(fc: dict, bbox: list[float], expected_horizons: set[int] | None = None) -> tuple[list[str], list[str]]:
    """Return (errors, warnings)."""
    errors: list[str] = []
    warnings: list[str] = []
    if fc.get("type") != "FeatureCollection":
        errors.append("type must be 'FeatureCollection'")
    meta = fc.get("metadata", {})
    ts = meta.get("timestamp")
    if not isinstance(ts, str) or not ts.endswith("Z"):
        errors.append("metadata.timestamp must be an ISO-8601 UTC string ending in 'Z'")
    source = meta.get("source_model")
    if source not in ALLOWED_MODELS:
        errors.append(f"metadata.source_model must be one of {sorted(ALLOWED_MODELS)}")
    features = fc.get("features")
    if not isinstance(features, list):
        return errors + ["features must be a list"], warnings
    seen_horizons = set()
    eps = 1e-9
    for i, feat in enumerate(features):
        where = f"features[{i}]"
        props = feat.get("properties", {})
        geom = feat.get("geometry", {})
        gid = props.get("grid_id")
        if not isinstance(gid, int) or isinstance(gid, bool) or gid <= 0:
            errors.append(f"{where}.properties.grid_id must be a positive int")
        p = props.get("fire_probability")
        if not isinstance(p, (int, float)) or isinstance(p, bool) or not 0.0 <= p <= 1.0:
            errors.append(f"{where}.properties.fire_probability must be a number in [0, 1]")
        h = props.get("horizon_hours")
        if h not in ALLOWED_HORIZONS or isinstance(h, bool):
            errors.append(f"{where}.properties.horizon_hours must be one of {sorted(ALLOWED_HORIZONS)}")
        else:
            seen_horizons.add(h)
        if props.get("source_model") != source:
            errors.append(f"{where}.properties.source_model must match metadata.source_model")
        if geom.get("type") != "Polygon":
            errors.append(f"{where}.geometry.type must be 'Polygon'")
            continue
        rings = geom.get("coordinates") or []
        if not rings or len(rings[0]) < 4:
            errors.append(f"{where}: polygon needs a ring of at least 4 positions")
            continue
        ring = rings[0]
        if ring[0] != ring[-1]:
            errors.append(f"{where}: ring is not closed")
        for lon, lat in ring:
            if not (bbox[0] - eps <= lon <= bbox[2] + eps and bbox[1] - eps <= lat <= bbox[3] + eps):
                errors.append(f"{where}: vertex {lon, lat} outside bbox (or [lat, lon] order)")
                break
    if expected_horizons and features and not expected_horizons <= seen_horizons:
        warnings.append(f"horizons missing from this response: {sorted(expected_horizons - seen_horizons)}")
    if not features:
        warnings.append("empty FeatureCollection (no fire in bbox?)")
    return errors, warnings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--bbox", nargs=4, type=float, default=[-122.9, 36.9, -121.5, 38.3])
    args = parser.parse_args()
    with open(args.path, encoding="utf-8-sig") as handle:  # tolerate the BOM Windows PowerShell adds
        body = json.load(handle)
    collections = body["predictions"] if "predictions" in body else [body]
    failed = False
    for k, fc in enumerate(collections):
        errors, warnings = validate(fc, args.bbox, {0, 1, 3, 6})
        for w in warnings:
            print(f"[{k}] warning: {w}")
        for e in errors[:50]:
            print(f"[{k}] ERROR: {e}")
        failed |= bool(errors)
        print(f"[{k}] {len(fc.get('features', []))} features, {len(errors)} errors")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
