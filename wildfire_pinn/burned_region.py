"""Persistent burned-area targets from active-fire detections.

A VIIRS hotspot says a pixel was hot on that pass. It does not say the whole
concave hull burned, and it does not say the pixel cooled back to unburned.
The estimate here keeps every accepted pixel footprint inside the burned
region at every later time. Gaps between pixels stay outside, because those
gaps were not observed to burn.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import Point
from shapely.ops import unary_union

from wildfire_pinn.data_pipeline import filter_detections, trim_outliers
from wildfire_pinn.geometry import cell_size_m, signed_distance, zero_contours

# Nadir I-band size for VIIRS_SNPP_SP, used only when a row has no scan/track.
VIIRS_NADIR_KM = 0.375


def footprint_radius_m(scan_km, track_km) -> np.ndarray:
    """Radius of the circle around a VIIRS pixel center, in meters.

    ``scan`` and ``track`` are the footprint sides in kilometers, as FIRMS
    records them. The circle circumscribes that rectangle, so the flagged
    pixel is inside the estimate. Corners outside the rectangle are included
    only because a circle is the simple pixel stand-in.
    """
    scan = np.asarray(scan_km, dtype=np.float64)
    track = np.asarray(track_km, dtype=np.float64)
    scan = np.where(np.isfinite(scan) & (scan > 0), scan, VIIRS_NADIR_KM)
    track = np.where(np.isfinite(track) & (track > 0), track, VIIRS_NADIR_KM)
    return 500.0 * np.hypot(scan, track)


def accumulate_footprints(batches: list[list[tuple[float, float, float]]]):
    """Union pixel disks through time. Each batch is ``(x, y, radius_m)``.

    ``B(t_i)`` contains ``B(t_{i-1})`` by construction. An empty first batch
    raises. A later empty batch keeps the region already accumulated.
    """
    if not batches:
        raise ValueError("need at least one pass")
    current = None
    regions = []
    for batch in batches:
        disks = [Point(x, y).buffer(radius, quad_segs=8) for x, y, radius in batch if radius > 0]
        added = unary_union(disks) if disks else None
        if current is None:
            current = added
        elif added is not None and not added.is_empty:
            current = unary_union([current, added])
        if current is None or current.is_empty:
            raise ValueError("the first pass produced an empty burned region")
        regions.append(current)
    return regions


def _accepted_detections(firms_csv: Path, crs: str) -> pd.DataFrame:
    raw = pd.read_csv(firms_csv)
    kept = filter_detections(raw)
    if "scan" not in kept.columns or "track" not in kept.columns:
        kept["scan"] = VIIRS_NADIR_KM
        kept["track"] = VIIRS_NADIR_KM
    pieces = []
    for overpass, group in kept.groupby("overpass", sort=True):
        lonlat = group[["longitude", "latitude"]].to_numpy(np.float64)
        mask = trim_outliers(lonlat, crs=crs)
        accepted = group.iloc[np.flatnonzero(mask)].copy()
        accepted["trimmed"] = False
        rejected = group.iloc[np.flatnonzero(~mask)].copy()
        rejected["trimmed"] = True
        pieces.append(pd.concat([accepted, rejected], ignore_index=True))
    out = pd.concat(pieces, ignore_index=True)
    from wildfire_pinn.data_pipeline import overpass_timestamps

    out["dt"] = overpass_timestamps(out["acq_date"], out["acq_time"])
    return out


def persistent_burned_regions(firms_csv: Path, perimeter_times, crs: str = "EPSG:3310"):
    """Burned-region estimate at each perimeter time, in ``crs``.

    A detection is assigned to the first perimeter time at or after it.
    Detections after the last perimeter are not used. Spatial outliers use
    the same neighbor cut as the concave-hull builder, and those points are
    returned separately so a plot can show them. They do not enter ``B(t)``.
    """
    import geopandas as gpd

    times = pd.to_datetime(pd.Series(perimeter_times)).reset_index(drop=True)
    detections = _accepted_detections(firms_csv, crs)
    projected = gpd.GeoSeries(
        gpd.points_from_xy(detections["longitude"], detections["latitude"]),
        crs="EPSG:4326",
    ).to_crs(crs)
    detections = detections.copy()
    detections["x"] = projected.x.to_numpy()
    detections["y"] = projected.y.to_numpy()
    detections["radius_m"] = footprint_radius_m(detections["scan"], detections["track"])
    assigned = np.searchsorted(times.to_numpy(), detections["dt"].to_numpy(), side="left")
    detections["pass_index"] = assigned
    used = detections[assigned < len(times)].copy()
    batches = []
    for index in range(len(times)):
        group = used[(used["pass_index"] == index) & (~used["trimmed"])]
        batches.append(list(zip(group["x"], group["y"], group["radius_m"], strict=True)))
    regions = accumulate_footprints(batches)
    return regions, used


def target_checks(regions, x: np.ndarray, y: np.ndarray, hulls=None) -> list[dict]:
    """Monotonicity, contour, sign, and area for each persistent burned region.

    A cell counts as unexpectedly unburned only when it was more than one cell
    inside the previous region and is more than one cell outside the next.
    Boundary cells can change sign from the distance transform without the
    polygon losing them.
    """
    pixel = cell_size_m(x, y)
    rows = []
    previous = None
    for index, geom in enumerate(regions):
        distance = signed_distance(geom, x, y)
        negative = distance < 0
        area_km2 = float(geom.area) / 1e6
        lost = 0
        if previous is not None:
            lost = int(np.count_nonzero((previous < -pixel) & (distance > pixel)))
        contours = zero_contours(distance, x, y)
        hull_km2 = None
        if hulls is not None:
            hull_km2 = float(hulls.iloc[index].area) / 1e6
        rows.append({
            "index": index,
            "area_km2": area_km2,
            "hull_km2": hull_km2,
            "area_change_km2": None if index == 0 else area_km2 - rows[index - 1]["area_km2"],
            "cells_clearly_unburned": lost,
            "zero_contour": bool(contours),
            "n_contours": len(contours),
            "sdf_min_m": float(distance.min()),
            "sdf_max_m": float(distance.max()),
            "inside_negative": bool(np.all(distance[negative] < 0)),
            "outside_positive": bool(np.all(distance[~negative] >= 0)),
            "negative_cells": int(negative.sum()),
        })
        previous = distance
    return rows


def write_target_review(
    firms_csv: Path,
    perimeters,
    x: np.ndarray,
    y: np.ndarray,
    out_dir: Path,
    passes: tuple[int, ...] = (0, 1, 2, 3, 9, 10, 11, 12, 13),
) -> dict:
    """Write inspection figures and a check table. Does not train."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    crs = str(perimeters.crs)
    regions, detections = persistent_burned_regions(firms_csv, perimeters["dt"], crs)
    checks = target_checks(regions, x, y, perimeters.geometry)
    (out_dir / "checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")

    final = regions[-1].bounds
    margin = 3000.0
    extent = (final[0] - margin, final[2] + margin, final[1] - margin, final[3] + margin)

    for index in passes:
        distance = signed_distance(regions[index], x, y)
        _write_pass_figure(
            out_dir / f"pass_{index:02d}.png",
            index,
            perimeters,
            detections,
            regions,
            distance,
            x,
            y,
            extent,
            plt,
            Line2D,
        )

    _write_area_figure(out_dir / "area_comparison.png", perimeters, checks, plt)
    return {
        "passes": len(regions),
        "area_km2": [row["area_km2"] for row in checks],
        "hull_km2": [row["hull_km2"] for row in checks],
        "cells_clearly_unburned": [row["cells_clearly_unburned"] for row in checks],
        "zero_contour": [row["zero_contour"] for row in checks],
        "sdf_negative_inside": all(row["inside_negative"] and row["outside_positive"] and row["sdf_min_m"] < 0 and row["sdf_max_m"] > 0 for row in checks),
    }


def _write_pass_figure(path, index, perimeters, detections, regions, distance, x, y, extent, plt, Line2D):
    stamp = pd.Timestamp(perimeters["dt"].iloc[index]).strftime("%Y-%m-%d %H:%M")
    this_pass = detections[detections["pass_index"] == index]
    accepted = this_pass[~this_pass["trimmed"]]
    rejected = this_pass[this_pass["trimmed"]]
    # The domain is tens of kilometers across, so an unclipped scale hides the
    # negative interior. Display is capped at ±5 km; the stored field is not.
    limit = 5000.0
    fig, axes = plt.subplots(1, 5, figsize=(16.5, 4.0), layout="constrained")
    panels = (
        (axes[0], "Hotspots added by this pass"),
        (axes[1], "Burned estimate, previous pass"),
        (axes[2], "Burned estimate through this pass"),
        (axes[3], "Signed distance of that estimate"),
        (axes[4], "Zero contour"),
    )
    for axis, title in panels:
        axis.set_title(title, fontsize=9)
        axis.set_xlim(extent[0], extent[1])
        axis.set_ylim(extent[2], extent[3])
        axis.set_aspect("equal")
        axis.set_xticks([])
        axis.set_yticks([])
    if len(accepted):
        axes[0].scatter(accepted["x"], accepted["y"], s=6, c="black", linewidths=0, label="Accepted")
    if len(rejected):
        axes[0].scatter(rejected["x"], rejected["y"], s=12, c="0.6", marker="x", linewidths=0.6, label="Outlier, excluded")
    axes[0].legend(loc="upper right", fontsize=7, frameon=False)
    if index == 0:
        axes[1].text(0.5, 0.5, "No earlier pass", ha="center", va="center", transform=axes[1].transAxes, fontsize=8)
    else:
        _fill(axes[1], regions[index - 1], "0.75")
    _fill(axes[2], regions[index], "tab:orange")
    hull = perimeters.geometry.iloc[index]
    _outline(axes[2], hull, "tab:blue")
    axes[2].legend(
        handles=[
            Line2D([0], [0], color="tab:orange", lw=4, label="Persistent estimate"),
            Line2D([0], [0], color="tab:blue", lw=1, label="Old hotspot hull"),
        ],
        loc="upper right",
        fontsize=7,
        frameon=False,
    )
    field, east, north = _north_up(distance, x, y)
    mesh = axes[3].pcolormesh(east, north, field, cmap="RdBu", vmin=-limit, vmax=limit, shading="auto")
    fig.colorbar(mesh, ax=axes[3], fraction=0.046, pad=0.04, label="phi target (m)")
    _fill(axes[4], regions[index], "0.85")
    _contours(axes[4], distance, x, y)
    fig.suptitle(f"Kincade pass {index}  {stamp}", fontsize=11)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _write_area_figure(path, perimeters, checks, plt):
    hours = (pd.to_datetime(perimeters["dt"]) - pd.Timestamp(perimeters["dt"].iloc[0])).dt.total_seconds() / 3600
    fig, ax = plt.subplots(figsize=(7.2, 4.2), layout="constrained")
    ax.plot(hours, [row["hull_km2"] for row in checks], marker="o", ms=3, label="Hotspot hull this pass")
    ax.plot(hours, [row["area_km2"] for row in checks], marker="o", ms=3, label="Persistent burned estimate")
    ax.set_xlabel("Hours since first pass")
    ax.set_ylabel("Area (km²)")
    ax.set_title("Kincade target area")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _fill(axis, geom, color):
    geoms = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
    for part in geoms:
        if part.geom_type != "Polygon":
            continue
        coords = np.asarray(part.exterior.coords)
        axis.fill(coords[:, 0], coords[:, 1], color=color, alpha=0.85, lw=0)


def _outline(axis, geom, color):
    geoms = [geom] if geom.geom_type == "Polygon" else list(getattr(geom, "geoms", []))
    for part in geoms:
        if part.geom_type != "Polygon":
            continue
        coords = np.asarray(part.exterior.coords)
        axis.plot(coords[:, 0], coords[:, 1], color=color, lw=0.8)


def _contours(axis, distance, x, y):
    for line in zero_contours(distance, x, y):
        axis.plot(line[:, 0], line[:, 1], color="black", lw=0.8)


def _north_up(field, x, y):
    field = np.asarray(field)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x[0] > x[-1]:
        x = x[::-1]
        field = field[:, ::-1]
    if y[0] > y[-1]:
        y = y[::-1]
        field = field[::-1, :]
    return field, x, y
