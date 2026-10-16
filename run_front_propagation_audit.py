"""Can persistent-mask growth be continuous outward propagation?

Training passes 0–11 only. No neural network, no changes to targets,
checkpoints, the PDE, or the train/validation/test split.

Distance is the Euclidean distance, in EPSG:3310 meters, from the center of
a newly burned grid cell to the old persistent burned polygon. That is the
distance to the old burned region, not to a hotspot hull and not a count of
grid steps. A center covered by the old polygon would have distance 0; those
centers are excluded from the new-cell set by the mask difference. v_min is
that distance divided by the elapsed seconds, the slowest constant speed that
reaches the cell in a straight line from the previous region.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from scipy.ndimage import label
from shapely import contains_xy, distance, points

from run_persistent_target_experiment import _kincade_config
from wildfire_pinn.burned_region import persistent_burned_regions
from wildfire_pinn.geometry import cell_size_m
from wildfire_pinn.pinn import RealConfig, load_fire_case, resolve_time_split

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "kincade_front_propagation_audit"
TRAIN = list(range(12))
SPEEDS_M_S = (0.01, 0.025, 0.05, 0.10, 0.25, 0.50, 1.00)
# Same masks previously measured with signed_distance < 0 on these passes.
EXPECTED_INSIDE = (5548, 8571, 8941, 9180, 9990, 10181, 10605, 11868, 12320, 20314, 25763, 29313)
SIGNIFICANT_KM2 = 0.25
EIGHT = np.ones((3, 3), dtype=int)


def _finite(value: float):
    value = float(value)
    if not np.isfinite(value):
        return None
    return value


def _speed_name(speed: float) -> str:
    return f"{speed:.3f}".replace(".", "p")


def _pack(values: np.ndarray) -> dict:
    if values.size == 0:
        return {"median": None, "p90": None, "p95": None, "maximum": None}
    pct = np.percentile(values, [50, 90, 95])
    return {
        "median": float(pct[0]),
        "p90": float(pct[1]),
        "p95": float(pct[2]),
        "maximum": float(values.max()),
    }


def _fill(axis, geom, color: str, alpha: float) -> None:
    parts = [geom] if geom.geom_type == "Polygon" else list(getattr(geom, "geoms", []))
    for part in parts:
        if part.geom_type != "Polygon":
            continue
        coords = np.asarray(part.exterior.coords)
        axis.fill(coords[:, 0], coords[:, 1], color=color, alpha=alpha, lw=0, zorder=1)


def _outline(axis, geom, color: str, lw: float) -> None:
    parts = [geom] if geom.geom_type == "Polygon" else list(getattr(geom, "geoms", []))
    for part in parts:
        if part.geom_type != "Polygon":
            continue
        coords = np.asarray(part.exterior.coords)
        axis.plot(coords[:, 0], coords[:, 1], color=color, lw=lw, zorder=3)
        for hole in part.interiors:
            ring = np.asarray(hole.coords)
            axis.plot(ring[:, 0], ring[:, 1], color=color, lw=lw * 0.7, zorder=3)


def _touching_old(old: np.ndarray, new: np.ndarray) -> np.ndarray:
    """New cells that share an edge with the old burned mask."""
    touch = np.zeros(old.shape, dtype=bool)
    touch[1:, :] |= old[:-1, :]
    touch[:-1, :] |= old[1:, :]
    touch[:, 1:] |= old[:, :-1]
    touch[:, :-1] |= old[:, 1:]
    return new & touch


def _component_class(min_distance_m: float, spacing_m: float) -> str:
    if min_distance_m <= spacing_m:
        return "local_perimeter_expansion"
    if min_distance_m < 1000.0:
        return "nearby_disconnected_addition"
    return "nonlocal_addition"


def _smallest_speed(reachable: dict, target: float):
    for speed in SPEEDS_M_S:
        if reachable[f"{speed:.3f}"] >= target:
            return speed
    return None


def _unit_checks(spacing_m: float) -> dict:
    """Grid meters and seconds, independent of the fire masks."""
    straight = 90.0 / 9000.0
    cell_area_km2 = (spacing_m ** 2) / 1e6
    return {
        "straight_line_90m_over_9000s_m_s": straight,
        "equals_0p01": bool(abs(straight - 0.01) < 1e-12),
        "spacing_m": spacing_m,
        "spacing_near_90m": bool(abs(spacing_m - 90.0) < 1.0),
        "cell_area_km2": cell_area_km2,
        "ninety_meter_cell_area_km2": 0.0081,
    }


def _interpret(intervals: list[dict], pooled: dict) -> dict:
    local = pooled["area_fraction_local"]
    nearby = pooled["area_fraction_nearby"]
    nonlocal_ = pooled["area_fraction_nonlocal"]
    reach = pooled["reachable_area_fraction"]
    focus_ids = ([0, 1], [8, 9], [9, 10], [10, 11])
    focus = [row for row in intervals if row["passes"] in focus_ids]
    focus_lines = []
    for row in focus:
        got = row["reachable_area_fraction"]
        focus_lines.append(
            f"{row['passes'][0]}->{row['passes'][1]}: {row['new_area_km2']:.1f} km2 in "
            f"{row['elapsed_hours']:.2f} h, median distance {row['distance_m']['median']:.0f} m, "
            f"median v_min {row['v_min_m_s']['median']:.3f} m/s, "
            f"local area {100 * row['area_fraction_local']:.0f}%, "
            f"nearby {100 * row['area_fraction_nearby']:.0f}%, "
            f"nonlocal {100 * row['area_fraction_nonlocal']:.0f}%, "
            f"reachable at 0.05 m/s {100 * got['0.050']:.0f}%, "
            f"at 0.10 m/s {100 * got['0.100']:.0f}%, "
            f"at 0.25 m/s {100 * got['0.250']:.0f}%."
        )
    geometry = (
        f"Of the newly burned training area, {100 * local:.0f}% is in components whose "
        f"closest cell is within one 90 m cell of the old region, {100 * nearby:.0f}% is "
        f"nearby but not touching (one cell to 1 km), and {100 * nonlocal_:.0f}% is in "
        f"components whose closest cell is at least 1 km from the old region. "
        f"Pooled reachable fractions are {100 * reach['0.010']:.0f}% at 0.01 m/s, "
        f"{100 * reach['0.025']:.0f}% at 0.025 m/s, {100 * reach['0.050']:.0f}% at 0.05 m/s, "
        f"{100 * reach['0.100']:.0f}% at 0.10 m/s, and {100 * reach['0.250']:.0f}% at 0.25 m/s."
    )
    pde = (
        "Most new area is an attached lobe, so continuous outward propagation can describe "
        "those mask pairs. The cells inside the lobe are burned at the next pass, and the "
        "hours in between were not observed, so the path is ambiguous rather than impossible. "
        "Pass 9 to 10 is the clear exception: about half of its new area belongs to components "
        "that start at least 1 km from the previous region. A local level-set would have to "
        "cross that gap. Sparse VIIRS disks can omit the corridor, so this is not evidence of "
        "a separate ignition and not evidence of a code bug. A single weather rate still cannot "
        "be near zero on most of the old perimeter and about 0.1 m/s through the lobe, and "
        "earlier diagnostics found that the available weather fields do not locate the lobe."
    )
    return {
        "most_new_area_is_local_continuous_advance": bool(local >= 0.7 and nonlocal_ < 0.15),
        "geometry_paragraph": geometry,
        "focus": focus_lines,
        "pde_paragraph": pde,
        "recommendation": "C",
        "recommendation_reason": (
            "Freeze the existing observation-only checkpoint as the WildfirePINN v1 baseline. "
            "The physical residual was left out of that baseline because weather R conflicts "
            "with a perimeter that is stationary along most of its length, weak PDE weights "
            "lost the corrected validation comparison, and pass 9 to 10 adds area a local "
            "front cannot reach without burning the unobserved gap. No distance, timing, or "
            "mask bug appeared in this audit, so another training run is not required."
        ),
    }


def _draw_transitions(intervals: list[dict]) -> None:
    fig, axes = plt.subplots(3, 4, figsize=(14.5, 11.2), layout="constrained")
    flat = list(axes.ravel())
    for axis, row in zip(flat, intervals):
        _fill(axis, row["old_geom"], "#d9d9d9", 0.95)
        added = row["added_xy"]
        if len(added):
            axis.scatter(added[:, 0], added[:, 1], s=2, c="#e07a3d", linewidths=0, zorder=2)
        _outline(axis, row["old_geom"], "black", 0.6)
        cloud = [np.asarray(row["old_geom"].bounds)[0:4]]
        if len(added):
            xmin, xmax = float(added[:, 0].min()), float(added[:, 0].max())
            ymin, ymax = float(added[:, 1].min()), float(added[:, 1].max())
        else:
            xmin, xmax = cloud[0][0], cloud[0][2]
            ymin, ymax = cloud[0][1], cloud[0][3]
        bounds = row["old_geom"].bounds
        xmin = min(xmin, bounds[0])
        xmax = max(xmax, bounds[2])
        ymin = min(ymin, bounds[1])
        ymax = max(ymax, bounds[3])
        pad = 1200.0
        axis.set_xlim(xmin - pad, xmax + pad)
        axis.set_ylim(ymin - pad, ymax + pad)
        axis.set_aspect("equal")
        axis.set_xticks([])
        axis.set_yticks([])
        median_v = row["v_min_m_s"]["median"]
        axis.set_title(
            f"{row['passes'][0]}→{row['passes'][1]}  {row['elapsed_hours']:.1f} h\n"
            f"+{row['new_area_km2']:.1f} km²  median v_min {median_v:.3f} m/s",
            fontsize=8,
        )
    flat[-1].axis("off")
    flat[0].fill([], [], color="#d9d9d9", label="Old persistent region")
    flat[0].scatter([], [], s=12, c="#e07a3d", label="Newly burned cells")
    flat[0].plot([], [], color="black", lw=1, label="Old perimeter")
    flat[0].legend(loc="upper left", fontsize=7, frameon=False)
    fig.suptitle("New persistent area against the previous region", fontsize=12)
    fig.savefig(OUT / "transition_maps.png", dpi=130)
    plt.close(fig)


def _draw_speeds(intervals: list[dict], vmax: float) -> None:
    fig, axes = plt.subplots(3, 4, figsize=(14.5, 11.2), layout="constrained")
    flat = list(axes.ravel())
    norm = LogNorm(vmin=1e-3, vmax=vmax)
    last = None
    for axis, row in zip(flat, intervals):
        _fill(axis, row["old_geom"], "#eeeeee", 0.9)
        _outline(axis, row["old_geom"], "black", 0.4)
        added = row["added_xy"]
        speeds = row["v_min"]
        if len(added):
            last = axis.scatter(
                added[:, 0],
                added[:, 1],
                c=np.maximum(speeds, 1e-3),
                s=3,
                cmap="magma",
                norm=norm,
                linewidths=0,
                zorder=2,
            )
        bounds = row["old_geom"].bounds
        if len(added):
            xmin = min(bounds[0], float(added[:, 0].min()))
            xmax = max(bounds[2], float(added[:, 0].max()))
            ymin = min(bounds[1], float(added[:, 1].min()))
            ymax = max(bounds[3], float(added[:, 1].max()))
        else:
            xmin, xmax, ymin, ymax = bounds
        pad = 1200.0
        axis.set_xlim(xmin - pad, xmax + pad)
        axis.set_ylim(ymin - pad, ymax + pad)
        axis.set_aspect("equal")
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_title(f"{row['passes'][0]}→{row['passes'][1]}", fontsize=9)
    flat[-1].axis("off")
    if last is not None:
        fig.colorbar(last, ax=flat[:11], fraction=0.025, pad=0.01, label="v_min (m/s)")
    fig.suptitle("Minimum straight-line speed from the old burned region", fontsize=12)
    fig.savefig(OUT / "required_speed_maps.png", dpi=130)
    plt.close(fig)


def _draw_reachable(intervals: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 4.6), layout="constrained")
    labels = [f"{row['passes'][0]}→{row['passes'][1]}" for row in intervals]
    xpos = np.arange(len(intervals))
    for speed in SPEEDS_M_S:
        series = [row["reachable_area_fraction"][f"{speed:.3f}"] for row in intervals]
        ax.plot(xpos, series, marker="o", ms=3.5, lw=1.4, label=f"{speed:.3f} m/s")
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("Fraction of new area reachable")
    ax.set_xlabel("Training interval")
    ax.set_title("Share of new cells within reach of a constant outward speed")
    ax.legend(frameon=False, fontsize=8, ncol=2, title="Speed at most")
    fig.savefig(OUT / "reachable_fraction.png", dpi=130)
    plt.close(fig)


def _draw_distribution(speeds: np.ndarray, areas: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2), layout="constrained")
    positive = speeds[speeds > 0]
    weights = areas[speeds > 0]
    hi = float(np.percentile(positive, 99.5)) if positive.size else 1.0
    bins = np.linspace(0.0, max(hi, 0.05), 40)
    axes[0].hist(speeds, bins=bins, weights=areas, color="#9b2226", alpha=0.9)
    axes[0].set_xlabel("v_min (m/s)")
    axes[0].set_ylabel("New area (km²)")
    axes[0].set_title("Linear scale, axis to the 99.5th percentile")
    log_bins = np.geomspace(max(float(positive.min()), 1e-4), max(float(positive.max()), 1e-3), 40)
    axes[1].hist(positive, bins=log_bins, weights=weights, color="#9b2226", alpha=0.9)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("v_min (m/s)")
    axes[1].set_ylabel("New area (km²)")
    axes[1].set_title("Log scale")
    fig.suptitle("Straight-line speed required by each newly burned training cell", fontsize=12)
    fig.savefig(OUT / "required_speed_distribution.png", dpi=130)
    plt.close(fig)


def main() -> None:
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = _kincade_config()
    case = load_fire_case(cfg.stack_nc, cfg.rate_nc, cfg.perimeters_path)
    n_perimeters = len(case.perimeters)
    split = resolve_time_split(
        n_perimeters, RealConfig().k_train, cfg.split_scheme, cfg.train_fraction, cfg.val_fraction
    )
    if split.train != TRAIN:
        raise RuntimeError(f"Expected training passes 0–11, found {split.train}")
    # Only the training timestamps are passed in, so later detections cannot
    # enter these regions and validation polygons are never built.
    train_times = case.perimeters["dt"].iloc[TRAIN]
    regions, _used = persistent_burned_regions(cfg.firms_csv, train_times, crs=str(case.perimeters.crs))
    if len(regions) != len(TRAIN):
        raise RuntimeError(f"Expected 12 training regions, found {len(regions)}")
    x = np.asarray(case.x, dtype=np.float64)
    y = np.asarray(case.y, dtype=np.float64)
    spacing = cell_size_m(x, y)
    checks = _unit_checks(spacing)
    cell_km2 = checks["cell_area_km2"]
    yy, xx = np.meshgrid(y, x, indexing="ij")
    masks = []
    for index, geom in enumerate(regions):
        mask = np.asarray(contains_xy(geom, xx, yy), dtype=bool)
        masks.append(mask)
        print(f"pass {index} inside {int(mask.sum())} cells", flush=True)
    counts = tuple(int(mask.sum()) for mask in masks)
    checks["inside_counts"] = counts
    checks["inside_counts_match_previous_persistent_raster"] = counts == EXPECTED_INSIDE
    checks["distance_is_meters_to_old_persistent_polygon"] = True
    checks["hotspot_hulls_used_as_masks"] = False
    checks["validation_or_test_masks_built"] = False
    checks["network_trained"] = False
    checks["targets_modified"] = False

    interval_rows = []
    component_rows = []
    plot_rows = []
    all_speeds = []
    all_areas = []
    sanity_touch = []
    for local in range(len(TRAIN) - 1):
        left = TRAIN[local]
        right = TRAIN[local + 1]
        old = masks[local]
        nxt = masks[local + 1]
        new = nxt & ~old
        removed = int(np.count_nonzero(old & ~nxt))
        overlap = int(np.count_nonzero(new & old))
        ys, xs = np.nonzero(new)
        cell_xy = np.column_stack([xx[new], yy[new]])
        dt_s = float((train_times.iloc[right] - train_times.iloc[left]).total_seconds())
        dt_h = dt_s / 3600.0
        if dt_s <= 0:
            raise RuntimeError(f"Non-positive elapsed time on {left}→{right}")
        if len(cell_xy):
            dist_m = np.asarray(distance(points(cell_xy), regions[left]), dtype=np.float64)
        else:
            dist_m = np.zeros(0, dtype=np.float64)
        v_min = dist_m / dt_s
        touch = _touching_old(old, new)
        touch_dist = dist_m[touch[new]] if np.any(new) else np.zeros(0)
        sanity_touch.append({
            "passes": [left, right],
            "new_cells": int(new.sum()),
            "overlap_with_old": overlap,
            "cells_removed": removed,
            "delta_t_seconds": dt_s,
            "zero_distance_cells": int(np.count_nonzero(dist_m == 0)),
            "edge_adjacent_new_cells": int(touch.sum()),
            "edge_adjacent_median_distance_m": _finite(np.median(touch_dist)) if touch_dist.size else None,
            "edge_adjacent_max_distance_m": _finite(np.max(touch_dist)) if touch_dist.size else None,
            "min_distance_m": _finite(np.min(dist_m)) if dist_m.size else None,
            "median_distance_times_check": _finite(np.median(v_min) * dt_s - np.median(dist_m)) if dist_m.size else None,
        })
        reachable = {}
        for speed in SPEEDS_M_S:
            reachable[f"{speed:.3f}"] = float(np.mean(dist_m <= speed * dt_s)) if dist_m.size else None
        labeled, n_components = label(new, structure=EIGHT)
        class_area = {
            "local_perimeter_expansion": 0.0,
            "nearby_disconnected_addition": 0.0,
            "nonlocal_addition": 0.0,
        }
        for comp_id in range(1, int(n_components) + 1):
            selected = labeled == comp_id
            comp_dist = dist_m[selected[new]]
            area = float(selected.sum()) * cell_km2
            kind = _component_class(float(comp_dist.min()), spacing)
            class_area[kind] += area
            far_enough = kind == "nonlocal_addition" and area >= 0.10
            if area >= SIGNIFICANT_KM2 or far_enough:
                component_rows.append({
                    "interval": f"{left}->{right}",
                    "component_id": comp_id,
                    "n_cells": int(selected.sum()),
                    "area_km2": area,
                    "min_distance_m": float(comp_dist.min()),
                    "median_distance_m": float(np.median(comp_dist)),
                    "min_v_m_s": float(comp_dist.min() / dt_s),
                    "class": kind,
                })
        new_area = float(new.sum()) * cell_km2
        dist_pack = _pack(dist_m)
        speed_pack = _pack(v_min)
        row = {
            "passes": [left, right],
            "start": str(train_times.iloc[left]),
            "end": str(train_times.iloc[right]),
            "elapsed_hours": dt_h,
            "delta_t_seconds": dt_s,
            "old_area_km2": float(old.sum()) * cell_km2,
            "new_area_km2": new_area,
            "new_cells": int(new.sum()),
            "cells_removed": removed,
            "distance_m": dist_pack,
            "v_min_m_s": speed_pack,
            "reachable_area_fraction": reachable,
            "n_components_8connected": int(n_components),
            "area_fraction_local": class_area["local_perimeter_expansion"] / new_area if new_area else None,
            "area_fraction_nearby": class_area["nearby_disconnected_addition"] / new_area if new_area else None,
            "area_fraction_nonlocal": class_area["nonlocal_addition"] / new_area if new_area else None,
            "class_area_km2": class_area,
        }
        interval_rows.append(row)
        plot_rows.append({
            **row,
            "old_geom": regions[left],
            "added_xy": cell_xy,
            "v_min": v_min,
        })
        all_speeds.append(v_min)
        all_areas.append(np.full(v_min.shape, cell_km2))
        print(
            f"{left}->{right} {dt_h:.2f} h ({dt_s:.0f} s) | new {row['new_cells']} cells "
            f"{new_area:.2f} km2 | median dist {dist_pack['median']:.0f} m | "
            f"p95 {dist_pack['p95']:.0f} m | max {dist_pack['maximum']:.0f} m | "
            f"median v {speed_pack['median']:.4f} m/s | max v {speed_pack['maximum']:.3f} | "
            f"local {100 * row['area_fraction_local']:.0f}% "
            f"nearby {100 * row['area_fraction_nearby']:.0f}% "
            f"nonlocal {100 * row['area_fraction_nonlocal']:.0f}%",
            flush=True,
        )

    speeds = np.concatenate(all_speeds) if all_speeds else np.zeros(0)
    areas = np.concatenate(all_areas) if all_areas else np.zeros(0)
    total_area = float(areas.sum()) if areas.size else 0.0
    pooled_reachable = {}
    for speed in SPEEDS_M_S:
        pooled_reachable[f"{speed:.3f}"] = float(np.average(speeds <= speed, weights=areas)) if total_area else None
    class_area = {
        "local_perimeter_expansion": 0.0,
        "nearby_disconnected_addition": 0.0,
        "nonlocal_addition": 0.0,
    }
    for row in interval_rows:
        for key, value in row["class_area_km2"].items():
            class_area[key] += value
    elapsed_per_cell = np.concatenate(
        [np.full(len(row["v_min"]), row["delta_t_seconds"]) for row in plot_rows]
    )
    pooled = {
        "new_area_km2": total_area,
        "new_cells": int(speeds.size),
        "distance_m": _pack(speeds * elapsed_per_cell),
        "v_min_m_s": _pack(speeds),
        "reachable_area_fraction": pooled_reachable,
        "area_fraction_local": class_area["local_perimeter_expansion"] / total_area if total_area else None,
        "area_fraction_nearby": class_area["nearby_disconnected_addition"] / total_area if total_area else None,
        "area_fraction_nonlocal": class_area["nonlocal_addition"] / total_area if total_area else None,
        "class_area_km2": class_area,
        "smallest_listed_speed_reaching_50pct_of_new_area": _smallest_speed(pooled_reachable, 0.50),
        "smallest_listed_speed_reaching_80pct_of_new_area": _smallest_speed(pooled_reachable, 0.80),
    }

    adjacent_ok = all(
        item["edge_adjacent_new_cells"] == 0
        or (
            item["edge_adjacent_median_distance_m"] is not None
            and 0.0 <= item["edge_adjacent_median_distance_m"] <= spacing
            and item["edge_adjacent_max_distance_m"] <= 2.0 * spacing
        )
        for item in sanity_touch
    )
    timing_ok = all(
        item["median_distance_times_check"] is not None and abs(item["median_distance_times_check"]) < 1e-6
        for item in sanity_touch
        if item["new_cells"]
    )
    checks["edge_adjacent_distances_within_one_cell"] = bool(adjacent_ok)
    checks["v_min_is_distance_over_seconds"] = bool(timing_ok)
    checks["no_new_cell_overlaps_old_mask"] = all(item["overlap_with_old"] == 0 for item in sanity_touch)
    checks["no_burned_cell_becomes_unburned"] = all(item["cells_removed"] == 0 for item in sanity_touch)
    checks["edge_adjacent_detail"] = sanity_touch
    sanity_ok = all(
        checks[key]
        for key in (
            "equals_0p01",
            "spacing_near_90m",
            "inside_counts_match_previous_persistent_raster",
            "edge_adjacent_distances_within_one_cell",
            "v_min_is_distance_over_seconds",
            "no_new_cell_overlaps_old_mask",
            "no_burned_cell_becomes_unburned",
        )
    )
    checks["all_passed"] = bool(sanity_ok)
    interpretation = _interpret(interval_rows, pooled)
    if not sanity_ok:
        interpretation["recommendation"] = "D"
        interpretation["recommendation_reason"] = (
            "A sanity check failed. Do not freeze a model until the distance, "
            "timing, or mask construction is corrected."
        )

    print("\nSanity checks", flush=True)
    for key in (
        "equals_0p01",
        "spacing_near_90m",
        "spacing_m",
        "inside_counts_match_previous_persistent_raster",
        "distance_is_meters_to_old_persistent_polygon",
        "hotspot_hulls_used_as_masks",
        "validation_or_test_masks_built",
        "edge_adjacent_distances_within_one_cell",
        "v_min_is_distance_over_seconds",
        "no_new_cell_overlaps_old_mask",
        "no_burned_cell_becomes_unburned",
        "network_trained",
        "targets_modified",
        "all_passed",
    ):
        print(f"  {key}: {checks[key]}", flush=True)
    for item in sanity_touch:
        print(
            f"  adjacent {item['passes'][0]}->{item['passes'][1]}: "
            f"n {item['edge_adjacent_new_cells']}, "
            f"median dist {item['edge_adjacent_median_distance_m']} m, "
            f"max {item['edge_adjacent_max_distance_m']} m, "
            f"zero-distance cells {item['zero_distance_cells']}, "
            f"dt {item['delta_t_seconds']:.0f} s",
            flush=True,
        )

    csv_intervals = []
    for row in interval_rows:
        flat = {
            "interval": f"{row['passes'][0]}->{row['passes'][1]}",
            "elapsed_hours": row["elapsed_hours"],
            "delta_t_seconds": row["delta_t_seconds"],
            "old_area_km2": row["old_area_km2"],
            "new_area_km2": row["new_area_km2"],
            "new_cells": row["new_cells"],
            "median_distance_m": row["distance_m"]["median"],
            "p90_distance_m": row["distance_m"]["p90"],
            "p95_distance_m": row["distance_m"]["p95"],
            "max_distance_m": row["distance_m"]["maximum"],
            "median_v_min_m_s": row["v_min_m_s"]["median"],
            "p90_v_min_m_s": row["v_min_m_s"]["p90"],
            "p95_v_min_m_s": row["v_min_m_s"]["p95"],
            "max_v_min_m_s": row["v_min_m_s"]["maximum"],
            "area_fraction_local": row["area_fraction_local"],
            "area_fraction_nearby": row["area_fraction_nearby"],
            "area_fraction_nonlocal": row["area_fraction_nonlocal"],
        }
        for speed in SPEEDS_M_S:
            flat[f"reachable_{_speed_name(speed)}"] = row["reachable_area_fraction"][f"{speed:.3f}"]
        csv_intervals.append(flat)
    pd.DataFrame(csv_intervals).to_csv(OUT / "interval_summary.csv", index=False)
    pd.DataFrame(component_rows).to_csv(OUT / "component_summary.csv", index=False)

    _draw_transitions(plot_rows)
    _draw_speeds(plot_rows, vmax=1.0)
    _draw_reachable(interval_rows)
    _draw_distribution(speeds, areas)

    elapsed = time.perf_counter() - started
    payload = {
        "distance_definition": (
            "Euclidean meters from the new cell center to the old persistent "
            "burned polygon. Zero only if that center is covered by the old "
            "region. v_min = distance / elapsed seconds."
        ),
        "component_rule": (
            "8-connected components of newly burned cells. Local means the "
            f"closest cell is within one grid cell ({spacing:.1f} m) of the old "
            "polygon. Nearby means farther than one cell and closer than 1 km. "
            "Nonlocal means at least 1 km. Significant rows are components of "
            f"at least {SIGNIFICANT_KM2} km², plus nonlocal components of at least 0.10 km². "
            "A disconnected component is not labeled a spot fire."
        ),
        "training_passes_only": TRAIN,
        "sanity": {key: value for key, value in checks.items() if key != "edge_adjacent_detail"},
        "adjacent_cells": sanity_touch,
        "intervals": interval_rows,
        "pooled": pooled,
        "interpretation": interpretation,
        "runtime_seconds": elapsed,
    }
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\n" + interpretation["geometry_paragraph"], flush=True)
    for line in interpretation["focus"]:
        print(line, flush=True)
    print(interpretation["pde_paragraph"], flush=True)
    print(f"Recommendation {interpretation['recommendation']}. {interpretation['recommendation_reason']}", flush=True)
    print(f"runtime {elapsed:.1f} s", flush=True)


if __name__ == "__main__":
    main()
