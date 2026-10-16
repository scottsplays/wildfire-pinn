"""Consecutive persistent training masks, passes 0 through 11.

No training. The stored targets are only read. Each pair reports how many
grid cells changed, whether the masks are identical, and how far the
boundary moved, together with the new VIIRS footprints assigned to the
later pass.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from shapely import contains_xy, distance, points
from shapely.geometry import Point

from run_final_candidate_audit import _parts, _sample_lines
from run_persistent_target_experiment import _kincade_config
from wildfire_pinn.burned_region import persistent_burned_regions
from wildfire_pinn.geometry import cell_size_m, signed_distance
from wildfire_pinn.pinn import RealConfig, domain_scales, load_fire_case, resolve_time_split

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "kincade_mask_intervals"
TRAIN = list(range(12))


def _boundary_stats(earlier, later, spacing_m: float) -> dict:
    earlier_pts = _sample_lines(_parts(earlier.boundary), spacing_m)
    later_pts = _sample_lines(_parts(later.boundary), spacing_m)
    forward = np.asarray(distance(points(earlier_pts), later.boundary), dtype=np.float64)
    backward = np.asarray(distance(points(later_pts), earlier.boundary), dtype=np.float64)

    def pack(values: np.ndarray) -> dict:
        if values.size == 0:
            return {"n": 0}
        pct = np.percentile(values, [50, 90, 95])
        return {
            "n": int(values.size),
            "mean_m": float(values.mean()),
            "median_m": float(pct[0]),
            "p90_m": float(pct[1]),
            "p95_m": float(pct[2]),
            "max_m": float(values.max()),
            "fraction_within_one_cell": float(np.mean(values <= spacing_m)),
            "fraction_within_500m": float(np.mean(values <= 500.0)),
        }

    return {
        "earlier_boundary_to_later": pack(forward),
        "later_boundary_to_earlier": pack(backward),
    }


def _new_footprints(used, earlier, later_index: int) -> dict:
    group = used[(used["pass_index"] == later_index) & (~used["trimmed"])]
    trimmed = used[(used["pass_index"] == later_index) & (used["trimmed"])]
    added_area = 0.0
    disks_that_extend = 0
    centers_inside = 0
    for x_coord, y_coord, radius in zip(group["x"], group["y"], group["radius_m"], strict=True):
        center = Point(float(x_coord), float(y_coord))
        if earlier.covers(center):
            centers_inside += 1
        extra = center.buffer(float(radius), quad_segs=8).difference(earlier)
        if extra.area > 1.0:
            disks_that_extend += 1
            added_area += float(extra.area)
    return {
        "new_detections": int(len(group)),
        "trimmed_detections": int(len(trimmed)),
        "centers_already_inside": centers_inside,
        "disks_that_extend_the_region": disks_that_extend,
        "added_disk_area_km2": added_area / 1e6,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = _kincade_config()
    case = load_fire_case(cfg.stack_nc, cfg.rate_nc, cfg.perimeters_path)
    regions, used = persistent_burned_regions(
        cfg.firms_csv, case.perimeters["dt"], crs=str(case.perimeters.crs)
    )
    split = resolve_time_split(
        len(regions), RealConfig().k_train, cfg.split_scheme, cfg.train_fraction, cfg.val_fraction
    )
    if split.train != TRAIN:
        raise RuntimeError(f"Expected training passes 0–11, found {split.train}")
    x = np.asarray(case.x, dtype=np.float64)
    y = np.asarray(case.y, dtype=np.float64)
    spacing = cell_size_m(x, y)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    cell_km2 = (spacing ** 2) / 1e6
    n_cells = int(xx.size)
    fields = []
    masks = []
    for index in TRAIN:
        distance_m = signed_distance(regions[index], x, y)
        fields.append(distance_m)
        masks.append(distance_m < 0)
        print(f"pass {index} inside {int(masks[-1].sum())} cells", flush=True)

    rows = []
    for local in range(len(TRAIN) - 1):
        left = TRAIN[local]
        right = TRAIN[local + 1]
        earlier = masks[local]
        later = masks[local + 1]
        added = ~earlier & later
        removed = earlier & ~later
        changed = added | removed
        intersection = int(np.count_nonzero(earlier & later))
        union = int(np.count_nonzero(earlier | later))
        dt_hours = float(
            (case.perimeters["dt"].iloc[right] - case.perimeters["dt"].iloc[left]).total_seconds()
        ) / 3600.0
        near = np.abs(fields[local]) <= 500.0
        unchanged_near = near & (fields[local] == fields[local + 1])
        footprints = _new_footprints(used, regions[left], right)
        boundary = _boundary_stats(regions[left], regions[right], spacing)
        row = {
            "passes": [left, right],
            "start": str(case.perimeters["dt"].iloc[left]),
            "end": str(case.perimeters["dt"].iloc[right]),
            "elapsed_hours": dt_hours,
            "polygon_equal": bool(regions[left].equals(regions[right])),
            "raster_mask_equal": bool(np.array_equal(earlier, later)),
            "sdf_equal": bool(np.array_equal(fields[local], fields[local + 1])),
            "changed_cells": int(np.count_nonzero(changed)),
            "changed_percent_of_domain": 100.0 * float(np.count_nonzero(changed)) / n_cells,
            "changed_percent_of_union": 100.0 * float(np.count_nonzero(changed)) / union if union else None,
            "cells_added": int(np.count_nonzero(added)),
            "cells_removed": int(np.count_nonzero(removed)),
            "intersection_cells": intersection,
            "union_cells": union,
            "iou": intersection / union if union else None,
            "area_earlier_km2": float(regions[left].area) / 1e6,
            "area_later_km2": float(regions[right].area) / 1e6,
            "area_change_km2": float(regions[right].area - regions[left].area) / 1e6,
            "raster_area_change_km2": float(np.count_nonzero(added) - np.count_nonzero(removed)) * cell_km2,
            "near_front_cells": int(np.count_nonzero(near)),
            "near_front_sdf_unchanged_fraction": (
                float(np.count_nonzero(unchanged_near) / np.count_nonzero(near)) if np.any(near) else None
            ),
            "boundary": boundary,
            "footprints": footprints,
        }
        rows.append(row)
        stationary = boundary["earlier_boundary_to_later"]["fraction_within_one_cell"]
        print(
            f"{left}->{right} {dt_hours:.2f} h | changed {row['changed_cells']} "
            f"({row['changed_percent_of_domain']:.3f}% domain) | "
            f"IoU {row['iou']:.4f} | dA {row['area_change_km2']:.2f} km2 | "
            f"removed {row['cells_removed']} | identical {row['raster_mask_equal']} | "
            f"boundary within 1 cell {100 * stationary:.1f}% | "
            f"new detections {footprints['new_detections']} "
            f"extending {footprints['disks_that_extend_the_region']} | "
            f"front SDF unchanged {100 * row['near_front_sdf_unchanged_fraction']:.1f}%",
            flush=True,
        )
    payload = {
        "cell_m": spacing,
        "domain_cells": n_cells,
        "construction": (
            "Each pass is the cumulative union of accepted VIIRS pixel disks "
            "assigned to that pass or an earlier pass. Trimmed spatial outliers "
            "are excluded. An empty later batch keeps the previous region."
        ),
        "intervals": rows,
        "any_raster_mask_identical": any(row["raster_mask_equal"] for row in rows),
        "any_polygon_identical": any(row["polygon_equal"] for row in rows),
    }
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
