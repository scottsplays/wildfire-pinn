"""Recompute held-out metrics from the three saved Kincade checkpoints.

No training. Every score uses the persistent burned-region geometry.
Predicted area, target area, IoU, and Dice share the model grid.
Contour distance is the symmetric mean of predicted-front to persistent
boundary and persistent boundary to predicted front.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
from shapely import contains_xy, distance, points
from shapely.geometry import LineString, MultiLineString

from run_persistent_target_experiment import _front_distance_m, _kincade_config
from wildfire_pinn.burned_region import persistent_burned_regions
from wildfire_pinn.geometry import cell_size_m, zero_contours
from wildfire_pinn.pinn import (
    LevelSetPINN,
    RealConfig,
    domain_scales,
    get_device,
    load_checkpoint,
    load_fire_case,
    predict_phi_meters,
    resolve_time_split,
)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "kincade_final_candidates"
CANDIDATES = (
    ("observation", "Observation only", 0.0),
    ("weak_pde_0p001", "Weak PDE 0.001", 0.001),
    ("weak_pde_0p01", "Weak PDE 0.01", 0.01),
)
HELD_OUT = ("validation", "test")


def _parts(geom):
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type == "LineString":
        return [geom]
    if geom.geom_type == "MultiLineString":
        return [part for part in geom.geoms if not part.is_empty]
    if hasattr(geom, "boundary"):
        return _parts(geom.boundary)
    return []


def _sample_lines(lines, spacing_m: float) -> np.ndarray:
    samples = []
    for line in lines:
        if line.length <= 0:
            continue
        count = max(8, int(np.ceil(line.length / spacing_m)))
        for place in np.linspace(0.0, line.length, count, endpoint=False):
            point = line.interpolate(float(place))
            samples.append((point.x, point.y))
    if not samples:
        return np.zeros((0, 2), dtype=np.float64)
    return np.asarray(samples, dtype=np.float64)


def _predicted_lines(contours):
    lines = [LineString(np.asarray(piece, dtype=np.float64)) for piece in contours if len(piece) >= 2]
    return lines


def _symmetric_distance_m(contours, target, spacing_m: float) -> float | None:
    """Chamfer mean, in meters, between the full predicted front and the persistent boundary."""
    predicted = _predicted_lines(contours)
    observed = _parts(target.boundary)
    if not predicted or not observed:
        return None
    predicted_geom = predicted[0] if len(predicted) == 1 else MultiLineString(predicted)
    observed_geom = target.boundary
    predicted_points = _sample_lines(predicted, spacing_m)
    observed_points = _sample_lines(observed, spacing_m)
    if len(predicted_points) == 0 or len(observed_points) == 0:
        return None
    forward = float(np.mean(distance(points(predicted_points), observed_geom)))
    backward = float(np.mean(distance(points(observed_points), predicted_geom)))
    if not np.isfinite(forward) or not np.isfinite(backward):
        return None
    return 0.5 * (forward + backward)


def _overlap(predicted_mask: np.ndarray, target_mask: np.ndarray) -> tuple[float, float]:
    intersection = int(np.count_nonzero(predicted_mask & target_mask))
    predicted_count = int(np.count_nonzero(predicted_mask))
    target_count = int(np.count_nonzero(target_mask))
    union = int(np.count_nonzero(predicted_mask | target_mask))
    iou = float(intersection / union) if union else 0.0
    denominator = predicted_count + target_count
    dice = float(2.0 * intersection / denominator) if denominator else 0.0
    return iou, dice


def _load_model(name: str) -> LevelSetPINN:
    hp = RealConfig()
    device = get_device()
    model = LevelSetPINN(hp.hidden, hp.layers).to(device)
    load_checkpoint(OUT / name / "levelset_pinn.pt", model, device)
    model.eval()
    return model


def _score_candidate(name, label, weight, case, regions, masks, scales, previous) -> dict:
    model = _load_model(name)
    cell_km2 = scales["cell_km2"]
    rows = []
    for index, geometry in enumerate(regions):
        split = scales["time_split"].label(index)
        if split not in HELD_OUT:
            continue
        phi_m = predict_phi_meters(model, case, float(scales["tau"][index]))
        contours = zero_contours(phi_m, case.x, case.y)
        predicted_mask = phi_m < 0
        target_mask = masks[index]
        predicted_km2 = float(predicted_mask.sum()) * cell_km2
        target_km2 = float(target_mask.sum()) * cell_km2
        polygon_km2 = float(geometry.area) / 1e6
        absolute = abs(predicted_km2 - target_km2)
        relative = absolute / target_km2 if target_km2 > 0 else None
        iou, dice = _overlap(predicted_mask, target_mask)
        one_way = _front_distance_m(contours, geometry.boundary)
        symmetric = _symmetric_distance_m(contours, geometry, scales["spacing_m"])
        prior = previous.get((name, index), {})
        rows.append({
            "candidate": label,
            "w_pde": weight,
            "index": index,
            "time": str(case.perimeters["dt"].iloc[index]),
            "split": split,
            "zero_contour": bool(contours),
            "predicted_km2": predicted_km2,
            "persistent_target_km2": target_km2,
            "persistent_polygon_km2": polygon_km2,
            "absolute_area_error_km2": absolute,
            "relative_area_error": relative,
            "symmetric_distance_m": symmetric,
            "previous_one_way_distance_m": one_way,
            "stored_target_km2": prior.get("target_km2"),
            "stored_distance_m": prior.get("distance_to_target_m"),
            "iou": iou,
            "dice": dice,
        })
        print(
            f"{label} k{index} {split} contour={bool(contours)} "
            f"area {predicted_km2:.1f}/{target_km2:.1f} "
            f"sym {symmetric if symmetric is None else round(symmetric)} m "
            f"IoU {iou:.3f}",
            flush=True,
        )
    del model
    return {
        "name": name,
        "label": label,
        "w_pde": weight,
        "passes": rows,
        "validation": _aggregate(rows, "validation"),
        "test": _aggregate(rows, "test"),
    }


def _aggregate(rows, split: str) -> dict:
    group = [row for row in rows if row["split"] == split]
    survivors = [row["symmetric_distance_m"] for row in group if row["symmetric_distance_m"] is not None]
    return {
        "n": len(group),
        "contour_survival": (
            sum(1 for row in group if row["zero_contour"]) / len(group) if group else None
        ),
        "mean_symmetric_distance_m": float(np.mean(survivors)) if survivors else None,
        "median_symmetric_distance_m": float(np.median(survivors)) if survivors else None,
        "distance_passes": len(survivors),
        "area_mae_km2": float(np.mean([row["absolute_area_error_km2"] for row in group])) if group else None,
        "mean_relative_area_error": float(np.mean([
            row["relative_area_error"] for row in group if row["relative_area_error"] is not None
        ])) if group else None,
        "mean_iou": float(np.mean([row["iou"] for row in group])) if group else None,
        "mean_dice": float(np.mean([row["dice"] for row in group])) if group else None,
    }


def _recommend(results: list[dict]) -> dict:
    qualified = [item for item in results if item["validation"]["contour_survival"] == 1.0]
    if not qualified:
        raise RuntimeError("No candidate keeps a zero contour on every validation pass.")

    def key(item):
        validation = item["validation"]
        return (
            -validation["mean_iou"],
            -validation["mean_dice"],
            validation["area_mae_km2"],
            validation["mean_symmetric_distance_m"],
        )

    winner = min(qualified, key=key)
    return {
        "name": winner["name"],
        "label": winner["label"],
        "w_pde": winner["w_pde"],
        "rule": (
            "Validation only. Require a zero contour on every validation pass. "
            "Then prefer higher mean IoU, higher mean Dice, lower area MAE, "
            "and lower symmetric contour distance, in that order. Test is not used."
        ),
    }


def _fmt(value, spec: str) -> str:
    if value is None:
        return ""
    return format(value, spec)


def _write_tables(results, winner_name: str) -> None:
    headers = [
        "candidate", "w_pde", "contour %", "mean dist m", "median dist m",
        "area MAE km2", "rel area", "mean IoU", "mean Dice",
    ]
    for split, filename in (("validation", "validation_comparison"), ("test", "test_comparison")):
        lines = [",".join(headers)]
        display = []
        for item in results:
            stats = item[split]
            values = [
                item["label"],
                f"{item['w_pde']:g}",
                f"{100.0 * stats['contour_survival']:.0f}",
                _fmt(stats["mean_symmetric_distance_m"], ".0f"),
                _fmt(stats["median_symmetric_distance_m"], ".0f"),
                _fmt(stats["area_mae_km2"], ".1f"),
                _fmt(stats["mean_relative_area_error"], ".3f"),
                _fmt(stats["mean_iou"], ".3f"),
                _fmt(stats["mean_dice"], ".3f"),
            ]
            lines.append(",".join(values))
            display.append(values)
        lines.append("")
        lines.append("pass,candidate,contour,predicted_km2,persistent_km2,abs_error_km2,rel_error,symmetric_m,iou,dice")
        for item in results:
            for row in item["passes"]:
                if row["split"] != split:
                    continue
                lines.append(",".join([
                    str(row["index"]),
                    item["label"],
                    "Y" if row["zero_contour"] else "N",
                    f"{row['predicted_km2']:.2f}",
                    f"{row['persistent_target_km2']:.2f}",
                    f"{row['absolute_area_error_km2']:.2f}",
                    _fmt(row["relative_area_error"], ".4f"),
                    _fmt(row["symmetric_distance_m"], ".1f"),
                    f"{row['iou']:.4f}",
                    f"{row['dice']:.4f}",
                ]))
        (OUT / f"{filename}.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
        fig, ax = plt.subplots(figsize=(13.6, 2.4), layout="constrained")
        ax.axis("off")
        widget = ax.table(cellText=display, colLabels=headers, loc="center")
        widget.auto_set_font_size(False)
        widget.set_fontsize(8)
        widget.scale(1, 1.45)
        if split == "validation":
            for row_index, item in enumerate(results):
                if item["name"] == winner_name:
                    for column in range(len(headers)):
                        widget[row_index + 1, column].set_facecolor("#d9ead3")
            ax.set_title("Validation. Persistent targets. Highlighted row is the selected candidate.")
        else:
            ax.set_title("Untouched test. Persistent targets. Not used for selection.")
        fig.savefig(OUT / f"{filename}.png", dpi=140)
        plt.close(fig)


def _previous_rows() -> dict:
    path = OUT / "comparison.json"
    if not path.exists():
        return {}
    stored = json.loads(path.read_text(encoding="utf-8"))
    found = {}
    for candidate in stored["candidates"]:
        for row in candidate["passes"]:
            found[(candidate["name"], row["index"])] = row
    return found


def _geometry_audit(results) -> dict:
    mismatches = []
    for item in results:
        for row in item["passes"]:
            stored = row["stored_target_km2"]
            polygon = row["persistent_polygon_km2"]
            if stored is None:
                continue
            if abs(stored - polygon) > 0.05:
                mismatches.append({
                    "candidate": item["name"],
                    "index": row["index"],
                    "stored_target_km2": stored,
                    "persistent_polygon_km2": polygon,
                })
    return {
        "previous_comparison_target": "persistent burned-region polygon",
        "previous_comparison_distance": (
            "one-way mean distance from the longest predicted phi=0 piece, "
            "every fifth vertex, to the persistent boundary"
        ),
        "previous_predicted_area": "grid cells with phi < 0",
        "evaluate_real_target": "per-pass hotspot hull stored on the case",
        "evaluate_real_used_by_candidate_comparison": False,
        "corrected_target_area": "grid cells inside the persistent burned region",
        "corrected_distance": "symmetric mean of both front-to-boundary directions",
        "stored_targets_match_persistent_polygons": not mismatches,
        "stored_target_mismatches": mismatches,
    }


def main() -> None:
    cfg = _kincade_config()
    case = load_fire_case(cfg.stack_nc, cfg.rate_nc, cfg.perimeters_path)
    regions, _used = persistent_burned_regions(
        cfg.firms_csv, case.perimeters["dt"], crs=str(case.perimeters.crs)
    )
    _cx, _cy, _length, _duration, _offset, tau = domain_scales(case)
    time_split = resolve_time_split(
        len(case.perimeters), RealConfig().k_train, cfg.split_scheme, cfg.train_fraction, cfg.val_fraction
    )
    yy, xx = np.meshgrid(np.asarray(case.y, dtype=np.float64), np.asarray(case.x, dtype=np.float64), indexing="ij")
    masks = [np.asarray(contains_xy(geometry, xx, yy), dtype=bool) for geometry in regions]
    scales = {
        "tau": tau,
        "time_split": time_split,
        "cell_km2": (cell_size_m(case.x, case.y) ** 2) / 1e6,
        "spacing_m": cell_size_m(case.x, case.y),
    }
    previous = _previous_rows()
    results = [
        _score_candidate(name, label, weight, case, regions, masks, scales, previous)
        for name, label, weight in CANDIDATES
    ]
    winner = _recommend(results)
    payload = {
        "geometry_audit": _geometry_audit(results),
        "selection": winner,
        "candidates": results,
    }
    (OUT / "audit.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_tables(results, winner["name"])
    print(f"\nVALIDATION WINNER {winner['label']} w_pde={winner['w_pde']:g}", flush=True)
    for split in HELD_OUT:
        print(split.upper(), flush=True)
        for item in results:
            stats = item[split]
            print(
                f"{item['label']}\t{100.0 * stats['contour_survival']:.0f}%\t"
                f"dist {stats['mean_symmetric_distance_m']:.0f}/"
                f"{stats['median_symmetric_distance_m']:.0f}\t"
                f"MAE {stats['area_mae_km2']:.1f}\t"
                f"rel {stats['mean_relative_area_error']:.3f}\t"
                f"IoU {stats['mean_iou']:.3f}\tDice {stats['mean_dice']:.3f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
