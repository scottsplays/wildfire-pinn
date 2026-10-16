"""Evaluate the frozen WildfirePINN v1 checkpoint. No training.

v1 is the observation-only level-set network at
outputs/kincade_final_candidates/observation/levelset_pinn.pt.
This script loads that file, scores persistent burned-region targets on
passes 0–19, and writes figures. It does not select a model and it does
not read the test split in order to change anything.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely import contains_xy

from run_final_candidate_audit import _overlap, _symmetric_distance_m
from run_persistent_target_experiment import _kincade_config
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
OUT = ROOT / "outputs" / "wildfirepinn_v1"
CHECKPOINT = ROOT / "outputs" / "kincade_final_candidates" / "observation" / "levelset_pinn.pt"
EXPECTED_TRAIN = list(range(12))
EXPECTED_VAL = list(range(12, 16))
EXPECTED_TEST = list(range(16, 20))
SPLIT_COLOR = {"train": "#1d3557", "validation": "#bc6c25", "test": "#9b2226"}
SPLIT_ROLE = {
    "train": "training fit; not a selection metric",
    "validation": "model-selection split; v1 was chosen here before this script",
    "test": "untouched test; not used to select v1",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _north_up(field: np.ndarray, x: np.ndarray, y: np.ndarray):
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


def _outline(axis, geom, color: str, lw: float) -> None:
    parts = [geom] if geom.geom_type == "Polygon" else list(getattr(geom, "geoms", []))
    for part in parts:
        if part.geom_type != "Polygon":
            continue
        coords = np.asarray(part.exterior.coords)
        axis.plot(coords[:, 0], coords[:, 1], color=color, lw=lw, solid_capstyle="round", zorder=3)
        for hole in part.interiors:
            ring = np.asarray(hole.coords)
            axis.plot(ring[:, 0], ring[:, 1], color=color, lw=max(0.4, lw * 0.7), zorder=3)


def _extent(target_masks, predicted_masks, x, y, pad: float = 4500.0):
    yy, xx = np.meshgrid(y, x, indexing="ij")
    occupied = np.zeros(target_masks[0].shape, dtype=bool)
    for mask in list(target_masks) + list(predicted_masks):
        occupied |= mask
    if not occupied.any():
        return float(x.min()), float(x.max()), float(y.min()), float(y.max())
    return (
        float(xx[occupied].min()) - pad,
        float(xx[occupied].max()) + pad,
        float(yy[occupied].min()) - pad,
        float(yy[occupied].max()) + pad,
    )


def _style_panel(axis, extent, split: str) -> None:
    axis.set_xlim(extent[0], extent[1])
    axis.set_ylim(extent[2], extent[3])
    axis.set_aspect("equal")
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_color(SPLIT_COLOR[split])
        spine.set_linewidth(1.4)


def _draw_pass(axis, row, field_xy, hot_x, hot_y, extent, show_metrics: bool) -> None:
    field, east, north = field_xy
    axis.contourf(east, north, field, levels=[-1.0e9, 0.0], colors=["#f4a261"], alpha=0.55, zorder=1)
    axis.contour(east, north, field, levels=[0.0], colors=["#9b2226"], linewidths=0.9, zorder=4)
    _outline(axis, row["geometry"], "#222222", 0.7)
    if hot_x.size:
        axis.scatter(hot_x, hot_y, s=3, c="#4a4a4a", linewidths=0, alpha=0.55, zorder=2)
    _style_panel(axis, extent, row["split"])
    stamp = pd.Timestamp(row["time"]).strftime("%b %d  %H:%M")
    title = f"{stamp}"
    if show_metrics and row["iou"] is not None:
        title = f"{stamp}\nIoU {row['iou']:.2f}"
    axis.set_title(title, fontsize=8, color="#222222", pad=3)
    axis.text(
        0.02,
        0.98,
        row["split"].upper(),
        transform=axis.transAxes,
        va="top",
        ha="left",
        color="white",
        fontsize=6,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.18", "fc": SPLIT_COLOR[row["split"]], "ec": "none"},
        zorder=5,
    )


def _legend(fig) -> None:
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor="#f4a261", alpha=0.7, label="Predicted burned"),
        plt.Line2D([], [], color="#9b2226", lw=1.4, label="Predicted perimeter"),
        plt.Line2D([], [], color="#222222", lw=1.2, label="Persistent target"),
        plt.Line2D([], [], color="#4a4a4a", marker="o", lw=0, ms=3, label="VIIRS hotspots this pass"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False, fontsize=8)


def _finite_metric(rows: list[dict], key: str) -> list[float]:
    values = []
    for row in rows:
        value = row[key]
        if value is None:
            continue
        if not np.isfinite(value):
            raise RuntimeError(f"Non-finite {key} on pass {row['pass']}")
        values.append(float(value))
    return values


def _aggregate(rows: list[dict], split: str) -> dict:
    group = [row for row in rows if row["split"] == split]
    distances = _finite_metric(group, "symmetric_distance_m")
    return {
        "split": split,
        "role": SPLIT_ROLE[split],
        "n_passes": len(group),
        "contour_fraction": float(np.mean([row["zero_contour"] for row in group])) if group else None,
        "mean_iou": float(np.mean(_finite_metric(group, "iou"))) if group else None,
        "mean_dice": float(np.mean(_finite_metric(group, "dice"))) if group else None,
        "mean_symmetric_distance_m": float(np.mean(distances)) if distances else None,
        "median_symmetric_distance_m": float(np.median(distances)) if distances else None,
        "mean_absolute_area_error_km2": float(np.mean(_finite_metric(group, "absolute_area_error_km2"))) if group else None,
        "mean_signed_area_error_km2": float(np.mean(_finite_metric(group, "area_error_km2"))) if group else None,
        "mean_target_area_km2": float(np.mean(_finite_metric(group, "target_area_km2"))) if group else None,
        "mean_predicted_area_km2": float(np.mean(_finite_metric(group, "predicted_area_km2"))) if group else None,
    }


def _draw_all(rows, fields, hotspots, extent) -> None:
    fig, axes = plt.subplots(4, 5, figsize=(15.2, 12.4), layout="constrained")
    for axis, row, field in zip(axes.ravel(), rows, fields):
        _draw_pass(axis, row, field, hotspots[row["pass"]][0], hotspots[row["pass"]][1], extent, show_metrics=False)
    _legend(fig)
    fig.suptitle(
        "WildfirePINN v1  ·  Kincade observation-only level set  ·  one map extent, north up",
        fontsize=13,
    )
    fig.savefig(OUT / "wildfirepinn_v1_predictions.png", dpi=160)
    plt.close(fig)


def _draw_heldout(rows, fields, hotspots, extent) -> None:
    held = [(row, field) for row, field in zip(rows, fields) if row["split"] != "train"]
    fig, axes = plt.subplots(2, 4, figsize=(13.6, 7.4), layout="constrained")
    for axis, (row, field) in zip(axes.ravel(), held):
        _draw_pass(axis, row, field, hotspots[row["pass"]][0], hotspots[row["pass"]][1], extent, show_metrics=True)
    _legend(fig)
    fig.suptitle(
        "Held-out Kincade passes  ·  validation selected v1  ·  test was not used",
        fontsize=13,
    )
    fig.savefig(OUT / "wildfirepinn_v1_heldout.png", dpi=170)
    plt.close(fig)


def _draw_area(rows) -> None:
    frame = pd.DataFrame(rows)
    hours = frame["hours_since_start"].to_numpy()
    fig, ax = plt.subplots(figsize=(8.4, 4.6), layout="constrained")
    train_end = float(frame.loc[frame["split"] == "train", "hours_since_start"].max())
    val_end = float(frame.loc[frame["split"] == "validation", "hours_since_start"].max())
    ax.axvspan(hours.min(), train_end, color="#e8eef4", zorder=0)
    ax.axvspan(train_end, val_end, color="#f6ead8", zorder=0)
    ax.axvspan(val_end, hours.max(), color="#f8e6e4", zorder=0)
    ax.plot(hours, frame["target_area_km2"], color="#222222", lw=1.8, marker="o", ms=3.5, label="Persistent target")
    ax.plot(hours, frame["predicted_area_km2"], color="#9b2226", lw=1.8, marker="o", ms=3.5, label="WildfirePINN v1")
    ymin, ymax = ax.get_ylim()
    y_text = ymin + 0.92 * (ymax - ymin)
    ax.text((hours.min() + train_end) / 2, y_text, "Train", ha="center", va="top", color=SPLIT_COLOR["train"], fontsize=9)
    ax.text((train_end + val_end) / 2, y_text, "Validation", ha="center", va="top", color=SPLIT_COLOR["validation"], fontsize=9)
    ax.text((val_end + hours.max()) / 2, y_text, "Test", ha="center", va="top", color=SPLIT_COLOR["test"], fontsize=9)
    ax.set_xlabel("Hours since first overpass")
    ax.set_ylabel("Burned area (km²)")
    ax.set_title("Kincade burned area, persistent target and v1")
    ax.legend(frameon=False, fontsize=9)
    fig.savefig(OUT / "wildfirepinn_v1_area.png", dpi=160)
    plt.close(fig)


def main() -> None:
    started = time.perf_counter()
    if not CHECKPOINT.is_file():
        raise RuntimeError(f"Missing frozen checkpoint {CHECKPOINT}")
    digest_before = _sha256(CHECKPOINT)
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = _kincade_config()
    case = load_fire_case(cfg.stack_nc, cfg.rate_nc, cfg.perimeters_path)
    split = resolve_time_split(
        len(case.perimeters), RealConfig().k_train, cfg.split_scheme, cfg.train_fraction, cfg.val_fraction
    )
    if split.train != EXPECTED_TRAIN or split.validation != EXPECTED_VAL or split.test != EXPECTED_TEST:
        raise RuntimeError(
            f"Split is train {split.train}, validation {split.validation}, test {split.test}"
        )
    regions, detections = persistent_burned_regions(
        cfg.firms_csv, case.perimeters["dt"], crs=str(case.perimeters.crs)
    )
    if len(regions) != 20:
        raise RuntimeError(f"Expected 20 persistent regions, found {len(regions)}")
    x = np.asarray(case.x, dtype=np.float64)
    y = np.asarray(case.y, dtype=np.float64)
    spacing = cell_size_m(x, y)
    cell_km2 = (spacing ** 2) / 1e6
    yy, xx = np.meshgrid(y, x, indexing="ij")
    _cx, _cy, _length, _duration, _offset, tau = domain_scales(case)
    device = get_device()
    hp = RealConfig()
    model = LevelSetPINN(hp.hidden, hp.layers).to(device)
    saved = load_checkpoint(CHECKPOINT, model, device)
    if float(saved.get("w_pde", -1)) != 0.0 or saved.get("observation_target") != "persistent_burn":
        raise RuntimeError(f"Checkpoint config is not the observation-only v1 model: {saved}")
    model.eval()
    digest_after = _sha256(CHECKPOINT)
    if digest_after != digest_before:
        raise RuntimeError("Checkpoint bytes changed while loading. v1 must stay immutable.")

    hotspots = {}
    accepted = detections[~detections["trimmed"]]
    for index in range(20):
        group = accepted[accepted["pass_index"] == index]
        hotspots[index] = (
            group["x"].to_numpy(dtype=np.float64),
            group["y"].to_numpy(dtype=np.float64),
        )

    start = pd.Timestamp(case.perimeters["dt"].iloc[0])
    rows = []
    fields = []
    target_masks = []
    predicted_masks = []
    for index, geometry in enumerate(regions):
        phi_m = predict_phi_meters(model, case, float(tau[index]))
        target = np.asarray(contains_xy(geometry, xx, yy), dtype=bool)
        predicted = phi_m < 0
        contours = zero_contours(phi_m, x, y)
        iou, dice = _overlap(predicted, target)
        distance_m = _symmetric_distance_m(contours, geometry, spacing)
        target_km2 = float(target.sum()) * cell_km2
        predicted_km2 = float(predicted.sum()) * cell_km2
        signed_error = predicted_km2 - target_km2
        split_name = split.label(index)
        row = {
            "pass": index,
            "split": split_name,
            "time": str(case.perimeters["dt"].iloc[index]),
            "hours_since_start": float((pd.Timestamp(case.perimeters["dt"].iloc[index]) - start).total_seconds()) / 3600.0,
            "target_area_km2": target_km2,
            "predicted_area_km2": predicted_km2,
            "area_error_km2": signed_error,
            "absolute_area_error_km2": abs(signed_error),
            "iou": iou,
            "dice": dice,
            "symmetric_distance_m": distance_m,
            "min_phi_m": float(np.min(phi_m)),
            "fraction_phi_negative": float(np.mean(predicted)),
            "zero_contour": bool(contours),
            "geometry": geometry,
        }
        for key in ("target_area_km2", "predicted_area_km2", "area_error_km2", "iou", "dice", "min_phi_m", "fraction_phi_negative"):
            if not np.isfinite(row[key]):
                raise RuntimeError(f"Non-finite {key} on pass {index}")
        if distance_m is not None and not np.isfinite(distance_m):
            raise RuntimeError(f"Non-finite symmetric distance on pass {index}")
        rows.append(row)
        fields.append(_north_up(phi_m, x, y))
        target_masks.append(target)
        predicted_masks.append(predicted)
        print(
            f"k{index:02d} {split_name:10s} contour={row['zero_contour']} "
            f"area {predicted_km2:.1f}/{target_km2:.1f} "
            f"IoU {iou:.3f} Dice {dice:.3f} "
            f"dist {None if distance_m is None else round(distance_m)} m "
            f"min phi {row['min_phi_m']:.0f}",
            flush=True,
        )

    summaries = [_aggregate(rows, name) for name in ("train", "validation", "test")]
    extent = _extent(target_masks, predicted_masks, x, y)
    _draw_all(rows, fields, hotspots, extent)
    _draw_heldout(rows, fields, hotspots, extent)
    _draw_area(rows)

    per_pass = []
    for row in rows:
        item = {key: value for key, value in row.items() if key != "geometry"}
        per_pass.append(item)
    pd.DataFrame(per_pass).to_csv(OUT / "per_pass_metrics.csv", index=False)
    pd.DataFrame(summaries).to_csv(OUT / "split_summary.csv", index=False)
    elapsed = time.perf_counter() - started
    payload = {
        "model": "WildfirePINN v1",
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)).replace("\\", "/"),
        "sha256": digest_before,
        "checkpoint_unchanged": digest_after == digest_before,
        "selection": {
            "chosen_on": "validation",
            "test_used_for_selection": False,
            "this_script_trains": False,
            "this_script_tunes": False,
            "note": (
                "v1 was frozen as the observation-only checkpoint before this "
                "evaluation. Test metrics are reported and were not used to choose the model."
            ),
        },
        "split": {
            "train": EXPECTED_TRAIN,
            "validation": EXPECTED_VAL,
            "test": EXPECTED_TEST,
        },
        "parameter_count": int(sum(tensor.numel() for tensor in model.state_dict().values())),
        "saved_config": saved,
        "per_pass": per_pass,
        "splits": {item["split"]: item for item in summaries},
        "nan_free": True,
        "runtime_seconds": elapsed,
    }
    (OUT / "final_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"sha256 {digest_before}", flush=True)
    for item in summaries:
        print(
            f"{item['split']:10s} IoU {item['mean_iou']:.3f} Dice {item['mean_dice']:.3f} "
            f"dist {item['mean_symmetric_distance_m']:.0f} m "
            f"area MAE {item['mean_absolute_area_error_km2']:.1f} km2",
            flush=True,
        )
    print(f"runtime {elapsed:.1f} s", flush=True)


if __name__ == "__main__":
    main()
