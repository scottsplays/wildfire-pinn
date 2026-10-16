"""Old hotspot-hull targets versus persistent burned-region targets.

Same network, same R, same loss weights, same seed. The only change is the
geometry the signed-distance observation is built from.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

from wildfire_pinn.burned_region import persistent_burned_regions
from wildfire_pinn.cli import setup_logging
from wildfire_pinn.config import DATASETS, ProjectConfig
from wildfire_pinn.geometry import cell_size_m, mean_distance_m, zero_contours
from wildfire_pinn.pinn import (
    RealConfig,
    _read_history,
    domain_scales,
    get_device,
    load_checkpoint,
    load_fire_case,
    predict_phi_meters,
    resolve_time_split,
    set_seed,
    train_real,
)
from wildfire_pinn.plots import _increasing_axes
from wildfire_pinn.pinn import LevelSetPINN

ROOT = Path(__file__).resolve().parent
OLD_DIR = ROOT / "outputs" / "kincade_target_old"
NEW_DIR = ROOT / "outputs" / "kincade_target_new"
COMPARE_DIR = ROOT / "outputs" / "kincade_target_compare"


def _kincade_config() -> ProjectConfig:
    spec = DATASETS["kincade"]
    return ProjectConfig(
        name=spec.name,
        bbox=spec.bbox,
        date_start=spec.date_start,
        date_end=spec.date_end,
        split_scheme=spec.split_scheme,
        train_fraction=spec.train_fraction,
        val_fraction=spec.val_fraction,
    )


def _train_pair(case, regions) -> None:
    cfg = _kincade_config()
    hp = RealConfig()
    shared = dict(
        retrain=False,
        split_scheme=cfg.split_scheme,
        train_fraction=cfg.train_fraction,
        val_fraction=cfg.val_fraction,
    )
    # Model weights are drawn before train_real reseeds. Seed here so both
    # runs start from the same initialization; train_real seeds the batches.
    set_seed(hp.seed)
    train_real(case, OLD_DIR, hp, target_geometries=None, **shared)
    set_seed(hp.seed)
    train_real(case, NEW_DIR, hp, target_geometries=regions, **shared)


def _load_model(out_dir: Path) -> LevelSetPINN:
    hp = RealConfig()
    device = get_device()
    model = LevelSetPINN(hp.hidden, hp.layers).to(device)
    load_checkpoint(out_dir / "levelset_pinn.pt", model, device)
    model.eval()
    return model


def _polygon_parts(geom):
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type == "Polygon":
        return [geom]
    if geom.geom_type in ("MultiPolygon", "GeometryCollection"):
        parts = []
        for piece in geom.geoms:
            parts.extend(_polygon_parts(piece))
        return parts
    return []


def _draw_rings(ax, geom, color, linewidth, label):
    drawn = False
    for part in _polygon_parts(geom):
        xs, ys = part.exterior.xy
        ax.plot(xs, ys, color=color, linewidth=linewidth, label=label if not drawn else None)
        drawn = True
        for ring in part.interiors:
            ax.plot(ring.xy[0], ring.xy[1], color=color, linewidth=linewidth, linestyle=":")
    if not drawn and label:
        ax.plot([], [], color=color, linewidth=linewidth, label=label)


def _front_distance_m(contours, boundary) -> float | None:
    if not contours:
        return None
    main = max(contours, key=len)
    stride = main[::5] if len(main) > 5 else main
    distance = mean_distance_m(stride, boundary)
    if not np.isfinite(distance):
        return None
    return float(distance)


def _pass_record(phi_m, contours, target, split, stamp, index, cell_km2) -> dict:
    negative = phi_m < 0
    return {
        "index": index,
        "time": stamp,
        "split": split,
        "zero_contour": bool(contours),
        "n_pieces": len(contours),
        "negative_km2": float(negative.sum()) * cell_km2,
        "min_phi_m": float(phi_m.min()),
        "max_phi_m": float(phi_m.max()),
        "target_km2": float(target.area) / 1e6,
        "distance_to_target_m": _front_distance_m(contours, target.boundary),
    }


def _compare(case, regions) -> dict:
    cfg = _kincade_config()
    hp = RealConfig()
    time_split = resolve_time_split(
        len(case.perimeters), hp.k_train, cfg.split_scheme, cfg.train_fraction, cfg.val_fraction
    )
    _cx, _cy, _length, duration, _offset, tau = domain_scales(case)
    side = cell_size_m(case.x, case.y)
    cell_km2 = (side * side) / 1e6
    domain_km2 = float(phi_shape_cells(case) * cell_km2)
    old_model = _load_model(OLD_DIR)
    new_model = _load_model(NEW_DIR)
    rows = []
    fields = []
    for index in range(len(case.perimeters)):
        stamp = str(case.perimeters["dt"].iloc[index])
        split = time_split.label(index)
        phi_old = predict_phi_meters(old_model, case, float(tau[index]))
        phi_new = predict_phi_meters(new_model, case, float(tau[index]))
        old_contours = zero_contours(phi_old, case.x, case.y)
        new_contours = zero_contours(phi_new, case.x, case.y)
        hull = case.perimeters.geometry.iloc[index]
        burned = regions[index]
        rows.append({
            "index": index,
            "time": stamp,
            "split": split,
            "hours": float((case.perimeters["dt"].iloc[index] - case.perimeters["dt"].iloc[0]).total_seconds()) / 3600.0,
            "old": _pass_record(phi_old, old_contours, hull, split, stamp, index, cell_km2),
            "new": _pass_record(phi_new, new_contours, burned, split, stamp, index, cell_km2),
        })
        fields.append((phi_old, phi_new, hull, burned, old_contours, new_contours))
    held = [row for row in rows if row["split"] != "train"]
    _write_heldout_figures(case, held, fields)
    _write_area_figure(rows, domain_km2)
    _write_loss_figure()
    payload = {
        "settings": {
            "epochs": hp.epochs,
            "seed": hp.seed,
            "lr": hp.lr,
            "hidden": hp.hidden,
            "layers": hp.layers,
            "n_pde": hp.n_pde,
            "n_ic": hp.n_ic,
            "n_obs": hp.n_obs,
            "w_ic": hp.w_ic,
            "w_pde": hp.w_pde,
            "w_obs": hp.w_obs,
            "c_wdir": hp.c_wdir,
            "sdf_sampling": "thirds",
            "split": time_split.stamp_groups(case),
            "T_days": duration / 86400.0,
            "domain_km2": domain_km2,
            "observation_target_old": "hotspot_hull",
            "observation_target_new": "persistent_burn",
        },
        "final_losses": {
            "old": _final_losses(OLD_DIR),
            "new": _final_losses(NEW_DIR),
        },
        "passes": rows,
        "held_out": [
            {
                "index": row["index"],
                "time": row["time"],
                "split": row["split"],
                "old_contour": row["old"]["zero_contour"],
                "new_contour": row["new"]["zero_contour"],
                "old_negative_km2": row["old"]["negative_km2"],
                "new_negative_km2": row["new"]["negative_km2"],
                "old_distance_to_hull_m": row["old"]["distance_to_target_m"],
                "new_distance_to_persistent_m": row["new"]["distance_to_target_m"],
                "hull_km2": row["old"]["target_km2"],
                "persistent_km2": row["new"]["target_km2"],
            }
            for row in held
        ],
    }
    COMPARE_DIR.mkdir(parents=True, exist_ok=True)
    (COMPARE_DIR / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def phi_shape_cells(case) -> int:
    return int(len(case.x) * len(case.y))


def _final_losses(out_dir: Path) -> dict:
    history = _read_history(out_dir / "history.json")
    out = {}
    for key in ("total", "pde", "ic", "obs"):
        values = history.get(key) or []
        out[key] = float(values[-1]) if values else None
        out["epochs"] = len(values)
    return out


def _write_loss_figure() -> None:
    old = _read_history(OLD_DIR / "history.json")
    new = _read_history(NEW_DIR / "history.json")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), layout="constrained")
    for ax, key, title in (
        (axes[0], "pde", "PDE loss"),
        (axes[1], "obs", "Observation loss"),
    ):
        ax.plot(old[key], color="C0", label="old hotspot hull")
        ax.plot(new[key], color="C1", label="new persistent region")
        ax.set_yscale("log")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.set_title(title)
        ax.legend()
    fig.suptitle("Same PINN settings. Only the signed-distance target changes.")
    COMPARE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(COMPARE_DIR / "losses.png", dpi=140)
    plt.close(fig)


def _write_area_figure(rows: list[dict], domain_km2: float) -> None:
    hours = [row["hours"] for row in rows]
    fig, ax = plt.subplots(figsize=(8.2, 4.6), layout="constrained")
    ax.plot(hours, [row["old"]["negative_km2"] for row in rows], color="C0", marker="o", label="old predicted burned area")
    ax.plot(hours, [row["new"]["negative_km2"] for row in rows], color="C1", marker="o", label="new predicted burned area")
    ax.plot(hours, [row["old"]["target_km2"] for row in rows], color="C0", linestyle="--", label="hotspot hull area")
    ax.plot(hours, [row["new"]["target_km2"] for row in rows], color="C1", linestyle="--", label="persistent estimate area")
    held_hours = [row["hours"] for row in rows if row["split"] != "train"]
    if held_hours:
        ax.axvspan(min(held_hours), max(hours), color="0.85", zorder=0, label="held-out passes")
    ax.set_xlabel("Hours since first pass")
    ax.set_ylabel("Area (km²)")
    ax.set_title(f"Predicted negative area. The training window is {domain_km2:.0f} km².")
    ax.legend(fontsize=8)
    COMPARE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(COMPARE_DIR / "area_over_time.png", dpi=140)
    plt.close(fig)


def _write_heldout_figures(case, held_rows, fields) -> None:
    COMPARE_DIR.mkdir(parents=True, exist_ok=True)
    x = np.asarray(case.x, dtype=np.float64)
    y = np.asarray(case.y, dtype=np.float64)
    for row in held_rows:
        index = row["index"]
        phi_old, phi_new, hull, burned, _old_c, _new_c = fields[index]
        fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.4), layout="constrained")
        for ax, phi, record, name in (
            (axes[0], phi_old, row["old"], "Old hotspot-hull target"),
            (axes[1], phi_new, row["new"], "New persistent target"),
        ):
            shown, xd, yd = _increasing_axes(phi, x, y)
            mesh = ax.pcolormesh(xd, yd, shown, cmap="RdBu", vmin=-5000, vmax=5000, shading="auto")
            ax.contour(xd, yd, shown, levels=[0], colors="black", linewidths=1.3)
            _draw_rings(ax, hull, "royalblue", 1.0, "hotspot hull")
            _draw_rings(ax, burned, "darkorange", 1.0, "persistent estimate")
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlim(float(x.min()), float(x.max()))
            ax.set_ylim(float(y.min()), float(y.max()))
            distance = record["distance_to_target_m"]
            distance_text = "no φ=0" if distance is None else f"dist {distance / 1000:.1f} km"
            contour_text = "contour" if record["zero_contour"] else "no contour"
            ax.set_title(
                f"{name}\n{record['negative_km2']:.0f} km² negative, {contour_text}, {distance_text}",
                fontsize=10,
            )
            ax.set_xlabel("Easting (m)")
            ax.set_ylabel("Northing (m)")
            ax.legend(loc="upper right", fontsize=7)
        fig.colorbar(mesh, ax=axes, label="predicted φ (m), color capped at ±5 km")
        fig.suptitle(f"Held-out pass {index}  {row['split']}  {row['time']}")
        fig.savefig(COMPARE_DIR / f"heldout_{index:02d}.png", dpi=140)
        plt.close(fig)


def main() -> None:
    COMPARE_DIR.mkdir(parents=True, exist_ok=True)
    setup_logging(COMPARE_DIR / "train.log")
    cfg = _kincade_config()
    case = load_fire_case(cfg.stack_nc, cfg.rate_nc, cfg.perimeters_path)
    regions, _used = persistent_burned_regions(
        cfg.firms_csv, case.perimeters["dt"], crs=str(case.perimeters.crs)
    )
    _train_pair(case, regions)
    payload = _compare(case, regions)
    print(json.dumps({
        "held_out": payload["held_out"],
        "final_losses": payload["final_losses"],
    }, indent=2))


if __name__ == "__main__":
    main()
