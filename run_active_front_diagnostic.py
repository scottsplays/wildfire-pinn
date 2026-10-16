"""Active-front predictability on Kincade training passes 0 through 11.

No PINN training. Persistent targets, the PDE, and checkpoints are only read.
Validation and test perimeters are not labels and are not features.

A sample on the old persistent boundary is active when the next training
mask gains a burned cell within one diagonal cell of that sample. That is
the same added-cell mask as the interval audit (later and not earlier).
Global area change is not the label.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely import covers, distance, points
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from run_final_candidate_audit import _parts, _sample_lines
from run_pde_residual_audit import _day_index, _grad, _local_downwind
from run_persistent_target_experiment import _kincade_config
from wildfire_pinn.burned_region import persistent_burned_regions
from wildfire_pinn.data_pipeline import load_dataset
from wildfire_pinn.geometry import cell_size_m, signed_distance
from wildfire_pinn.pinn import RealConfig, domain_scales, downwind_unit_vectors, load_fire_case, resolve_time_split

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "kincade_active_front_diagnostic"
TRAIN = list(range(12))
FEATURE_COLUMNS = (
    "wind_speed_m_s",
    "local_wind_alignment",
    "domain_wind_alignment",
    "R_m_s",
    "slope_deg",
    "fuel_moisture_pct",
    "relative_humidity_pct",
    "air_temperature_c",
    "elevation_m",
)
# Modest bar, fixed before looking at the scores. Ranking and the rare
# active class both have to clear it. A tie with the inactive baseline does not.
SIGNAL_ROC_AUC = 0.65
SIGNAL_BALANCED_ACCURACY = 0.60
SIGNAL_RECALL = 0.25


def _interpolator(values: np.ndarray, x: np.ndarray, y: np.ndarray, method: str):
    from scipy.interpolate import RegularGridInterpolator

    y_axis = np.asarray(y, dtype=np.float64)
    x_axis = np.asarray(x, dtype=np.float64)
    data = np.asarray(values, dtype=np.float64)
    if y_axis[0] > y_axis[-1]:
        y_axis = y_axis[::-1]
        data = data[::-1, :]
    if x_axis[0] > x_axis[-1]:
        x_axis = x_axis[::-1]
        data = data[:, ::-1]
    return RegularGridInterpolator(
        (y_axis, x_axis), data, method=method, bounds_error=False, fill_value=np.nan
    )


def _at(interp, xq: np.ndarray, yq: np.ndarray) -> np.ndarray:
    return np.asarray(interp(np.column_stack([yq, xq])), dtype=np.float64)


def _unit(east: np.ndarray, north: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    norm = np.hypot(east, north)
    ok = norm > 1e-6
    east_hat = np.divide(east, norm, out=np.zeros_like(east), where=ok)
    north_hat = np.divide(north, norm, out=np.zeros_like(north), where=ok)
    return east_hat, north_hat


def _distribution(values: np.ndarray, mask: np.ndarray) -> dict:
    chosen = values[mask & np.isfinite(values)]
    if chosen.size == 0:
        return {"n": 0}
    pct = np.percentile(chosen, [10, 50, 90])
    return {
        "n": int(chosen.size),
        "mean": float(chosen.mean()),
        "p10": float(pct[0]),
        "median": float(pct[1]),
        "p90": float(pct[2]),
    }


def _ranking(y_true: np.ndarray, scores: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=np.float64)
    valid = np.isfinite(scores)
    y_true = y_true[valid]
    scores = scores[valid]
    if y_true.size == 0 or y_true.min() == y_true.max() or np.unique(scores).size < 2:
        return {"roc_auc": None, "pr_auc": None}
    return {
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "pr_auc": float(average_precision_score(y_true, scores)),
    }


def _classification(y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray | None) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    positive = y_true == 1
    negative = ~positive
    tp = int(np.count_nonzero(positive & (y_pred == 1)))
    fp = int(np.count_nonzero(negative & (y_pred == 1)))
    fn = int(np.count_nonzero(positive & (y_pred == 0)))
    tn = int(np.count_nonzero(negative & (y_pred == 0)))
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    specificity = tn / (tn + fp) if (tn + fp) else None
    if precision is None or recall is None or (precision + recall) == 0:
        f1 = 0.0 if (precision == 0 or recall == 0) else None
    else:
        f1 = 2.0 * precision * recall / (precision + recall)
    balanced = None
    if recall is not None and specificity is not None:
        balanced = 0.5 * (recall + specificity)
    ranking = {"roc_auc": None, "pr_auc": None}
    if scores is not None:
        ranking = _ranking(y_true, scores)
    return {
        "n": int(y_true.size),
        "positives": int(positive.sum()),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "balanced_accuracy": balanced,
        **ranking,
    }


def _leave_one_interval(frame: pd.DataFrame) -> dict:
    usable = frame.dropna(subset=list(FEATURE_COLUMNS)).copy()
    y = usable["active"].to_numpy(dtype=int)
    groups = usable["interval"].to_numpy(dtype=int)
    x = usable[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float64)
    stored_pred = np.full(y.shape, -1, dtype=int)
    stored_score = np.full(y.shape, np.nan, dtype=np.float64)
    stored_baseline = np.full(y.shape, -1, dtype=int)
    coefficients = []
    folds = []
    for held in range(len(TRAIN) - 1):
        test = groups == held
        train = ~test
        if not np.any(test):
            continue
        y_train = y[train]
        majority = int(np.bincount(y_train).argmax()) if y_train.size else 0
        stored_baseline[test] = majority
        fold = {
            "held_out_interval": f"{held}->{held + 1}",
            "train_positives": int(y_train.sum()),
            "train_n": int(train.sum()),
            "majority_class": majority,
        }
        if np.unique(y_train).size < 2:
            fold["fit"] = False
            folds.append(fold)
            continue
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x[train])
        x_test = scaler.transform(x[test])
        model = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            solver="lbfgs",
            max_iter=1000,
            random_state=0,
        )
        model.fit(x_train, y_train)
        score = model.predict_proba(x_test)[:, 1]
        pred = (score >= 0.5).astype(int)
        stored_score[test] = score
        stored_pred[test] = pred
        coefficients.append(model.coef_.ravel().copy())
        fold["fit"] = True
        fold["classifier"] = _classification(y[test], pred, score)
        fold["majority_baseline"] = _classification(y[test], np.full(int(test.sum()), majority), None)
        fold["inactive_baseline"] = _classification(y[test], np.zeros(int(test.sum()), dtype=int), None)
        folds.append(fold)

    scored = stored_pred >= 0
    pooled = _classification(y[scored], stored_pred[scored], stored_score[scored])
    inactive = _classification(y[scored], np.zeros(int(scored.sum()), dtype=int), None)
    majority_pred = stored_baseline[scored]
    majority = _classification(y[scored], majority_pred, None)
    coef = np.vstack(coefficients) if coefficients else np.zeros((0, len(FEATURE_COLUMNS)))
    coef_rows = []
    for index, name in enumerate(FEATURE_COLUMNS):
        column = coef[:, index] if coef.size else np.array([])
        coef_rows.append({
            "feature": name,
            "mean_standardized_coefficient": float(column.mean()) if column.size else None,
            "std_across_folds": float(column.std(ddof=0)) if column.size else None,
        })
    both_classes = [
        fold["classifier"]["balanced_accuracy"]
        for fold in folds
        if fold.get("fit") and fold["classifier"]["balanced_accuracy"] is not None
        and fold["classifier"]["positives"] not in (0, fold["classifier"]["n"])
    ]
    return {
        "rows_with_finite_features": int(len(usable)),
        "rows_scored": int(scored.sum()),
        "pooled": pooled,
        "pooled_inactive_baseline": inactive,
        "pooled_majority_baseline": majority,
        "mean_balanced_accuracy_over_mixed_intervals": (
            float(np.mean(both_classes)) if both_classes else None
        ),
        "mixed_intervals": len(both_classes),
        "folds": folds,
        "coefficients": coef_rows,
        "reference": {
            "inactive_balanced_accuracy_when_both_classes_exist": 0.5,
            "uninformative_roc_auc": 0.5,
            "uninformative_pr_auc": "positive prevalence on the scored rows",
        },
    }


def _fuel_table(fuel: np.ndarray, active: np.ndarray) -> list[dict]:
    rows = []
    valid = np.isfinite(fuel)
    codes = np.rint(fuel[valid]).astype(int)
    labels = active[valid]
    for code in sorted(set(codes.tolist())):
        chosen = codes == code
        count = int(chosen.sum())
        if count < 30:
            continue
        rows.append({
            "fuel_code": int(code),
            "n": count,
            "active_fraction": float(labels[chosen].mean()),
        })
    rows.sort(key=lambda row: row["n"], reverse=True)
    return rows


def _draw_intervals(samples: list[dict], spacing: float) -> None:
    fig, axes = plt.subplots(3, 4, figsize=(14.5, 11.0), layout="constrained")
    flat = axes.ravel()
    for index, sample in enumerate(samples):
        axis = flat[index]
        added = sample["added_xy"]
        if len(added):
            axis.scatter(added[:, 0], added[:, 1], s=2, c="#f4a261", linewidths=0, alpha=0.45, zorder=1)
        pts = sample["xy"]
        active = sample["active"]
        if len(pts) >= 2:
            jump = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))
            for point_index in range(len(pts) - 1):
                if jump[point_index] > 2.5 * spacing:
                    continue
                color = "#d62728" if active[point_index] else "#b0b0b0"
                axis.plot(
                    pts[point_index:point_index + 2, 0],
                    pts[point_index:point_index + 2, 1],
                    color=color,
                    lw=1.3,
                    solid_capstyle="round",
                    zorder=2,
                )
        pad = 800.0
        pieces = [pts] if len(pts) else []
        if len(added):
            pieces.append(added)
        cloud = np.vstack(pieces)
        axis.set_xlim(float(cloud[:, 0].min()) - pad, float(cloud[:, 0].max()) + pad)
        axis.set_ylim(float(cloud[:, 1].min()) - pad, float(cloud[:, 1].max()) + pad)
        axis.set_aspect("equal")
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_title(
            f"{sample['left']}→{sample['right']}  {sample['hours']:.1f} h\n"
            f"active {100 * sample['active_fraction']:.1f}%",
            fontsize=9,
        )
    flat[-1].axis("off")
    flat[0].plot([], [], color="#d62728", lw=2, label="Active old boundary")
    flat[0].plot([], [], color="#b0b0b0", lw=2, label="Inactive old boundary")
    flat[0].scatter([], [], s=12, c="#f4a261", label="Cells added on the next pass")
    flat[0].legend(loc="upper left", fontsize=7, frameon=False)
    fig.suptitle("Old persistent perimeter, training transitions", fontsize=12)
    fig.savefig(OUT / "perimeters.png", dpi=130)
    plt.close(fig)


def _draw_fraction(rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 4.2), layout="constrained")
    labels = [f"{row['left']}→{row['right']}\n{row['elapsed_hours']:.1f} h" for row in rows]
    fractions = [100.0 * row["active_fraction"] for row in rows]
    ax.bar(np.arange(len(rows)), fractions, color="#d62728", width=0.72)
    ax.set_xticks(np.arange(len(rows)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Active fraction of the old boundary (%)")
    ax.set_title("Local outward growth along the previous perimeter")
    fig.savefig(OUT / "active_fraction.png", dpi=130)
    plt.close(fig)


def _draw_predictors(frame: pd.DataFrame) -> None:
    names = (
        ("wind_speed_m_s", "Wind speed (m/s)"),
        ("local_wind_alignment", "Local wind · outward normal"),
        ("domain_wind_alignment", "Domain-mean wind · outward normal"),
        ("R_m_s", "Weather R (m/s)"),
        ("R_eff_m_s", "R_eff (m/s)"),
        ("slope_deg", "Slope (deg)"),
        ("fuel_moisture_pct", "100-h moisture (%)"),
        ("relative_humidity_pct", "Min RH (%)"),
        ("air_temperature_c", "Max temperature (°C)"),
    )
    fig, axes = plt.subplots(3, 3, figsize=(11.2, 8.4), layout="constrained")
    active = frame["active"].to_numpy(dtype=bool)
    for axis, (column, title) in zip(axes.ravel(), names, strict=True):
        values = frame[column].to_numpy(dtype=np.float64)
        finite = np.isfinite(values)
        lo, hi = np.nanpercentile(values[finite], [1, 99])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = 0.0, 1.0
        bins = np.linspace(lo, hi, 28)
        axis.hist(values[finite & ~active], bins=bins, density=True, color="#b0b0b0", alpha=0.85, label="Inactive")
        axis.hist(values[finite & active], bins=bins, density=True, color="#d62728", alpha=0.55, label="Active")
        axis.set_title(title, fontsize=9)
        axis.tick_params(labelsize=7)
    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.suptitle("Predictors sampled on the old boundary", fontsize=12)
    fig.savefig(OUT / "predictors.png", dpi=130)
    plt.close(fig)


def _draw_classifier(result: dict, prevalence: float) -> None:
    labels = ["Balanced acc.", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]
    keys = ["balanced_accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    model = [result["pooled"].get(key) for key in keys]
    baseline = [
        result["pooled_inactive_baseline"].get("balanced_accuracy"),
        result["pooled_inactive_baseline"].get("precision") or 0.0,
        result["pooled_inactive_baseline"].get("recall") or 0.0,
        result["pooled_inactive_baseline"].get("f1") or 0.0,
        0.5,
        prevalence,
    ]
    fig, ax = plt.subplots(figsize=(8.2, 4.2), layout="constrained")
    xpos = np.arange(len(labels))
    ax.bar(xpos - 0.18, [0.0 if value is None else value for value in baseline], width=0.36, color="#b0b0b0", label="Inactive baseline")
    ax.bar(xpos + 0.18, [0.0 if value is None else value for value in model], width=0.36, color="#1f4e79", label="Logistic regression")
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Leave-one-interval-out, pooled boundary samples")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(OUT / "classifier.png", dpi=130)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = _kincade_config()
    case = load_fire_case(cfg.stack_nc, cfg.rate_nc, cfg.perimeters_path)
    regions, _used = persistent_burned_regions(
        cfg.firms_csv, case.perimeters["dt"], crs=str(case.perimeters.crs)
    )
    split = resolve_time_split(
        len(regions), RealConfig().k_train, cfg.split_scheme, cfg.train_fraction, cfg.val_fraction
    )
    if split.train != TRAIN:
        raise RuntimeError(f"Expected training passes 0–11, found {split.train}")
    stack = load_dataset(cfg.stack_nc)
    x = np.asarray(case.x, dtype=np.float64)
    y = np.asarray(case.y, dtype=np.float64)
    spacing = cell_size_m(x, y)
    neighborhood = spacing * np.sqrt(2.0)
    _cx, _cy, _length, duration, t0_offset, tau = domain_scales(case)
    c_wdir = float(RealConfig().c_wdir)
    wind = np.asarray(case.wind_from_deg, dtype=np.float64)
    local_east, local_north = _local_downwind(wind)
    domain_wind = np.asarray(downwind_unit_vectors(wind), dtype=np.float64)
    n_days = int(case.spread.shape[0])
    masks = []
    fields = []
    for index in TRAIN:
        field = signed_distance(regions[index], x, y)
        fields.append(field)
        masks.append(field < 0)
        print(f"pass {index} inside {int(masks[-1].sum())} cells", flush=True)

    cache: dict[tuple, object] = {}

    def sample(values: np.ndarray, key: tuple, xq, yq, method: str) -> np.ndarray:
        if key not in cache:
            cache[key] = _interpolator(values, x, y, method)
        return _at(cache[key], xq, yq)

    samples = []
    tables = []
    for local in range(len(TRAIN) - 1):
        left = TRAIN[local]
        right = TRAIN[local + 1]
        boundary = _sample_lines(_parts(regions[left].boundary), spacing)
        if len(boundary) == 0:
            raise RuntimeError(f"pass {left} has no boundary samples")
        added = (~masks[local]) & masks[local + 1]
        removed = int(np.count_nonzero(masks[local] & ~masks[local + 1]))
        yy, xx = np.meshgrid(y, x, indexing="ij")
        added_xy = np.column_stack([xx[added], yy[added]])
        if len(added_xy):
            dist_added, _ = cKDTree(added_xy).query(boundary, k=1)
        else:
            dist_added = np.full(len(boundary), np.inf)
        active = dist_added <= neighborhood
        to_later = np.asarray(distance(points(boundary), regions[right].boundary), dtype=np.float64)
        covered = np.asarray(covers(regions[right], points(boundary)), dtype=bool)
        engulfed = covered & (to_later > spacing)
        gx, gy, _mag = _grad(fields[local], x, y)
        normal_x = sample(gx, ("gx", left), boundary[:, 0], boundary[:, 1], "linear")
        normal_y = sample(gy, ("gy", left), boundary[:, 0], boundary[:, 1], "linear")
        normal_norm = np.hypot(normal_x, normal_y)
        usable_normal = normal_norm >= 0.25
        normal_x, normal_y = _unit(normal_x, normal_y)
        day = _day_index(float(tau[left]), t0_offset, duration, n_days)
        later_day = _day_index(float(tau[right]), t0_offset, duration, n_days)
        wind_east = sample(local_east[day], ("east", day), boundary[:, 0], boundary[:, 1], "linear")
        wind_north = sample(local_north[day], ("north", day), boundary[:, 0], boundary[:, 1], "linear")
        wind_east, wind_north = _unit(wind_east, wind_north)
        domain_east = float(domain_wind[day, 0])
        domain_north = float(domain_wind[day, 1])
        local_alignment = wind_east * normal_x + wind_north * normal_y
        domain_alignment = domain_east * normal_x + domain_north * normal_y
        rate = sample(np.asarray(case.spread[day], dtype=np.float64), ("R", day), boundary[:, 0], boundary[:, 1], "linear")
        r_eff = np.maximum(rate * (1.0 + c_wdir * domain_alignment), 0.0)
        wind_speed = sample(np.asarray(stack["wind_speed"].values[day]), ("ws", day), boundary[:, 0], boundary[:, 1], "linear")
        slope = sample(np.asarray(stack["slope_deg"].values), ("slope",), boundary[:, 0], boundary[:, 1], "linear")
        moisture = sample(np.asarray(stack["fuel_moisture"].values[day]), ("fm", day), boundary[:, 0], boundary[:, 1], "linear")
        humidity = sample(np.asarray(stack["relative_humidity"].values[day]), ("rh", day), boundary[:, 0], boundary[:, 1], "linear")
        temperature = sample(np.asarray(stack["air_temperature"].values[day]), ("t", day), boundary[:, 0], boundary[:, 1], "linear")
        elevation = sample(np.asarray(stack["elevation_m"].values), ("elev",), boundary[:, 0], boundary[:, 1], "linear")
        fuel = sample(np.asarray(stack["fuel_model"].values), ("fuel",), boundary[:, 0], boundary[:, 1], "nearest")
        if np.nanmedian(temperature) > 200.0:
            temperature = temperature - 273.15
        local_alignment = np.where(usable_normal, local_alignment, np.nan)
        domain_alignment = np.where(usable_normal, domain_alignment, np.nan)
        r_eff = np.where(usable_normal, r_eff, np.nan)
        hours = float((case.perimeters["dt"].iloc[right] - case.perimeters["dt"].iloc[left]).total_seconds()) / 3600.0
        row = {
            "interval": local,
            "left": left,
            "right": right,
            "start": str(case.perimeters["dt"].iloc[left]),
            "end": str(case.perimeters["dt"].iloc[right]),
            "elapsed_hours": hours,
            "day_index_at_old_pass": day,
            "later_pass_day_index": later_day,
            "boundary_samples": int(len(boundary)),
            "active_samples": int(active.sum()),
            "active_fraction": float(active.mean()),
            "engulfed_fraction": float(engulfed.mean()),
            "cells_added": int(np.count_nonzero(added)),
            "cells_removed": removed,
            "area_change_km2": float(regions[right].area - regions[left].area) / 1e6,
            "median_distance_to_added_cell_m": float(np.median(dist_added)),
            "median_distance_to_later_boundary_m": float(np.median(to_later)),
            "valid_normal_fraction": float(np.mean(usable_normal)),
        }
        tables.append(row)
        samples.append({
            "left": left,
            "right": right,
            "hours": hours,
            "xy": boundary,
            "active": active,
            "active_fraction": float(active.mean()),
            "added_xy": added_xy,
        })
        block = pd.DataFrame({
            "interval": np.full(len(boundary), local, dtype=np.int16),
            "active": active.astype(np.int8),
            "engulfed": engulfed.astype(np.int8),
            "x": boundary[:, 0],
            "y": boundary[:, 1],
            "distance_to_added_cell_m": dist_added,
            "distance_to_later_boundary_m": to_later,
            "wind_speed_m_s": wind_speed,
            "local_wind_alignment": local_alignment,
            "domain_wind_alignment": domain_alignment,
            "R_m_s": rate,
            "R_eff_m_s": r_eff,
            "slope_deg": slope,
            "fuel_moisture_pct": moisture,
            "relative_humidity_pct": humidity,
            "air_temperature_c": temperature,
            "elevation_m": elevation,
            "fuel_code": fuel,
        })
        samples[-1]["frame"] = block
        print(
            f"{left}->{right} {hours:.2f} h | active {row['active_samples']}/{row['boundary_samples']} "
            f"({100 * row['active_fraction']:.2f}%) | engulfed {100 * row['engulfed_fraction']:.2f}% | "
            f"added cells {row['cells_added']} | removed {removed}",
            flush=True,
        )

    frame = pd.concat([sample["frame"] for sample in samples], ignore_index=True)
    active_mask = frame["active"].to_numpy(dtype=bool)
    predictor_names = (
        "wind_speed_m_s",
        "local_wind_alignment",
        "domain_wind_alignment",
        "R_m_s",
        "R_eff_m_s",
        "slope_deg",
        "fuel_moisture_pct",
        "relative_humidity_pct",
        "air_temperature_c",
        "elevation_m",
    )
    distributions = {}
    for name in predictor_names:
        values = frame[name].to_numpy(dtype=np.float64)
        distributions[name] = {
            "active": _distribution(values, active_mask),
            "inactive": _distribution(values, ~active_mask),
            "univariate_roc_auc": _ranking(frame["active"].to_numpy(), values)["roc_auc"],
        }
    classifier = _leave_one_interval(frame)
    prevalence = classifier["pooled"]["positives"] / classifier["pooled"]["n"] if classifier["pooled"]["n"] else None
    pooled = classifier["pooled"]
    enough = bool(
        pooled.get("roc_auc") is not None
        and pooled["roc_auc"] >= SIGNAL_ROC_AUC
        and pooled.get("balanced_accuracy") is not None
        and pooled["balanced_accuracy"] >= SIGNAL_BALANCED_ACCURACY
        and pooled.get("recall") is not None
        and pooled["recall"] >= SIGNAL_RECALL
    )
    interval_rows = [row for row in tables if "active_fraction" in row]
    payload = {
        "label": (
            "A point spaced along the old persistent boundary is active when the "
            "nearest cell that is outside the old mask and inside the next mask "
            f"is within one diagonal cell ({neighborhood:.1f} m). "
            "Engulfed is the separate check that the same point lies more than "
            "one cell inside the next boundary. Neither label uses total area."
        ),
        "features": (
            "Weather, terrain, R, and wind are sampled at the old boundary point, "
            "on the day of the old pass. The later pass supplies "
            "only the activity label. Domain-mean wind is the vector the PDE uses. "
            "Local wind is the stack direction at that boundary point. "
            "Validation and test passes are not used."
        ),
        "cell_m": spacing,
        "neighborhood_m": neighborhood,
        "c_wdir": c_wdir,
        "intervals": interval_rows,
        "class_balance": {
            "boundary_samples": int(len(frame)),
            "active_samples": int(active_mask.sum()),
            "active_fraction": float(active_mask.mean()),
            "inactive_fraction": float((~active_mask).mean()),
        },
        "distributions": distributions,
        "fuel_at_boundary": _fuel_table(frame["fuel_code"].to_numpy(dtype=np.float64), active_mask),
        "classifier": classifier,
        "positive_prevalence_on_scored_rows": prevalence,
        "enough_signal_for_spatially_varying_R": enough,
        "signal_rule": {
            "roc_auc_at_least": SIGNAL_ROC_AUC,
            "balanced_accuracy_at_least": SIGNAL_BALANCED_ACCURACY,
            "recall_at_least": SIGNAL_RECALL,
            "compared_with": "leave-one-interval-out logistic regression versus predicting every point inactive",
        },
    }
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pd.DataFrame(interval_rows).to_csv(OUT / "intervals.csv", index=False)
    frame.to_csv(OUT / "boundary_points.csv", index=False)
    _draw_fraction(interval_rows)
    _draw_intervals(samples, spacing)
    _draw_predictors(frame)
    _draw_classifier(classifier, float(prevalence or 0.0))
    print(
        f"active fraction {100 * payload['class_balance']['active_fraction']:.2f}% | "
        f"LOIO ROC-AUC {pooled.get('roc_auc')} | "
        f"balanced accuracy {pooled.get('balanced_accuracy')} | "
        f"enough signal {enough}",
        flush=True,
    )


if __name__ == "__main__":
    main()
