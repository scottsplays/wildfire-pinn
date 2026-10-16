"""Final WildfirePINN v1 selection on chronological Kincade.

Three production candidates, same network and persistent burned-region targets.
The only change is the PDE weight on the existing directed residual:

    A. w_pde = 0
    B. w_pde = 0.001
    C. w_pde = 0.01

The winner is chosen from validation. A candidate that loses a validation
zero contour is disqualified. Test metrics are reported after that choice.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import torch

from run_persistent_target_experiment import _draw_rings, _front_distance_m, _kincade_config
from run_three_pass_diagnostic import _capture_probe, _probe_loss
from wildfire_pinn.burned_region import persistent_burned_regions
from wildfire_pinn.cli import setup_logging
from wildfire_pinn.geometry import cell_size_m, signed_distance, zero_contours
from wildfire_pinn.pinn import (
    LevelSetPINN,
    RealConfig,
    _read_history,
    _supervision_pools,
    day_index_from_tau,
    directed_residual,
    domain_scales,
    downwind_unit_vectors,
    get_device,
    load_checkpoint,
    load_fire_case,
    predict_phi_meters,
    resolve_time_split,
    sample_volume,
    set_seed,
    train_real,
)
from wildfire_pinn.plots import _increasing_axes

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "kincade_final_candidates"
SDF_CACHE = ROOT / "outputs" / "kincade_target_new" / "sdf_train.npz"
CANDIDATES = (
    ("observation", "Observation only", 0.0),
    ("weak_pde_0p001", "Weak PDE 0.001", 0.001),
    ("weak_pde_0p01", "Weak PDE 0.01", 0.01),
)


def _hours(case) -> list[float]:
    start = case.perimeters["dt"].iloc[0]
    return [
        float((stamp - start).total_seconds()) / 3600.0
        for stamp in case.perimeters["dt"]
    ]


def _build_probes(case, regions, scales):
    cache = OUT / "persistent_sdfs.npz"
    length = scales["length"]
    if cache.exists():
        fields = np.load(cache)["sdf"].astype(np.float32)
    else:
        built = []
        for index, geometry in enumerate(regions):
            distance_m = signed_distance(geometry, case.x, case.y)
            built.append((distance_m / length).astype(np.float32))
            print(f"signed distance pass {index} [{distance_m.min():.0f}, {distance_m.max():.0f}] m", flush=True)
        fields = np.stack(built)
        np.savez_compressed(cache, sdf=fields)
    device = scales["device"]
    hp = RealConfig()
    set_seed(123)
    probes = []
    pools = []
    for index, geometry in enumerate(regions):
        pool = _supervision_pools(geometry, fields[index], case.x, case.y, device)
        pools.append(pool)
        probes.append(_capture_probe(pool, hp.n_obs))
    return fields, probes


def _train_one(case, regions, name: str, w_pde: float) -> None:
    out_dir = OUT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    destination = out_dir / "sdf_train.npz"
    if SDF_CACHE.exists() and not destination.exists() and not (out_dir / "levelset_pinn.pt").exists():
        shutil.copyfile(SDF_CACHE, destination)
    cfg = _kincade_config()
    hp = RealConfig(w_pde=w_pde)
    print(f"\n=== {name} | w_pde={w_pde:g} | epochs={hp.epochs} | seed={hp.seed} ===", flush=True)
    set_seed(hp.seed)
    train_real(
        case,
        out_dir,
        hp,
        retrain=False,
        split_scheme=cfg.split_scheme,
        train_fraction=cfg.train_fraction,
        val_fraction=cfg.val_fraction,
        sdf_sampling="thirds",
        target_geometries=regions,
    )


def _load_model(name: str) -> LevelSetPINN:
    hp = RealConfig()
    device = get_device()
    model = LevelSetPINN(hp.hidden, hp.layers).to(device)
    load_checkpoint(OUT / name / "levelset_pinn.pt", model, device)
    model.eval()
    return model


def _split_collocation(case, tau, indexes, count: int, seed: int):
    lo = float(tau[indexes[0]])
    hi = float(tau[indexes[-1]])
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    x = torch.rand(count, generator=generator)
    y = torch.rand(count, generator=generator)
    x = x * (float(case.x.max()) - float(case.x.min())) + float(case.x.min())
    y = y * (float(case.y.max()) - float(case.y.min())) + float(case.y.min())
    if hi <= lo:
        times = torch.full((count,), lo)
    else:
        times = torch.rand(count, generator=generator) * (hi - lo) + lo
    return x, y, times


def _pde_loss(model, case, scales, points) -> float:
    device = scales["device"]
    length = scales["length"]
    duration = scales["duration"]
    x_pde = points[0].to(device)
    y_pde = points[1].to(device)
    tau_pde = points[2].to(device).requires_grad_(True)
    x_norm = ((x_pde - scales["center_x"]) / length).detach().requires_grad_(True)
    y_norm = ((y_pde - scales["center_y"]) / length).detach().requires_grad_(True)
    phi = model(tau_pde, x_norm, y_norm)
    phi_t, phi_x, phi_y = torch.autograd.grad(
        phi, (tau_pde, x_norm, y_norm), torch.ones_like(phi), create_graph=False
    )
    n_days = int(scales["spread"].shape[0])
    days = day_index_from_tau(tau_pde, scales["t0_offset"], duration, n_days)
    rate = sample_volume(scales["spread"], days, case.x, case.y, x_pde, y_pde).detach()
    wind_now = scales["wind"][days]
    residual = directed_residual(
        phi_t, phi_x, phi_y, rate,
        wind_now[:, 0], wind_now[:, 1], scales["c_wdir"], duration / length,
    )
    return float(torch.mean(residual ** 2).detach())


def _evaluate_one(name, label, w_pde, case, regions, scales, fields, probes, collocation) -> dict:
    model = _load_model(name)
    hp = RealConfig()
    cell_km2 = scales["cell_km2"]
    rows = []
    model.eval()
    for index, geometry in enumerate(regions):
        phi_m = predict_phi_meters(model, case, float(scales["tau"][index]))
        contours = zero_contours(phi_m, case.x, case.y)
        distance = _front_distance_m(contours, geometry.boundary)
        obs = _probe_loss(
            model, float(scales["tau"][index]), probes[index], fields[index],
            case, scales["center_x"], scales["center_y"], scales["length"],
        )
        rows.append({
            "index": index,
            "time": str(case.perimeters["dt"].iloc[index]),
            "hours": scales["hours"][index],
            "split": scales["time_split"].label(index),
            "zero_contour": bool(contours),
            "predicted_km2": float((phi_m < 0).sum()) * cell_km2,
            "target_km2": float(geometry.area) / 1e6,
            "distance_to_target_m": distance,
            "min_phi_m": float(phi_m.min()),
            "observation_loss": obs,
            "phi": phi_m,
        })
    pde = {
        split_label: _pde_loss(model, case, scales, points)
        for split_label, points in collocation.items()
    }
    history = _read_history(OUT / name / "history.json")
    training_final = {}
    for key in ("total", "ic", "pde", "obs"):
        values = history.get(key) or []
        training_final[key] = float(values[-1]) if values else None
    payload = {
        "name": name,
        "label": label,
        "w_pde": w_pde,
        "epochs": hp.epochs,
        "passes": rows,
        "pde_loss_by_split": pde,
        "training_final": training_final,
        "history": {key: history.get(key) or [] for key in ("total", "ic", "pde", "obs")},
    }
    payload["splits"] = {
        split_label: _aggregate(rows, split_label, pde[split_label])
        for split_label in ("train", "validation", "test")
    }
    del model
    return payload


def _aggregate(rows, split_label: str, pde_loss: float) -> dict:
    group = [row for row in rows if row["split"] == split_label]
    distances = [row["distance_to_target_m"] for row in group if row["distance_to_target_m"] is not None]
    area_errors = [abs(row["predicted_km2"] - row["target_km2"]) for row in group]
    relative = [
        abs(row["predicted_km2"] - row["target_km2"]) / row["target_km2"]
        for row in group if row["target_km2"] > 0
    ]
    observations = [row["observation_loss"] for row in group]
    valid = sum(1 for row in group if row["zero_contour"])
    return {
        "n": len(group),
        "valid_zero_contour_fraction": valid / len(group) if group else None,
        "mean_contour_distance_m": float(np.mean(distances)) if distances else None,
        "median_contour_distance_m": float(np.median(distances)) if distances else None,
        "burned_area_mae_km2": float(np.mean(area_errors)) if area_errors else None,
        "burned_area_relative_error": float(np.mean(relative)) if relative else None,
        "observation_loss": float(np.mean(observations)) if observations else None,
        "pde_loss": pde_loss,
    }


def _qualified(item) -> bool:
    validation = [row for row in item["passes"] if row["split"] == "validation"]
    return bool(validation) and all(
        row["zero_contour"] and row["distance_to_target_m"] is not None for row in validation
    )


def _select(results: list[dict]) -> dict:
    qualified = [item for item in results if _qualified(item)]
    if not qualified:
        raise RuntimeError("Every candidate lost a validation zero contour.")

    def key(item):
        validation = item["splits"]["validation"]
        return (
            validation["mean_contour_distance_m"],
            validation["burned_area_mae_km2"],
            -item["w_pde"],
        )

    return min(qualified, key=key)


def _public_pass(row) -> dict:
    return {key: value for key, value in row.items() if key != "phi"}


def _public_result(item, selected: bool) -> dict:
    return {
        "name": item["name"],
        "label": item["label"],
        "w_pde": item["w_pde"],
        "epochs": item["epochs"],
        "selected": selected,
        "validation_qualified": _qualified(item),
        "splits": item["splits"],
        "training_final": item["training_final"],
        "passes": [_public_pass(row) for row in item["passes"]],
    }


def _write_maps(case, regions, results) -> None:
    map_dir = OUT / "maps"
    map_dir.mkdir(parents=True, exist_ok=True)
    x = np.asarray(case.x, dtype=np.float64)
    y = np.asarray(case.y, dtype=np.float64)
    held = [row for row in results[0]["passes"] if row["split"] != "train"]
    for row in held:
        index = row["index"]
        fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.8), layout="constrained")
        mesh = None
        for ax, item in zip(axes, results):
            record = item["passes"][index]
            shown, xd, yd = _increasing_axes(record["phi"], x, y)
            mesh = ax.pcolormesh(xd, yd, shown, cmap="RdBu", vmin=-5000, vmax=5000, shading="auto")
            ax.contour(xd, yd, shown, levels=[0], colors="black", linewidths=0.9)
            _draw_rings(ax, regions[index], "darkorange", 0.9, None)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlim(float(x.min()), float(x.max()))
            ax.set_ylim(float(y.min()), float(y.max()))
            distance = record["distance_to_target_m"]
            distance_text = "no contour" if distance is None else f"{distance / 1000:.2f} km"
            ax.set_title(
                f"{item['label']}\n{record['predicted_km2']:.0f} / {record['target_km2']:.0f} km², {distance_text}",
                fontsize=9,
            )
        fig.colorbar(mesh, ax=axes, fraction=0.02, label="predicted φ (m), capped at ±5 km")
        fig.suptitle(f"Pass {index}  {row['split']}  {row['time']}  |  black φ=0, orange persistent target")
        fig.savefig(map_dir / f"pass_{index:02d}_{row['split']}.png", dpi=130)
        plt.close(fig)


def _write_curves(results) -> None:
    hours = [row["hours"] for row in results[0]["passes"]]
    splits = [row["split"] for row in results[0]["passes"]]
    val_hours = [hour for hour, split in zip(hours, splits) if split == "validation"]
    test_hours = [hour for hour, split in zip(hours, splits) if split == "test"]
    fig, ax = plt.subplots(figsize=(8.4, 4.6), layout="constrained")
    if val_hours:
        ax.axvspan(min(val_hours), max(val_hours), color="#f4e4c4", label="validation", zorder=0)
    if test_hours:
        ax.axvspan(min(test_hours), max(test_hours), color="#d9e6f5", label="test", zorder=0)
    ax.plot(hours, [row["target_km2"] for row in results[0]["passes"]], color="black", linestyle="--", label="persistent target")
    for item in results:
        ax.plot(hours, [row["predicted_km2"] for row in item["passes"]], marker="o", label=item["label"])
    ax.set_xlabel("Hours since first pass")
    ax.set_ylabel("Burned area (km²)")
    ax.set_title("Predicted negative area")
    ax.legend(fontsize=8)
    fig.savefig(OUT / "area_vs_time.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.4, 4.6), layout="constrained")
    if val_hours:
        ax.axvspan(min(val_hours), max(val_hours), color="#f4e4c4", zorder=0)
    if test_hours:
        ax.axvspan(min(test_hours), max(test_hours), color="#d9e6f5", zorder=0)
    for item in results:
        distances = [
            np.nan if row["distance_to_target_m"] is None else row["distance_to_target_m"] / 1000.0
            for row in item["passes"]
        ]
        ax.plot(hours, distances, marker="o", label=item["label"])
    ax.set_xlabel("Hours since first pass")
    ax.set_ylabel("Contour distance (km)")
    ax.set_title("Distance from the predicted φ=0 contour to the persistent target")
    ax.legend(fontsize=8)
    fig.savefig(OUT / "distance_vs_time.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.0), layout="constrained")
    for ax, key, title in (
        (axes[0], "total", "Total loss"),
        (axes[1], "obs", "Observation loss"),
        (axes[2], "pde", "PDE loss"),
    ):
        for item in results:
            ax.plot(item["history"][key], label=item["label"])
        ax.set_yscale("log")
        ax.set_xlabel("epoch")
        ax.set_title(title)
        ax.legend(fontsize=7)
    fig.savefig(OUT / "losses.png", dpi=140)
    plt.close(fig)


def _write_table(results, winner_name: str) -> None:
    headers = [
        "candidate", "w_pde", "split", "contour %", "mean dist m", "median dist m",
        "area MAE km2", "area rel", "obs loss", "pde loss",
    ]
    lines = [",".join(headers)]
    table_rows = []
    for item in results:
        for split_label in ("train", "validation", "test"):
            stats = item["splits"][split_label]
            values = [
                item["label"],
                f"{item['w_pde']:g}",
                split_label,
                f"{100.0 * stats['valid_zero_contour_fraction']:.0f}",
                _fmt(stats["mean_contour_distance_m"], ".0f"),
                _fmt(stats["median_contour_distance_m"], ".0f"),
                _fmt(stats["burned_area_mae_km2"], ".1f"),
                _fmt(stats["burned_area_relative_error"], ".3f"),
                _fmt(stats["observation_loss"], ".3e"),
                _fmt(stats["pde_loss"], ".3e"),
            ]
            lines.append(",".join(values))
            if split_label == "validation":
                mark = "  <-- selected" if item["name"] == winner_name else ""
                table_rows.append(values + [mark])
    (OUT / "comparison_table.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    fig, ax = plt.subplots(figsize=(14.5, 2.6), layout="constrained")
    ax.axis("off")
    displayed = [row[:-1] for row in table_rows]
    widget = ax.table(cellText=displayed, colLabels=headers, loc="center")
    widget.auto_set_font_size(False)
    widget.set_fontsize(8)
    widget.scale(1, 1.4)
    for row_index, row in enumerate(table_rows):
        if row[-1]:
            for column in range(len(headers)):
                widget[row_index + 1, column].set_facecolor("#d9ead3")
    ax.set_title("Validation comparison. The selected candidate is highlighted.")
    fig.savefig(OUT / "summary_table.png", dpi=140)
    plt.close(fig)


def _fmt(value, spec: str) -> str:
    if value is None:
        return ""
    return format(value, spec)


def _print_summary(results, winner) -> None:
    print("\nVALIDATION", flush=True)
    print("candidate\tw\tcontours\tmean_m\tmedian_m\tarea_mae\trel\tobs\tpde", flush=True)
    for item in results:
        stats = item["splits"]["validation"]
        flag = "  SELECTED" if item["name"] == winner["name"] else ""
        print(
            f"{item['label']}\t{item['w_pde']:g}\t"
            f"{100.0 * stats['valid_zero_contour_fraction']:.0f}%\t"
            f"{stats['mean_contour_distance_m']:.0f}\t{stats['median_contour_distance_m']:.0f}\t"
            f"{stats['burned_area_mae_km2']:.1f}\t{stats['burned_area_relative_error']:.3f}\t"
            f"{stats['observation_loss']:.3e}\t{stats['pde_loss']:.3e}{flag}",
            flush=True,
        )
    test = winner["splits"]["test"]
    print(
        f"\nSELECTED {winner['label']} w_pde={winner['w_pde']:g}\n"
        f"TEST contours {100.0 * test['valid_zero_contour_fraction']:.0f}% | "
        f"mean {test['mean_contour_distance_m']:.0f} m | median {test['median_contour_distance_m']:.0f} m | "
        f"area MAE {test['burned_area_mae_km2']:.1f} km2 | rel {test['burned_area_relative_error']:.3f}",
        flush=True,
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    setup_logging(OUT / "train.log")
    cfg = _kincade_config()
    case = load_fire_case(cfg.stack_nc, cfg.rate_nc, cfg.perimeters_path)
    regions, _used = persistent_burned_regions(
        cfg.firms_csv, case.perimeters["dt"], crs=str(case.perimeters.crs)
    )
    if len(regions) != 20:
        raise RuntimeError(f"Expected 20 Kincade passes, found {len(regions)}.")
    center_x, center_y, length, duration, t0_offset, tau = domain_scales(case)
    device = get_device()
    time_split = resolve_time_split(
        len(case.perimeters), RealConfig().k_train, cfg.split_scheme, cfg.train_fraction, cfg.val_fraction
    )
    if time_split.train != list(range(12)) or time_split.validation != list(range(12, 16)):
        raise RuntimeError(f"Unexpected split: {time_split.train} {time_split.validation} {time_split.test}")
    scales = {
        "center_x": center_x,
        "center_y": center_y,
        "length": length,
        "duration": duration,
        "t0_offset": t0_offset,
        "tau": tau,
        "device": device,
        "hours": _hours(case),
        "time_split": time_split,
        "cell_km2": (cell_size_m(case.x, case.y) ** 2) / 1e6,
        "spread": torch.tensor(np.asarray(case.spread, dtype=np.float32), device=device),
        "wind": torch.tensor(downwind_unit_vectors(case.wind_from_deg), device=device),
        "c_wdir": RealConfig().c_wdir,
    }
    fields, probes = _build_probes(case, regions, scales)
    for name, _label, weight in CANDIDATES:
        _train_one(case, regions, name, weight)
    hp = RealConfig()
    collocation = {
        "train": _split_collocation(case, tau, time_split.train, hp.n_pde, 101),
        "validation": _split_collocation(case, tau, time_split.validation, hp.n_pde, 102),
        "test": _split_collocation(case, tau, time_split.test, hp.n_pde, 103),
    }
    results = [
        _evaluate_one(name, label, weight, case, regions, scales, fields, probes, collocation)
        for name, label, weight in CANDIDATES
    ]
    winner = _select(results)
    payload = {
        "selection_rule": (
            "Disqualify a candidate that lacks a zero contour on any validation pass. "
            "Among the rest, choose the lowest validation mean contour distance, "
            "then burned-area MAE, then the larger PDE weight. Test is not used."
        ),
        "selected": winner["name"],
        "selected_w_pde": winner["w_pde"],
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
            "w_obs": hp.w_obs,
            "c_wdir": hp.c_wdir,
            "sdf_sampling": "thirds",
            "observation_target": "persistent_burn",
            "split": {"train": time_split.train, "validation": time_split.validation, "test": time_split.test},
        },
        "candidates": [_public_result(item, item["name"] == winner["name"]) for item in results],
        "selected_test": winner["splits"]["test"],
    }
    (OUT / "comparison.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_maps(case, regions, results)
    _write_curves(results)
    _write_table(results, winner["name"])
    _print_summary(results, winner)
    for item in results:
        for row in item["passes"]:
            row.pop("phi", None)
            item.pop("history", None)


if __name__ == "__main__":
    main()
