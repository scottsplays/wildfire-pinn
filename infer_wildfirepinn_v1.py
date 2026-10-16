"""Load frozen WildfirePINN v1 and run inference. No training.

The canonical checkpoint is
outputs/kincade_final_candidates/observation/levelset_pinn.pt.
This script checks its SHA-256, builds the network from the architecture
stored in that file, and uses the same normalization as predict_phi_meters.
Pass 0 is the smoke test. Pixel counts for train, validation, and test are
reported afterward. They are not used to change the model.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from shapely import contains_xy

from run_persistent_target_experiment import _kincade_config
from wildfire_pinn.burned_region import persistent_burned_regions
from wildfire_pinn.geometry import cell_size_m, zero_contours
from wildfire_pinn.pinn import (
    LevelSetPINN,
    domain_scales,
    get_device,
    load_checkpoint,
    load_fire_case,
    predict_phi_meters,
    resolve_time_split,
)
from wildfire_pinn.pinn import RealConfig

ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "outputs" / "kincade_final_candidates" / "observation" / "levelset_pinn.pt"
METRICS = ROOT / "outputs" / "wildfirepinn_v1" / "per_pass_metrics.csv"
HANDOFF = ROOT / "outputs" / "wildfirepinn_v1" / "handoff"
EXPECTED_SHA256 = "93de4791f6017664174b6dec0bc9434bd2bbe5fffd52dc63d795168f4890e996"
SMOKE_PASS = 0
ACCURACY_WARNING = (
    "Accuracy is high because most grid cells are outside the fire. "
    "True negatives dominate. Use IoU, Dice, symmetric boundary distance, "
    "and area error as the primary metrics."
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rates(tp: int, fp: int, tn: int, fn: int) -> dict:
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    specificity = tn / (tn + fp) if (tn + fp) else None
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total else None
    f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else None
    return {
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "accuracy": accuracy,
        "f1": f1,
        "positive_fraction": (tp + fn) / total if total else None,
    }


def _load_saved_model():
    device = get_device()
    try:
        blob = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    except TypeError:
        blob = torch.load(CHECKPOINT, map_location=device)
    arch = blob.get("arch") or {}
    hidden = int(arch["hidden"])
    layers = int(arch["layers"])
    model = LevelSetPINN(hidden, layers).to(device)
    saved = load_checkpoint(CHECKPOINT, model, device)
    model.eval()
    return model, saved, {"hidden": hidden, "layers": layers}


def _confusion(predicted: np.ndarray, target: np.ndarray) -> dict:
    pred = np.asarray(predicted, dtype=bool)
    truth = np.asarray(target, dtype=bool)
    tp = int(np.count_nonzero(pred & truth))
    fp = int(np.count_nonzero(pred & ~truth))
    tn = int(np.count_nonzero(~pred & ~truth))
    fn = int(np.count_nonzero(~pred & truth))
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, **_rates(tp, fp, tn, fn)}


def _write_handoff(saved: dict, confusion_frame: pd.DataFrame) -> None:
    HANDOFF.mkdir(parents=True, exist_ok=True)
    copy_path = HANDOFF / "levelset_pinn.pt"
    shutil.copy2(CHECKPOINT, copy_path)
    copy_digest = _sha256(copy_path)
    if copy_digest != EXPECTED_SHA256 or _sha256(CHECKPOINT) != EXPECTED_SHA256:
        raise RuntimeError("Checkpoint hash changed while preparing the handoff copy.")
    (HANDOFF / "SHA256.txt").write_text(EXPECTED_SHA256 + "\n", encoding="utf-8")
    (HANDOFF / "config.json").write_text(json.dumps(saved, indent=2), encoding="utf-8")
    confusion_frame.to_csv(HANDOFF / "confusion_matrix.csv", index=False)
    for name in ("MODEL_CARD.md", "HANDOFF.md"):
        source = ROOT / name
        if source.is_file():
            shutil.copy2(source, HANDOFF / name)
    metrics = ROOT / "outputs" / "wildfirepinn_v1" / "final_metrics.json"
    if metrics.is_file():
        shutil.copy2(metrics, HANDOFF / "final_metrics.json")
    readme = """# WildfirePINN v1 handoff bundle

Run this from the repository root, not from this folder:

```powershell
python infer_wildfirepinn_v1.py
```

`levelset_pinn.pt` in this folder is a copy of
`outputs/kincade_final_candidates/observation/levelset_pinn.pt`.
Both must match SHA-256 in `SHA256.txt`. The inference script loads the
canonical path. It does not train.

`config.json` is the config stored inside the checkpoint.
`final_metrics.json` is the frozen evaluation. IoU, Dice, symmetric boundary
distance, and area error are the primary scores.
`confusion_matrix.csv` is an extra pixel count. Accuracy is inflated by the
large unburned area.

The Kincade grids and the FIRMS table are not copied here. See `HANDOFF.md`
in the repository root for the file list. Full instructions are in that file
and in `MODEL_CARD.md`.
"""
    (HANDOFF / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    if not CHECKPOINT.is_file():
        raise RuntimeError(f"Missing frozen checkpoint {CHECKPOINT}")
    digest = _sha256(CHECKPOINT)
    print(f"sha256 {digest}", flush=True)
    if digest != EXPECTED_SHA256:
        raise RuntimeError(
            "Checkpoint hash does not match the frozen v1 file. "
            f"Expected {EXPECTED_SHA256}."
        )

    cfg = _kincade_config()
    case = load_fire_case(cfg.stack_nc, cfg.rate_nc, cfg.perimeters_path)
    split = resolve_time_split(
        len(case.perimeters),
        RealConfig().k_train,
        cfg.split_scheme,
        cfg.train_fraction,
        cfg.val_fraction,
    )
    if split.train != list(range(12)) or split.validation != list(range(12, 16)) or split.test != list(range(16, 20)):
        raise RuntimeError(f"Unexpected split: {split.train}, {split.validation}, {split.test}")
    regions, _detections = persistent_burned_regions(
        cfg.firms_csv, case.perimeters["dt"], crs=str(case.perimeters.crs)
    )
    center_x, center_y, length, _duration, _offset, tau = domain_scales(case)
    model, saved, arch = _load_saved_model()
    if float(saved.get("w_pde", -1)) != 0.0 or saved.get("observation_target") != "persistent_burn":
        raise RuntimeError("Loaded config is not the observation-only v1 checkpoint.")
    if arch != {"hidden": 128, "layers": 4}:
        raise RuntimeError(f"Saved architecture is {arch}, expected hidden 128 and 4 layers.")

    x = np.asarray(case.x, dtype=np.float64)
    y = np.asarray(case.y, dtype=np.float64)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    phi = predict_phi_meters(model, case, float(tau[SMOKE_PASS]))
    contours = zero_contours(phi, x, y)
    target = np.asarray(contains_xy(regions[SMOKE_PASS], xx, yy), dtype=bool)
    print(
        "smoke pass {pass_} time {time} tau {tau:.6f} L {length:.1f} m "
        "center ({cx:.1f}, {cy:.1f}) arch {arch} w_pde {w_pde} "
        "phi min {mn:.1f} max {mx:.1f} fraction_negative {frac:.6f} "
        "contour {contour} predicted_km2 {pred:.2f} target_km2 {tgt:.2f}".format(
            pass_=SMOKE_PASS,
            time=case.perimeters["dt"].iloc[SMOKE_PASS],
            tau=float(tau[SMOKE_PASS]),
            length=length,
            cx=center_x,
            cy=center_y,
            arch=arch,
            w_pde=saved.get("w_pde"),
            mn=float(phi.min()),
            mx=float(phi.max()),
            frac=float(np.mean(phi < 0)),
            contour=bool(contours),
            pred=float(np.count_nonzero(phi < 0)) * (cell_size_m(x, y) ** 2) / 1e6,
            tgt=float(target.sum()) * (cell_size_m(x, y) ** 2) / 1e6,
        ),
        flush=True,
    )

    stored = pd.read_csv(METRICS) if METRICS.is_file() else None
    per_pass = []
    pooled = {name: {"tp": 0, "fp": 0, "tn": 0, "fn": 0} for name in ("train", "validation", "test")}
    for index, geometry in enumerate(regions):
        field = phi if index == SMOKE_PASS else predict_phi_meters(model, case, float(tau[index]))
        truth = target if index == SMOKE_PASS else np.asarray(contains_xy(geometry, xx, yy), dtype=bool)
        counts = _confusion(field < 0, truth)
        split_name = split.label(index)
        for key in ("tp", "fp", "tn", "fn"):
            pooled[split_name][key] += counts[key]
        dice_stored = None
        if stored is not None:
            dice_stored = float(stored.loc[stored["pass"] == index, "dice"].iloc[0])
            if abs(counts["f1"] - dice_stored) > 1e-6:
                raise RuntimeError(
                    f"Pass {index} F1 {counts['f1']} disagrees with stored Dice {dice_stored}."
                )
        per_pass.append({
            "pass": index,
            "split": split_name,
            **counts,
            "dice_existing": dice_stored,
            "abs_f1_minus_dice": None if dice_stored is None else abs(counts["f1"] - dice_stored),
        })
        print(
            f"pass {index:02d} {split_name:10s} "
            f"TP {counts['tp']} FP {counts['fp']} TN {counts['tn']} FN {counts['fn']} "
            f"F1 {counts['f1']:.6f}",
            flush=True,
        )

    summary_rows = []
    for split_name, counts in pooled.items():
        rates = _rates(counts["tp"], counts["fp"], counts["tn"], counts["fn"])
        group = [row for row in per_pass if row["split"] == split_name]
        mean_dice = float(np.mean([row["dice_existing"] for row in group]))
        mean_f1 = float(np.mean([row["f1"] for row in group]))
        summary_rows.append({
            "split": split_name,
            "n_passes": len(group),
            "n_pixels": counts["tp"] + counts["fp"] + counts["tn"] + counts["fn"],
            **counts,
            **rates,
            "mean_f1_agrees_with_mean_dice": abs(mean_f1 - mean_dice) <= 1e-6,
            "mean_dice_existing": mean_dice,
            "pooled_f1_not_the_reported_dice": rates["f1"],
            "primary_metrics": "IoU, Dice, symmetric boundary distance, area error",
            "accuracy_warning": ACCURACY_WARNING,
        })
        print(
            f"{split_name:10s} precision {rates['precision']:.4f} recall {rates['recall']:.4f} "
            f"specificity {rates['specificity']:.4f} accuracy {rates['accuracy']:.4f} "
            f"mean F1 {mean_f1:.4f} mean Dice {mean_dice:.4f}",
            flush=True,
        )
    print(ACCURACY_WARNING, flush=True)

    out_dir = ROOT / "outputs" / "wildfirepinn_v1"
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(summary_rows)
    frame.to_csv(out_dir / "confusion_matrix.csv", index=False)
    pd.DataFrame(per_pass).to_csv(out_dir / "confusion_matrix_per_pass.csv", index=False)
    (out_dir / "confusion_matrix.json").write_text(
        json.dumps({"warning": ACCURACY_WARNING, "splits": summary_rows, "per_pass": per_pass}, indent=2),
        encoding="utf-8",
    )
    _write_handoff(saved, frame)
    if _sha256(CHECKPOINT) != EXPECTED_SHA256:
        raise RuntimeError("Canonical checkpoint hash changed.")
    print("canonical checkpoint unchanged", flush=True)


if __name__ == "__main__":
    main()
