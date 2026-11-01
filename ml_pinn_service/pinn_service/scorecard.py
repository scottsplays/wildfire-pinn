"""Forecast scorecard: how accurate were the model's forecasts, checked against later satellite passes.

For a past forecast time (``issued_at``) and lead times (default 12 h and 24 h):

  1. The forecast is re-run exactly as the service would have issued it at ``issued_at``: same code,
     same model, only satellite data from before that moment (the ingestion never returns later
     detections), same random seed.
  2. The VIIRS overpasses after ``issued_at`` are found. For each lead, the verifying pass is the
     last one within ``lead + tolerance`` hours; the forecast is evaluated at that pass's time.
  3. Predicted and observed NEW burned area are compared on the service's 1 km output grid (the
     cells the dashboard draws). "New" = not burning or burned before ``issued_at``, so a forecast
     of "no change" scores 0 and the existing fire gives no free credit.

Metrics per lead (new burned area only):
    new-area IoU = TP / (TP + FP + FN)       F1 = 2 TP / (2 TP + FP + FN) = 2 IoU / (1 + IoU)
    precision = TP / (TP + FP)               recall = TP / (TP + FN)
    growth ratio = predicted new area / observed new area (1 = right amount)
    Brier score of the cell probabilities near the fire
and the same overlap metrics for a no-skill reference, the current fire grown by one grid cell
(the "one-pixel dilation" baseline the U-Net is also compared with).

Status per lead: scored | no_growth (the fire did not grow: IoU undefined, false-alarm area still
reported) | pending (the verifying pass has not happened yet) | no_observation (no VIIRS pass in
the window) | no_fire (no wildland fire at issued_at).

Requests (stateless; the backend stores the results):
    POST /verify   {"bbox": [...], "date": "2026-08-04T12:00:00Z", "leads_hours": [12, 24],
                    "method": "calibrated"}                   (or a list of such objects)
    POST /predict  the same object with "task": "verify" (Vertex AI only routes /predict)

Replay mode (scores the deployed model on saved past fires, no network):
    python -m pinn_service.scorecard --replay data/fires --catalogue data/fire_catalogue_2026.csv
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy import ndimage
from pydantic import BaseModel, ConfigDict, Field, field_validator
from scipy.spatial import cKDTree

from . import calibrated as cal
from .contract import MODEL_VERSION, PredictionRequest, ServiceError, parse_reference_time
from .forecaster import forecast, metres_per_degree, search_bbox
from .native_ingestion import Detections, firms_detections
from .postprocessing import GridSpec

TOLERANCE_H = 3.0          # a pass up to this long after issued_at + lead still verifies the lead
EARLY_H = 6.0              # ... and one up to this long before it (VIIRS passes come in clusters ~12 h apart)
DOMAIN_KM = 10.0           # observed growth is counted within this distance of the fire at issued_at
FIRMS_MAX_HOURS = 115.0    # FIRMS area API serves at most 5 days per request


class VerifyInput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    bbox: list[float]
    date: str = Field(..., description="issued_at: the time the forecast was (or would have been) made, ISO-8601 UTC")
    leads_hours: list[int] = Field(default_factory=lambda: [12, 24])
    method: str = Field("calibrated", pattern="^(calibrated|nowcast)$")
    threshold: float | None = Field(None, gt=0.0, le=1.0, description="probability at or above which a cell counts as "
                                    "predicted to burn; default = the model's best-guess area (see median_threshold)")
    tolerance_hours: float = Field(TOLERANCE_H, ge=0.0, le=12.0)
    early_hours: float = Field(EARLY_H, ge=0.0, le=12.0)
    include_cells: bool = Field(False, description="also return the grid_ids of hit, false-alarm and missed cells (for map overlays)")
    additional_params: dict = Field(default_factory=dict)

    @field_validator("leads_hours")
    @classmethod
    def _leads(cls, value: list[int]) -> list[int]:
        value = sorted({int(v) for v in value})
        if not value or value[0] < 1 or value[-1] > 48:
            raise ValueError("leads_hours must be between 1 and 48")
        return value

    @field_validator("date")
    @classmethod
    def _date(cls, value: str) -> str:
        parse_reference_time(value)
        return value


class HorizonRequest(PredictionRequest):
    """A PredictionRequest whose horizons are set explicitly (the scorecard needs 12-30 h)."""

    fixed_horizons: list[int] = Field(default_factory=list)

    @property
    def horizons(self) -> list[int]:
        return self.fixed_horizons


def median_threshold(method: str) -> float:
    """The probability that marks the model's own best-guess (median-member) burned area: half the
    ensemble for 'calibrated'; for 'nowcast' the calibrated probability of half the members (the
    v0.4 probabilities are calibrated, so they rarely exceed 0.4 even where fire is expected)."""
    if method == "nowcast":
        c = cal.nowcast_config()["prob_calibration"]
        return round(float(np.interp(0.5, c["raw"], c["calibrated"])), 4)
    return 0.5


# --------------------------------------------------------------------------- satellite observations

def observations(source, area: list[float], start_s: float, end_s: float) -> Detections:
    if hasattr(source, "window"):                        # replay
        return source.window(area, start_s, end_s)
    end = dt.datetime.fromtimestamp(end_s, dt.timezone.utc)
    hours = min((end_s - start_s) / 3600.0, FIRMS_MAX_HOURS)
    return firms_detections(area, end, hours)


def detection_cells(dets: Detections, grid: GridSpec, radius_m: float = cal.VIIRS_RADIUS_M) -> set[int]:
    """1 km cells touched by each detection's 375 m pixel (a 212 m disk, as the forecast paints it:
    centre plus four points on the disk edge)."""
    if len(dets) == 0:
        return set()
    m_lat, m_lon = metres_per_degree(float(np.mean(dets.lat)))
    lon, lat, r = dets.lon, dets.lat, radius_m
    xs = [lon, lon + r / m_lon, lon - r / m_lon, lon, lon]
    ys = [lat, lat, lat, lat + r / m_lat, lat - r / m_lat]
    flat = grid.locate(np.concatenate(xs), np.concatenate(ys))
    return {int(c) for c in flat[flat >= 0]}


def dilate(cells: set[int], grid: GridSpec, steps: int = 1) -> set[int]:
    """Cells within ``steps`` cells (8-neighbourhood) of ``cells``, on a local raster window."""
    if not cells or steps <= 0:
        return set(cells)
    flat = np.fromiter(cells, dtype=np.int64)
    rows, cols = np.divmod(flat, grid.ncols)
    r0, c0 = max(rows.min() - steps, 0), max(cols.min() - steps, 0)
    r1, c1 = min(rows.max() + steps + 1, grid.nrows), min(cols.max() + steps + 1, grid.ncols)
    box = np.zeros((r1 - r0, c1 - c0), dtype=bool)
    box[rows - r0, cols - c0] = True
    box = ndimage.binary_dilation(box, structure=np.ones((3, 3), bool), iterations=steps)
    box &= grid.ids[r0:r1, c0:c1] > 0
    rr, cc = np.nonzero(box)
    return {int(v) for v in (rr + r0) * grid.ncols + (cc + c0)}


def centres_m(cells, grid: GridSpec, lat0: float) -> np.ndarray:
    m_lat, m_lon = metres_per_degree(lat0)
    pts = []
    for c in cells:
        x0, y0, x1, y1 = grid.bounds(c)
        pts.append(((x0 + x1) / 2 * m_lon, (y0 + y1) / 2 * m_lat))
    return np.array(pts, dtype=float).reshape(-1, 2)


def near(cells: set[int], anchor: set[int], grid: GridSpec, km: float) -> set[int]:
    if not cells or not anchor:
        return set()
    lat0 = grid.min_lat + grid.nrows * grid.dlat / 2
    tree = cKDTree(centres_m(anchor, grid, lat0))
    cells = list(cells)
    dist, _ = tree.query(centres_m(cells, grid, lat0))
    return {c for c, d in zip(cells, dist) if d <= km * 1000.0}


# --------------------------------------------------------------------------- metrics

def overlap(pred: set[int], obs: set[int], cell_km2: float) -> dict:
    tp, fp, fn = len(pred & obs), len(pred - obs), len(obs - pred)
    out = {
        "new_area_iou": round(tp / (tp + fp + fn), 4) if (tp + fp + fn) else None,
        "f1": round(2 * tp / (2 * tp + fp + fn), 4) if (tp + fp + fn) else None,
        "precision": round(tp / (tp + fp), 4) if (tp + fp) else None,
        "recall": round(tp / (tp + fn), 4) if (tp + fn) else None,
        "growth_ratio": round(len(pred) / len(obs), 3) if obs else None,
        "predicted_new_km2": round(len(pred) * cell_km2, 2),
        "observed_new_km2": round(len(obs) * cell_km2, 2),
        "hit_km2": round(tp * cell_km2, 2),
        "false_alarm_km2": round(fp * cell_km2, 2),
        "missed_km2": round(fn * cell_km2, 2),
    }
    return out


def brier(prob: dict[int, float], obs: set[int], domain: set[int]) -> dict:
    if not domain:
        return {"brier": None, "base_rate": None}
    p = np.array([prob.get(c, 0.0) for c in domain])
    o = np.array([c in obs for c in domain], dtype=float)
    return {"brier": round(float(np.mean((p - o) ** 2)), 5), "base_rate": round(float(o.mean()), 4)}


# --------------------------------------------------------------------------- verify one forecast

def verify(payload: dict, source, grid: GridSpec, now: dt.datetime | None = None) -> dict:
    started = time.perf_counter()
    try:
        vin = VerifyInput.model_validate(payload)
    except Exception as exc:  # pydantic ValidationError
        raise ServiceError(422, f"verify request: {exc}") from exc
    now = now or dt.datetime.now(dt.timezone.utc)
    params = {"method": vin.method, "seed": None} | {k: v for k, v in vin.additional_params.items() if k != "method"}
    base = PredictionRequest.model_validate({"bbox": vin.bbox, "time_horizon": 6, "model_type": "pinn",
                                             "date": vin.date, "additional_params": params})
    p = base.additional_params
    issued = base.reference_time
    t0 = issued.timestamp()
    threshold = vin.threshold if vin.threshold is not None else median_threshold(vin.method)
    area = search_bbox(base.bbox, p.search_buffer_m)
    window_end = t0 + (max(vin.leads_hours) + vin.tolerance_hours) * 3600.0
    history_h = max(p.lookback_hours, min(p.history_hours, FIRMS_MAX_HOURS - (window_end - t0) / 3600.0))
    dets = observations(source, area, t0 - history_h * 3600.0, window_end)
    before = dets.subset(dets.time_s <= t0)
    after = dets.subset(dets.time_s > t0)
    passes = cal.pass_times(after.time_s)
    cell_km2 = grid.resolution_km ** 2

    # verifying pass and integer forecast horizon for each lead
    plan = {}
    for lead in vin.leads_hours:
        cands = [t for t in passes
                 if t0 + max(lead - vin.early_hours, 0.0) * 3600.0 < t <= t0 + (lead + vin.tolerance_hours) * 3600.0]
        if cands:
            v = cands[-1]
            plan[lead] = {"valid_s": v, "horizon": max(1, int(round((v - t0) / 3600.0)))}
        else:
            plan[lead] = None

    out = {
        "type": "pinn_scorecard",
        "model_version": MODEL_VERSION,
        "method": vin.method,
        "issued_at": issued.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "bbox": base.bbox,
        "threshold": threshold,
        "grid": grid.name,
        "grid_cell_km2": round(cell_km2, 4),
        "results": [],
    }
    horizons = sorted({v["horizon"] for v in plan.values() if v})
    req = HorizonRequest.model_validate(base.model_dump() | {"fixed_horizons": [0] + horizons})
    probs, meta = forecast(req, source, grid)
    out["fires_detected"] = meta.get("fires_detected", 0)
    out["forecast_notes"] = [f"fire {f.get('fire')}: calibration {f.get('calibration_mode', f.get('nowcast', ''))}, "
                             f"alpha {f.get('alpha', '-')}" for f in meta.get("fires", [])]

    prior = detection_cells(before, grid)
    recent = detection_cells(before.subset(before.time_s >= t0 - p.lookback_hours * 3600.0), grid)
    for lead in vin.leads_hours:
        row: dict = {"lead_hours": lead}
        due = issued + dt.timedelta(hours=lead + vin.tolerance_hours)
        if not meta.get("fires_detected") or not recent:
            row["status"] = "no_fire"
        elif plan[lead] is None:
            row["status"] = "pending" if now < due else "no_observation"
        else:
            v_s, h = plan[lead]["valid_s"], plan[lead]["horizon"]
            seen = detection_cells(dets.subset(dets.time_s <= v_s + cal.SEEN_SLACK_S), grid)
            obs_new = near(seen - prior, prior, grid, DOMAIN_KM)
            prob_h = probs.get(h, {})
            pred_new = {c for c, q in prob_h.items() if q >= threshold} - prior
            base_new = dilate(recent, grid) - prior
            row.update({
                "status": "scored" if obs_new else "no_growth",
                "valid_at": dt.datetime.fromtimestamp(v_s, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "forecast_horizon_hours": h,
                "burned_km2_at_issue": round(len(prior) * cell_km2, 2),
            })
            row["model"] = overlap(pred_new, obs_new, cell_km2) | brier(
                prob_h, obs_new, dilate(prior, grid, int(math.ceil(DOMAIN_KM / grid.resolution_km))) - prior)
            row["baseline_dilation"] = overlap(base_new, obs_new, cell_km2)
            if vin.include_cells:
                gid = lambda cells: sorted(grid.grid_id(c) for c in cells)  # noqa: E731
                row["cells"] = {"burned_at_issue": gid(prior), "hit": gid(pred_new & obs_new),
                                "false_alarm": gid(pred_new - obs_new), "missed": gid(obs_new - pred_new)}
        out["results"].append(row)
    out["runtime_s"] = round(time.perf_counter() - started, 2)
    return out


# --------------------------------------------------------------------------- summaries

METRICS = ("new_area_iou", "f1", "precision", "recall")


def summarize(records: list[dict], by_fire: bool = True) -> list[dict]:
    """Mean scores per (method, lead) over scored forecasts. With by_fire, each fire counts once
    (forecasts averaged within a fire first), as in the report's tests."""
    rows = []
    for rec in records:
        for r in rec.get("results", []):
            rows.append({"method": rec["method"], "lead": r["lead_hours"], "fire": rec.get("fire_id", rec["issued_at"]),
                         "status": r["status"], "model": r.get("model"), "base": r.get("baseline_dilation")})
    out = []
    for method in sorted({r["method"] for r in rows}):
        for lead in sorted({r["lead"] for r in rows if r["method"] == method}):
            sel = [r for r in rows if r["method"] == method and r["lead"] == lead]
            scored = [r for r in sel if r["status"] == "scored"]
            entry = {"method": method, "lead_hours": lead, "forecasts_scored": len(scored),
                     "fires_scored": len({r["fire"] for r in scored}),
                     "no_growth": sum(r["status"] == "no_growth" for r in sel),
                     "no_observation": sum(r["status"] in ("no_observation", "pending") for r in sel)}
            for who in ("model", "base"):
                for m in METRICS + ("growth_ratio",):
                    vals = {}
                    for r in scored:
                        x = r[who].get(m)
                        if x is not None:
                            vals.setdefault(r["fire"] if by_fire else id(r), []).append(x)
                    per = [float(np.mean(v)) for v in vals.values()]
                    if m == "growth_ratio":
                        entry[f"{who}_{m}_median"] = round(float(np.median([x for v in vals.values() for x in v])), 3) if per else None
                    else:
                        entry[f"{who}_{m}"] = round(float(np.mean(per)), 4) if per else None
            ng = [r["model"]["false_alarm_km2"] for r in sel if r["status"] == "no_growth"]
            entry["false_alarm_km2_when_no_growth"] = round(float(np.mean(ng)), 2) if ng else None
            out.append(entry)
    return out


def bootstrap_ci(values: list[float], n: int = 2000, seed: int = 0) -> tuple[float, float] | None:
    if len(values) < 3:
        return None
    rng = np.random.default_rng(seed)
    v = np.asarray(values, float)
    means = v[rng.integers(0, len(v), (n, len(v)))].mean(axis=1)
    return round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4)


# --------------------------------------------------------------------------- replay mode

def replay_issue_times(src, hours_utc=(0, 12), max_days: float | None = None) -> list[dt.datetime]:
    """Issue times while the fire is active: every day at the given UTC hours (just after the night
    and afternoon VIIRS passes over California), from the first detection to the last."""
    t = src.all.time_s
    if len(t) == 0:
        return []
    first = dt.datetime.fromtimestamp(t.min(), dt.timezone.utc)
    last = dt.datetime.fromtimestamp(t.max(), dt.timezone.utc)
    if max_days:
        last = min(last, first + dt.timedelta(days=max_days))
    day = first.replace(hour=0, minute=0, second=0, microsecond=0)
    out = []
    while day <= last:
        for h in hours_utc:
            moment = day + dt.timedelta(hours=h)
            if first < moment < last:
                out.append(moment)
        day += dt.timedelta(days=1)
    return out


def replay(fires_dir: Path, fire_ids: list[str], methods: list[str], leads: list[int], out_path: Path,
           max_days: float | None = None, log=print) -> list[dict]:
    from .postprocessing import active_grid
    from .replay import ReplaySource

    grid = active_grid()
    done = set()
    records = []
    if out_path.exists():                                  # resume
        for line in out_path.read_text().splitlines():
            rec = json.loads(line)
            records.append(rec)
            done.add((rec["fire_id"], rec["method"], rec["issued_at"]))
    with open(out_path, "a") as fh:
        for fid in fire_ids:
            src = ReplaySource(fires_dir / fid)
            bbox = src.bbox()
            times = replay_issue_times(src, max_days=max_days)
            log(f"{fid}: {len(times)} issue times")
            for moment in times:
                for method in methods:
                    stamp = moment.strftime("%Y-%m-%dT%H:%M:%SZ")
                    if (fid, method, stamp) in done:
                        continue
                    rec = verify({"bbox": bbox, "date": stamp, "leads_hours": leads, "method": method,
                                  "additional_params": {"search_buffer_m": 0.0}}, src, grid,
                                 now=dt.datetime.now(dt.timezone.utc))
                    rec["fire_id"] = fid
                    records.append(rec)
                    fh.write(json.dumps(rec) + "\n")
                    fh.flush()
                    s = " ".join(f"{r['lead_hours']}h:{r['status']}"
                                 + (f"/IoU {r['model']['new_area_iou']}" if r["status"] == "scored" else "")
                                 for r in rec["results"])
                    log(f"  {stamp} {method:10s} {s}  ({rec['runtime_s']} s)")
    return records


def write_summary(records: list[dict], path: Path) -> str:
    lines = [f"PINN scorecard replay: model {MODEL_VERSION}, {len({r['fire_id'] for r in records})} fires, "
             f"{len(records)} forecast runs; 1 km output grid; new burned area only",
             "", "Each fire counts once (its forecasts are averaged first); 95% bootstrap CI over fires.", ""]
    summary = summarize(records)
    header = (f"{'method':11s} {'lead':>4s} {'fires':>5s} {'fcsts':>5s} {'IoU':>6s} {'95% CI':>17s} {'F1':>6s} "
              f"{'prec':>6s} {'recall':>6s} {'growth':>6s} | {'dilation IoU':>12s} {'F1':>6s}")
    lines.append(header)
    for s in summary:
        per_fire = {}
        for rec in records:
            if rec["method"] != s["method"]:
                continue
            for r in rec["results"]:
                if r["lead_hours"] == s["lead_hours"] and r["status"] == "scored":
                    per_fire.setdefault(rec["fire_id"], []).append(r["model"]["new_area_iou"])
        ci = bootstrap_ci([float(np.mean(v)) for v in per_fire.values()])
        fmt = lambda x: "  -   " if x is None else f"{x:6.3f}"  # noqa: E731
        lines.append(f"{s['method']:11s} {s['lead_hours']:>3d}h {s['fires_scored']:>5d} {s['forecasts_scored']:>5d} "
                     f"{fmt(s['model_new_area_iou'])} {('[%.3f, %.3f]' % ci) if ci else '':>17s} {fmt(s['model_f1'])} "
                     f"{fmt(s['model_precision'])} {fmt(s['model_recall'])} {fmt(s['model_growth_ratio_median'])} | "
                     f"{fmt(s['base_new_area_iou']):>12s} {fmt(s['base_f1'])}")
    lines += ["", "growth = median predicted / observed new area; dilation = current fire grown by one 1 km cell.",
              "Forecasts where the fire did not grow are excluded from IoU (undefined) and counted separately:"]
    for s in summary:
        lines.append(f"  {s['method']} {s['lead_hours']}h: {s['no_growth']} no-growth forecasts, mean false alarm "
                     f"{s['false_alarm_km2_when_no_growth']} km2; {s['no_observation']} without a verifying pass")
    text = "\n".join(lines)
    path.write_text(text + "\n")
    (path.with_suffix(".json")).write_text(json.dumps(summary, indent=2))
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="PINN forecast scorecard (replay mode)")
    parser.add_argument("--replay", required=True, help="folder of saved fires, e.g. data/fires")
    parser.add_argument("--catalogue", help="CSV with a fire_id column (default: every folder starting with 2026)")
    parser.add_argument("--fires", nargs="*", help="fire ids (overrides --catalogue)")
    parser.add_argument("--methods", nargs="+", default=["calibrated", "nowcast"])
    parser.add_argument("--leads", nargs="+", type=int, default=[12, 24])
    parser.add_argument("--max-days", type=float, default=None, help="only the first N days of each fire")
    parser.add_argument("--out", default="outputs/scorecard")
    args = parser.parse_args()
    fires_dir = Path(args.replay)
    if args.fires:
        ids = args.fires
    elif args.catalogue:
        with open(args.catalogue, newline="") as fh:
            ids = [r["fire_id"] for r in csv.DictReader(fh)]
    else:
        ids = sorted(p.name for p in fires_dir.iterdir() if p.name.startswith("2026"))
    ids = [i for i in ids if (fires_dir / i / "static.npz").exists()]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    records = replay(fires_dir, ids, args.methods, args.leads, out / "replay_forecasts.jsonl", args.max_days)
    print()
    print(write_summary(records, out / "replay_summary.txt"))


if __name__ == "__main__":
    main()
