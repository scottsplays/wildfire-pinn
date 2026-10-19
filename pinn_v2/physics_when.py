"""When does physics help? Splits the v0.4 VALIDATION forecasts by how much the weather changed.

    python -m pinn_v2.physics_when

Hypothesis (written before looking, 19 Oct 2026): persistence (edge:last) wins when the next hours
look like the last ones; the physics level set (levelset:alpha) should do relatively better when the
weather CHANGES between the observed interval and the forecast interval, because only the physics
model sees the new conditions.

"Changed" (fixed thresholds) = between the 12 h before the forecast start and the forecast interval:
  mean 10 m wind speed changed by >= 2 m/s, or mean wind direction turned by >= 45 deg,
  or mean equilibrium fuel moisture changed by >= 2 percentage points.
Reads outputs/realfire/v4_dev/forecasts.csv and data/fires/<id>/weather.csv; writes
outputs/realfire/v4_dev/physics_when.txt. No forecasts are re-run and no 2026 fire is touched.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pinn_service.native_ingestion import emc_percent
from pinn_v2 import realfire_round2 as r2

REPO = Path(__file__).resolve().parent.parent
PAIRS = [("levelset:alpha", "edge:last"), ("edge:alpha", "edge:last"), ("soft25:alpha", "edge:last")]
LEAD_H = {"12h": 12.0, "24h": 24.0}


def weather_change(folder: Path, h_start: float, span: float) -> dict:
    t0 = pd.Timestamp(str(np.load(folder / "arrival.npz")["t0_utc"]))          # h_start is hours after this
    w = pd.read_csv(folder / "weather.csv", parse_dates=["time_utc"])
    w["h"] = (w.time_utc - t0).dt.total_seconds() / 3600.0
    w["emc"] = np.clip(emc_percent(w.temp_c.to_numpy(), w.rh_pct.to_numpy()), 2, 35)

    def stats(a, b):
        x = w[(w.h >= a) & (w.h < b)]
        if not len(x):
            return None
        ang = np.deg2rad(x.wind_from_deg)
        return x.wind_speed_m_s.mean(), np.rad2deg(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())), x.emc.mean()

    before, after = stats(h_start - 12, h_start), stats(h_start, h_start + span)
    if before is None or after is None:
        return {}
    turn = abs((after[1] - before[1] + 180) % 360 - 180)
    return {"d_wind": after[0] - before[0], "turn_deg": turn, "d_emc": after[2] - before[2]}


def main():
    out = REPO / "outputs" / "realfire" / "v4_dev"
    df = pd.read_csv(out / "forecasts.csv")
    df = df[df.observed_new_cells > 0]
    keys = df[["fire_id", "h_start", "lead"]].drop_duplicates()
    rows = []
    for _, k in keys.iterrows():
        c = weather_change(r2.FIRES / k.fire_id, k.h_start, LEAD_H[k.lead])
        if c:
            rows.append({**k.to_dict(), **c})
    cond = pd.DataFrame(rows)
    cond["changed"] = (cond.d_wind.abs() >= 2) | (cond.turn_deg >= 45) | (cond.d_emc.abs() >= 2)
    df = df.merge(cond, on=["fire_id", "h_start", "lead"])
    L = ["WHEN DOES PHYSICS HELP? (validation forecasts, split by weather change; thresholds fixed in advance)",
         f"forecast starts: {len(cond)}, weather changed in {int(cond.changed.sum())} ({cond.changed.mean():.0%})"]
    for lead in ("12h", "24h"):
        for changed in (False, True):
            d = df[(df.lead == lead) & (df.changed == changed)]
            if not len(d):
                continue
            pf = d.pivot_table(index="fire_id", columns="method", values="change_iou", aggfunc="mean")
            L.append(f"\n--- lead {lead}, weather {'CHANGED' if changed else 'steady'}: {d.fire_id.nunique()} fires, "
                     f"{d[d.method == 'edge:last'].shape[0]} forecasts")
            for m in ("edge:last", "levelset:alpha", "edge:alpha", "soft25:alpha", "buffer", "physics"):
                if m in pf:
                    L.append(f"  {m:16s} new-area IoU {pf[m].mean():.3f}")
            for a, b in PAIRS:
                if a in pf and b in pf:
                    diff = (pf[a] - pf[b]).dropna()
                    if len(diff) >= 3:
                        lo, hi = r2.bootstrap_ci(diff)
                        L.append(f"  {a} - {b}: {diff.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]{' *' if lo > 0 else ''}"
                                 f" ({(diff > 0).mean():.0%} of fires)")
    L.append("\n* = 95% CI above zero. Read the CHANGED rows: if physics minus persistence is positive there and not\n"
             "in the steady rows, physics earns its place exactly where persistence has no information.")
    text = "\n".join(L)
    print(text)
    (out / "physics_when.txt").write_text(text)


if __name__ == "__main__":
    main()
