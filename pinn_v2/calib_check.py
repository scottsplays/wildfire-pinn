"""Gate 2: does the fire's own recent history fix the spread rate?  (numpy only, no torch)

    python -m pinn_v2.calib_check --quick      # 1 hidden map, 1 scenario: checks it runs (~5 min)
    python -m pinn_v2.calib_check              # full study on real LANDFIRE fuel (~45-60 min CPU)
    python -m pinn_v2.calib_check --case synthetic   # same on a made-up patchy fuel map (no static file needed)

Set-up (a "twin experiment": we make a truth we know, then see what each method recovers)
  * The model's spread rate R comes from LANDFIRE fuel, slope, wind and moisture, as in pinn_service.
  * The TRUE fire burns at R * m(x, t). The hidden multiplier m is the model error: overall ~1.5x too
    slow, with smooth patches from ~0.7x to ~3x. Methods never see m.
  * The true fire is stepped hour by hour from t = -6 h to t = +6 h; we "observe" it every hour.

Two directions, each using k = 1, 2, 3 or 6 hourly observations:
  forward   use the observations at -k ... 0 h to calibrate, then forecast +1 ... +6 h from the 0 h fire,
            stepping one hour at a time (each hour's prediction is the start of the next).
  backward  use the observations at 0 ... +k h to calibrate, then hindcast -1 ... -6 h from the 0 h fire
            (the physics run backwards: the burned area shrinks). Useful for filling the gaps between
            satellite passes and reconstructing how a fire grew, and a check on the calibration.

Methods
  persistence  the 0 h fire, unchanged
  model        uncalibrated physics (what pinn_service runs today)
  global       one rate factor for the whole fire, the one that best reproduces each observed hour
               from the hour before
  local        a smooth correction MAP, nudged up where the fire outran the physics and down where it
               lagged (a stand-in for the PINN / differentiable-solver correction)
  oracle       the true rate: the best any calibration could do on this grid

Scenarios
  clean   hidden m fixed in time, perfect 90 m observations
  drift   m grows ~6 % per hour (afternoon drying): old history is stale, so more k is not always better
  coarse  m fixed, observations blurred to 360 m blocks (roughly VIIRS 375 m pixels)

Scored against the TRUE fire: IoU, and change IoU = IoU of the area that burned between 0 h and the
target hour (forward) or that had not yet burned at the target hour (backward). Persistence scores 0
change IoU by definition, so any positive value is skill.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
from scipy import ndimage

from pinn_service import physics
from pinn_service.forecaster import metres_per_degree, signed_distance

REPO = Path(__file__).resolve().parent.parent
HOUR = 3600.0
KS = (1, 2, 3, 6)
SPAN = 6                                     # hours of history on each side of t = 0
ALPHAS = np.exp(np.linspace(np.log(0.25), np.log(6.0), 41))   # global factor search grid (~8 % steps)
SCENARIOS = ("clean", "drift", "coarse")
DRIFT_PER_H = 0.06


# --------------------------------------------------------------------------- terrain

def fuel_terrain(half: float = 12000.0, res: float = 90.0):
    from pinn_service.features import load_npz

    path = Path(os.environ.get("STATIC_LAYERS", REPO / "pinn_service" / "static" / "bay_area_static.npz"))
    if not path.exists():
        raise SystemExit(f"static layers not found at {path}; use --case synthetic or run python -m pinn_service.build_static_layers")
    static = load_npz(path, str(path))
    lon0, lat0 = -121.935, 38.004
    m_lat, m_lon = metres_per_degree(lat0)
    xs = np.arange(-half, half + res / 2, res)
    X, Y = np.meshgrid(xs, xs)
    fuel = static.sample("fuel_fbfm40", lon0 + X / m_lon, lat0 + Y / m_lat, fill=0.0)
    slope = static.sample("slope_deg", lon0 + X / m_lon, lat0 + Y / m_lat, fill=0.0)
    rate, _ = physics.spread_rate(fuel, slope, np.full(X.shape, 7.0), np.full(X.shape, 11.0))
    return xs, res, rate


def synthetic_terrain(half: float = 12000.0, res: float = 90.0, seed: int = 7):
    rng = np.random.default_rng(seed)
    xs = np.arange(-half, half + res / 2, res)
    n = len(xs)
    patches = ndimage.gaussian_filter(rng.normal(size=(n, n)), 5)
    patches /= patches.std()
    rate = np.clip(0.06 * np.exp(0.6 * patches), 0.01, 0.3)
    rate[patches < -1.2] = 0.0                                   # rock, water, roads
    X, Y = np.meshgrid(xs, xs)
    rate[np.hypot(X, Y) < 800] = np.maximum(rate[np.hypot(X, Y) < 800], 0.03)
    return xs, res, rate


def hidden_multiplier(shape, res: float, seed: int) -> np.ndarray:
    """The model error the methods must recover: ~1.5x overall, smooth patches (~1.5 km) from ~0.7x to ~3x."""
    rng = np.random.default_rng(1000 + seed)
    field = ndimage.gaussian_filter(rng.normal(size=shape), 1500.0 / res)
    field /= field.std()
    return 1.5 * np.exp(0.4 * field)


# --------------------------------------------------------------------------- physics helpers

class World:
    def __init__(self, xs, res, rate, wind_from_deg: float = 320.0):
        self.xs, self.res, self.rate = xs, res, rate
        east, north = physics.wind_toward_from_met(wind_from_deg)
        self.wind = (np.full(rate.shape, float(east)), np.full(rate.shape, float(north)))
        self.back_wind = (-self.wind[0], -self.wind[1])

    def step(self, phi, mult, hours=1.0, backward=False):
        """Advance phi by `hours` with rate R * mult. Backward: the burned area shrinks; equivalent to
        growing the unburned side with the wind reversed."""
        rate = self.rate * mult
        if backward:
            return -physics.propagate(-phi, self.res, rate, self.back_wind, [hours * HOUR])[hours * HOUR]
        return physics.propagate(phi, self.res, rate, self.wind, [hours * HOUR])[hours * HOUR]


def truth_run(world: World, m: np.ndarray, drift: bool) -> dict[int, np.ndarray]:
    X, Y = np.meshgrid(world.xs, world.xs)
    phi = np.hypot(X, Y) - 600.0
    fields = {-SPAN: phi}
    for t in range(-SPAN, SPAN):
        mult = m * np.exp(DRIFT_PER_H * (t + 0.5)) if drift else m
        phi = world.step(phi, mult)
        fields[t + 1] = phi
    return fields


def true_mult(m, t, drift):
    return m * np.exp(DRIFT_PER_H * (t + 0.5)) if drift else m


def observe(phi: np.ndarray, coarse: bool, block: int = 4) -> np.ndarray:
    burned = phi < 0
    if not coarse:
        return burned
    ny, nx = burned.shape
    py, px = (-ny) % block, (-nx) % block
    padded = np.pad(burned.astype(float), ((0, py), (0, px)))
    frac = padded.reshape(padded.shape[0] // block, block, padded.shape[1] // block, block).mean(axis=(1, 3))
    return np.kron(frac >= 0.5, np.ones((block, block), dtype=bool))[:ny, :nx]


# --------------------------------------------------------------------------- calibration

def sym_diff(pred_phi, target_mask):
    return int(((pred_phi < 0) ^ target_mask).sum())


class Calibrator:
    """Fits on observed hourly pairs (obs_i -> obs_{i+1}), always forward in time."""

    def __init__(self, world: World, obs: dict[int, np.ndarray]):
        self.world, self.obs = world, obs
        self.sd = {t: signed_distance(mask, world.res) for t, mask in obs.items()}
        self._cache: dict[tuple[int, int], int] = {}

    def pair_loss(self, t: int, a: int) -> int:
        key = (t, a)
        if key not in self._cache:
            self._cache[key] = sym_diff(self.world.step(self.sd[t], ALPHAS[a]), self.obs[t + 1])
        return self._cache[key]

    def global_factor(self, pairs: list[int]) -> float:
        losses = [sum(self.pair_loss(t, a) for t in pairs) for a in range(len(ALPHAS))]
        return float(ALPHAS[int(np.argmin(losses))])

    def local_map(self, pairs: list[int], alpha: float, iters: int = 8, eta: float = 0.6, smooth_m: float = 600.0):
        res = self.world.res
        m = np.full(self.world.rate.shape, alpha)
        sigma = smooth_m / res
        for _ in range(iters):
            push = np.zeros_like(m)
            weight = np.zeros_like(m)
            for t in pairs:
                pred = self.world.step(self.sd[t], m) < 0
                target, start = self.obs[t + 1], self.obs[t]
                push += (target & ~pred).astype(float) - (pred & ~target).astype(float)
                weight += ((target | pred) & ~start).astype(float)
            s_push = ndimage.gaussian_filter(push, sigma)
            s_w = ndimage.gaussian_filter(weight, sigma)
            confident = s_w > 0.02
            step = np.zeros_like(m)
            step[confident] = np.clip(s_push[confident] / s_w[confident], -1, 1)
            m = np.clip(m * np.exp(eta * step), 0.1, 12.0)
        return m


# --------------------------------------------------------------------------- scoring

def score(pred_phi, true_phi, true_phi0, res, backward: bool):
    a, b = pred_phi < 0, true_phi < 0
    union = (a | b).sum()
    iou = float((a & b).sum() / union) if union else 1.0
    start = true_phi0 < 0
    if backward:
        ca, cb = start & ~a, start & ~b         # area not yet burned at the target hour
    else:
        ca, cb = a & ~start, b & ~start         # area that burned after 0 h
    cu = (ca | cb).sum()
    change = float((ca & cb).sum() / cu) if cu else 1.0
    return round(iou, 4), round(change, 4)


def roll(world: World, phi0, mults, backward: bool) -> dict[int, np.ndarray]:
    """Step hour by hour from the 0 h fire; mults(h) gives the rate multiplier for hour h (1-based)."""
    out, phi = {}, phi0
    for h in range(1, SPAN + 1):
        phi = world.step(phi, mults(h), backward=backward)
        out[h] = phi
    return out


# --------------------------------------------------------------------------- study

def run_one(world: World, seed: int, scenario: str, rows: list, report: dict) -> None:
    m = hidden_multiplier(world.rate.shape, world.res, seed)
    drift, coarse = scenario == "drift", scenario == "coarse"
    truth = truth_run(world, m, drift)
    obs = {t: observe(phi, coarse) for t, phi in truth.items()}
    cal = Calibrator(world, obs)
    phi_start = cal.sd[0]                                   # every method starts from the OBSERVED 0 h fire
    burnable = world.rate > 0
    for direction in ("forward", "backward"):
        back = direction == "backward"
        sign = -1 if back else 1

        def record(method, k, fields, extra=None):
            for h in (1, 3, 6):
                iou, change = score(fields[h], truth[sign * h], truth[0], world.res, back)
                rows.append({"scenario": scenario, "direction": direction, "seed": seed, "method": method,
                             "k": k, "hours": sign * h, "iou": iou, "change_iou": change, **(extra or {})})

        persistence = {h: phi_start for h in range(1, SPAN + 1)}
        record("persistence", 0, persistence)
        record("model", 0, roll(world, phi_start, lambda h: 1.0, back))
        # oracle: the true multiplier for each hour stepped through
        record("oracle", 0, roll(world, phi_start, lambda h: true_mult(m, (-h if back else h - 1), drift), back))
        for k in KS:
            pairs = list(range(-k, 0)) if not back else list(range(0, k))
            alpha = cal.global_factor(pairs)
            record("global", k, roll(world, phi_start, lambda h: alpha, back), {"alpha": round(alpha, 3)})
            local = cal.local_map(pairs, alpha)
            err = float(np.median(np.abs(np.log(local[burnable] / m[burnable]))))
            record("local", k, roll(world, phi_start, lambda h: local, back), {"alpha": round(alpha, 3)})
            report.setdefault(f"{scenario}/{direction}/seed{seed}", {})[f"k{k}"] = {
                "global_factor": round(alpha, 3), "true_median_factor": round(float(np.median(m[burnable])), 3),
                "local_map_median_log_error": round(err, 3)}
        print(f"  {scenario:6s} {direction:8s} seed {seed}: done", flush=True)


def summarise(rows: list, out_dir: Path) -> None:
    def mean(sel):
        vals = [r["change_iou"] for r in rows if all(r[k] == v for k, v in sel.items())]
        return np.mean(vals) if vals else float("nan")

    lines = []
    for scenario in dict.fromkeys(r["scenario"] for r in rows):
        for direction in ("forward", "backward"):
            hrs = (6, 3, 1) if direction == "forward" else (-6, -3, -1)
            lines.append(f"\n--- {scenario}, {direction}: change IoU vs TRUE fire (mean over hidden maps), persistence = 0")
            lines.append(f"{'method':12s} {'k':>2s} " + " ".join(f"{h:+4d} h" for h in hrs))
            for method in ("model", "oracle"):
                lines.append(f"{method:12s} {'-':>2s} " + " ".join(f"{mean(dict(scenario=scenario, direction=direction, method=method, hours=h)):6.3f}" for h in hrs))
            for method in ("global", "local"):
                for k in KS:
                    lines.append(f"{method:12s} {k:2d} " + " ".join(
                        f"{mean(dict(scenario=scenario, direction=direction, method=method, k=k, hours=h)):6.3f}" for h in hrs))
    text = "\n".join(lines)
    print(text)
    (out_dir / "summary.txt").write_text(text)


def figure(rows: list, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scenarios = list(dict.fromkeys(r["scenario"] for r in rows))
    fig, axes = plt.subplots(2, len(scenarios), figsize=(4.6 * len(scenarios), 7.5), squeeze=False, sharey=True)
    for col, scenario in enumerate(scenarios):
        for row, direction in enumerate(("forward", "backward")):
            ax = axes[row][col]
            h = 6 if direction == "forward" else -6
            pick = lambda **sel: np.mean([r["change_iou"] for r in rows if r["scenario"] == scenario and r["direction"] == direction and r["hours"] == h and all(r[k] == v for k, v in sel.items())])
            for method, colour in (("global", "#2a6fdb"), ("local", "#d9480f")):
                ax.plot(KS, [pick(method=method, k=k) for k in KS], "o-", color=colour, label=f"{method} (calibrated)")
            ax.axhline(pick(method="model"), color="0.4", ls="--", label="model (uncalibrated)")
            ax.axhline(pick(method="oracle"), color="0.1", ls=":", label="oracle (true rate)")
            ax.set_title(f"{scenario}, {direction} ({h:+d} h)")
            ax.set_xticks(KS)
            ax.set_xlabel("hourly observations used (k)")
            if col == 0:
                ax.set_ylabel("change IoU vs truth")
            ax.set_ylim(0, 1)
            ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Gate 2: calibrating the spread rate from the fire's own history (persistence = 0)")
    fig.tight_layout()
    fig.savefig(out_dir / "calibration.png", dpi=110)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Gate 2 calibration study")
    p.add_argument("--case", choices=["fuel", "synthetic"], default="fuel")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="hidden multiplier maps")
    p.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    p.add_argument("--quick", action="store_true", help="1 map, clean scenario only")
    p.add_argument("--out", default=str(REPO / "outputs" / "pinn_v2_gate"))
    a = p.parse_args()
    if a.quick:
        a.seeds, a.scenarios = [0], ["clean"]
    xs, res, rate = fuel_terrain() if a.case == "fuel" else synthetic_terrain()
    world = World(xs, res, rate)
    out = Path(a.out) / f"calib_{a.case}" / ("quick" if a.quick else "full")
    out.mkdir(parents=True, exist_ok=True)
    print(f"Gate 2 calibration: {a.case} terrain, grid {rate.shape}, {res:g} m cells, burnable {float((rate > 0).mean()):.0%}; "
          f"scenarios {a.scenarios}, hidden maps {a.seeds}")
    rows, report, started = [], {}, time.perf_counter()
    for scenario in a.scenarios:
        for seed in a.seeds:
            run_one(world, seed, scenario, rows, report)
    with open(out / "calibration.csv", "w", newline="") as fh:
        fields = list(dict.fromkeys(k for r in rows for k in r))
        wr = csv.DictWriter(fh, fieldnames=fields)
        wr.writeheader()
        wr.writerows(rows)
    (out / "factors.json").write_text(json.dumps(report, indent=2))
    summarise(rows, out)
    figure(rows, out)
    print(f"\n({time.perf_counter() - started:.0f} s; table, factors and figure in {out})")


if __name__ == "__main__":
    main()