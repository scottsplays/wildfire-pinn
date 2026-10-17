"""WildfirePINN v2, gate 1: does the hard-IC level-set PINN solve the spread equation?

    python -m pinn_v2.gate_check                 # both cases, ~5-15 min on a laptop CPU
    python -m pinn_v2.gate_check --quick         # 600 steps, a 1-2 min smoke run
    python -m pinn_v2.gate_check --case fuel     # one case

The PINN (plan doc, "The v2 approach", steps 1-2):

    plain:  phi(x, y, t) = phi0(x, y)                + tau * S * g_theta(tau, X, Y)
    local:  phi(x, y, t) = phi0(x, y) - t * F0(x, y) + tau * S * g_theta(tau, X, Y)   (default)

    tau = t / H,  S = R_ref * H,  F0 = R (1 + 0.6 n0.w)_+  (local front speed, normal of phi0,
    smoothed over one cell)

phi0 is the signed distance to the current fire (metres, negative inside), so
tau = 0 is the observed fire exactly. The "local" form gives the network the
physics' first guess (every point moves at its own local speed): it already stops
at non-burnable ground and leans downwind, so g only learns the non-local
correction, e.g. a front that arrives late because it crossed slow fuel first.
Plain networks learn smooth averages first (spectral bias) and on a patchy fuel
map produce a round blob. g is trained ONLY on the level-set residual

    phi_t + R (1 + 0.6 n.w)_+ |grad phi| = 0

the same equation, rate and wind factor as pinn_service, written as a front-speed error
(F - V_n) so it stays bounded where phi is steep. No future observation is used.

The reference is the Godunov solver in pinn_service.physics (boundary bug fixed).
Gate (decided before running): at 6 h, IoU(PINN, solver) >= 0.90, IoU of the NEW
burned area >= 0.70, and mean boundary distance <= 2 cells. Total-area IoU alone is
too easy: the hard initial condition gets the starting fire right for free.

Cases
  circle  uniform R = 0.05 m/s, no wind; exact answer r(t) = r0 + R t
  fuel    real LANDFIRE fuel/slope around today's Pittsburg-hills detection
          (-121.935, 38.004), NW wind 7 m/s, FM100 11 % (gridMET's value there today):
          patchy, heterogeneous R, the hard case.

Outputs: outputs/pinn_v2_gate/<case>_<ansatz>.json, .png, _model.pt
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import ndimage

from pinn_service import physics
from pinn_service.forecaster import metres_per_degree, signed_distance

HORIZONS_H = (1, 3, 6)
C_WDIR = physics.C_WDIR
GATE = {"iou": 0.90, "growth_iou": 0.70, "boundary_cells": 2.0}
REPO = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- cases

@dataclass
class Case:
    name: str
    res: float                       # m
    xs: np.ndarray                   # cell centres, m (relative to window centre)
    ys: np.ndarray
    phi0: np.ndarray                 # m, negative inside
    rate: np.ndarray                 # m/s
    wind: tuple[np.ndarray, np.ndarray] | None   # unit vector toward (east, north)
    horizon_s: float = 6 * 3600.0
    notes: dict = field(default_factory=dict)

    @property
    def half_width(self) -> float:
        return float(max(self.xs[-1] - self.xs[0], self.ys[-1] - self.ys[0]) / 2)


def circle_case() -> Case:
    res, half = 60.0, 6000.0
    xs = np.arange(-half, half + res / 2, res)
    X, Y = np.meshgrid(xs, xs)
    r0, rate = 500.0, 0.05
    phi0 = np.hypot(X, Y) - r0
    return Case("circle", res, xs, xs.copy(), phi0, np.full(X.shape, rate), None,
                notes={"r0_m": r0, "R_m_s": rate, "exact_radius_6h_m": r0 + rate * 6 * 3600})


def fuel_case(static_path: Path | None = None) -> Case:
    from pinn_service.features import load_npz

    path = Path(static_path or os.environ.get("STATIC_LAYERS", REPO / "pinn_service" / "static" / "bay_area_static.npz"))
    if not path.exists():
        raise SystemExit(f"static layers not found at {path}; run python -m pinn_service.build_static_layers")
    static = load_npz(path, str(path))
    lon0, lat0 = -121.935, 38.004
    res, half = 90.0, 8000.0
    m_lat, m_lon = metres_per_degree(lat0)
    xs = np.arange(-half, half + res / 2, res)
    X, Y = np.meshgrid(xs, xs)
    lon, lat = lon0 + X / m_lon, lat0 + Y / m_lat
    fuel = static.sample("fuel_fbfm40", lon, lat, fill=0.0)
    slope = static.sample("slope_deg", lon, lat, fill=0.0)
    wind_speed, wind_from, fm100 = 7.0, 320.0, 11.0
    rate, unknown = physics.spread_rate(fuel, slope, np.full(X.shape, wind_speed), np.full(X.shape, fm100))
    east, north = physics.wind_toward_from_met(wind_from)
    burned = np.hypot(X, Y) <= 600.0
    burnable = rate > 0
    return Case(
        "fuel", res, xs, xs.copy(), signed_distance(burned, res), rate,
        (np.full(X.shape, float(east)), np.full(X.shape, float(north))),
        notes={
            "centre_lon_lat": [lon0, lat0], "static_layers": str(path),
            "wind_m_s": wind_speed, "wind_from_deg": wind_from, "fm100_pct": fm100,
            "burnable_fraction": round(float(burnable.mean()), 3),
            "R_median_m_s": round(float(np.median(rate[burnable])), 4) if burnable.any() else 0.0,
            "R_p95_m_s": round(float(np.percentile(rate[burnable], 95)), 4) if burnable.any() else 0.0,
            "unknown_fuel_codes": unknown,
        },
    )


# --------------------------------------------------------------------------- reference + metrics

def reference(case: Case) -> dict[int, np.ndarray]:
    fields = physics.propagate(case.phi0, case.res, case.rate, case.wind, [h * 3600.0 for h in HORIZONS_H])
    return {h: fields[h * 3600.0] for h in HORIZONS_H}


def _boundary(mask: np.ndarray) -> np.ndarray:
    return mask & ~ndimage.binary_erosion(mask)


def compare(pred_phi: np.ndarray, ref_phi: np.ndarray, phi0: np.ndarray, res: float) -> dict:
    a, b, start = pred_phi < 0, ref_phi < 0, phi0 < 0
    union = (a | b).sum()
    iou = float((a & b).sum() / union) if union else 1.0
    ga, gb = a & ~start, b & ~start
    gunion = (ga | gb).sum()
    growth_iou = float((ga & gb).sum() / gunion) if gunion else 1.0
    ba, bb = _boundary(a), _boundary(b)
    if ba.any() and bb.any():
        d_ab = ndimage.distance_transform_edt(~bb)[ba].mean()
        d_ba = ndimage.distance_transform_edt(~ba)[bb].mean()
        boundary_cells = float((d_ab + d_ba) / 2)
    else:
        boundary_cells = float("inf")
    return {
        "iou": round(iou, 4),
        "growth_iou": round(growth_iou, 4),
        "boundary_cells": round(boundary_cells, 3),
        "boundary_m": round(boundary_cells * res, 1),
        "area_pred_km2": round(float(a.sum()) * res * res / 1e6, 3),
        "area_ref_km2": round(float(b.sum()) * res * res / 1e6, 3),
        "growth_pred_km2": round(float(ga.sum()) * res * res / 1e6, 3),
        "growth_ref_km2": round(float(gb.sum()) * res * res / 1e6, 3),
    }


def gate_passed(m: dict) -> bool:
    return m["iou"] >= GATE["iou"] and m["growth_iou"] >= GATE["growth_iou"] and m["boundary_cells"] <= GATE["boundary_cells"]


# --------------------------------------------------------------------------- PINN (torch)

def local_speed(case: Case, smooth_cells: float = 1.0) -> np.ndarray:
    """F0 = R (1 + c n0.w)_+ with n0 the outward normal of phi0, smoothed over ~1 cell (prior only;
    the PDE residual always uses the exact R)."""
    gy, gx = np.gradient(case.phi0, case.res)
    mag = np.hypot(gx, gy) + 1e-12
    if case.wind is None:
        f0 = case.rate.copy()
    else:
        f0 = case.rate * np.clip(1 + C_WDIR * (case.wind[0] * gx + case.wind[1] * gy) / mag, 0, None)
    return ndimage.gaussian_filter(f0, smooth_cells) if smooth_cells else f0


def train_and_predict(case: Case, steps: int, batch: int, seed: int, ansatz: str = "local", log_every: int = 200):
    import torch
    from torch import nn

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    f32 = dict(dtype=torch.float32, device=device)

    H = case.horizon_s
    L = case.half_width
    burnable = case.rate > 0
    r_ref = float(np.percentile(case.rate[burnable], 90)) if burnable.any() else 1e-3
    S = r_ref * H                                                        # metres moved by a typical fast front
    gy, gx = np.gradient(case.phi0, case.res)
    east, north = case.wind if case.wind is not None else (np.zeros_like(case.rate), np.zeros_like(case.rate))
    f0 = local_speed(case) if ansatz == "local" else np.zeros_like(case.rate)
    f0y, f0x = np.gradient(f0, case.res)
    grids = {k: torch.as_tensor(v, **f32) for k, v in
             {"phi0": case.phi0, "phi0_x": gx, "phi0_y": gy, "rate": case.rate, "east": east, "north": north,
              "f0": f0, "f0_x": f0x, "f0_y": f0y}.items()}
    ny, nx = case.phi0.shape
    x_min, y_min = float(case.xs[0]), float(case.ys[0])

    def sample(name, x, y):
        """Bilinear lookup at metre coordinates (no autograd through the lookup)."""
        fi = ((y - y_min) / case.res).clamp(0, ny - 1.0001)
        fj = ((x - x_min) / case.res).clamp(0, nx - 1.0001)
        i0, j0 = fi.floor().long(), fj.floor().long()
        wy, wx = fi - i0, fj - j0
        g = grids[name]
        return ((1 - wy) * (1 - wx) * g[i0, j0] + (1 - wy) * wx * g[i0, j0 + 1]
                + wy * (1 - wx) * g[i0 + 1, j0] + wy * wx * g[i0 + 1, j0 + 1])

    # multi-scale random Fourier features: wavelengths from the window down to ~3 cells
    scales = [0.5, 2.0, 8.0, min(32.0, L / (3 * case.res))]
    bank = torch.cat([torch.randn(16, 2) * s for s in scales]).to(device)

    class HardICPINN(nn.Module):
        def __init__(self, hidden=128, layers=4):
            super().__init__()
            mods, width = [], 3 + 2 * bank.shape[0]
            for _ in range(layers):
                mods += [nn.Linear(width, hidden), nn.Tanh()]
                width = hidden
            mods.append(nn.Linear(width, 1))
            self.net = nn.Sequential(*mods)
            for m in self.net:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain("tanh"))
                    nn.init.zeros_(m.bias)
            nn.init.normal_(self.net[-1].weight, std=1e-3)

        def g(self, tau, X, Y):
            proj = 2 * math.pi * torch.stack([X, Y], dim=1) @ bank.T
            feats = torch.cat([tau[:, None], X[:, None], Y[:, None], torch.sin(proj), torch.cos(proj)], dim=1)
            return self.net(feats).squeeze(-1)

    model = HardICPINN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=1e-5)

    # collocation pool: cells the front can plausibly reach (plus a margin) and a uniform share
    reach = 1.3 * float(case.rate.max()) * (1 + C_WDIR) * H + 3 * case.res
    band = np.argwhere((case.phi0 < reach) & (case.phi0 > -4 * case.res))
    every = np.argwhere(np.ones_like(case.phi0, dtype=bool))

    def draw(n):
        n_band = int(0.8 * n)
        pick = np.concatenate([band[rng.integers(0, len(band), n_band)], every[rng.integers(0, len(every), n - n_band)]])
        jitter = rng.uniform(-0.5, 0.5, size=pick.shape) * case.res
        y = case.ys[pick[:, 0]] + jitter[:, 0]
        x = case.xs[pick[:, 1]] + jitter[:, 1]
        return torch.as_tensor(x, **f32), torch.as_tensor(y, **f32), torch.rand(n, **f32)

    def residual(x, y, tau):
        X = (x / L).requires_grad_(True)
        Y = (y / L).requires_grad_(True)
        tau = tau.requires_grad_(True)
        g = model.g(tau, X, Y)
        g_tau, g_X, g_Y = torch.autograd.grad(g.sum(), (tau, X, Y), create_graph=True)
        xd, yd = x.detach(), y.detach()
        phi_t = -sample("f0", xd, yd) + r_ref * (g + tau * g_tau)          # (S / H) = r_ref
        phi_x = sample("phi0_x", xd, yd) - tau * H * sample("f0_x", xd, yd) + tau * S * g_X / L
        phi_y = sample("phi0_y", xd, yd) - tau * H * sample("f0_y", xd, yd) + tau * S * g_Y / L
        mag = torch.sqrt(phi_x ** 2 + phi_y ** 2 + 1e-12)
        align = (sample("east", xd, yd) * phi_x + sample("north", xd, yd) * phi_y) / mag
        speed = sample("rate", xd, yd) * torch.clamp(1 + C_WDIR * align, min=0.0)
        # front-speed form: (phi_t + F|grad phi|) / |grad phi| = F - V_n, the error in normal speed.
        # Same zero set as the raw residual, but bounded where phi is steep (fuel edges under the
        # "local" first guess) or flat; |grad phi| is floored at 0.2 so kinks do not blow it up.
        return (phi_t + speed * mag) / (r_ref * torch.clamp(mag, min=0.2))

    history = []
    started = time.perf_counter()
    for step in range(1, steps + 1):
        x, y, tau = draw(batch)
        loss = torch.mean(residual(x, y, tau) ** 2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % log_every == 0 or step == 1 or step == steps:
            history.append((step, float(loss.detach())))
            print(f"  [{case.name}] step {step:5d}/{steps}  residual^2 {float(loss.detach()):.3e}  ({time.perf_counter() - started:.0f} s)", flush=True)

    # predict phi on the grid at each horizon
    X, Y = np.meshgrid(case.xs, case.ys)
    xt = torch.as_tensor(X.ravel() / L, **f32)
    yt = torch.as_tensor(Y.ravel() / L, **f32)
    preds = {}
    with torch.no_grad():
        for h in HORIZONS_H:
            tau_val = h * 3600.0 / H
            out = []
            for k in range(0, len(xt), 65536):
                xs_, ys_ = xt[k:k + 65536], yt[k:k + 65536]
                out.append(model.g(torch.full_like(xs_, tau_val), xs_, ys_))
            g = torch.cat(out).cpu().numpy().reshape(X.shape)
            preds[h] = case.phi0 - tau_val * H * f0 + tau_val * S * g
    meta = {"ansatz": ansatz, "steps": steps, "batch": batch, "seed": seed, "device": str(device), "r_ref_m_s": r_ref,
            "train_seconds": round(time.perf_counter() - started, 1), "loss_history": history}
    return preds, meta, model


# --------------------------------------------------------------------------- report

def figure(case: Case, preds, refs, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(HORIZONS_H), figsize=(5 * len(HORIZONS_H), 5))
    extent = [case.xs[0] / 1000, case.xs[-1] / 1000, case.ys[0] / 1000, case.ys[-1] / 1000]
    for ax, h in zip(axes, HORIZONS_H):
        ax.imshow(case.rate, origin="lower", extent=extent, cmap="Greens", alpha=0.5)
        X, Y = np.meshgrid(case.xs / 1000, case.ys / 1000)
        ax.contour(X, Y, case.phi0, [0], colors="k", linewidths=1)
        ax.contour(X, Y, refs[h], [0], colors="tab:blue", linewidths=2)
        ax.contour(X, Y, preds[h], [0], colors="tab:red", linewidths=1.5, linestyles="--")
        ax.set_title(f"{case.name}: +{h} h")
        ax.set_xlabel("km")
        ax.set_aspect("equal")
    axes[0].set_ylabel("km")
    fig.suptitle("black = start, blue = Godunov solver, red dashed = hard-IC PINN; green = spread rate R")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def run(case: Case, steps: int, batch: int, seed: int, out_dir: Path, ansatz: str = "local") -> dict:
    print(f"\n== {case.name} ({ansatz} ansatz): grid {case.phi0.shape}, {case.res:g} m cells, horizon 6 h; {case.notes}")
    refs = reference(case)
    prior = {}
    if ansatz == "local":
        f0 = local_speed(case)
        prior = {h: compare(case.phi0 - h * 3600 * f0, refs[h], case.phi0, case.res) for h in HORIZONS_H}
        print("   first guess alone (no network): " + "; ".join(
            f"+{h} h IoU {prior[h]['iou']:.3f} / new {prior[h]['growth_iou']:.3f}" for h in HORIZONS_H))
    preds, meta, model = train_and_predict(case, steps, batch, seed, ansatz)
    results = {h: compare(preds[h], refs[h], case.phi0, case.res) for h in HORIZONS_H}
    if case.name == "circle":
        for h in HORIZONS_H:
            exact = case.notes["r0_m"] + case.notes["R_m_s"] * h * 3600
            for key, field_ in (("pinn", preds[h]), ("solver", refs[h])):
                results[h][f"{key}_radius_m"] = round(float(np.sqrt((field_ < 0).sum() * case.res ** 2 / np.pi)), 1)
            results[h]["exact_radius_m"] = exact
    passed = gate_passed(results[6])
    report = {"case": case.name, "ansatz": ansatz, "gate": GATE, "passed_at_6h": passed, "horizons": results,
              "first_guess_only": prior, "notes": case.notes, "training": meta}
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{case.name}_{ansatz}"
    (out_dir / f"{stem}.json").write_text(json.dumps(report, indent=2))
    figure(case, preds, refs, out_dir / f"{stem}.png")
    try:
        import torch

        torch.save({"state_dict": model.state_dict(), "case": case.name, "meta": meta}, out_dir / f"{stem}_model.pt")
    except Exception:  # noqa: BLE001
        pass
    print(f"-- {case.name} vs solver:")
    for h in HORIZONS_H:
        r = results[h]
        extra = f", radius PINN {r['pinn_radius_m']} / solver {r['solver_radius_m']} / exact {r['exact_radius_m']} m" if "exact_radius_m" in r else ""
        print(f"   +{h} h: IoU {r['iou']:.3f}, new-area IoU {r['growth_iou']:.3f}, boundary {r['boundary_cells']:.2f} cells ({r['boundary_m']:.0f} m){extra}")
    print(f"   GATE at 6 h: {'PASS' if passed else 'FAIL'}   (figure: {out_dir / (stem + '.png')})")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--case", choices=["circle", "fuel", "all"], default="all")
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ansatz", choices=["local", "plain"], default="local")
    parser.add_argument("--quick", action="store_true", help="600 steps, batch 2048")
    parser.add_argument("--out", default=str(REPO / "outputs" / "pinn_v2_gate"))
    args = parser.parse_args()
    if args.quick:
        args.steps, args.batch = 600, 2048
    builders = {"circle": circle_case, "fuel": fuel_case}
    names = list(builders) if args.case == "all" else [args.case]
    summary = {n: run(builders[n](), args.steps, args.batch, args.seed, Path(args.out), args.ansatz)["passed_at_6h"] for n in names}
    print("\nSUMMARY:", ", ".join(f"{n} {'PASS' if ok else 'FAIL'}" for n, ok in summary.items()))
    if args.quick:
        print("(quick run: too few steps to judge the gate; run without --quick for the real check)")


if __name__ == "__main__":
    main()
