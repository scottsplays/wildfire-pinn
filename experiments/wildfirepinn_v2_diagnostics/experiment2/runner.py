"""Train and score hard-IC PINNs against the numerical reference.

One scenario per process. Experiment 1 and 1B checkpoints are not touched.

    python -m experiments.wildfirepinn_v2_diagnostics.experiment2 --scenario 2a
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from experiments.wildfirepinn_v2_diagnostics.circle import CircleProblem, analytical_phi
from experiments.wildfirepinn_v2_diagnostics.experiment1 import HIDDEN, LAYERS, N_PDE, _sample_pde
from experiments.wildfirepinn_v2_diagnostics.experiment2.directional import propagate_directional
from experiments.wildfirepinn_v2_diagnostics.experiment2.fields import (
    R0_M_S,
    WIND_C,
    WAVELENGTH_M,
    spatial_rate_np,
    spatial_rate_torch,
    wind_torch,
)
from experiments.wildfirepinn_v2_diagnostics.experiment2.metrics import score_against_reference, support_growth
from experiments.wildfirepinn_v2_diagnostics.experiment2.physics import (
    circle_phi0,
    coded_residual,
    grid_phi0,
    hard_ic,
)
from experiments.wildfirepinn_v2_diagnostics.experiment2.sdf import irregular_polygon, irregular_sdf
from experiments.wildfirepinn_v2_diagnostics.solver import propagate
from shapely import distance, points
from wildfire_pinn.geometry import zero_contours
from wildfire_pinn.pinn import LevelSetPINN, load_checkpoint, save_checkpoint, set_seed

HERE = Path(__file__).resolve().parent
SEED = 0
EPOCHS = 1500
LR = 1e-3
FRONT_JITTER_M = 100.0


def _problem() -> CircleProblem:
    # 20 m cells. The refinement check compares this grid with coarser and finer ones.
    return CircleProblem(n=201)


def _times(problem: CircleProblem) -> list[float]:
    span = problem.t_final
    return [0.0, span / 3.0, 2.0 * span / 3.0, span]


SCENARIOS = {
    "2a": {
        "label": "spatial R, circular initial fire, uniform samples",
        "rate": "spatial",
        "geometry": "circle",
        "wind": None,
        "sampling": "uniform",
    },
    "2a-zero": {
        "label": "R = 0 preservation, circular initial fire",
        "rate": "zero",
        "geometry": "circle",
        "wind": None,
        "sampling": "uniform",
    },
    "2b-constant": {
        "label": "constant east wind, circular initial fire",
        "rate": "constant",
        "geometry": "circle",
        "wind": {"mode": "constant", "first": (1.0, 0.0), "second": None, "switch_s": None},
        "sampling": "uniform",
    },
    "2b-switch": {
        "label": "wind turns from east to north at T/2",
        "rate": "constant",
        "geometry": "circle",
        "wind": {"mode": "switch", "first": (1.0, 0.0), "second": (0.0, 1.0), "switch_s": None},
        "sampling": "uniform",
    },
    "2c-isotropic": {
        "label": "spatial R, irregular initial fire",
        "rate": "spatial",
        "geometry": "irregular",
        "wind": None,
        "sampling": "uniform",
    },
    "2c-wind": {
        "label": "constant east wind, irregular initial fire",
        "rate": "constant",
        "geometry": "irregular",
        "wind": {"mode": "constant", "first": (1.0, 0.0), "second": None, "switch_s": None},
        "sampling": "uniform",
    },
    "2d": {
        "label": "spatial R, samples near the PINN's own predicted front",
        "rate": "spatial",
        "geometry": "circle",
        "wind": None,
        "sampling": "predicted",
    },
}


class FrontCache:
    def __init__(self):
        self.xy = None

    def refresh(self, model, problem, phi0_fn, grid_x, grid_y):
        points_xy = []
        for moment in _times(problem):
            field = predict_phi(model, problem, phi0_fn, moment, grid_x, grid_y)
            for line in zero_contours(field, grid_x, grid_y):
                if len(line) >= 2:
                    points_xy.append(line)
        if not points_xy:
            self.xy = None
            return
        self.xy = torch.tensor(np.concatenate(points_xy, axis=0), dtype=torch.float32).detach()


def sample_mixed(n: int, problem: CircleProblem, cache: FrontCache, device):
    """Half uniform, half jittered around the detached predicted front. No reference contour."""
    n_front = n // 2
    x_u, y_u, t_u = _sample_pde(n - n_front, problem, device)
    if cache.xy is None or len(cache.xy) < 8:
        x_f, y_f, t_f = _sample_pde(n_front, problem, device)
        return torch.cat([x_u, x_f]), torch.cat([y_u, y_f]), torch.cat([t_u, t_f])
    choice = torch.randint(0, len(cache.xy), (n_front,))
    base = cache.xy[choice].to(device)
    jitter = (torch.rand(n_front, 2, device=device) * 2.0 - 1.0) * FRONT_JITTER_M
    xy = base + jitter
    x = xy[:, 0]
    y = xy[:, 1]
    t = torch.rand(n_front, device=device) * problem.t_final
    inside = (
        (x >= -problem.half_extent) & (x <= problem.half_extent)
        & (y >= -problem.half_extent) & (y <= problem.half_extent)
        & (torch.hypot(x - problem.x0, y - problem.y0) > problem.pde_exclusion_m)
    )
    if int(inside.sum()) < n_front // 4:
        x_f, y_f, t_f = _sample_pde(n_front, problem, device)
    else:
        x_f, y_f, t_f = x[inside], y[inside], t[inside]
        while x_f.shape[0] < n_front:
            x_f = torch.cat([x_f, x_f])
            y_f = torch.cat([y_f, y_f])
            t_f = torch.cat([t_f, t_f])
        x_f, y_f, t_f = x_f[:n_front], y_f[:n_front], t_f[:n_front]
    batch_x = torch.cat([x_u, x_f]).detach()
    batch_y = torch.cat([y_u, y_f]).detach()
    batch_t = torch.cat([t_u, t_f]).detach()
    return batch_x, batch_y, batch_t


def _rate_on(kind: str, x, y):
    if kind == "spatial":
        return spatial_rate_torch(x, y)
    if kind == "constant":
        return torch.full_like(x, R0_M_S)
    if kind == "zero":
        return torch.zeros_like(x)
    raise ValueError(kind)


def _rate_grid(kind: str, xx, yy):
    if kind == "spatial":
        return spatial_rate_np(xx, yy)
    if kind == "constant":
        return np.full(xx.shape, R0_M_S, dtype=np.float64)
    if kind == "zero":
        return np.zeros(xx.shape, dtype=np.float64)
    raise ValueError(kind)


def _wind_schedule(spec, problem):
    wind = spec["wind"]
    if wind is None:
        return None, ()
    if wind["mode"] == "constant":
        east, north = wind["first"]
        return (lambda _t, east=east, north=north: (east, north)), ()
    switch = problem.t_final / 2.0
    first, second = wind["first"], wind["second"]
    return (lambda t, switch=switch, first=first, second=second: first if t < switch else second), (switch,)


def build_phi0(spec, problem, x, y):
    if spec["geometry"] == "circle":
        xx, yy = np.meshgrid(x, y)
        phi0 = analytical_phi(xx, yy, 0.0, problem, 0.0)
        return phi0, lambda xn, yn: circle_phi0(xn, yn, problem), None
    sdf = irregular_sdf(x, y, problem.x0, problem.y0, problem.r0)
    polygon = irregular_polygon(problem.x0, problem.y0, problem.r0)
    lines = [line for line in zero_contours(sdf, x, y) if len(line) >= 2]
    if lines:
        pts = np.concatenate(lines, axis=0)
        miss = float(np.mean(distance(points(pts), polygon.boundary)))
    else:
        miss = float("inf")
    cell = float(x[1] - x[0])
    if miss > 1.5 * cell:
        raise RuntimeError(f"irregular zero contour is {miss:.1f} m from the polygon, cell {cell:.1f} m")
    fn = lambda xn, yn, sdf=sdf, x=x, y=y, problem=problem: grid_phi0(xn, yn, sdf, x, y, problem)
    return sdf, fn, {"contour_to_polygon_m": miss, "cell_m": cell}


def reference_fields(spec, problem, phi0, x, y):
    xx, yy = np.meshgrid(x, y)
    rate = _rate_grid(spec["rate"], xx, yy)
    moments = _times(problem)
    started = time.perf_counter()
    wind_at, switches = _wind_schedule(spec, problem)
    if wind_at is None:
        fields = propagate(phi0, x, y, rate, moments)
    else:
        fields = propagate_directional(
            phi0, x, y, rate, moments, wind_at, WIND_C, switch_times=switches
        )
    elapsed = time.perf_counter() - started
    if spec["wind"] is None:
        frozen = propagate(phi0, x, y, np.zeros_like(rate), [problem.t_final])[problem.t_final]
    else:
        frozen = propagate_directional(
            phi0, x, y, np.zeros_like(rate), [problem.t_final], wind_at, WIND_C, switch_times=switches
        )[problem.t_final]
    return fields, rate, elapsed, float(np.max(np.abs(frozen - phi0)))


def predict_phi(model, problem, phi0_fn, t, x, y) -> np.ndarray:
    device = next(model.parameters()).device
    yy, xx = np.meshgrid(y, x, indexing="ij")
    tau = torch.tensor(np.full(xx.shape, float(t) / problem.t_final).ravel(), dtype=torch.float32, device=device)
    x_norm = torch.tensor(((xx - problem.center_x) / problem.length).ravel(), dtype=torch.float32, device=device)
    y_norm = torch.tensor(((yy - problem.center_y) / problem.length).ravel(), dtype=torch.float32, device=device)
    was = model.training
    model.eval()
    with torch.no_grad():
        raw = model(tau, x_norm, y_norm)
        phi = hard_ic(raw, tau, phi0_fn(x_norm, y_norm))
    if was:
        model.train()
    return phi.detach().cpu().numpy().reshape(xx.shape) * problem.length


def _batch_residual(model, problem, phi0_fn, spec, x, y, t):
    length = problem.length
    duration = problem.t_final
    tau = (t / duration).detach().requires_grad_(True)
    x_norm = ((x - problem.center_x) / length).detach().requires_grad_(True)
    y_norm = ((y - problem.center_y) / length).detach().requires_grad_(True)
    raw = model(tau, x_norm, y_norm)
    phi = hard_ic(raw, tau, phi0_fn(x_norm, y_norm))
    phi_t, phi_x, phi_y = torch.autograd.grad(
        phi, (tau, x_norm, y_norm), torch.ones_like(phi), create_graph=model.training
    )
    rate = _rate_on(spec["rate"], x.detach(), y.detach())
    wind_east = wind_north = None
    c_wind = 0.0
    if spec["wind"] is not None:
        c_wind = WIND_C
        wind = spec["wind"]
        switch = None if wind["mode"] == "constant" else duration / 2.0
        wind_east, wind_north = wind_torch(t.detach(), switch, wind["first"], wind["second"])
    return coded_residual(phi_t, phi_x, phi_y, rate, duration / length, wind_east, wind_north, c_wind)


def train_scenario(spec, problem, phi0_fn, x, y, run_dir: Path, epochs: int):
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = run_dir / "levelset_pinn.pt"
    history_path = run_dir / "history.json"
    state_path = run_dir / "trainer_state.pt"
    device = torch.device("cpu")
    config = {"scenario": spec["name"], "epochs": epochs, "hidden": HIDDEN, "layers": LAYERS, "sampling": spec["sampling"]}
    start = 1
    history = {"pde": []}
    if checkpoint.is_file() and history_path.is_file():
        history = json.loads(history_path.read_text(encoding="utf-8"))
        done = len(history.get("pde", []))
        if done == epochs:
            model = LevelSetPINN(HIDDEN, LAYERS).to(device)
            load_checkpoint(checkpoint, model, device)
            model.eval()
            print(f"resume finished {spec['name']}", flush=True)
            return model, history, 0.0
        history = {"pde": []}
    set_seed(SEED)
    model = LevelSetPINN(HIDDEN, LAYERS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    cache = FrontCache()
    model.train()
    started = time.perf_counter()
    log_every = 1 if epochs <= 5 else 250
    for epoch in range(start, epochs + 1):
        if spec["sampling"] == "predicted" and (epoch == 1 or epoch % 50 == 0):
            cache.refresh(model, problem, phi0_fn, x, y)
        if spec["sampling"] == "predicted":
            x_pde, y_pde, t_pde = sample_mixed(N_PDE, problem, cache, device)
        else:
            x_pde, y_pde, t_pde = _sample_pde(N_PDE, problem, device)
        _coded, physical, _mag = _batch_residual(model, problem, phi0_fn, spec, x_pde, y_pde, t_pde)
        loss = torch.mean(_coded ** 2)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at epoch {epoch}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        history["pde"].append(float(loss.detach()))
        if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
            print(f"{spec['name']} epoch {epoch:4d} | pde {history['pde'][-1]:.3e}", flush=True)
            if epoch % log_every == 0 or epoch == epochs:
                save_checkpoint(checkpoint, model, config)
                history_path.write_text(json.dumps(history), encoding="utf-8")
                torch.save({"optimizer": optimizer.state_dict(), "epoch": epoch}, state_path)
    model.eval()
    return model, history, time.perf_counter() - started


def _probe(model, problem, phi0_fn, spec, x, y, reference, mode: str) -> float:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(12345 if mode == "uniform" else 99)
    if mode == "uniform":
        xs = torch.rand(4096, generator=generator) * problem.length - problem.half_extent
        ys = torch.rand(4096, generator=generator) * problem.length - problem.half_extent
        ts = torch.rand(4096, generator=generator) * problem.t_final
        keep = torch.hypot(xs - problem.x0, ys - problem.y0) > problem.pde_exclusion_m
        xs, ys, ts = xs[keep][:3000], ys[keep][:3000], ts[keep][:3000]
    else:
        lines = [line for line in zero_contours(reference[_times(problem)[-1]], x, y) if len(line) >= 2]
        if not lines:
            return float("nan")
        base = torch.tensor(np.concatenate(lines), dtype=torch.float32)
        choice = torch.randint(0, len(base), (3000,), generator=generator)
        jitter = (torch.rand(3000, 2, generator=generator) * 2 - 1) * FRONT_JITTER_M
        xy = base[choice] + jitter
        xs, ys = xy[:, 0], xy[:, 1]
        ts = torch.rand(3000, generator=generator) * problem.t_final
    was = model.training
    model.eval()
    _coded, physical, _mag = _batch_residual(model, problem, phi0_fn, spec, xs, ys, ts)
    if was:
        model.train()
    return float(physical.detach().abs().mean())


def _growth_by_rate(phi, x, y, problem, rate_grid) -> dict:
    lines = [line for line in zero_contours(phi, x, y) if len(line) >= 2]
    if not lines:
        return {"high_r_growth_m": None, "low_r_growth_m": None}
    pts = np.concatenate(lines, axis=0)
    radial = np.hypot(pts[:, 0] - problem.x0, pts[:, 1] - problem.y0) - problem.r0
    local = spatial_rate_np(pts[:, 0], pts[:, 1])
    high = local >= R0_M_S
    low = ~high
    return {
        "high_r_growth_m": float(radial[high].mean()) if high.any() else None,
        "low_r_growth_m": float(radial[low].mean()) if low.any() else None,
    }


def evaluate(spec, model, problem, phi0_fn, phi0, fields, rate_grid, x, y, train_seconds, solver_seconds, stationary_change, history):
    rows = []
    predictions = {}
    for moment in _times(problem):
        pred = predict_phi(model, problem, phi0_fn, moment, x, y)
        predictions[moment] = pred
        row = score_against_reference(pred, fields[moment], x, y)
        row["t_s"] = moment
        rows.append(row)
    wind_stats = None
    if spec["wind"] is not None:
        east, north = spec["wind"]["first"]
        wind_stats = support_growth(phi0, predictions[problem.t_final], x, y, problem.x0, problem.y0, east, north)
        numerical_wind = support_growth(phi0, fields[problem.t_final], x, y, problem.x0, problem.y0, east, north)
    else:
        numerical_wind = None
    growth = _growth_by_rate(predictions[problem.t_final], x, y, problem, rate_grid) if spec["rate"] == "spatial" else None
    numerical_growth = _growth_by_rate(fields[problem.t_final], x, y, problem, rate_grid) if spec["rate"] == "spatial" else None
    record = {
        "scenario": spec["name"],
        "label": spec["label"],
        "sampling": spec["sampling"],
        "oracle_front_sampling": False,
        "train_seconds": train_seconds,
        "solver_seconds": solver_seconds,
        "numerical_r0_max_change_m": stationary_change,
        "initial_iou": rows[0]["iou"],
        "final_iou": rows[-1]["iou"],
        "final_symmetric_m": rows[-1]["symmetric_distance_m"],
        "final_p95_m": rows[-1]["p95_distance_m"],
        "final_area_error_km2": rows[-1]["area_error_km2"],
        "final_components": rows[-1]["burned_components"],
        "final_contours": rows[-1]["n_contours"],
        "volume_residual_m_s": _probe(model, problem, phi0_fn, spec, x, y, fields, "uniform"),
        "front_residual_m_s": _probe(model, problem, phi0_fn, spec, x, y, fields, "front"),
        "final_train_pde": history["pde"][-1] if history.get("pde") else None,
        "pinn_growth": growth,
        "numerical_growth": numerical_growth,
        "pinn_wind": wind_stats,
        "numerical_wind": numerical_wind,
        "times": rows,
    }
    return record, predictions


def refine_spatial(problem_template: CircleProblem) -> dict:
    """Compare spatial-R fronts at 40 m, 20 m, and 10 m cells. No PINN training."""
    out = {}
    stored = {}
    for count, label in ((101, "40m"), (201, "20m"), (401, "10m")):
        problem = CircleProblem(n=count, t_final=problem_template.t_final, r0=problem_template.r0, x0=problem_template.x0, y0=problem_template.y0)
        x, y = problem.grid()
        xx, yy = np.meshgrid(x, y)
        phi0 = analytical_phi(xx, yy, 0.0, problem, 0.0)
        rate = spatial_rate_np(xx, yy)
        started = time.perf_counter()
        final = propagate(phi0, x, y, rate, [problem.t_final])[problem.t_final]
        stored[label] = (final, x, y, time.perf_counter() - started)
    fine, xf, yf, fine_seconds = stored["10m"]
    from shapely.geometry import LineString, MultiLineString
    from shapely import distance as shp_distance, points as shp_points

    fine_lines = [line for line in zero_contours(fine, xf, yf) if len(line) >= 2]
    fine_geom = MultiLineString([LineString(line) for line in fine_lines])
    fine_pts = np.concatenate(fine_lines)
    for label in ("40m", "20m"):
        field, x, y, seconds = stored[label]
        coarse_lines = [line for line in zero_contours(field, x, y) if len(line) >= 2]
        coarse = MultiLineString([LineString(line) for line in coarse_lines])
        coarse_pts = np.concatenate(coarse_lines)
        forward = np.asarray(shp_distance(shp_points(coarse_pts), fine_geom), dtype=np.float64)
        backward = np.asarray(shp_distance(shp_points(fine_pts), coarse), dtype=np.float64)
        out[label] = {
            "solver_seconds": seconds,
            "symmetric_to_10m_m": 0.5 * (float(forward.mean()) + float(backward.mean())),
        }
    out["10m_solver_seconds"] = fine_seconds
    return out


def plot_scenario(spec, problem, phi0, predictions, fields, rate_grid, x, y, path: Path):
    moments = _times(problem)
    fig, axes = plt.subplots(2, 4, figsize=(14, 7.2))
    extent = [x[0], x[-1], y[0], y[-1]]
    window = 1100.0
    for col, moment in enumerate(moments):
        ax = axes[0, col]
        image = ax.imshow(rate_grid, origin="lower", extent=extent, cmap="viridis", vmin=0.02, vmax=0.08)
        for line in zero_contours(fields[moment], x, y):
            ax.plot(line[:, 0], line[:, 1], color="white", lw=1.4)
        for line in zero_contours(predictions[moment], x, y):
            ax.plot(line[:, 0], line[:, 1], color="crimson", lw=1.2)
        ax.set_xlim(problem.x0 - window, problem.x0 + window)
        ax.set_ylim(problem.y0 - window, problem.y0 + window)
        ax.set_aspect("equal")
        ax.set_title(f"t = {moment:.0f} s")
        if col == 0:
            ax.set_ylabel("R and fronts")
        err = axes[1, col]
        gap = np.abs(predictions[moment] - fields[moment])
        err.imshow(gap, origin="lower", extent=extent, cmap="magma", vmin=0, vmax=50)
        for line in zero_contours(fields[moment], x, y):
            err.plot(line[:, 0], line[:, 1], color="cyan", lw=1.0)
        for line in zero_contours(predictions[moment], x, y):
            err.plot(line[:, 0], line[:, 1], color="lime", lw=1.0)
        err.set_xlim(problem.x0 - window, problem.x0 + window)
        err.set_ylim(problem.y0 - window, problem.y0 + window)
        err.set_aspect("equal")
        if col == 0:
            err.set_ylabel("|phi PINN − numerical| (m)")
    fig.colorbar(image, ax=axes[0, :].tolist(), fraction=0.02, pad=0.01, label="R (m/s)")
    fig.suptitle(f"{spec['name']}: white/cyan numerical, red/lime PINN. {spec['label']}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _fmt(value, digits=3):
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def write_reports(records: list[dict]):
    lines = [
        "# Experiment 2",
        "",
        "Hard-IC PINN against a numerical reference. R is prescribed. "
        "Oracle future-front sampling is not used. The R-scaled ansatz is not used. "
        "Experiment 1, Experiment 1B, and the Kincade model were not modified.",
        "",
        f"Spatial R uses R0 = {R0_M_S} m/s, amplitude {0.5}, wavelength {WAVELENGTH_M:.0f} m, "
        f"so R stays in [0.025, 0.075] m/s. Wind cases use a constant base rate {R0_M_S} m/s and c = {WIND_C}. "
        "Wind points in the direction the wind blows toward. The initial circle has radius 400 m. "
        "The irregular fire is the polar curve r(1 + 0.15 cos 3θ + 0.1 sin 5θ), converted to a signed-distance grid.",
        "",
        "| scenario | init IoU | final IoU | symmetric m | p95 m | area error km² | volume residual m/s | front residual m/s | components | train s | solver s |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for record in records:
        lines.append(
            "| {name} | {ii} | {fi} | {sym} | {p95} | {area} | {vol} | {front} | {comp} | {train} | {solver} |".format(
                name=record["scenario"],
                ii=_fmt(record["initial_iou"], 3),
                fi=_fmt(record["final_iou"], 3),
                sym=_fmt(record["final_symmetric_m"], 2),
                p95=_fmt(record["final_p95_m"], 2),
                area=_fmt(record["final_area_error_km2"], 3),
                vol=_fmt(record["volume_residual_m_s"], 4),
                front=_fmt(record["front_residual_m_s"], 4),
                comp=record["final_components"],
                train=_fmt(record["train_seconds"], 1),
                solver=_fmt(record["solver_seconds"], 2),
            )
        )
    lines.append("")
    for record in records:
        lines.append(f"## {record['scenario']}")
        lines.append("")
        lines.append(record["label"] + ".")
        lines.append("")
        lines.append("| t s | IoU | symmetric m | p95 m | area error km² |")
        lines.append("| ---: | ---: | ---: | ---: | ---: |")
        for row in record["times"]:
            lines.append(
                f"| {row['t_s']:.0f} | {_fmt(row['iou'], 3)} | {_fmt(row['symmetric_distance_m'], 2)} | {_fmt(row['p95_distance_m'], 2)} | {_fmt(row['area_error_km2'], 3)} |"
            )
        if record.get("pinn_growth"):
            lines.append("")
            lines.append(
                f"Radial growth beyond 400 m where R ≥ R0: PINN {_fmt(record['pinn_growth']['high_r_growth_m'], 1)} m, "
                f"numerical {_fmt(record['numerical_growth']['high_r_growth_m'], 1)} m. "
                f"Where R < R0: PINN {_fmt(record['pinn_growth']['low_r_growth_m'], 1)} m, "
                f"numerical {_fmt(record['numerical_growth']['low_r_growth_m'], 1)} m."
            )
        if record.get("pinn_wind"):
            lines.append("")
            lines.append(
                f"Advance along the initial wind and opposite it: "
                f"PINN down {_fmt(record['pinn_wind']['downwind_growth_m'], 1)} m, up {_fmt(record['pinn_wind']['upwind_growth_m'], 1)} m; "
                f"numerical down {_fmt(record['numerical_wind']['downwind_growth_m'], 1)} m, up {_fmt(record['numerical_wind']['upwind_growth_m'], 1)} m."
            )
        lines.append("")
        lines.append(
            f"Numerical R = 0 changes the field by at most {_fmt(record['numerical_r0_max_change_m'], 4)} m. "
            f"Final burned components: {record['final_components']}. Contours: {record['final_contours']}."
        )
        if record.get("refinement"):
            lines.append("")
            lines.append("Grid refinement of the spatial-R numerical front against the 10 m run:")
            lines.append("")
            for key, value in record["refinement"].items():
                lines.append(f"- {key}: {value}")
        lines.append("")
    lines.append("Figures are in `experiment2/figures/`. Finished runs are skipped when the history already has the requested epoch count.")
    lines.append("")
    path = HERE / "EXPERIMENT_2_RESULTS.md"
    if path.is_file() and "\n## Reading\n" in path.read_text(encoding="utf-8"):
        reading = path.read_text(encoding="utf-8").split("\n## Reading\n", 1)[1].strip()
        lines.extend(["## Reading", "", reading, ""])
    path.write_text("\n".join(lines), encoding="utf-8")
    summary = {"experiment": "2", "runs": records, "experiments_3_to_5": "not_started"}
    (HERE / "EXPERIMENT_2_SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {path}", flush=True)


def run_scenario(name: str, epochs: int, with_refinement: bool):
    spec = dict(SCENARIOS[name])
    spec["name"] = name
    problem = _problem()
    x, y = problem.grid()
    phi0, phi0_fn, geometry_note = build_phi0(spec, problem, x, y)
    run_dir = HERE / "runs" / name
    print(f"start {name}: {spec['label']}", flush=True)
    model, history, train_seconds = train_scenario(spec, problem, phi0_fn, x, y, run_dir, epochs)
    fields, rate_grid, solver_seconds, stationary = reference_fields(spec, problem, phi0, x, y)
    cache = HERE / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache / f"{name}_reference.npz",
        **{f"t{index}": fields[moment] for index, moment in enumerate(_times(problem))},
        x=x,
        y=y,
        rate=rate_grid,
    )
    record, predictions = evaluate(
        spec, model, problem, phi0_fn, phi0, fields, rate_grid, x, y, train_seconds, solver_seconds, stationary, history
    )
    if geometry_note:
        record["geometry_check"] = geometry_note
    if with_refinement and name == "2a":
        print("grid refinement", flush=True)
        record["refinement"] = refine_spatial(problem)
    plot_scenario(spec, problem, phi0, predictions, fields, rate_grid, x, y, HERE / "figures" / f"{name}.png")
    metrics_path = HERE / "metrics.json"
    existing = []
    if metrics_path.is_file():
        existing = [row for row in json.loads(metrics_path.read_text(encoding="utf-8")) if row.get("scenario") != name]
    existing.append(record)
    existing.sort(key=lambda row: row["scenario"])
    metrics_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    write_reports(existing)
    print(
        f"{name} init IoU {record['initial_iou']:.3f} final IoU {record['final_iou']:.3f} "
        f"symmetric {record['final_symmetric_m']} m front residual {record['front_residual_m_s']:.4f}",
        flush=True,
    )
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True, choices=sorted(SCENARIOS))
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()
    run_scenario(args.scenario, args.epochs, with_refinement=args.scenario == "2a" and args.epochs >= EPOCHS)


if __name__ == "__main__":
    main()
