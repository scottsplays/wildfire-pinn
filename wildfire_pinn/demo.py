"""A synthetic fire on a projected grid, using the same training code as Palisades.

No FIRMS key or terrain download is required. The run checks that the real-data
loss (initial perimeter, later perimeters, and the level-set residual) executes.
It is not a stand-in for the Palisades forecast.

Perimeters are circles of radius 300 m + R_demo * elapsed time. R_demo is read
from the spread field this function computes, so the observations and that field
describe the same expansion. The training residual still multiplies R by the
downwind factor stored on RealConfig; these circles do not include that factor.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import Point

from wildfire_pinn.data_pipeline import compute_spread_rate
from wildfire_pinn.pinn import (
    RealConfig,
    FireCase,
    evaluate_real,
    train_real,
    write_metrics,
)
from wildfire_pinn.plots import plot_dashboard, plot_losses, plot_scalar_grid

log = logging.getLogger("wildfire_pinn")


def make_demo_case():
    """Six overpasses, 12 h apart. Circle radii follow the computed spread rate.

    The window is sized from that same rate. A 4 km grid cannot hold a front
    that advances at the fuel-101 rate for two and a half days, and a front
    that fills the window has no zero contour to score.
    """
    import geopandas as gpd

    cell_m = 90.0
    initial_radius_m = 300.0
    times = pd.date_range("2025-01-07 21:00", periods=6, freq="12h")
    elapsed_s = (times - times[0]).total_seconds().to_numpy(dtype=np.float64)
    day0 = pd.Timestamp("2025-01-07")
    n_days = int((times.max().normalize() - day0).days) + 1
    # One burnable cell is enough to learn how large the window must be.
    # The radius below is read from the field actually stored on the case.
    probe, _stats = compute_spread_rate(
        np.full((2, 2), 101.0),
        np.zeros((2, 2)),
        np.full((n_days, 2, 2), 2.237),
        np.zeros((n_days, 2, 2)),
    )
    margin_m = 8.0 * cell_m
    half_m = initial_radius_m + float(probe.reshape(-1)[0]) * float(elapsed_s[-1]) + margin_m
    n = int(np.ceil(2.0 * half_m / cell_m)) + 1
    x = np.linspace(0.0, cell_m * (n - 1), n)
    y = np.linspace(cell_m * (n - 1), 0.0, n)  # north first, same layout as the reprojected stack
    center_x = float(x.mean())
    center_y = float(0.5 * (y.min() + y.max()))
    fuel = np.full((n, n), 101.0)
    slope = np.zeros((n, n))
    wind_speed = np.full((n_days, n, n), 2.237)
    moisture = np.zeros((n_days, n, n))
    spread, _stats = compute_spread_rate(fuel, slope, wind_speed, moisture)
    r_demo = float(spread[0, spread.shape[1] // 2, spread.shape[2] // 2])
    radii = initial_radius_m + r_demo * elapsed_s
    half_width = 0.5 * float(x.max() - x.min())
    if float(radii.max()) >= half_width:
        raise RuntimeError(
            "Demo front radius %.0f m does not fit in a half-width of %.0f m, "
            "so the signed distance would have no exterior."
            % (float(radii.max()), half_width)
        )
    log.info(
        "demo R = %.4f m/s read from the spread field; radii %.0f m -> %.0f m over %d overpasses",
        r_demo, float(radii[0]), float(radii[-1]), len(times),
    )
    perimeters = gpd.GeoDataFrame(
        {
            "acq_date": times.strftime("%Y-%m-%d"),
            "acq_time": times.strftime("%H%M"),
            "n_points": np.full(len(times), 40),
            "dt": times,
            "geometry": [Point(center_x, center_y).buffer(float(radius)) for radius in radii],
        },
        crs="EPSG:3310",
    )
    wind_from = np.zeros((n_days, n, n), dtype=np.float32)
    easting, _northing = np.meshgrid(x, y)
    elevation = 80.0 + 0.01 * (easting - easting.mean())
    case = FireCase(
        x=x,
        y=y,
        spread=spread,
        wind_from_deg=wind_from,
        perimeters=perimeters,
        day0=day0,
    )
    return case, elevation


def run_demo(out_dir: Path, epochs: int = 60) -> list[dict]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    case, elevation = make_demo_case()
    plot_dashboard(case, k_train=4, path=out_dir / "dashboard.png", elevation=elevation)
    initial = case.perimeters.geometry.iloc[0]
    from wildfire_pinn.geometry import signed_distance

    plot_scalar_grid(
        case.x, case.y, signed_distance(initial, case.x, case.y),
        out_dir / "initial_sdf.png",
        title="Demo initial signed distance",
        colorbar="signed distance (m)",
        zero=True,
    )
    hp = RealConfig(
        k_train=4,
        epochs=epochs,
        hidden=32,
        layers=2,
        n_pde=256,
        n_ic=128,
        n_obs=64,
        log_every=max(1, epochs // 4),
    )
    log.info(
        "demo fire: %d perimeters on a %d x %d grid, %d epochs. "
        "This exercises the Palisades loss; it does not fit the real fire.",
        len(case.perimeters), len(case.y), len(case.x), epochs,
    )
    model, history = train_real(case, out_dir, hp, retrain=True)
    plot_losses(history, out_dir / "loss_curves.png", "Demo level-set PINN")
    rows = evaluate_real(
        model, case, hp.k_train,
        figure_path=out_dir / "heldout_fronts.png",
        geojson_path=out_dir / "fronts.geojson",
        case_name="demo",
    )
    write_metrics(out_dir / "metrics.json", rows)
    log.info("demo figures and metrics are in %s", out_dir)
    return rows
