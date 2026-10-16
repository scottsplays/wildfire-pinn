"""Spread-rate proxy and an anisotropic level-set integrator.

The spread-rate tables and formula are copied from
``wildfire_pinn/data_pipeline.py::compute_spread_rate`` so this service can be
built without the training package (xarray, rioxarray, torch, ...).
``tests/test_service.py`` checks that the two stay identical when the training
package is importable.

R = C_REF * r_rel[fuel] * W(wind speed) * S(slope) * M(moisture)          [m/s]

The front moves with the outward normal speed

    F = R * max(0, 1 + C_WDIR * (n . w))

where n = grad(phi) / |grad(phi)| is the outward normal (phi < 0 inside the
burned region) and w is the unit vector the wind blows toward. This is the same
downwind factor the PINN residual uses (``wildfire_pinn.pinn.directed_residual``
with c_wdir = 0.6). F >= 0, so the burned region can only grow.
"""

from __future__ import annotations

import numpy as np

# --- copied from wildfire_pinn/data_pipeline.py (keep in sync; tested) -------
GRASS_SHRUB = (101, 102, 103, 121, 122, 123, 141, 142, 144, 145, 147)
TIMBER_UNDERSTORY = (161, 163, 165)
TIMBER_LITTER = (181, 182, 183, 184, 185, 186, 188, 189)
NONBURNABLE = frozenset({0, 91, 92, 93, 98, 99})

MX = {
    101: 15, 102: 15, 103: 30, 121: 15, 122: 15, 123: 15,
    141: 15, 142: 15, 143: 40, 144: 30, 145: 15, 146: 30, 147: 15,
    161: 20, 162: 30, 163: 30, 165: 25,
    181: 30, 182: 25, 183: 20, 184: 25, 185: 25, 186: 25, 187: 25, 188: 35, 189: 35,
    201: 25, 202: 25,
}
R_REL = {code: 1.0 for code in GRASS_SHRUB}
R_REL.update({code: 0.2 for code in TIMBER_UNDERSTORY})
R_REL.update({code: 0.07 for code in TIMBER_LITTER})
R_REL.update({146: 1.0, 162: 0.2, 201: 0.2, 202: 0.2, 143: 0.07, 187: 0.07})

C_REF = 0.177   # m/s, calibration knob
C_W = 0.5
U_REF = 2.237   # m/s (5 mi/h)
C_S = 1.0
# ---------------------------------------------------------------------------

C_WDIR = 0.6    # downwind factor, as in the PINN residual


def spread_rate(fuel, slope_deg, wind_speed, fm100_pct) -> tuple[np.ndarray, list[int]]:
    """R in m/s on one 2-D grid. Returns (rate, unknown_fuel_codes).

    Differs from the training-package version in one deliberate way: an
    unknown FBFM40 code is treated as non-burnable and reported, instead of
    raising, because a live service must not fail on one odd pixel.
    """
    fuel = np.asarray(fuel, dtype=np.float64)
    codes = np.clip(np.where(np.isfinite(fuel), np.rint(fuel), 0), 0, None).astype(int)
    slope = np.asarray(slope_deg, dtype=np.float64)
    wind = np.asarray(wind_speed, dtype=np.float64)
    moist = np.asarray(fm100_pct, dtype=np.float64)

    present = set(np.unique(codes).tolist()) - set(NONBURNABLE)
    unknown = sorted(present - set(R_REL))

    lut_n = max(256, int(codes.max()) + 1)
    lut_rel = np.zeros(lut_n)
    lut_mx = np.full(lut_n, 30.0)
    for code, value in R_REL.items():
        lut_rel[code] = value
    for code, value in MX.items():
        lut_mx[code] = value

    terrain_bad = ~np.isfinite(slope) | (slope < 0) | (slope > 90)
    burnable = (lut_rel[codes] > 0) & ~terrain_bad
    extinction = lut_mx[codes]

    wind = np.where(np.isfinite(wind), np.maximum(wind, 0.0), 0.0)
    moist = np.where(np.isfinite(moist), moist, extinction)
    slope = np.where(terrain_bad, 0.0, slope)

    wind_factor = (1.0 + C_W * wind) / (1.0 + C_W * U_REF)
    slope_factor = 1.0 + C_S * np.tan(np.deg2rad(slope))
    moisture_factor = np.clip(1.0 - moist / extinction, 0.0, 1.0)
    rate = C_REF * lut_rel[codes] * wind_factor * slope_factor * moisture_factor
    rate = np.where(burnable, rate, 0.0)
    return np.nan_to_num(rate, nan=0.0, posinf=0.0, neginf=0.0), unknown


def _upwind_magnitude(phi: np.ndarray, h: float) -> np.ndarray:
    """Osher-Sethian upwind |grad phi| for an outward (F >= 0) speed.

    Boundary: zero-gradient ghost cells (the one-sided difference that would
    reach outside the window is 0). Copying the neighbouring difference
    instead (linear extrapolation) is unstable once R varies cell to cell:
    an edge cell that drops below its neighbour never registers as a
    minimum and runs away. See tests/test_service.py::test_heterogeneous_rate_stable.
    """
    dxm = np.zeros_like(phi)
    dxp = np.zeros_like(phi)
    dym = np.zeros_like(phi)
    dyp = np.zeros_like(phi)
    dx = np.diff(phi, axis=1) / h
    dy = np.diff(phi, axis=0) / h
    dxm[:, 1:] = dx
    dxp[:, :-1] = dx
    dym[1:, :] = dy
    dyp[:-1, :] = dy
    return np.sqrt(
        np.maximum(dxm, 0.0) ** 2 + np.minimum(dxp, 0.0) ** 2
        + np.maximum(dym, 0.0) ** 2 + np.minimum(dyp, 0.0) ** 2
    )


def propagate(
    phi0: np.ndarray,
    spacing_m: float,
    rate: np.ndarray,
    wind_toward: tuple[np.ndarray, np.ndarray] | None,
    times_s,
    cfl: float = 0.45,
    c_wdir: float = C_WDIR,
) -> dict[float, np.ndarray]:
    """Advance phi (meters, negative inside) to each time in ``times_s``.

    ``rate`` is R in m/s on the same grid. ``wind_toward`` is (east, north) unit
    components, or None for no direction. Axis 0 of every array is northing
    (increasing), axis 1 is easting (increasing).
    """
    field = np.asarray(phi0, dtype=np.float64).copy()
    rate = np.asarray(rate, dtype=np.float64)
    if field.shape != rate.shape or min(field.shape) < 3:
        raise ValueError("phi and rate must share a 2-D shape of at least 3x3")
    targets = sorted({float(t) for t in times_s})
    f_max = float(rate.max()) * (1.0 + (c_wdir if wind_toward is not None else 0.0))
    out: dict[float, np.ndarray] = {}
    current = 0.0
    if f_max <= 0:
        return {t: field.copy() for t in targets}
    step = cfl * spacing_m / (np.sqrt(2.0) * f_max)
    for target in targets:
        while current < target - 1e-9:
            dt = min(step, target - current)
            if wind_toward is not None:
                gy, gx = np.gradient(field, spacing_m)
                norm = np.sqrt(gx * gx + gy * gy) + 1e-12
                align = (wind_toward[0] * gx + wind_toward[1] * gy) / norm
                speed = rate * np.maximum(1.0 + c_wdir * align, 0.0)
            else:
                speed = rate
            field -= dt * speed * _upwind_magnitude(field, spacing_m)
            current += dt
        out[target] = field.copy()
    return out


def wind_toward_from_met(direction_from_deg) -> tuple[np.ndarray, np.ndarray]:
    """Meteorological 'from' direction (deg clockwise from north) -> toward unit vector."""
    rad = np.deg2rad(np.asarray(direction_from_deg, dtype=np.float64)) + np.pi
    return np.sin(rad), np.cos(rad)
