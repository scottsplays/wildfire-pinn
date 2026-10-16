"""Figures written by the training and evaluation commands."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _increasing_axes(field: np.ndarray, x: np.ndarray, y: np.ndarray):
    field = np.asarray(field)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x[0] > x[-1]:
        x = x[::-1]
        field = field[:, ::-1]
    if y[0] > y[-1]:
        y = y[::-1]
        field = field[::-1, :]
    return field, x, y


def plot_losses(history: dict, path: Path, title: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.4), layout="constrained")
    series = []
    for key, label in (
        ("total", "total"),
        ("pde", "PDE residual"),
        ("ic", "initial condition"),
        ("obs", "later perimeters"),
        ("eikonal", "eikonal"),
    ):
        values = history.get(key) or []
        if values:
            ax.plot(values, label=label)
            series.append(np.asarray(values, dtype=float))
    if series and all(np.all(values > 0) for values in series):
        ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title(title)
    ax.legend()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_scalar_grid(
    x, y, field, path: Path, title: str, colorbar: str, cmap: str = "RdBu_r", zero: bool = False
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    field, x, y = _increasing_axes(field, x, y)
    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    mesh = ax.pcolormesh(x, y, field, cmap=cmap, shading="auto")
    fig.colorbar(mesh, ax=ax, label=colorbar)
    if zero:
        ax.contour(x, y, field, levels=[0], colors="black", linewidths=1.5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_synthetic_fronts(snapshots: list[tuple[float, np.ndarray, np.ndarray, np.ndarray]], path: Path, rate: float, radius: float) -> None:
    """Each snapshot is (time, X, Y, predicted phi). The dashed circle is the analytical front."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(snapshots), figsize=(4 * len(snapshots), 4), squeeze=False, layout="constrained")
    for ax, (time, x_grid, y_grid, phi) in zip(axes[0], snapshots):
        ax.contour(x_grid, y_grid, phi, levels=[0], colors="crimson", linewidths=2)
        ax.add_patch(plt.Circle((0, 0), radius + rate * time, fill=False, color="black", linestyle="--"))
        ax.set_aspect("equal")
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_title(f"t = {time:g}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    fig.suptitle("Predicted front (red) and analytical circle (dashed)")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_heldout_fronts(panels: list[dict], path: Path) -> None:
    """Each panel has x, y, phi (meters), an observed geometry, and a title."""
    if not panels:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(panels), figsize=(4.4 * len(panels), 4.6), squeeze=False, layout="constrained")
    for ax, panel in zip(axes[0], panels):
        field, x, y = _increasing_axes(panel["phi"], panel["x"], panel["y"])
        ax.contour(x, y, field, levels=[0], colors="crimson", linewidths=2)
        geom = panel["geometry"]
        parts = [geom] if geom.geom_type == "Polygon" else list(getattr(geom, "geoms", []))
        for part in parts:
            if part.geom_type != "Polygon":
                continue
            xs, ys = part.exterior.xy
            ax.plot(xs, ys, color="black", linewidth=1.2)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(panel["title"])
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")
    fig.suptitle("Held-out front: prediction (red) and observed hull (black)")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_playback_gif(frames: list[dict], path: Path, duration_ms: int = 600) -> None:
    """Animated GIF. Each frame has x, y, contours from zero_contours, a geometry, and a title.

    Observed hull boundary is black. Predicted phi=0 polylines are red. Pillow is
    already installed with matplotlib; this does not add a dependency.
    """
    from PIL import Image

    if not frames:
        raise ValueError("playback has no frames")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    images = []
    x_all = np.asarray(frames[0]["x"], dtype=np.float64)
    y_all = np.asarray(frames[0]["y"], dtype=np.float64)
    for frame in frames:
        fig, ax = plt.subplots(figsize=(6.2, 6.2), layout="constrained")
        geom = frame["geometry"]
        parts = [geom] if geom.geom_type == "Polygon" else list(getattr(geom, "geoms", []))
        drew_hull = False
        for part in parts:
            if part.geom_type != "Polygon":
                continue
            xs, ys = part.exterior.xy
            ax.plot(xs, ys, color="black", linewidth=1.2, label="observed hull" if not drew_hull else None)
            drew_hull = True
        drew_front = False
        for polyline in frame["contours"]:
            ax.plot(
                polyline[:, 0], polyline[:, 1], color="crimson", linewidth=1.6,
                label="predicted front" if not drew_front else None,
            )
            drew_front = True
        if not drew_front:
            ax.plot([], [], color="crimson", linewidth=1.6, label="predicted front")
        ax.set_xlim(float(x_all.min()), float(x_all.max()))
        ax.set_ylim(float(y_all.min()), float(y_all.max()))
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(frame["title"])
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")
        ax.legend(loc="upper right", fontsize=8)
        fig.canvas.draw()
        image = np.asarray(fig.canvas.buffer_rgba())
        images.append(Image.fromarray(image.copy()))
        plt.close(fig)
    images[0].save(
        path, save_all=True, append_images=images[1:],
        duration=duration_ms, loop=0, disposal=2,
    )


def plot_dashboard(
    case, k_train: int, path: Path, elevation: np.ndarray | None = None, boundary_label: str | None = None,
) -> None:
    """Four-panel summary: perimeters, detection counts, initial SDF, area growth."""
    from wildfire_pinn.geometry import signed_distance

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    perims = case.perimeters
    times = pd.to_datetime(perims["dt"])
    elapsed = (times - times.min()).dt.total_seconds().to_numpy()
    span = elapsed / elapsed.max() if elapsed.max() > 0 else np.zeros(len(times))
    colors = plt.cm.plasma(span)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), layout="constrained")
    top_left, top_right, bottom_left, bottom_right = axes.ravel()

    if elevation is not None:
        shade = _hillshade(np.nan_to_num(elevation, nan=0.0), case.x, case.y)
        shade, x_shade, y_shade = _increasing_axes(shade, case.x, case.y)
        top_left.pcolormesh(x_shade, y_shade, shade, cmap="gray", shading="auto")
    for color, geom in zip(colors, perims.geometry):
        parts = [geom] if geom.geom_type == "Polygon" else list(getattr(geom, "geoms", []))
        for part in parts:
            if part.geom_type != "Polygon":
                continue
            xs, ys = part.exterior.xy
            top_left.plot(xs, ys, color=color, linewidth=1.6)
    ignition = perims.geometry.iloc[0].centroid
    top_left.plot(ignition.x, ignition.y, marker="*", color="white", markersize=12, markeredgecolor="black")
    top_left.set_aspect("equal", adjustable="box")
    top_left.set_title(f"Fire progression — {len(perims)} overpass perimeters")
    top_left.set_xlabel("Easting (m)")
    top_left.set_ylabel("Northing (m)")

    counts = perims["n_points"] if "n_points" in perims.columns else np.ones(len(perims))
    top_right.bar(range(len(perims)), counts, color=colors)
    top_right.set_xticks(range(len(perims)))
    top_right.set_xticklabels([t.strftime("%m-%d %H:%M") for t in times], rotation=45, ha="right", fontsize=8)
    top_right.axvline(
        k_train - 0.5, color="crimson", linestyle="--",
        label=boundary_label or f"train | held-out (k={k_train})",
    )
    top_right.set_ylabel("detections in the hull")
    top_right.set_title("Detections per satellite overpass")
    top_right.legend(fontsize=8)

    initial = signed_distance(perims.geometry.iloc[0], case.x, case.y)
    initial, x_sdf, y_sdf = _increasing_axes(initial, case.x, case.y)
    mesh = bottom_left.pcolormesh(x_sdf, y_sdf, initial, cmap="RdBu_r", shading="auto")
    bottom_left.contour(x_sdf, y_sdf, initial, levels=[0], colors="black", linewidths=1.4)
    fig.colorbar(mesh, ax=bottom_left, label="signed distance (m)")
    bottom_left.set_aspect("equal", adjustable="box")
    bottom_left.set_title("PINN initial condition (black = phi 0)")
    bottom_left.set_xlabel("Easting (m)")
    bottom_left.set_ylabel("Northing (m)")

    area_km2 = np.array([geom.area for geom in perims.geometry], dtype=float) / 1e6
    bottom_right.plot(times, area_km2, marker="o", color="darkred")
    bottom_right.set_ylabel("hull area (km^2)")
    bottom_right.set_title("Observed burned-area growth (alpha-hull proxy)")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _hillshade(elevation: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    dy = float(np.mean(np.diff(np.asarray(y, dtype=np.float64))))
    dx = float(np.mean(np.diff(np.asarray(x, dtype=np.float64))))
    gy, gx = np.gradient(elevation, dy, dx)
    azimuth, altitude = np.deg2rad(315), np.deg2rad(45)
    slope = np.pi / 2 - np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    shaded = np.sin(altitude) * np.sin(slope) + np.cos(altitude) * np.cos(slope) * np.cos(azimuth - aspect)
    return np.clip(shaded, 0, 1)
