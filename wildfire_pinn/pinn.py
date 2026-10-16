"""Level-set PINN: analytical circle, and the Palisades forecast.

The network takes normalized coordinates and returns one scalar, phi.
The spread rate is an input field, not a second network.

Synthetic problem (constant rate, circular ignition), in the network's own coordinates:

    phi_t + R |grad phi| = 0
    phi(x, y, 0) = sqrt(x^2 + y^2) - r0

Analytical front: a circle of radius r0 + R t. An eikonal penalty keeps phi a
signed distance. No boundary loss is used; the front stays inside [-1, 1]^2.

Palisades problem, with phi the signed distance in meters:

    phi_t + R_eff |grad phi| = 0
    R_eff = R(x, y, day) * (1 + C_WDIR * wind_hat · n)

Training uses dimensionless variables Phi = phi / L, X = (x - xc) / L,
Y = (y - yc) / L, tau = t / T. Substituting gives

    Phi_tau + (R_eff * T / L) |grad_XY Phi| = 0

which is the residual below. L is the longer side of the domain and T is the
span from the first overpass to the last one, including held-out times.
Observation loss sees only the first k_train perimeters.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from wildfire_pinn.geometry import cell_size_m, mean_distance_m, signed_distance, zero_contours

log = logging.getLogger("wildfire_pinn")


def _scalar(value) -> float:
    if torch.is_tensor(value):
        return float(value.detach())
    return float(value)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LevelSetPINN(nn.Module):
    """MLP from (t, x, y) to phi. ``layers`` counts hidden tanh blocks, not the output."""

    def __init__(self, hidden: int = 64, layers: int = 3):
        super().__init__()
        if layers < 1:
            raise ValueError("layers must be at least 1")
        if hidden < 1:
            raise ValueError("hidden width must be at least 1")
        blocks: list[nn.Module] = [nn.Linear(3, hidden), nn.Tanh()]
        for _ in range(layers - 1):
            blocks.extend([nn.Linear(hidden, hidden), nn.Tanh()])
        blocks.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*blocks)
        self.hidden = int(hidden)
        self.layers = int(layers)
        self.reset_parameters()

    @property
    def arch(self) -> dict:
        return {"hidden": self.hidden, "layers": self.layers}

    def reset_parameters(self) -> None:
        # Kaiming init is scaled for ReLU. These layers are tanh, so use Xavier.
        linear = [module for module in self.net if isinstance(module, nn.Linear)]
        gain = nn.init.calculate_gain("tanh")
        for module in linear[:-1]:
            nn.init.xavier_normal_(module.weight, gain=gain)
            nn.init.zeros_(module.bias)
        nn.init.xavier_normal_(linear[-1].weight, gain=1.0)
        nn.init.zeros_(linear[-1].bias)

    def forward(self, t, x, y) -> torch.Tensor:
        device = next(self.parameters()).device
        columns = []
        for value in (t, x, y):
            if not torch.is_tensor(value):
                value = torch.as_tensor(value, dtype=torch.float32, device=device)
            columns.append(value.reshape(-1, 1).to(device=device, dtype=torch.float32))
        return self.net(torch.cat(columns, dim=1)).squeeze(-1)


def save_checkpoint(path: Path, model: LevelSetPINN, config: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "arch": model.arch, "config": config},
        path,
    )


def load_checkpoint(path: Path, model: LevelSetPINN, device: torch.device) -> dict:
    """Load weights saved for this exact width and depth.

    The Colab crash was a 64-unit synthetic checkpoint applied to the 128-unit
    Palisades network (and a model object that was not the one just constructed).
    A mismatch raises, with an instruction to retrain, instead of a raw state_dict error.
    """
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
        saved_arch = checkpoint.get("arch")
        saved_config = checkpoint.get("config") or {}
    else:
        state = checkpoint
        saved_arch = None
        saved_config = {}
    if saved_arch is not None and dict(saved_arch) != model.arch:
        raise RuntimeError(
            f"Checkpoint {path} was trained with architecture {saved_arch}, "
            f"but this run builds {model.arch}. "
            "The synthetic circle model and the Palisades model are different widths. "
            "Pass --retrain to discard the old weights."
        )
    try:
        model.load_state_dict(state)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Could not load {path} into a {model.arch} network. "
            "This is the checkpoint mismatch from the Colab notebook "
            "(a 64-unit file loaded into a 128-unit model, or the reverse). "
            "Pass --retrain.\n"
            f"{exc}"
        ) from exc
    model.to(device)
    return saved_config


def isotropic_residual(phi_t, phi_x, phi_y, rate: float):
    """phi_t + R |grad phi|, and the gradient magnitude (for the eikonal term)."""
    magnitude = torch.sqrt(phi_x ** 2 + phi_y ** 2 + 1e-8)
    return phi_t + rate * magnitude, magnitude


def directed_residual(phi_t, phi_x, phi_y, rate, wind_east, wind_north, c_wdir: float, t_over_l: float):
    """Dimensionless level-set residual with a downwind factor.

    ``rate`` is in m/s, ``t_over_l`` is T/L in s/m, and the wind components are
    a unit vector in the direction the wind blows toward (east, north).
    The normal is grad(phi) / |grad(phi)|, which points out of the burned region
    when phi is negative inside.
    """
    magnitude = torch.sqrt(phi_x ** 2 + phi_y ** 2 + 1e-12)
    alignment = wind_east * phi_x / magnitude + wind_north * phi_y / magnitude
    effective = torch.clamp(rate * (1.0 + c_wdir * alignment), min=0.0)
    return phi_t + (effective * t_over_l) * magnitude


def eikonal_loss_from_gradients(phi_x, phi_y):
    """Mean squared departure of |grad_XY Phi| from 1, and that magnitude.

    ``phi_x`` and ``phi_y`` are derivatives of dimensionless Phi with respect to
    X = (x - xc) / L and Y = (y - yc) / L. For a signed distance in meters,
    those derivatives already have magnitude 1, so this residual is dimensionless.
    """
    magnitude = torch.sqrt(phi_x ** 2 + phi_y ** 2 + 1e-12)
    return torch.mean((magnitude - 1.0) ** 2), magnitude


def analytic_signed_distance_eikonal(length: float = 1000.0, radius_m: float = 200.0):
    """Eikonal loss of an exact circular signed distance, independent of the network.

    Phi = sqrt(X^2 + Y^2) - r/L. Points stay off the origin, where the gradient
    of a circle is undefined. Returns the scalar loss and the gradient magnitudes.
    """
    angles = torch.linspace(0, 2 * np.pi, 17)[:-1]
    radii = torch.tensor([0.05, 0.15, 0.35, 0.6, 0.85], dtype=torch.float32)
    xs = []
    ys = []
    for radius in radii:
        xs.append(radius * torch.cos(angles))
        ys.append(radius * torch.sin(angles))
    x_norm = torch.cat(xs).detach().requires_grad_(True)
    y_norm = torch.cat(ys).detach().requires_grad_(True)
    phi = torch.sqrt(x_norm ** 2 + y_norm ** 2) - float(radius_m) / float(length)
    phi_x, phi_y = torch.autograd.grad(phi, (x_norm, y_norm), torch.ones_like(phi), create_graph=True)
    loss, magnitude = eikonal_loss_from_gradients(phi_x, phi_y)
    return loss.detach(), magnitude.detach()


def classify_point(phi: float, tol: float = 1e-3) -> str:
    if phi < -tol:
        return "inside"
    if phi > tol:
        return "outside"
    return "on_interface"


def outward_normals(phi_fn, t, x, y):
    """Unit outward normal of a callable phi(t, x, y)."""
    def leaf(value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value, dtype=torch.float32)
        return value.reshape(-1).detach().float().requires_grad_(True)

    t, x, y = leaf(t), leaf(x), leaf(y)
    phi = phi_fn(t, x, y)
    phi_x, phi_y = torch.autograd.grad(phi, (x, y), torch.ones_like(phi))
    magnitude = torch.sqrt(phi_x ** 2 + phi_y ** 2 + 1e-8)
    return (phi_x / magnitude).detach(), (phi_y / magnitude).detach()


def sample_grid(field, x_coords, y_coords, x_query, y_query) -> torch.Tensor:
    """Bilinear sample of a (row=y, col=x) field at physical coordinates.

    grid_sample's -1 is the first column and the first row when align_corners
    is true. Mapping query coordinates from ``coords[0]`` to ``coords[-1]``
    therefore works whether y increases or decreases.
    """
    if not torch.is_tensor(x_query):
        x_query = torch.as_tensor(np.asarray(x_query), dtype=torch.float32)
    x_query = x_query.reshape(-1).to(dtype=torch.float32)
    if not torch.is_tensor(y_query):
        y_query = torch.as_tensor(np.asarray(y_query), dtype=torch.float32, device=x_query.device)
    y_query = y_query.reshape(-1).to(device=x_query.device, dtype=torch.float32)
    if x_query.numel() == 0:
        return x_query
    if not torch.is_tensor(field):
        field = torch.as_tensor(field, dtype=torch.float32, device=x_query.device)
    else:
        field = field.to(device=x_query.device, dtype=torch.float32)
    x_axis = torch.as_tensor(x_coords, dtype=torch.float32, device=x_query.device)
    y_axis = torch.as_tensor(y_coords, dtype=torch.float32, device=x_query.device)
    x_span = x_axis[-1] - x_axis[0]
    y_span = y_axis[-1] - y_axis[0]
    if float(x_span) == 0.0 or float(y_span) == 0.0:
        raise ValueError("cannot sample a grid with a zero-length axis")
    grid_x = 2.0 * (x_query - x_axis[0]) / x_span - 1.0
    grid_y = 2.0 * (y_query - y_axis[0]) / y_span - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1).view(1, -1, 1, 2)
    image = field.reshape(1, 1, field.shape[-2], field.shape[-1])
    sampled = F.grid_sample(
        image, grid, mode="bilinear", padding_mode="border", align_corners=True
    )
    return sampled.view(-1)


def sample_volume(volume, day_index, x_coords, y_coords, x_query, y_query) -> torch.Tensor:
    """Sample a (day, y, x) volume. One grid_sample per distinct day, not one copy per point."""
    if not torch.is_tensor(volume):
        volume = torch.as_tensor(volume, dtype=torch.float32, device=x_query.device)
    else:
        volume = volume.to(device=x_query.device, dtype=torch.float32)
    out = torch.empty(x_query.reshape(-1).shape[0], dtype=torch.float32, device=x_query.device)
    for day in day_index.detach().unique():
        selected = day_index == day
        out[selected] = sample_grid(
            volume[int(day)], x_coords, y_coords, x_query[selected], y_query[selected]
        )
    return out


def day_index_from_tau(tau, t0_offset_s: float, duration_s: float, n_days: int) -> torch.Tensor:
    seconds = t0_offset_s + tau.detach() * duration_s
    index = torch.floor(seconds / 86400.0).to(dtype=torch.long)
    return index.clamp(0, n_days - 1)


@dataclass
class SyntheticConfig:
    epochs: int = 2001
    lr: float = 1e-3
    hidden: int = 64
    layers: int = 3
    n_pde: int = 5000
    n_ic: int = 1000
    w_ic: float = 10.0
    w_pde: float = 1.0
    w_eik: float = 0.1
    rate: float = 1.0
    radius: float = 0.2
    t_final: float = 0.5
    seed: int = 0
    log_every: int = 500


def _synthetic_targets(x, y, radius: float):
    return torch.sqrt(x ** 2 + y ** 2) - radius


def train_synthetic(out_dir: Path, hp: SyntheticConfig | None = None, retrain: bool = False):
    hp = hp or SyntheticConfig()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()
    checkpoint = out_dir / "levelset_pinn.pt"
    model = LevelSetPINN(hp.hidden, hp.layers).to(device)
    if checkpoint.exists() and not retrain:
        load_checkpoint(checkpoint, model, device)
        model.eval()
        log.info("loaded %s; training skipped", checkpoint)
        history = _read_history(out_dir / "history.json")
        return model, history

    set_seed(hp.seed)
    log.info("synthetic PINN on %s | arch %s | %d epochs", device, model.arch, hp.epochs)
    t_pde = torch.rand(hp.n_pde, device=device) * hp.t_final
    x_pde = torch.rand(hp.n_pde, device=device) * 2 - 1
    y_pde = torch.rand(hp.n_pde, device=device) * 2 - 1
    t_ic = torch.zeros(hp.n_ic, device=device)
    x_ic = torch.rand(hp.n_ic, device=device) * 2 - 1
    y_ic = torch.rand(hp.n_ic, device=device) * 2 - 1
    ic_target = _synthetic_targets(x_ic, y_ic, hp.radius)

    optimizer = torch.optim.Adam(model.parameters(), lr=hp.lr)
    history = {"total": [], "pde": [], "ic": [], "eikonal": []}
    model.train()
    for epoch in range(hp.epochs):
        t = t_pde.detach().requires_grad_(True)
        x = x_pde.detach().requires_grad_(True)
        y = y_pde.detach().requires_grad_(True)
        phi = model(t, x, y)
        phi_t, phi_x, phi_y = torch.autograd.grad(
            phi, (t, x, y), torch.ones_like(phi), create_graph=True
        )
        residual, magnitude = isotropic_residual(phi_t, phi_x, phi_y, hp.rate)
        loss_pde = torch.mean(residual ** 2)
        loss_eik = torch.mean((magnitude - 1.0) ** 2)
        loss_ic = torch.mean((model(t_ic, x_ic, y_ic) - ic_target) ** 2)
        loss = hp.w_ic * loss_ic + hp.w_pde * loss_pde + hp.w_eik * loss_eik
        if not torch.isfinite(loss):
            raise RuntimeError(f"Synthetic loss became non-finite at epoch {epoch}.")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        history["total"].append(_scalar(loss))
        history["pde"].append(_scalar(loss_pde))
        history["ic"].append(_scalar(loss_ic))
        history["eikonal"].append(_scalar(loss_eik))
        if epoch % hp.log_every == 0 or epoch == hp.epochs - 1:
            log.info(
                "epoch %4d | total %.3e | ic %.3e | pde %.3e | eikonal %.3e",
                epoch, _scalar(loss), _scalar(loss_ic), _scalar(loss_pde), _scalar(loss_eik),
            )
    save_checkpoint(checkpoint, model, asdict(hp))
    _write_history(out_dir / "history.json", history)
    log.info("wrote %s", checkpoint)
    model.eval()
    return model, history


def analytical_phi(x, y, t, radius: float, rate: float):
    return np.sqrt(x ** 2 + y ** 2) - (radius + rate * t)


def relative_l2_synthetic(model: LevelSetPINN, hp: SyntheticConfig, grid: int = 101) -> float:
    device = next(model.parameters()).device
    axis = np.linspace(-1, 1, grid)
    x_grid, y_grid = np.meshgrid(axis, axis)
    x_t = torch.tensor(x_grid.ravel(), dtype=torch.float32, device=device)
    y_t = torch.tensor(y_grid.ravel(), dtype=torch.float32, device=device)
    numerator = 0.0
    denominator = 0.0
    times = (0.0, hp.t_final / 3, 2 * hp.t_final / 3, hp.t_final)
    with torch.no_grad():
        for time in times:
            pred = model(torch.full((x_t.shape[0],), float(time), device=device), x_t, y_t)
            pred = pred.cpu().numpy().reshape(x_grid.shape)
            truth = analytical_phi(x_grid, y_grid, time, hp.radius, hp.rate)
            numerator += float(np.sum((pred - truth) ** 2))
            denominator += float(np.sum(truth ** 2))
    return float(np.sqrt(numerator / denominator))


def synthetic_snapshots(model, hp: SyntheticConfig, grid: int = 121):
    device = next(model.parameters()).device
    axis = np.linspace(-1, 1, grid)
    x_grid, y_grid = np.meshgrid(axis, axis)
    x_t = torch.tensor(x_grid.ravel(), dtype=torch.float32, device=device)
    y_t = torch.tensor(y_grid.ravel(), dtype=torch.float32, device=device)
    snapshots = []
    times = (0.0, hp.t_final / 3, 2 * hp.t_final / 3, hp.t_final)
    with torch.no_grad():
        for time in times:
            pred = model(torch.full((x_t.shape[0],), float(time), device=device), x_t, y_t)
            snapshots.append((time, x_grid, y_grid, pred.cpu().numpy().reshape(x_grid.shape)))
    return snapshots


@dataclass
class FireCase:
    """One fire, already on the model grid. Perimeters are in the same CRS as x and y."""

    x: np.ndarray
    y: np.ndarray
    spread: np.ndarray
    wind_from_deg: np.ndarray
    perimeters: pd.DataFrame
    day0: pd.Timestamp


@dataclass
class RealConfig:
    k_train: int = 10
    epochs: int = 4000
    lr: float = 1e-3
    hidden: int = 128
    layers: int = 4
    n_pde: int = 8192
    n_ic: int = 4096
    n_obs: int = 1024
    w_ic: float = 10.0
    w_pde: float = 1.0
    w_obs: float = 5.0
    c_wdir: float = 0.6
    seed: int = 0
    log_every: int = 500


def resolve_k_train(n_perimeters: int, k_train: int) -> int:
    if n_perimeters < 2:
        raise RuntimeError(f"Need at least 2 perimeters so one can be held out; found {n_perimeters}.")
    if k_train >= n_perimeters:
        adjusted = n_perimeters - 1
        log.warning("k_train reduced from %d to %d so the last perimeter is held out", k_train, adjusted)
        return adjusted
    if k_train < 1:
        return 1
    return k_train


@dataclass
class TimeSplit:
    """Chronological index groups. Train is always a prefix, so nothing is shuffled."""

    train: list[int]
    validation: list[int]
    test: list[int]
    scheme: str

    def label(self, index: int) -> str:
        if index in self.validation:
            return "validation"
        if index in self.test:
            return "held_out" if self.scheme == "k_train" else "test"
        return "train"

    def stamp_groups(self, case: FireCase) -> dict:
        def stamps(indices: list[int]) -> list[str]:
            return [_stamp(case.perimeters["dt"].iloc[index]) for index in indices]

        if self.scheme == "k_train":
            return {"train": stamps(self.train), "held_out": stamps(self.test)}
        return {
            "train": stamps(self.train),
            "validation": stamps(self.validation),
            "test": stamps(self.test),
        }


def resolve_time_split(
    n_perimeters: int,
    k_train: int,
    scheme: str = "k_train",
    train_fraction: float = 0.6,
    val_fraction: float = 0.2,
) -> TimeSplit:
    """Train, then validation, then test, in time order.

    ``k_train`` is the Palisades split: the first ``k_train`` overpasses train
    and the remainder is held out. ``chronological`` uses the fractions and
    keeps at least one test overpass. Validation is the block between them
    and is not used in the loss.
    """
    if scheme == "k_train":
        trained = resolve_k_train(n_perimeters, k_train)
        return TimeSplit(list(range(trained)), [], list(range(trained, n_perimeters)), "k_train")
    if scheme != "chronological":
        raise RuntimeError(f"Unknown split scheme {scheme!r}. Use k_train or chronological.")
    if n_perimeters < 3:
        trained = n_perimeters - 1
        return TimeSplit(list(range(trained)), [], [trained], "chronological")
    n_train = max(1, int(np.floor(n_perimeters * train_fraction)))
    n_val = max(1, int(np.floor(n_perimeters * val_fraction)))
    if n_train + n_val >= n_perimeters:
        n_train = n_perimeters - 2
        n_val = 1
    n_test = n_perimeters - n_train - n_val
    if n_test < 1:
        n_test = 1
        n_val = max(1, n_perimeters - n_train - n_test)
    return TimeSplit(
        list(range(n_train)),
        list(range(n_train, n_train + n_val)),
        list(range(n_train + n_val, n_perimeters)),
        "chronological",
    )


def domain_scales(case: FireCase):
    x = np.asarray(case.x, dtype=np.float64)
    y = np.asarray(case.y, dtype=np.float64)
    if np.any(np.diff(x) <= 0):
        raise RuntimeError("grid eastings must be strictly increasing")
    center_x = float(0.5 * (x.min() + x.max()))
    center_y = float(0.5 * (y.min() + y.max()))
    length = float(max(x.max() - x.min(), y.max() - y.min()))
    if length <= 0:
        raise RuntimeError("domain has zero extent")
    times = pd.to_datetime(case.perimeters["dt"])
    start = times.iloc[0]
    duration = (times.iloc[-1] - start).total_seconds()
    if duration <= 0:
        raise RuntimeError("perimeter timestamps do not span a positive duration")
    tau = ((times - start).dt.total_seconds() / duration).to_numpy(np.float64)
    t0_offset = (start - pd.Timestamp(case.day0)).total_seconds()
    return center_x, center_y, length, float(duration), float(t0_offset), tau


def downwind_unit_vectors(wind_from_deg: np.ndarray) -> np.ndarray:
    """Unit vector the wind blows toward, averaged over the domain each day.

    Meteorological direction is degrees clockwise from north, the direction the
    wind comes from. Adding pi reverses it: 0 degrees (from the north) blows
    toward the south, which is (east, north) = (0, -1).
    """
    radians = np.deg2rad(np.asarray(wind_from_deg, dtype=np.float64))
    east = np.sin(radians + np.pi)
    north = np.cos(radians + np.pi)
    with np.errstate(all="ignore"):
        mean_east = np.nanmean(east.reshape(east.shape[0], -1), axis=1)
        mean_north = np.nanmean(north.reshape(north.shape[0], -1), axis=1)
    vectors = np.stack([mean_east, mean_north], axis=1)
    missing = ~np.isfinite(vectors).all(axis=1)
    vectors[missing] = 0.0
    norm = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.maximum(norm, 1e-12)
    return vectors.astype(np.float32)


def _sample_box(count: int, x: np.ndarray, y: np.ndarray, device: torch.device):
    x_query = torch.rand(count, device=device) * (float(x.max()) - float(x.min())) + float(x.min())
    y_query = torch.rand(count, device=device) * (float(y.max()) - float(y.min())) + float(y.min())
    return x_query, y_query


def _region_counts(count: int) -> tuple[int, int, int]:
    """Inside, on the perimeter, and outside. The remainder goes outside."""
    if count < 3:
        raise ValueError(f"perimeter supervision needs at least 3 points, got {count}")
    inside = count // 3
    boundary = count // 3
    return inside, boundary, count - inside - boundary


def _boundary_coordinates(geom, count: int) -> np.ndarray:
    """Points spaced along the perimeter. Their signed distance is exactly zero."""
    boundary = geom.boundary
    parts = [boundary] if boundary.geom_type == "LineString" else list(getattr(boundary, "geoms", []))
    parts = [part for part in parts if part.length > 0]
    if not parts:
        raise RuntimeError("training perimeter has no boundary to supervise")
    lengths = np.array([part.length for part in parts], dtype=np.float64)
    shares = np.maximum(1, np.round(count * lengths / lengths.sum()).astype(int))
    while int(shares.sum()) > count:
        shares[int(np.argmax(shares))] -= 1
    while int(shares.sum()) < count:
        shares[int(np.argmax(lengths))] += 1
    coords = []
    for part, n_part in zip(parts, shares):
        if int(n_part) <= 0:
            continue
        for distance in np.linspace(0.0, part.length, int(n_part), endpoint=False):
            point = part.interpolate(float(distance))
            coords.append((point.x, point.y))
    return np.asarray(coords, dtype=np.float32)


def _supervision_pools(geom, sdf: np.ndarray, x: np.ndarray, y: np.ndarray, device: torch.device, boundary_count: int = 4096):
    """Cell centers inside and outside the fire, plus points that lie on the perimeter.

    A uniform draw on the whole window lands outside almost every time, because
    the fire is a small part of the box. Those draws supervise a positive
    distance and leave the interior free to stay positive too.
    """
    inside = sdf < 0
    outside = sdf > 0
    if not np.any(inside):
        raise RuntimeError("signed-distance field has no interior cells to supervise")
    if not np.any(outside):
        raise RuntimeError("signed-distance field has no exterior cells to supervise")
    yy, xx = np.meshgrid(np.asarray(y, dtype=np.float32), np.asarray(x, dtype=np.float32), indexing="ij")
    boundary = _boundary_coordinates(geom, boundary_count)

    def _coords(mask):
        return (
            torch.tensor(xx[mask], dtype=torch.float32, device=device),
            torch.tensor(yy[mask], dtype=torch.float32, device=device),
        )

    return {
        "inside": _coords(inside),
        "outside": _coords(outside),
        "boundary": (
            torch.tensor(boundary[:, 0], dtype=torch.float32, device=device),
            torch.tensor(boundary[:, 1], dtype=torch.float32, device=device),
        ),
        "n_inside": int(np.count_nonzero(inside)),
        "n_outside": int(np.count_nonzero(outside)),
    }


def _draw_pool(pool_x: torch.Tensor, pool_y: torch.Tensor, count: int):
    choice = torch.randint(0, pool_x.shape[0], (count,), device=pool_x.device)
    return pool_x[choice], pool_y[choice]


def _even_counts(total: int, groups: int) -> list[int]:
    if groups < 1:
        raise ValueError("need at least one group")
    if total < groups:
        raise ValueError(f"{total} points cannot cover {groups} groups")
    base, extra = divmod(int(total), int(groups))
    return [base + (1 if index < extra else 0) for index in range(groups)]


def _masked_coords(mask: np.ndarray, x: np.ndarray, y: np.ndarray, meters: np.ndarray, device):
    if not np.any(mask):
        return None
    yy, xx = np.meshgrid(np.asarray(y, dtype=np.float32), np.asarray(x, dtype=np.float32), indexing="ij")
    chosen = meters[mask]
    return {
        "x": torch.tensor(xx[mask], dtype=torch.float32, device=device),
        "y": torch.tensor(yy[mask], dtype=torch.float32, device=device),
        "n": int(np.count_nonzero(mask)),
        "range_m": (float(chosen.min()), float(chosen.max())),
    }


def _eikonal_region_pools(sdf, x, y, device, near_m: float, band_m: float, length: float) -> dict:
    """Cell centers for the front Eikonal loss.

    ``near`` is the one-cell supervision ring. ``shell`` is the rest of the
    symmetric ±band neighborhood, so deep interior and far exterior stay out.
    ``band`` is the union, used only to measure |grad Phi| near the front.
    """
    slack_m = 0.01
    meters = np.asarray(sdf, dtype=np.float64) * float(length)
    near_limit = float(near_m) + slack_m
    band_limit = float(band_m) + slack_m
    if band_limit <= near_limit:
        raise RuntimeError(
            f"Eikonal band {band_m:.0f} m must be wider than the near-front ring {near_m:.0f} m."
        )
    absolute = np.abs(meters)
    regions = {
        "near": absolute <= near_limit,
        "shell": (absolute > near_limit) & (absolute <= band_limit),
        "band": absolute <= band_limit,
    }
    pools = {}
    for name, mask in regions.items():
        pool = _masked_coords(mask, x, y, meters, device)
        if pool is None:
            raise RuntimeError(
                f"No cells in the Eikonal {name} pool "
                f"(|signed distance| near {near_m:.0f} m, band {band_m:.0f} m)."
            )
        pools[name] = pool
    return pools


def _eikonal_batch(model, region_pools, tau_values, counts, center_x: float, center_y: float, length: float):
    """mean((|grad_XY Phi| - 1)^2) on a stratified front sample.

    For each training time, half the points (the extra point, when the count is
    odd) come from the one-cell ring and half from the remainder of the ±band.
    Both halves are drawn with replacement. The mean treats the two halves
    equally, so the much larger ±band does not drown the ring.
    """
    xs = []
    ys = []
    taus = []
    device = next(model.parameters()).device
    for regions, tau_value, count in zip(region_pools, tau_values, counts):
        n_shell = count // 2
        n_near = count - n_shell
        for name, n_draw in (("near", n_near), ("shell", n_shell)):
            if n_draw <= 0:
                continue
            x_query, y_query = _draw_pool(regions[name]["x"], regions[name]["y"], n_draw)
            xs.append(x_query)
            ys.append(y_query)
            taus.append(torch.full((n_draw,), float(tau_value), device=device))
    x_query = torch.cat(xs)
    y_query = torch.cat(ys)
    tau = torch.cat(taus)
    x_norm = ((x_query - center_x) / length).detach().requires_grad_(True)
    y_norm = ((y_query - center_y) / length).detach().requires_grad_(True)
    phi = model(tau, x_norm, y_norm)
    phi_x, phi_y = torch.autograd.grad(
        phi, (x_norm, y_norm), torch.ones_like(phi), create_graph=True
    )
    loss, _magnitude = eikonal_loss_from_gradients(phi_x, phi_y)
    return loss


def _median_front_gradient(model, tau_value: float, pool: dict, center_x: float, center_y: float, length: float, count: int, generator: torch.Generator) -> float:
    """Median |grad_XY Phi| on a replacement sample. Does not change the training RNG."""
    choice = torch.randint(0, pool["n"], (count,), generator=generator)
    x_query = pool["x"][choice.to(pool["x"].device)]
    y_query = pool["y"][choice.to(pool["y"].device)]
    device = x_query.device
    x_norm = ((x_query - center_x) / length).detach().requires_grad_(True)
    y_norm = ((y_query - center_y) / length).detach().requires_grad_(True)
    tau = torch.full((count,), float(tau_value), device=device)
    phi = model(tau, x_norm, y_norm)
    phi_x, phi_y = torch.autograd.grad(phi, (x_norm, y_norm), torch.ones_like(phi), create_graph=False)
    _loss, magnitude = eikonal_loss_from_gradients(phi_x, phi_y)
    return float(magnitude.detach().median())


def _temporal_negative_summary(rows: list[dict]) -> dict:
    """Whether the predicted negative set retreats. Predictions are not edited.

    A step loses area where a cell was negative and the next overpass is not.
    The set is approximately non-decreasing when every step loses less than 5 km².
    A large region disappears when some overpass has at least 20 km² negative and
    a later overpass keeps less than a quarter of that area.
    """
    areas = [float(row["negative_km2"]) for row in rows]
    losses = [float(row.get("lost_km2", 0.0)) for row in rows[1:]]
    worst = None
    for earlier in rows:
        if earlier["negative_km2"] < 20.0:
            continue
        for later in rows[earlier["index"] + 1:]:
            if later["negative_km2"] < 0.25 * earlier["negative_km2"]:
                drop = earlier["negative_km2"] - later["negative_km2"]
                if worst is None or drop > worst["drop_km2"]:
                    worst = {
                        "from_index": earlier["index"],
                        "to_index": later["index"],
                        "from_km2": earlier["negative_km2"],
                        "to_km2": later["negative_km2"],
                        "drop_km2": drop,
                    }
    max_lost = max(losses) if losses else 0.0
    return {
        "approximately_nondecreasing": max_lost < 5.0 and worst is None,
        "steps_losing_at_least_5km2": sum(1 for value in losses if value >= 5.0),
        "max_lost_km2_one_step": max_lost,
        "large_region_disappears": worst is not None,
        "worst_shrink": worst,
        "negative_km2": areas,
    }


def _perimeter_batch(model, tau_value: float, pool: dict, sdf, case: FireCase, count: int, center_x: float, center_y: float, length: float):
    """MSE of normalized phi against the signed distance, averaged over three regions.

    Inside targets are negative, boundary targets are 0, outside targets are
    positive. Each region is its own mean so a large exterior distance cannot
    hide an interior that never crosses zero.
    """
    n_in, n_on, n_out = _region_counts(count)
    device = next(model.parameters()).device
    x_in, y_in = _draw_pool(*pool["inside"], n_in)
    x_on, y_on = _draw_pool(*pool["boundary"], n_on)
    x_out, y_out = _draw_pool(*pool["outside"], n_out)
    pieces = []
    for x_query, y_query, target in (
        (x_in, y_in, sample_grid(sdf, case.x, case.y, x_in, y_in)),
        (x_on, y_on, torch.zeros(n_on, device=device)),
        (x_out, y_out, sample_grid(sdf, case.x, case.y, x_out, y_out)),
    ):
        phi = model(
            torch.full((x_query.shape[0],), float(tau_value), device=device),
            (x_query - center_x) / length,
            (y_query - center_y) / length,
        )
        pieces.append(F.mse_loss(phi, target))
    return torch.stack(pieces).mean()


def near_front_band_m(x: np.ndarray, y: np.ndarray) -> float:
    """Width of the near-front band, equal to one grid cell.

    Cell-center signed distance changes by about one cell across the perimeter,
    so |phi| <= this band is the ring of cells the front actually cuts.
    """
    return float(cell_size_m(x, y))


def _balanced_supervision_pools(sdf, x, y, device: torch.device, band_m: float, length: float):
    """Disjoint cell-center pools for a balanced signed-distance batch.

    Interior is phi < -band, near-front is |phi| <= band, exterior is phi > band.
    The regression target stays the signed distance. A small fire is drawn with
    replacement later, so a short pool is allowed and an empty one is not.

    The field is stored as float32 phi/L. A one-cell ring of exactly ``band_m``
    comes back a fraction of a millimeter over the cutoff, which would empty
    the near-front pool. One centimeter of slack puts that ring back in the band
    and does not reach the next cell (about 127 m on a diagonal).
    """
    slack_m = 0.01
    meters = np.asarray(sdf, dtype=np.float64) * float(length)
    limit = float(band_m) + slack_m
    masks = {
        "interior": meters < -limit,
        "near": np.abs(meters) <= limit,
        "exterior": meters > limit,
    }
    if not np.any(masks["interior"]):
        raise RuntimeError(
            f"No cells with signed distance < -{band_m:.0f} m. "
            "The perimeter has no interior beyond the near-front band."
        )
    if not np.any(masks["near"]):
        raise RuntimeError(
            f"No cells with |signed distance| <= {band_m:.0f} m."
        )
    if not np.any(masks["exterior"]):
        raise RuntimeError(
            f"No cells with signed distance > {band_m:.0f} m."
        )
    yy, xx = np.meshgrid(np.asarray(y, dtype=np.float32), np.asarray(x, dtype=np.float32), indexing="ij")
    pools = {}
    for name, mask in masks.items():
        pools[name] = (
            torch.tensor(xx[mask], dtype=torch.float32, device=device),
            torch.tensor(yy[mask], dtype=torch.float32, device=device),
        )
        pools[f"n_{name}"] = int(np.count_nonzero(mask))
        pools[f"range_{name}"] = (
            float(meters[mask].min()),
            float(meters[mask].max()),
        )
    return pools


def _balanced_perimeter_batch(model, tau_value: float, pool: dict, sdf, case: FireCase, count: int, center_x: float, center_y: float, length: float, clip_m: float | None = None):
    """Three SDF losses: interior, near-front, exterior.

    Each third is its own mean. Targets are the signed-distance field, including
    inside the near-front band, where the target is a small signed distance
    rather than a forced zero. ``clip_m`` truncates those targets to a symmetric
    band in meters. The sample locations do not change, and nothing is added
    to the network output.
    """
    n_in, n_near, n_out = _region_counts(count)
    device = next(model.parameters()).device
    groups = (
        ("interior", n_in),
        ("near", n_near),
        ("exterior", n_out),
    )
    losses = []
    for name, n_draw in groups:
        x_query, y_query = _draw_pool(*pool[name], n_draw)
        target = sample_grid(sdf, case.x, case.y, x_query, y_query)
        if clip_m is not None:
            limit = float(clip_m) / float(length)
            target = target.clamp(-limit, limit)
        phi = model(
            torch.full((x_query.shape[0],), float(tau_value), device=device),
            (x_query - center_x) / length,
            (y_query - center_y) / length,
        )
        losses.append(F.mse_loss(phi, target))
    return losses[0], losses[1], losses[2]


def _perimeter_stamp(case: FireCase, k_train: int) -> str:
    times = [
        pd.Timestamp(case.perimeters["dt"].iloc[index]).isoformat()
        for index in range(k_train)
    ]
    return "|".join(times)


def build_training_sdfs(case: FireCase, k_train: int, length: float, cache_path: Path | None, geometries=None) -> np.ndarray:
    expected = (k_train, len(case.y), len(case.x))
    stamp = _perimeter_stamp(case, k_train)
    if geometries is not None:
        stamp += "|persistent_burn_v1"
    if cache_path is not None and cache_path.exists():
        cached = np.load(cache_path)
        cached_stamp = str(cached["stamp"]) if "stamp" in cached.files else ""
        if cached["sdf"].shape == expected and cached_stamp == stamp:
            log.info("loaded SDF cache %s %s", cache_path, cached["sdf"].shape)
            return cached["sdf"].astype(np.float32)
        log.warning("SDF cache %s does not match this grid or these perimeters; rebuilding", cache_path)
    log.info("building signed-distance fields for %d training perimeters", k_train)
    fields = []
    for index in range(k_train):
        geom = case.perimeters.geometry.iloc[index] if geometries is None else geometries[index]
        distance_m = signed_distance(geom, case.x, case.y)
        fields.append(distance_m / length)
        log.info(
            "  perimeter %d | phi in [%.0f, %.0f] m",
            index, float(distance_m.min()), float(distance_m.max()),
        )
    stacked = np.stack(fields).astype(np.float32)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, sdf=stacked, stamp=np.array(stamp))
        log.info("wrote %s", cache_path)
    return stacked


def _contour_summary(rows: list[dict]) -> dict:
    summary = {}
    for label in ("train", "validation", "test", "held_out"):
        group = [row for row in rows if row["split"] == label]
        if not group:
            continue
        summary[label] = {
            "contours": sum(1 for row in group if row["zero_contour"]),
            "n": len(group),
        }
    return summary


def train_real(
    case: FireCase,
    out_dir: Path,
    hp: RealConfig | None = None,
    retrain: bool = False,
    split_scheme: str = "k_train",
    train_fraction: float = 0.6,
    val_fraction: float = 0.2,
    sdf_sampling: str = "thirds",
    sign_epochs: tuple[int, ...] | None = None,
    sdf_clip_m: float | None = None,
    w_eik: float = 0.0,
    eik_band_m: float = 2000.0,
    n_eik: int = 4096,
    target_geometries=None,
):
    hp = hp or RealConfig()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()
    time_split = resolve_time_split(
        len(case.perimeters), hp.k_train, split_scheme, train_fraction, val_fraction
    )
    k_train = len(time_split.train)
    center_x, center_y, length, duration, t0_offset, tau = domain_scales(case)
    log.info(
        "overpasses: %d | train %d | validation %d | test %d | scheme %s | T=%.2f days | L=%.0f m | T/L=%.4f s/m",
        len(case.perimeters), len(time_split.train), len(time_split.validation), len(time_split.test),
        time_split.scheme, duration / 86400.0, length, duration / length,
    )
    log.info(
        "tau = 0 at the first overpass and tau = 1 at the last overpass. "
        "The observation loss uses only the first %d perimeters.",
        k_train,
    )
    checkpoint = out_dir / "levelset_pinn.pt"
    model = LevelSetPINN(hp.hidden, hp.layers).to(device)
    if checkpoint.exists() and not retrain:
        load_checkpoint(checkpoint, model, device)
        model.eval()
        log.info("loaded %s; training skipped", checkpoint)
        return model, _read_history(out_dir / "history.json")

    if sdf_sampling not in ("thirds", "balanced"):
        raise RuntimeError(f"Unknown sdf_sampling {sdf_sampling!r}. Use thirds or balanced.")
    if sdf_clip_m is not None and sdf_sampling != "balanced":
        raise RuntimeError("sdf_clip_m only applies to balanced SDF sampling.")
    if sdf_clip_m is not None and sdf_clip_m <= 0:
        raise RuntimeError("sdf_clip_m must be a positive distance in meters.")
    if w_eik < 0:
        raise RuntimeError("w_eik must be non-negative.")
    if w_eik > 0 and sdf_sampling != "balanced":
        raise RuntimeError("The front Eikonal loss is only used with balanced SDF sampling.")
    if w_eik > 0 and sdf_clip_m is None:
        raise RuntimeError("The front Eikonal loss keeps the clipped SDF targets. Pass sdf_clip_m.")
    balanced = sdf_sampling == "balanced"
    band_m = near_front_band_m(case.x, case.y) if balanced else None
    set_seed(hp.seed)
    if target_geometries is not None and len(target_geometries) != len(case.perimeters):
        raise RuntimeError(
            f"target_geometries has {len(target_geometries)} entries for {len(case.perimeters)} perimeters."
        )
    if target_geometries is not None:
        log.info(
            "observation targets are the persistent burned-region estimate. "
            "Hotspot hulls remain stored on the case. "
            "evaluate_real scores those hulls. "
            "A comparison that is given these persistent geometries scores the persistent boundaries instead."
        )
    sdfs = build_training_sdfs(
        case, k_train, length, out_dir / "sdf_train.npz",
        None if target_geometries is None else target_geometries[:k_train],
    )
    pools = []
    for index in range(k_train):
        target_geom = (
            case.perimeters.geometry.iloc[index]
            if target_geometries is None else target_geometries[index]
        )
        if balanced:
            pool = _balanced_supervision_pools(
                sdfs[index], case.x, case.y, device, band_m, length
            )
            log.info(
                "perimeter %d balanced pools | interior [%.0f, %.0f] m (%d cells) | "
                "near-front |phi|<=%.0f m [%.0f, %.0f] m (%d cells) | "
                "exterior [%.0f, %.0f] m (%d cells)",
                index,
                pool["range_interior"][0], pool["range_interior"][1], pool["n_interior"],
                band_m, pool["range_near"][0], pool["range_near"][1], pool["n_near"],
                pool["range_exterior"][0], pool["range_exterior"][1], pool["n_exterior"],
            )
        else:
            pool = _supervision_pools(
                target_geom, sdfs[index], case.x, case.y, device
            )
            inside = sdfs[index] < 0
            outside = sdfs[index] > 0
            log.info(
                "perimeter %d signed-distance targets | inside [%.0f, %.0f] m (%d cells) | "
                "boundary 0 | outside [%.0f, %.0f] m (%d cells)",
                index,
                float(sdfs[index][inside].min()) * length,
                float(sdfs[index][inside].max()) * length,
                pool["n_inside"],
                float(sdfs[index][outside].min()) * length,
                float(sdfs[index][outside].max()) * length,
                pool["n_outside"],
            )
        pools.append(pool)
    if balanced:
        n_in, n_near, n_out = _region_counts(hp.n_obs)
        clip_note = (
            f" SDF targets are clipped to ±{sdf_clip_m:.0f} m."
            if sdf_clip_m is not None else ""
        )
        log.info(
            "balanced SDF batches draw %d interior, %d near-front, %d exterior "
            "out of %d observation points (with replacement). "
            "Near-front band is %.0f m, one cell. PDE points stay uniform on the window.%s",
            n_in, n_near, n_out, hp.n_obs, band_m, clip_note,
        )
    else:
        log.info(
            "initial and observation points are split inside the perimeter, on it, and outside it. "
            "The PDE points stay uniform on the window."
        )
    eik_on = w_eik > 0
    eik_train_pools = []
    eik_probe_pools = []
    eik_counts: list[int] = []
    if eik_on:
        if n_eik < k_train:
            raise RuntimeError(f"n_eik={n_eik} cannot cover {k_train} training perimeters.")
        eik_counts = _even_counts(n_eik, k_train)
        for index in range(len(case.perimeters)):
            if index < k_train:
                field = sdfs[index]
            else:
                held_geom = (
                    case.perimeters.geometry.iloc[index]
                    if target_geometries is None else target_geometries[index]
                )
                distance_m = signed_distance(held_geom, case.x, case.y)
                field = distance_m / length
            regions = _eikonal_region_pools(
                field, case.x, case.y, device, band_m, eik_band_m, length
            )
            eik_probe_pools.append(regions)
            if index < k_train:
                eik_train_pools.append(regions)
                n_near_draw = eik_counts[index] - eik_counts[index] // 2
                n_shell_draw = eik_counts[index] // 2
                log.info(
                    "perimeter %d eikonal pools | near-front |phi|<=%.0f m (%d cells, %d draws) | "
                    "shell %.0f m < |phi| <= %.0f m (%d cells, %d draws)",
                    index, band_m, regions["near"]["n"], n_near_draw,
                    band_m, eik_band_m, regions["shell"]["n"], n_shell_draw,
                )
        log.info(
            "Eikonal loss is mean((|grad_XY Phi| - 1)^2), weight %.3g, on %d points per epoch. "
            "Points are split evenly across the %d training times. On each time, half the draws "
            "(plus one when the count is odd) come from the %.0f m supervision ring and half from "
            "the rest of the ±%.0f m neighborhood of that observed front. "
            "Deep interior and far exterior are not sampled. "
            "Validation and test neighborhoods are measured, not trained. "
            "The PDE residual, balanced sampler, and ±%.0f m SDF clip are unchanged.",
            w_eik, n_eik, k_train, band_m, eik_band_m, sdf_clip_m,
        )
    spread = torch.tensor(np.asarray(case.spread, dtype=np.float32), device=device)
    wind = torch.tensor(downwind_unit_vectors(case.wind_from_deg), device=device)
    if spread.ndim != 3:
        raise RuntimeError(f"spread rate must have shape (day, y, x); got {tuple(spread.shape)}")
    if wind.shape[0] != spread.shape[0]:
        raise RuntimeError(
            f"wind has {wind.shape[0]} days but the spread rate has {spread.shape[0]}"
        )
    n_days = int(spread.shape[0])
    t_over_l = duration / length
    optimizer = torch.optim.Adam(model.parameters(), lr=hp.lr)
    history = {"total": [], "pde": [], "ic": [], "obs": []}
    if balanced:
        history["obs_interior"] = []
        history["obs_near"] = []
        history["obs_exterior"] = []
    if eik_on:
        history["eikonal"] = []
    sign_at = set(sign_epochs or ())
    sign_snapshots: list[dict] = []

    def _batch_losses(tau_value: float, pool: dict, sdf, count: int):
        if balanced:
            loss_in, loss_near, loss_out = _balanced_perimeter_batch(
                model, tau_value, pool, sdf, case, count, center_x, center_y, length,
                clip_m=sdf_clip_m,
            )
            return (loss_in + loss_near + loss_out) / 3, loss_in, loss_near, loss_out
        loss = _perimeter_batch(
            model, tau_value, pool, sdf, case, count, center_x, center_y, length
        )
        return loss, None, None, None

    def _record_sign(epoch: int) -> None:
        sign_rows = phi_sign_diagnostics(model, case, time_split, track_negative_set=eik_on)
        if eik_on:
            probe = torch.Generator(device="cpu")
            probe.manual_seed(10_000 + int(epoch))
            model.eval()
            for row, regions in zip(sign_rows, eik_probe_pools):
                row["median_grad_2km"] = _median_front_gradient(
                    model, float(tau[row["index"]]), regions["band"],
                    center_x, center_y, length, 1024, probe,
                )
                row["median_grad_near"] = _median_front_gradient(
                    model, float(tau[row["index"]]), regions["near"],
                    center_x, center_y, length, 1024, probe,
                )
            grad_note = " ".join(
                f"k{row['index']}={row['median_grad_2km']:.2f}/{row['median_grad_near']:.2f}"
                for row in sign_rows
            )
            log.info("epoch %4d median |grad| in ±%.0f m band / %.0f m ring | %s", epoch, eik_band_m, band_m, grad_note)
            temporal = _temporal_negative_summary(sign_rows)
            log.info(
                "epoch %4d negative set | approximately nondecreasing=%s | "
                "steps losing >=5 km2=%d | max one-step loss %.1f km2 | large region disappears=%s",
                epoch, temporal["approximately_nondecreasing"],
                temporal["steps_losing_at_least_5km2"], temporal["max_lost_km2_one_step"],
                temporal["large_region_disappears"],
            )
        else:
            temporal = None
        summary = _contour_summary(sign_rows)
        parts = [
            f"{label} {item['contours']}/{item['n']}"
            for label, item in summary.items()
        ]
        log.info("epoch %4d contours | %s", epoch, " | ".join(parts))
        if _phi_is_degenerate(sign_rows):
            log.warning(
                "DEGENERATE at epoch %d: phi is positive at every grid cell on every overpass. "
                "A zero contour cannot exist until training drives phi below zero.",
                epoch,
            )
        snapshot = {"epoch": epoch, "summary": summary, "overpasses": sign_rows}
        if temporal is not None:
            snapshot["temporal"] = temporal
        sign_snapshots.append(snapshot)
        model.train()

    model.train()
    if 0 in sign_at:
        _record_sign(0)
    started = None
    import time
    for epoch in range(1, hp.epochs + 1):
        if epoch == 1:
            started = time.perf_counter()
        loss_ic, _ic_in, _ic_near, _ic_out = _batch_losses(0.0, pools[0], sdfs[0], hp.n_ic)

        x_pde, y_pde = _sample_box(hp.n_pde, case.x, case.y, device)
        tau_pde = torch.rand(hp.n_pde, device=device).requires_grad_(True)
        x_norm = ((x_pde - center_x) / length).detach().requires_grad_(True)
        y_norm = ((y_pde - center_y) / length).detach().requires_grad_(True)
        phi = model(tau_pde, x_norm, y_norm)
        phi_t, phi_x, phi_y = torch.autograd.grad(
            phi, (tau_pde, x_norm, y_norm), torch.ones_like(phi), create_graph=True
        )
        days = day_index_from_tau(tau_pde, t0_offset, duration, n_days)
        rate = sample_volume(spread, days, case.x, case.y, x_pde, y_pde).detach()
        wind_now = wind[days]
        residual = directed_residual(
            phi_t, phi_x, phi_y, rate,
            wind_now[:, 0], wind_now[:, 1], hp.c_wdir, t_over_l,
        )
        loss_pde = torch.mean(residual ** 2)

        if k_train == 1:
            loss_obs = torch.zeros((), device=device)
            loss_obs_in = loss_obs_near = loss_obs_out = loss_obs
        else:
            combined = []
            part_in = []
            part_near = []
            part_out = []
            for index in range(1, k_train):
                loss_i, loss_in, loss_near, loss_out = _batch_losses(
                    float(tau[index]), pools[index], sdfs[index], hp.n_obs
                )
                combined.append(loss_i)
                if loss_in is not None:
                    part_in.append(loss_in)
                    part_near.append(loss_near)
                    part_out.append(loss_out)
            loss_obs = torch.stack(combined).mean()
            if part_in:
                loss_obs_in = torch.stack(part_in).mean()
                loss_obs_near = torch.stack(part_near).mean()
                loss_obs_out = torch.stack(part_out).mean()
            else:
                loss_obs_in = loss_obs_near = loss_obs_out = None

        if eik_on:
            loss_eik = _eikonal_batch(
                model, eik_train_pools, tau[:k_train], eik_counts, center_x, center_y, length
            )
        else:
            loss_eik = None
        loss = hp.w_ic * loss_ic + hp.w_pde * loss_pde + hp.w_obs * loss_obs
        if loss_eik is not None:
            loss = loss + w_eik * loss_eik
        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Loss became non-finite at epoch {epoch}. "
                "Check the spread-rate field for non-finite values."
            )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        history["total"].append(_scalar(loss))
        history["pde"].append(_scalar(loss_pde))
        history["ic"].append(_scalar(loss_ic))
        history["obs"].append(_scalar(loss_obs))
        if balanced and loss_obs_in is not None:
            history["obs_interior"].append(_scalar(loss_obs_in))
            history["obs_near"].append(_scalar(loss_obs_near))
            history["obs_exterior"].append(_scalar(loss_obs_out))
        if loss_eik is not None:
            history["eikonal"].append(_scalar(loss_eik))
        if epoch == 1 and started is not None:
            elapsed = time.perf_counter() - started
            remaining = elapsed * hp.epochs
            if remaining < 90:
                estimate = f"about {remaining:.0f} s"
            else:
                estimate = f"about {remaining / 60.0:.0f} min"
            log.info(
                "first epoch took %.2f s; %d epochs is %s on this machine",
                elapsed, hp.epochs, estimate,
            )
        if epoch % hp.log_every == 0 or epoch == hp.epochs or epoch in sign_at:
            if balanced and loss_obs_in is not None and loss_eik is not None:
                log.info(
                    "epoch %4d | total %.3e | pde %.3e | eikonal %.3e | obs %.3e | "
                    "interior %.3e | near %.3e | exterior %.3e",
                    epoch, _scalar(loss), _scalar(loss_pde), _scalar(loss_eik), _scalar(loss_obs),
                    _scalar(loss_obs_in), _scalar(loss_obs_near), _scalar(loss_obs_out),
                )
            elif balanced and loss_obs_in is not None:
                log.info(
                    "epoch %4d | total %.3e | pde %.3e | obs %.3e | "
                    "interior %.3e | near %.3e | exterior %.3e",
                    epoch, _scalar(loss), _scalar(loss_pde), _scalar(loss_obs),
                    _scalar(loss_obs_in), _scalar(loss_obs_near), _scalar(loss_obs_out),
                )
            else:
                log.info(
                    "epoch %4d | total %.3e | ic %.3e | pde %.3e | obs %.3e",
                    epoch, _scalar(loss), _scalar(loss_ic), _scalar(loss_pde), _scalar(loss_obs),
                )
            # log_every is 500. Sign checks stay on that cadence, inside the 250–500 range,
            # unless the caller names epochs explicitly.
            scheduled = hp.log_every >= 250 and (epoch % hp.log_every == 0 or epoch == hp.epochs)
            if epoch in sign_at or scheduled:
                _record_sign(epoch)

    config = asdict(hp)
    config["k_train_resolved"] = k_train
    config["T_days"] = duration / 86400.0
    config["L_m"] = length
    config["sdf_sampling"] = sdf_sampling
    if balanced:
        config["near_front_band_m"] = band_m
        config["sdf_clip_m"] = sdf_clip_m
    config["w_eik"] = w_eik
    config["observation_target"] = "hotspot_hull" if target_geometries is None else "persistent_burn"
    if eik_on:
        config["eik_band_m"] = eik_band_m
        config["n_eik"] = n_eik
    save_checkpoint(checkpoint, model, config)
    _write_history(out_dir / "history.json", history)
    (out_dir / "split.json").write_text(
        json.dumps(time_split.stamp_groups(case), indent=2), encoding="utf-8"
    )
    log.info("wrote %s", checkpoint)
    if sign_snapshots:
        (out_dir / "phi_epochs.json").write_text(
            json.dumps(sign_snapshots, indent=2), encoding="utf-8"
        )
        log.info("wrote %s", out_dir / "phi_epochs.json")
    if sign_snapshots and sign_snapshots[-1]["epoch"] == hp.epochs:
        sign_rows = sign_snapshots[-1]["overpasses"]
    else:
        model.eval()
        sign_rows = phi_sign_diagnostics(model, case, time_split)
    degenerate = _phi_is_degenerate(sign_rows)
    if degenerate:
        log.warning(
            "DEGENERATE: phi is positive at every grid cell on every overpass. "
            "Predictions were not shifted."
        )
    sign_payload = {"degenerate": degenerate, "scheme": time_split.scheme, "overpasses": sign_rows}
    if sign_snapshots and "temporal" in sign_snapshots[-1] and sign_snapshots[-1]["epoch"] == hp.epochs:
        sign_payload["temporal"] = sign_snapshots[-1]["temporal"]
    elif eik_on and sign_rows and "negative_km2" in sign_rows[0]:
        sign_payload["temporal"] = _temporal_negative_summary(sign_rows)
    (out_dir / "phi_sign.json").write_text(
        json.dumps(sign_payload, indent=2),
        encoding="utf-8",
    )
    log.info("wrote %s", out_dir / "phi_sign.json")
    return model, history


def _phi_is_degenerate(rows: list[dict]) -> bool:
    return bool(rows) and all(row["min_phi_m"] >= 0.0 for row in rows)


def phi_sign_diagnostics(model, case: FireCase, time_split: TimeSplit, track_negative_set: bool = False) -> list[dict]:
    """min, max, negative fraction, and whether a zero contour exists at every overpass.

    Read from the network output. Nothing is added to phi. When
    ``track_negative_set`` is set, each row also records the negative area and
    how many square kilometers of the previous negative set became non-negative.
    """
    _cx, _cy, _length, _duration, _offset, tau = domain_scales(case)
    model.eval()
    rows = []
    previous = None
    cell_km2 = 0.0
    if track_negative_set:
        side = cell_size_m(case.x, case.y)
        cell_km2 = (side * side) / 1e6
    log.info("phi sign (index, split, min m, max m, fraction<0, zero contour):")
    for index in range(len(case.perimeters)):
        phi_m = predict_phi_meters(model, case, float(tau[index]))
        has_contour = bool(zero_contours(phi_m, case.x, case.y))
        split = time_split.label(index)
        negative = phi_m < 0
        row = {
            "index": index,
            "time": _stamp(case.perimeters["dt"].iloc[index]),
            "split": split,
            "min_phi_m": float(phi_m.min()),
            "max_phi_m": float(phi_m.max()),
            "fraction_negative": float(negative.mean()),
            "zero_contour": has_contour,
        }
        if track_negative_set:
            row["negative_km2"] = float(negative.sum()) * cell_km2
            if previous is None:
                row["lost_km2"] = 0.0
                row["gained_km2"] = 0.0
            else:
                row["lost_km2"] = float(np.logical_and(previous, ~negative).sum()) * cell_km2
                row["gained_km2"] = float(np.logical_and(~previous, negative).sum()) * cell_km2
            previous = negative
        rows.append(row)
        log.info(
            "  k=%d %s %s min=%.0f max=%.0f neg=%.4f contour=%s",
            index, split, row["time"], row["min_phi_m"], row["max_phi_m"],
            row["fraction_negative"], has_contour,
        )
    return rows


def predict_phi_meters(model, case: FireCase, tau_value: float) -> np.ndarray:
    center_x, center_y, length, _duration, _offset, _tau = domain_scales(case)
    device = next(model.parameters()).device
    x_norm, y_norm = np.meshgrid(
        (np.asarray(case.x) - center_x) / length,
        (np.asarray(case.y) - center_y) / length,
    )
    x_t = torch.tensor(x_norm.ravel(), dtype=torch.float32, device=device)
    y_t = torch.tensor(y_norm.ravel(), dtype=torch.float32, device=device)
    with torch.no_grad():
        phi = model(torch.full((x_t.shape[0],), float(tau_value), device=device), x_t, y_t)
    return phi.reshape(len(case.y), len(case.x)).cpu().numpy() * length


def evaluate_real(
    model,
    case: FireCase,
    k_train: int,
    figure_path: Path | None = None,
    geojson_path: Path | None = None,
    case_name: str = "",
    split_scheme: str = "k_train",
    train_fraction: float = 0.6,
    val_fraction: float = 0.2,
) -> list[dict]:
    """Mean distance from the predicted zero contour to each held-out hull boundary.

    When ``geojson_path`` is set, also writes every overpass as WGS84 GeoJSON:
    the observed hull and the predicted phi=0 lines. A time with no zero
    crossing is still a feature, with null geometry.
    """
    from wildfire_pinn.plots import plot_heldout_fronts

    time_split = resolve_time_split(
        len(case.perimeters), k_train, split_scheme, train_fraction, val_fraction
    )
    k_train = len(time_split.train)
    _cx, _cy, _length, _duration, _offset, tau = domain_scales(case)
    model.eval()
    rows = []
    panels = []
    predicted: dict[int, list[np.ndarray]] = {}
    scored = time_split.validation + time_split.test
    log.info("scored overpasses (index, split, time, mean front error m, hull area km^2):")
    for index in scored:
        phi_m = predict_phi_meters(model, case, float(tau[index]))
        contours = zero_contours(phi_m, case.x, case.y)
        predicted[index] = contours
        geometry = case.perimeters.geometry.iloc[index]
        boundary = geometry.boundary
        area_km2 = float(geometry.area) / 1e6
        stamp = _stamp(case.perimeters["dt"].iloc[index])
        if not contours:
            error = float("nan")
            used = 0
            log.info("  k=%d %s  no zero contour  hull=%.1f km^2", index, stamp, area_km2)
        else:
            main = max(contours, key=len)
            stride = main[::5] if len(main) > 5 else main
            error = mean_distance_m(stride, boundary)
            used = int(len(main))
            log.info(
                "  k=%d %s  err=%.0f m  hull=%.1f km^2  contour vertices=%d (of %d pieces)",
                index, stamp, error, area_km2, used, len(contours),
            )
        rows.append({
            "index": index,
            "time": stamp,
            "split": time_split.label(index),
            "front_error_m": error,
            "hull_area_km2": area_km2,
            "contour_vertices": used,
        })
        if figure_path is not None and len(panels) < 4:
            panels.append({
                "x": case.x, "y": case.y, "phi": phi_m, "geometry": geometry,
                "title": stamp,
            })
    finite = [row["front_error_m"] for row in rows if np.isfinite(row["front_error_m"])]
    if finite:
        log.info("mean front error on scored perimeters: %.0f m", float(np.mean(finite)))
    else:
        log.info("no scored perimeter produced a zero contour yet")
    if figure_path is not None and panels:
        plot_heldout_fronts(panels, figure_path)
        log.info("wrote %s", figure_path)
    if geojson_path is not None:
        for index in range(k_train):
            phi_m = predict_phi_meters(model, case, float(tau[index]))
            predicted[index] = zero_contours(phi_m, case.x, case.y)
        errors = {row["index"]: row["front_error_m"] for row in rows}
        write_fronts_geojson(
            case, geojson_path, predicted, k_train, errors, case_name, time_split=time_split
        )
    return rows


def write_fronts_geojson(
    case: FireCase,
    path: Path,
    predicted: dict[int, list[np.ndarray]],
    k_train: int,
    errors: dict[int, float] | None = None,
    case_name: str = "",
    time_split: TimeSplit | None = None,
) -> Path:
    """Observed hulls and predicted phi=0 lines, as one WGS84 GeoJSON file.

    Coordinates are longitude, latitude (EPSG:4326), which is what a map and
    the FIRMS perimeter file both use. ``front_error_m`` is set on the longest
    predicted piece at each held-out time, the same piece the score uses, and
    is null everywhere else. An empty prediction is a feature with null
    geometry rather than a dropped time.
    """
    from shapely.geometry import LineString
    from shapely.ops import transform
    from pyproj import Transformer

    if case.perimeters.crs is None:
        raise RuntimeError("Perimeters have no CRS, so the fronts cannot be written as lon/lat.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if time_split is None:
        time_split = resolve_time_split(len(case.perimeters), k_train, "k_train")
    _cx, _cy, _length, _duration, _offset, tau = domain_scales(case)
    to_wgs84 = Transformer.from_crs(case.perimeters.crs, "EPSG:4326", always_xy=True)
    errors = errors or {}

    def project(geom):
        if geom is None or geom.is_empty:
            return None
        return transform(to_wgs84.transform, geom)

    features = []
    for index in range(len(case.perimeters)):
        stamp = _stamp(case.perimeters["dt"].iloc[index])
        split = time_split.label(index)
        observed = project(case.perimeters.geometry.iloc[index])
        observed_vertices = 0
        if observed is not None and observed.geom_type == "Polygon":
            observed_vertices = int(len(observed.exterior.coords))
        features.append(_front_feature(
            observed,
            case=case_name, index=index, time=stamp, tau=float(tau[index]),
            split=split, source="observed", piece=0,
            n_vertices=observed_vertices,
            front_error_m=None, scored=False,
        ))
        pieces = list(predicted.get(index) or [])
        pieces = [piece for piece in pieces if len(piece) >= 2]
        error = _json_number(float(errors[index])) if index in errors else None
        if not pieces:
            features.append(_front_feature(
                None,
                case=case_name, index=index, time=stamp, tau=float(tau[index]),
                split=split, source="predicted", piece=0, n_vertices=0,
                front_error_m=error, scored=index in errors,
            ))
            continue
        scored = max(pieces, key=len)
        for piece_index, vertices in enumerate(pieces):
            line = project(LineString(np.asarray(vertices, dtype=np.float64)))
            is_scored = vertices is scored and index in errors
            features.append(_front_feature(
                line,
                case=case_name, index=index, time=stamp, tau=float(tau[index]),
                split=split, source="predicted", piece=piece_index,
                n_vertices=int(len(vertices)),
                front_error_m=error if is_scored else None,
                scored=is_scored,
            ))
    payload = {"type": "FeatureCollection", "features": features}
    path.write_text(json.dumps(payload), encoding="utf-8")
    log.info("wrote %s (%d features)", path, len(features))
    return path


def _round_coords(obj, places: int = 6):
    if isinstance(obj, (list, tuple)):
        if obj and isinstance(obj[0], (int, float)):
            return [round(float(value), places) for value in obj]
        return [_round_coords(value, places) for value in obj]
    return obj


def _front_feature(geom, **props) -> dict:
    from shapely.geometry import mapping

    if geom is None or geom.is_empty:
        geometry = None
    else:
        geometry = mapping(geom)
        geometry["coordinates"] = _round_coords(geometry["coordinates"])
    return {"type": "Feature", "properties": props, "geometry": geometry}


def _iter_grid_partials(model, case: FireCase, tau_value: float, batch_size: int = 20000):
    """Partial derivatives of normalized phi on the model grid, one batch at a time.

    Same normalization as training: tau, X = (x-xc)/L, Y = (y-yc)/L.
    Batching keeps the autograd graph off the full raster.
    """
    center_x, center_y, length, _duration, _offset, _tau = domain_scales(case)
    device = next(model.parameters()).device
    xx, yy = np.meshgrid(np.asarray(case.x, dtype=np.float64), np.asarray(case.y, dtype=np.float64))
    flat_x = torch.tensor(xx.reshape(-1), dtype=torch.float32, device=device)
    flat_y = torch.tensor(yy.reshape(-1), dtype=torch.float32, device=device)
    for start in range(0, flat_x.shape[0], batch_size):
        stop = start + batch_size
        xm = flat_x[start:stop]
        ym = flat_y[start:stop]
        tau = torch.full((xm.shape[0],), float(tau_value), device=device).requires_grad_(True)
        xn = ((xm - center_x) / length).detach().requires_grad_(True)
        yn = ((ym - center_y) / length).detach().requires_grad_(True)
        phi = model(tau, xn, yn)
        phi_t, phi_x, phi_y = torch.autograd.grad(
            phi, (tau, xn, yn), grad_outputs=torch.ones_like(phi), create_graph=False
        )
        yield phi_t, phi_x, phi_y, xm, ym


def pde_residual_grid(model, case: FireCase, tau_value: float, c_wdir: float) -> np.ndarray:
    """Absolute level-set residual at one tau, shaped (y, x)."""
    _cx, _cy, _length, duration, t0_offset, _tau = domain_scales(case)
    device = next(model.parameters()).device
    spread = torch.tensor(np.asarray(case.spread, dtype=np.float32), device=device)
    wind = torch.tensor(downwind_unit_vectors(case.wind_from_deg), device=device)
    t_over_l = duration / _length
    pieces = []
    for phi_t, phi_x, phi_y, xm, ym in _iter_grid_partials(model, case, tau_value):
        days = day_index_from_tau(
            torch.full(phi_t.shape, float(tau_value), device=device),
            t0_offset, duration, int(spread.shape[0]),
        )
        rate = sample_volume(spread, days, case.x, case.y, xm, ym).detach()
        wind_now = wind[days]
        residual = directed_residual(
            phi_t, phi_x, phi_y, rate,
            wind_now[:, 0], wind_now[:, 1], c_wdir, t_over_l,
        )
        pieces.append(residual.detach().abs().cpu())
    ny, nx = len(case.y), len(case.x)
    return torch.cat(pieces).reshape(ny, nx).numpy()


def eikonal_on_training_times(model, case: FireCase, taus) -> tuple[float, float]:
    """Mean and max of ||grad phi| - 1| on the grid at each training overpass.

    |grad phi| uses the same normalized coordinates as directed_residual, where
    a physical signed distance has unit gradient.
    """
    total = 0.0
    count = 0
    worst = 0.0
    for tau_value in taus:
        for _phi_t, phi_x, phi_y, _xm, _ym in _iter_grid_partials(model, case, float(tau_value)):
            magnitude = torch.sqrt(phi_x ** 2 + phi_y ** 2 + 1e-12)
            defect = (magnitude - 1.0).abs()
            total += float(defect.sum())
            count += int(defect.numel())
            worst = max(worst, float(defect.max()))
    if count == 0:
        return float("nan"), float("nan")
    return total / count, worst


def _json_number(value: float):
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def run_diagnostics(model, case: FireCase, out_dir: Path, k_train: int, c_wdir: float) -> dict:
    """Residual maps at three times, plus an eikonal check on the training grids.

    Does not train. early/late are the first and last training overpasses;
    held-out is the first perimeter the observation loss did not see.
    """
    from wildfire_pinn.plots import plot_scalar_grid

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    k_train = resolve_k_train(len(case.perimeters), k_train)
    _cx, _cy, _length, _duration, _offset, tau = domain_scales(case)
    slots = (
        ("early_train", 0),
        ("late_train", k_train - 1),
        ("held_out", k_train),
    )
    model.eval()
    report = {}
    for name, index in slots:
        tau_value = float(tau[index])
        field = pde_residual_grid(model, case, tau_value, c_wdir)
        finite = field[np.isfinite(field)]
        entry = {
            "tau": tau_value,
            "time": _stamp(case.perimeters["dt"].iloc[index]),
            "mean_abs_residual": _json_number(float(finite.mean()) if finite.size else float("nan")),
            "max_abs_residual": _json_number(float(finite.max()) if finite.size else float("nan")),
        }
        report[name] = entry
        figure = out_dir / f"residual_{name}.png"
        plot_scalar_grid(
            case.x, case.y, field, figure,
            title=f"|PDE residual| {name.replace('_', ' ')} ({entry['time']})",
            colorbar="|residual|",
            cmap="magma",
        )
        log.info(
            "%s tau=%.3f %s |residual| mean=%s max=%s -> %s",
            name, tau_value, entry["time"], entry["mean_abs_residual"], entry["max_abs_residual"], figure,
        )
    train_taus = [float(tau[index]) for index in range(k_train)]
    eik_mean, eik_max = eikonal_on_training_times(model, case, train_taus)
    report["eikonal"] = {
        "mean_abs_grad_minus_one": _json_number(eik_mean),
        "max_abs_grad_minus_one": _json_number(eik_max),
        "sampled_at": "every cell of the model grid, at each training overpass",
    }
    log.info("eikonal ||grad phi|-1| mean=%s max=%s", report["eikonal"]["mean_abs_grad_minus_one"], report["eikonal"]["max_abs_grad_minus_one"])
    path = out_dir / "diagnostics.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log.info("wrote %s", path)
    return report


def write_playback(model, case: FireCase, path: Path) -> None:
    """GIF of every overpass: observed hull in black, predicted zero contour in red."""
    from wildfire_pinn.plots import write_playback_gif

    _cx, _cy, _length, _duration, _offset, tau = domain_scales(case)
    model.eval()
    frames = []
    for index in range(len(case.perimeters)):
        phi_m = predict_phi_meters(model, case, float(tau[index]))
        frames.append({
            "x": case.x,
            "y": case.y,
            "contours": zero_contours(phi_m, case.x, case.y),
            "geometry": case.perimeters.geometry.iloc[index],
            "title": _stamp(case.perimeters["dt"].iloc[index]),
        })
    write_playback_gif(frames, path)
    log.info("wrote %s (%d frames)", path, len(frames))


def load_fire_case(stack_path: Path, rate_path: Path, perimeter_path: Path) -> FireCase:
    import geopandas as gpd

    from wildfire_pinn.data_pipeline import load_dataset

    stack = load_dataset(stack_path)
    rate = load_dataset(rate_path)
    perimeters = gpd.read_file(perimeter_path)
    if perimeters.crs is None:
        raise RuntimeError(
            f"{perimeter_path} has no CRS. Rebuild it; perimeters are saved as EPSG:4326."
        )
    grid_crs = stack.attrs.get("crs")
    if grid_crs is None:
        grid_crs = "EPSG:3310"
    perimeters = perimeters.to_crs(str(grid_crs))
    if "dt" in perimeters.columns:
        perimeters["dt"] = pd.to_datetime(perimeters["dt"])
    else:
        from wildfire_pinn.data_pipeline import overpass_timestamps
        perimeters["dt"] = overpass_timestamps(perimeters["acq_date"], perimeters["acq_time"])
    perimeters = perimeters.sort_values("dt").reset_index(drop=True)

    x = np.asarray(stack["x"].values, dtype=np.float64)
    y = np.asarray(stack["y"].values, dtype=np.float64)
    spread = np.asarray(rate["R"].values, dtype=np.float32)
    if spread.shape[-2:] != (len(y), len(x)):
        raise RuntimeError(
            f"Spread-rate grid {spread.shape} does not match the stack {(len(y), len(x))}. "
            "Rebuild the spread rate from the current stack."
        )
    wind_name = "wind_dir" if "wind_dir" in stack else "wind_from_direction"
    if wind_name not in stack:
        raise RuntimeError("stack is missing wind_dir")
    wind = np.asarray(stack[wind_name].values, dtype=np.float32)
    if wind.shape != spread.shape:
        raise RuntimeError(f"wind shape {wind.shape} does not match spread rate {spread.shape}")
    reference = stack.attrs.get("reference_date") or rate.attrs.get("reference_date") or "2025-01-07"
    return FireCase(
        x=x,
        y=y,
        spread=spread,
        wind_from_deg=wind,
        perimeters=perimeters,
        day0=pd.Timestamp(str(reference)),
    )


def _stamp(value) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d %H:%M")


def _write_history(path: Path, history: dict) -> None:
    path.write_text(json.dumps(history), encoding="utf-8")


def _read_history(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_metrics(path: Path, rows: list[dict]) -> None:
    cleaned = []
    for row in rows:
        item = {}
        for key, value in row.items():
            if isinstance(value, float) and not np.isfinite(value):
                item[key] = None
            else:
                item[key] = value
        cleaned.append(item)
    finite = [row["front_error_m"] for row in rows if np.isfinite(row["front_error_m"])]
    payload = {
        "perimeters": cleaned,
        "mean_front_error_m": float(np.mean(finite)) if finite else None,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
