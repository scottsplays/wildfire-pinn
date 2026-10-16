"""Experiment 2A.3 F1 seed-0 hard-IC PINN. Other checkpoints are not substituted.

Phi = phi0 / L + (t / T) * g_theta, then scaled back to meters.
Coordinates match the training normalization: X = (x - 0) / L, tau = t / T.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

from experiments.wildfirepinn_v2_diagnostics.circle import CircleProblem
from experiments.wildfirepinn_v2_diagnostics.experiment2.physics import circle_phi0, hard_ic
from experiments.wildfirepinn_v2_diagnostics.experiment2.stabilization.banks import bank_f1
from experiments.wildfirepinn_v2_diagnostics.experiment2.stabilization.net import BankFourierPINN
from wildfire_pinn.pinn import load_checkpoint

REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = (
    REPO_ROOT
    / "experiments"
    / "wildfirepinn_v2_diagnostics"
    / "experiment2"
    / "stabilization"
    / "runs"
    / "F1_seed0"
    / "levelset_pinn.pt"
)
EXPECTED_ARCH = {"hidden": 64, "layers": 4, "fourier_frequencies": 2, "variant": "F1"}
VERSION = "experiment2a3-f1-seed0"

_MODEL = None
_STATUS: dict | None = None


def checkpoint_sha256() -> str | None:
    if not CHECKPOINT.is_file():
        return None
    digest = hashlib.sha256()
    with CHECKPOINT.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def supported_problem() -> CircleProblem:
    """The exact circle used to train F1. Grid n=201 matches that experiment."""
    return CircleProblem(n=201, x0=200.0, y0=-150.0, r0=400.0, t_final=4000.0, half_extent=2000.0)


def status() -> dict:
    global _STATUS
    if _STATUS is not None:
        return dict(_STATUS)
    if not CHECKPOINT.is_file():
        _STATUS = {
            "available": False,
            "reason": f"checkpoint missing: {CHECKPOINT}",
            "sha256": None,
            "version": VERSION,
            "arch": EXPECTED_ARCH,
        }
        return dict(_STATUS)
    try:
        _load()
    except Exception as exc:
        _STATUS = {
            "available": False,
            "reason": f"checkpoint incompatible: {exc}",
            "sha256": checkpoint_sha256(),
            "version": VERSION,
            "arch": EXPECTED_ARCH,
        }
        return dict(_STATUS)
    _STATUS = {
        "available": True,
        "reason": None,
        "sha256": checkpoint_sha256(),
        "version": VERSION,
        "arch": EXPECTED_ARCH,
        "checkpoint": str(CHECKPOINT),
    }
    return dict(_STATUS)


def _load() -> BankFourierPINN:
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    problem = supported_problem()
    model = BankFourierPINN(bank_f1(problem.length), "F1", 64, 4)
    if model.arch != EXPECTED_ARCH:
        raise RuntimeError(f"constructed arch {model.arch} does not match {EXPECTED_ARCH}")
    load_checkpoint(CHECKPOINT, model, torch.device("cpu"))
    model.eval()
    _MODEL = model
    return model


def predict_phi_meters(t: float, x: np.ndarray | None = None, y: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """phi in meters on the training grid, unless x and y are the training axes."""
    info = status()
    if not info["available"]:
        raise RuntimeError(info["reason"])
    problem = supported_problem()
    if x is None or y is None:
        x, y = problem.grid()
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if not np.allclose(x, problem.grid()[0]) or not np.allclose(y, problem.grid()[1]):
        raise ValueError("direct_pinn_experimental evaluates only the training grid")
    if t < 0.0 or t > problem.t_final:
        raise ValueError("forecast time must lie in [0, 4000] seconds for this checkpoint")
    model = _load()
    yy, xx = np.meshgrid(y, x, indexing="ij")
    device = next(model.parameters()).device
    tau = torch.tensor(np.full(xx.shape, float(t) / problem.t_final).ravel(), dtype=torch.float32, device=device)
    x_norm = torch.tensor(((xx - problem.center_x) / problem.length).ravel(), dtype=torch.float32, device=device)
    y_norm = torch.tensor(((yy - problem.center_y) / problem.length).ravel(), dtype=torch.float32, device=device)
    phi0_fn = lambda xn, yn, problem=problem: circle_phi0(xn, yn, problem)
    model.eval()
    with torch.no_grad():
        raw = model(tau, x_norm, y_norm)
        phi = hard_ic(raw, tau, phi0_fn(x_norm, y_norm))
    field = phi.detach().cpu().numpy().reshape(xx.shape) * problem.length
    return field, x, y
