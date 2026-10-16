"""Frozen WildfirePINN v1 inference.

The network, normalization, and sign convention are the ones in
``infer_wildfirepinn_v1.py``: ``LevelSetPINN`` built from the checkpoint
architecture, ``load_checkpoint``, and ``predict_phi_meters``. φ is in
meters. φ < 0 is predicted burned. Nothing in this module trains.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from wildfire_pinn.config import DATASETS, ProjectConfig
from wildfire_pinn.geometry import cell_size_m
from wildfire_pinn.pinn import (
    LevelSetPINN,
    domain_scales,
    load_checkpoint,
    load_fire_case,
    predict_phi_meters,
)

log = logging.getLogger("wildfirepinn.serving")

MODEL_VERSION = "wildfirepinn-v1"
EXPECTED_SHA256 = "93de4791f6017664174b6dec0bc9434bd2bbe5fffd52dc63d795168f4890e996"
EXPERIMENTAL_WARNING = (
    "WildfirePINN v1 is an observation-only experimental level-set baseline "
    "(w_pde=0). It is not an operational wildfire forecast. Untouched Kincade "
    "test IoU is about 0.13. Predicted burned cells are phi < 0 on the fixed "
    "90 m grid. Do not use this response for incident decisions."
)


class RequestError(Exception):
    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_number(value):
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


@dataclass
class Prediction:
    timestamp: pd.Timestamp
    tau: float
    phi: np.ndarray

    @property
    def burned(self) -> np.ndarray:
        return self.phi < 0


class FrozenV1:
    """One loaded checkpoint. Safe to call from the request thread pool under ``lock``."""

    def __init__(self, root: Path, model, case, saved: dict, sha256: str):
        self.root = root
        self.model = model
        self.case = case
        self.saved = saved
        self.sha256 = sha256
        self.device = torch.device("cpu")
        self.lock = threading.Lock()
        center_x, center_y, length, duration, _offset, pass_tau = domain_scales(case)
        times = pd.to_datetime(case.perimeters["dt"])
        self.center_x = float(center_x)
        self.center_y = float(center_y)
        self.length_m = float(length)
        self.duration_s = float(duration)
        self.pass_tau = np.asarray(pass_tau, dtype=np.float64)
        self.window_start = pd.Timestamp(times.iloc[0])
        self.window_end = pd.Timestamp(times.iloc[-1])
        self.x = np.asarray(case.x, dtype=np.float64)
        self.y = np.asarray(case.y, dtype=np.float64)
        self.cell_m = float(cell_size_m(self.x, self.y))
        code = case.perimeters.crs.to_epsg() if case.perimeters.crs is not None else None
        self.crs = f"EPSG:{code}" if code else str(case.perimeters.crs)
        self._cached_key: str | None = None
        self._cached: Prediction | None = None
        if next(model.parameters()).device.type != "cpu":
            raise RuntimeError("WildfirePINN v1 inference is CPU-only.")
        if file_sha256(self.checkpoint_path) != sha256:
            raise RuntimeError("Checkpoint hash changed while the model was loading.")

    @property
    def checkpoint_path(self) -> Path:
        return (
            self.root
            / "outputs"
            / "kincade_final_candidates"
            / "observation"
            / "levelset_pinn.pt"
        )

    @classmethod
    def load(cls, root: Path) -> "FrozenV1":
        root = Path(root)
        checkpoint = (
            root / "outputs" / "kincade_final_candidates" / "observation" / "levelset_pinn.pt"
        )
        if not checkpoint.is_file():
            raise RuntimeError(f"Missing frozen checkpoint {checkpoint}")
        sha256 = file_sha256(checkpoint)
        if sha256 != EXPECTED_SHA256:
            raise RuntimeError(
                "Checkpoint SHA-256 does not match frozen WildfirePINN v1. "
                f"Expected {EXPECTED_SHA256}."
            )
        spec = DATASETS["kincade"]
        cfg = ProjectConfig(
            root=root,
            name=spec.name,
            bbox=spec.bbox,
            date_start=spec.date_start,
            date_end=spec.date_end,
            split_scheme=spec.split_scheme,
            train_fraction=spec.train_fraction,
            val_fraction=spec.val_fraction,
        )
        case = load_fire_case(cfg.stack_nc, cfg.rate_nc, cfg.perimeters_path)
        device = torch.device("cpu")
        try:
            blob = torch.load(checkpoint, map_location=device, weights_only=False)
        except TypeError:
            blob = torch.load(checkpoint, map_location=device)
        arch = blob.get("arch") or {}
        hidden = int(arch["hidden"])
        layers = int(arch["layers"])
        if hidden != 128 or layers != 4:
            raise RuntimeError(f"Saved architecture is {arch}, expected hidden 128 and 4 layers.")
        model = LevelSetPINN(hidden, layers).to(device)
        saved = load_checkpoint(checkpoint, model, device)
        model.eval()
        if float(saved.get("w_pde", -1)) != 0.0:
            raise RuntimeError("Refusing to serve a checkpoint whose w_pde is not 0.")
        if saved.get("observation_target") != "persistent_burn":
            raise RuntimeError("Refusing to serve a checkpoint that is not observation_target=persistent_burn.")
        log.info(
            "loaded %s sha256=%s device=cpu parameters=%d window=%s..%s",
            MODEL_VERSION,
            sha256,
            sum(parameter.numel() for parameter in model.parameters()),
            pd.Timestamp(case.perimeters["dt"].iloc[0]).isoformat(),
            pd.Timestamp(case.perimeters["dt"].iloc[-1]).isoformat(),
        )
        return cls(root, model, case, saved, sha256)

    def info(self) -> dict:
        return {
            "model_version": MODEL_VERSION,
            "model_sha256": self.sha256,
            "experimental": True,
            "warning": EXPERIMENTAL_WARNING,
            "training_enabled": False,
            "device": "cpu",
            "w_pde": _json_number(self.saved.get("w_pde")),
            "observation_target": self.saved.get("observation_target"),
            "architecture": {"hidden": 128, "layers": 4},
            "parameter_count": int(sum(parameter.numel() for parameter in self.model.parameters())),
            "crs": self.crs,
            "normalization": {
                "tau": "(t - t0) / T",
                "X": "(x - xc) / L",
                "Y": "(y - yc) / L",
                "phi_m": "Phi * L",
                "L_m": self.length_m,
                "T_s": self.duration_s,
                "center_x": self.center_x,
                "center_y": self.center_y,
            },
            "grid": self.grid_metadata(),
            "time_window": {
                "start": self._stamp(self.window_start),
                "end": self._stamp(self.window_end),
                "timezone": None,
                "note": "Send clock times in this window with no timezone suffix.",
            },
        }

    def grid_metadata(self) -> dict:
        return {
            "crs": self.crs,
            "height": int(len(self.y)),
            "width": int(len(self.x)),
            "cell_size_m": self.cell_m,
            "x_first": float(self.x[0]),
            "x_last": float(self.x[-1]),
            "y_first": float(self.y[0]),
            "y_last": float(self.y[-1]),
            "indexing": "row follows y, column follows x, cell centers",
        }

    def parse_timestamp(self, value: str) -> pd.Timestamp:
        text = value.strip()
        if not text or len(text) > 40:
            raise RequestError(422, "timestamp must be a date-time in the Kincade model window.")
        if text.endswith("Z") or text.endswith("UTC") or "+" in text[10:]:
            raise RequestError(
                422,
                "Send the timestamp without a timezone. "
                f"The model window is {self._stamp(self.window_start)} through {self._stamp(self.window_end)}.",
            )
        try:
            stamp = pd.Timestamp(text)
        except (ValueError, TypeError) as exc:
            raise RequestError(422, "timestamp is not a valid date-time.") from exc
        if stamp.tzinfo is not None:
            raise RequestError(
                422,
                "Send the timestamp without a timezone. "
                f"The model window is {self._stamp(self.window_start)} through {self._stamp(self.window_end)}.",
            )
        if stamp < self.window_start or stamp > self.window_end:
            raise RequestError(
                422,
                "timestamp is outside the Kincade model window "
                f"{self._stamp(self.window_start)} through {self._stamp(self.window_end)}.",
            )
        return stamp

    def predict(self, value: str) -> Prediction:
        stamp = self.parse_timestamp(value)
        key = self._stamp(stamp)
        with self.lock:
            if self._cached_key == key and self._cached is not None:
                return self._cached
            tau = (stamp - self.window_start).total_seconds() / self.duration_s
            phi = predict_phi_meters(self.model, self.case, float(tau))
            prediction = Prediction(timestamp=stamp, tau=float(tau), phi=np.asarray(phi))
            self._cached_key = key
            self._cached = prediction
            burned_cells = int(np.count_nonzero(prediction.burned))
            area = burned_cells * (self.cell_m ** 2) / 1e6
            log.info(
                "prediction timestamp=%s tau=%.6f burned_km2=%.3f cells=%d",
                key,
                prediction.tau,
                area,
                burned_cells,
            )
            return prediction

    def metadata(self, prediction: Prediction) -> dict:
        burned_cells = int(np.count_nonzero(prediction.burned))
        total = int(prediction.phi.size)
        return {
            "model_version": MODEL_VERSION,
            "model_sha256": self.sha256,
            "experimental": True,
            "warning": EXPERIMENTAL_WARNING,
            "crs": self.crs,
            "timestamp": self._stamp(prediction.timestamp),
            "tau": prediction.tau,
            "grid": self.grid_metadata(),
            "burned_cells": burned_cells,
            "unburned_cells": total - burned_cells,
            "burned_area_km2": burned_cells * (self.cell_m ** 2) / 1e6,
            "phi_min_m": float(np.min(prediction.phi)),
            "phi_max_m": float(np.max(prediction.phi)),
            "sign_convention": "phi < 0 is predicted burned; phi = 0 is the perimeter; phi > 0 is outside",
            "mask": {
                "method": "POST",
                "path": "/predict/mask",
                "format": "npz",
                "body": {"timestamp": self._stamp(prediction.timestamp)},
                "arrays": {
                    "burned": "uint8, 1 where phi < 0",
                    "x": "cell-center easting in CRS units",
                    "y": "cell-center northing in CRS units",
                },
            },
        }

    def mask_bytes(self, prediction: Prediction) -> bytes:
        payload = self.metadata(prediction)
        buffer = io.BytesIO()
        np.savez_compressed(
            buffer,
            burned=np.asarray(prediction.burned, dtype=np.uint8),
            x=self.x,
            y=self.y,
            metadata=np.asarray(json.dumps(payload)),
        )
        return buffer.getvalue()

    def mask_filename(self, prediction: Prediction) -> str:
        stamp = self._stamp(prediction.timestamp).replace("-", "").replace(":", "")
        return f"wildfirepinn-v1-burned-{stamp}.npz"

    @staticmethod
    def _stamp(value: pd.Timestamp) -> str:
        stamp = pd.Timestamp(value)
        if stamp.nanosecond or stamp.microsecond:
            return stamp.isoformat()
        return stamp.strftime("%Y-%m-%dT%H:%M:%S")
