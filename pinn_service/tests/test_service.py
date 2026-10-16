"""Contract and physics tests for the WildfirePINN inference service.

    python -m unittest pinn_service.tests.test_service -v      (from the repo root)

Uses the synthetic feature files from make_test_features.py (created if missing).
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path

import numpy as np

from pinn_service import physics
from pinn_service.forecaster import _signed_distance
from pinn_service.make_test_features import main as make_test_features
from pinn_service.serve import app, handle
from pinn_service.validate_contract import validate

HERE = Path(__file__).resolve().parent.parent
BBOX = [-122.9, 36.9, -121.5, 38.3]
FIRE_DAY = "2026-10-16"
QUIET_DAY = "2026-10-17"


def request(**overrides) -> dict:
    body = {"bbox": BBOX, "time_horizon": 6, "model_type": "pinn", "date": FIRE_DAY}
    body.update(overrides)
    return body


def setUpModule():
    os.environ.setdefault("FEATURE_DIR", str(HERE / "test_features"))
    if not (HERE / "test_features" / f"{FIRE_DAY}.npz").exists():
        make_test_features()


class ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.status, cls.fc = handle(request())

    def test_ok_and_valid(self):
        self.assertEqual(self.status, 200)
        errors, _ = validate(self.fc, BBOX, {0, 1, 3, 6})
        self.assertEqual(errors, [])

    def test_all_horizons_present(self):
        horizons = {f["properties"]["horizon_hours"] for f in self.fc["features"]}
        self.assertEqual(horizons, {0, 1, 3, 6})

    def test_metadata(self):
        meta = self.fc["metadata"]
        self.assertEqual(meta["source_model"], "pinn")
        self.assertTrue(meta["timestamp"].endswith("Z"))
        self.assertEqual(meta["total_cells"], len(self.fc["features"]))
        self.assertEqual(meta["fires_detected"], 1)

    def test_burned_area_never_shrinks(self):
        """Every cell burning at an earlier horizon burns later, with probability at least as high."""
        by_h: dict[int, dict[int, float]] = {}
        for f in self.fc["features"]:
            p = f["properties"]
            by_h.setdefault(p["horizon_hours"], {})[p["grid_id"]] = p["fire_probability"]
        for early, late in ((0, 1), (1, 3), (3, 6)):
            for gid, prob in by_h[early].items():
                if early == 0:
                    self.assertIn(gid, by_h[late])
                else:
                    self.assertGreaterEqual(by_h[late].get(gid, 0.0) + 1e-9, prob)

    def test_spreads_downwind(self):
        """Synthetic wind blows toward the SE, so the 6 h centroid sits SE of the 0 h centroid."""
        def centroid(h):
            pts = [np.mean(f["geometry"]["coordinates"][0][:4], axis=0)
                   for f in self.fc["features"] if f["properties"]["horizon_hours"] == h]
            return np.mean(pts, axis=0)
        c0, c6 = centroid(0), centroid(6)
        self.assertGreater(c6[0], c0[0])   # east
        self.assertLess(c6[1], c0[1])      # south

    def test_deterministic(self):
        _, again = handle(request())
        strip = lambda fc: [(f["properties"]["grid_id"], f["properties"]["horizon_hours"], f["properties"]["fire_probability"]) for f in fc["features"]]  # noqa: E731
        self.assertEqual(strip(self.fc), strip(again))


class RequestHandlingTests(unittest.TestCase):
    def test_vertex_wrapper(self):
        status, body = handle({"instances": [request(time_horizon=1)]})
        self.assertEqual(status, 200)
        self.assertEqual(len(body["predictions"]), 1)
        self.assertEqual({f["properties"]["horizon_hours"] for f in body["predictions"][0]["features"]}, {0, 1})

    def test_quiet_day_is_empty_not_an_error(self):
        status, fc = handle(request(date=QUIET_DAY))
        self.assertEqual(status, 200)
        self.assertEqual(fc["features"], [])
        self.assertEqual(fc["metadata"]["fires_detected"], 0)

    def test_horizon_subset(self):
        _, fc = handle(request(time_horizon=3))
        self.assertEqual({f["properties"]["horizon_hours"] for f in fc["features"]}, {0, 1, 3})

    def test_confidence_threshold(self):
        _, fc = handle(request(additional_params={"confidence_threshold": 0.5}))
        self.assertTrue(fc["features"])
        self.assertTrue(all(f["properties"]["fire_probability"] >= 0.5 for f in fc["features"]))

    def test_small_bbox_clamps_geometry(self):
        small = [-122.24, 37.15, -122.18, 37.20]
        status, fc = handle(request(bbox=small))
        self.assertEqual(status, 200)
        errors, _ = validate(fc, small)
        self.assertEqual(errors, [])

    def test_rejections(self):
        cases = [
            (request(model_type="unet"), 422),
            (request(bbox=[-121.5, 36.9, -122.9, 38.3]), 422),
            (request(time_horizon=12), 422),
            (request(date="yesterday"), 422),
            (request(date="1999-01-01"), 404),
            ({"instances": []}, 400),
            ([1, 2, 3], 400),
        ]
        for body, expected in cases:
            with self.subTest(body=str(body)[:60]):
                status, payload = handle(body)
                self.assertEqual(status, expected)
                self.assertIn("detail", payload)


class HttpTests(unittest.TestCase):
    def setUp(self):
        try:
            from fastapi.testclient import TestClient
        except Exception as exc:  # httpx missing
            self.skipTest(f"TestClient unavailable: {exc}")
        self.client = TestClient(app)

    def test_health(self):
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["status"], "ok")
        self.assertTrue(r.json()["models_loaded"])

    def test_predict_over_http(self):
        r = self.client.post("/predict", json=request(time_horizon=1))
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["type"], "FeatureCollection")
        bad = self.client.post("/predict", content=b"not json", headers={"content-type": "application/json"})
        self.assertEqual(bad.status_code, 400)


class PhysicsTests(unittest.TestCase):
    def test_reference_rate(self):
        """Grass (101), flat, reference wind, bone-dry fuel -> exactly C_REF."""
        rate, unknown = physics.spread_rate([[101]], [[0.0]], [[physics.U_REF]], [[0.0]])
        self.assertAlmostEqual(float(rate[0, 0]), physics.C_REF, places=9)
        self.assertEqual(unknown, [])

    def test_nonburnable_and_unknown(self):
        rate, unknown = physics.spread_rate([[98, 999]], [[0.0, 0.0]], [[5.0, 5.0]], [[5.0, 5.0]])
        self.assertTrue(np.all(rate == 0))
        self.assertEqual(unknown, [999])

    def test_expanding_circle_speed(self):
        """Uniform R, no wind: the front moves R*t (within one cell)."""
        n, h, r_m_s, t = 161, 30.0, 0.05, 3600.0
        yy, xx = np.mgrid[0:n, 0:n]
        burned = np.hypot(xx - 80, yy - 80) * h <= 300.0
        phi = physics.propagate(_signed_distance(burned, h), h, np.full((n, n), r_m_s), None, [t])[t]
        radius = np.sqrt((phi < 0).sum() * h * h / np.pi)
        self.assertAlmostEqual(radius, 300.0 + r_m_s * t, delta=h)

    def test_heterogeneous_rate_stable(self):
        """Patchy R must not blow up at the window edge (regression for the extrapolated-ghost bug)."""
        rng = np.random.default_rng(0)
        n, h = 200, 90.0
        burned = np.zeros((n, n), bool)
        burned[90:110, 90:110] = True
        rate = np.where(rng.random((n, n)) < 0.08, 0.0, 0.2)
        phi = physics.propagate(_signed_distance(burned, h), h, rate, None, [21600.0])[21600.0]
        self.assertTrue(np.isfinite(phi).all())
        self.assertFalse((phi[:, 0] < 0).any() or (phi[:, -1] < 0).any() or (phi[0] < 0).any() or (phi[-1] < 0).any())
        self.assertGreater(phi.min(), -2 * h * 20)

    def test_tables_match_training_package(self):
        """physics.py copies the spread-rate tables from wildfire_pinn/data_pipeline.py; they must agree.

        Reads the table block from the source file, so the training package's
        heavy dependencies (shapely, xarray, ...) are not needed here.
        """
        source = HERE.parent / "wildfire_pinn" / "data_pipeline.py"
        if not source.exists():
            self.skipTest(f"{source} not present (service copied without the training package)")
        text = source.read_text()
        ns: dict = {}
        exec(text[text.index("GRASS_SHRUB = "):text.index("if set(R_REL) - set(MX)")], ns)  # noqa: S102
        self.assertEqual(physics.R_REL, ns["R_REL"])
        self.assertEqual(physics.MX, ns["MX"])
        for name in ("C_REF", "C_W", "U_REF", "C_S"):
            self.assertEqual(getattr(physics, name), ns[name])
        self.assertTrue(set(ns["NONBURNABLE"]) <= set(physics.NONBURNABLE))


if __name__ == "__main__":
    unittest.main()
