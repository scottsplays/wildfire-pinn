"""Scorecard and replay-source tests (no network).

    python -m unittest pinn_service.tests.test_scorecard -v      (from the repo root)

A synthetic fire folder is written to a temp directory: uniform grass, steady west wind, and four
VIIRS overpasses with a growing burned disk (two before the forecast time, two after).
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pinn_service import scorecard as sc
from pinn_service.postprocessing import active_grid
from pinn_service.replay import ReplaySource

LON0, LAT0 = -121.0, 38.0
ISSUE = dt.datetime(2026, 8, 5, 0, 0, tzinfo=dt.timezone.utc)
# (time, radius of the burned disk in m): two passes before ISSUE, two after (+10 h, +22 h)
PASSES = [(ISSUE - dt.timedelta(hours=14), 600.0), (ISSUE - dt.timedelta(hours=3), 1300.0),
          (ISSUE + dt.timedelta(hours=10), 2200.0), (ISSUE + dt.timedelta(hours=22), 3000.0)]


def m_per_deg(lat):
    phi = math.radians(lat)
    return (111132.92 - 559.82 * math.cos(2 * phi) + 1.175 * math.cos(4 * phi),
            111412.84 * math.cos(phi) - 93.5 * math.cos(3 * phi))


def write_fire(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    xs = np.arange(-12000.0, 12001.0, 90.0)
    ys = np.arange(-12000.0, 12001.0, 90.0)
    np.savez(folder / "static.npz", fuel_fbfm40=np.full((len(ys), len(xs)), 102, np.int16),
             slope_deg=np.zeros((len(ys), len(xs)), np.float32), xs=xs, ys=ys, lon0=np.float64(LON0),
             lat0=np.float64(LAT0), source=np.array("synthetic"))
    m_lat, m_lon = m_per_deg(LAT0)
    with open(folder / "detections.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["latitude", "longitude", "scan", "track", "acq_date", "acq_time", "satellite", "instrument",
                    "confidence", "frp", "daynight", "time_utc"])
        for when, radius in PASSES:
            for dx in np.arange(-radius, radius + 1, 300.0):
                for dy in np.arange(-radius, radius + 1, 300.0):
                    if math.hypot(dx, dy) <= radius:
                        w.writerow([f"{LAT0 + dy / m_lat:.5f}", f"{LON0 + dx / m_lon:.5f}", 0.39, 0.36,
                                    when.date().isoformat(), when.strftime("%H%M"), "N", "VIIRS", "n", 5.0, "D",
                                    when.isoformat()])
    with open(folder / "weather.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["time_utc", "wind_speed_m_s", "wind_from_deg", "rh_pct", "temp_c", "source"])
        t = ISSUE - dt.timedelta(days=3)
        while t <= ISSUE + dt.timedelta(days=3):
            w.writerow([t.isoformat(), 5.0, 270.0, 20.0, 30.0, "synthetic"])
            t += dt.timedelta(hours=1)
    with open(folder / "fm100.csv", "w", newline="") as fh:
        fh.write("date,fm100_pct\n2026-08-03,7.0\n2026-08-04,7.0\n2026-08-05,7.0\n")
    (folder / "meta.json").write_text(json.dumps({"fire_id": folder.name}))


class ScorecardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.folder = Path(cls.tmp.name) / "20260805_synthetic"
        write_fire(cls.folder)
        cls.src = ReplaySource(cls.folder)
        cls.grid = active_grid()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_verify(self, **over):
        body = {"bbox": self.src.bbox(), "date": ISSUE.strftime("%Y-%m-%dT%H:%M:%SZ"), "leads_hours": [12, 24],
                "additional_params": {"search_buffer_m": 0.0}} | over
        return sc.verify(body, self.src, self.grid, now=ISSUE + dt.timedelta(days=30))

    def test_replay_never_sees_the_future(self):
        from pinn_service.contract import PredictionRequest

        req = PredictionRequest.model_validate({"bbox": self.src.bbox(), "time_horizon": 6, "model_type": "pinn",
                                                "date": ISSUE.isoformat()})
        dets = self.src.detections(req, self.src.bbox())
        self.assertGreater(len(dets), 0)
        self.assertLessEqual(float(dets.time_s.max()), ISSUE.timestamp())

    def test_scores_both_leads_against_the_right_passes(self):
        out = self.run_verify()
        self.assertEqual(out["fires_detected"], 1)
        r12, r24 = out["results"]
        self.assertEqual((r12["status"], r24["status"]), ("scored", "scored"))
        self.assertEqual(r12["valid_at"], PASSES[2][0].strftime("%Y-%m-%dT%H:%M:%SZ"))
        self.assertEqual(r24["valid_at"], PASSES[3][0].strftime("%Y-%m-%dT%H:%M:%SZ"))
        self.assertEqual((r12["forecast_horizon_hours"], r24["forecast_horizon_hours"]), (10, 22))
        for r in (r12, r24):
            m = r["model"]
            self.assertTrue(0.0 <= m["new_area_iou"] <= 1.0)
            self.assertAlmostEqual(m["f1"], 2 * m["new_area_iou"] / (1 + m["new_area_iou"]), places=3)
            self.assertGreater(m["observed_new_km2"], 0)
            self.assertIn("new_area_iou", r["baseline_dilation"])
        self.assertGreater(r24["model"]["observed_new_km2"], r12["model"]["observed_new_km2"])
        self.assertGreater(r12["model"]["new_area_iou"], 0.0)       # a growing fire in wind is partly predicted

    def test_cells_for_map_overlays(self):
        out = self.run_verify(leads_hours=[24], include_cells=True)
        r = out["results"][0]
        cells = r["cells"]
        km2 = out["grid_cell_km2"]
        self.assertAlmostEqual(len(cells["hit"]) * km2, r["model"]["hit_km2"], places=1)
        self.assertAlmostEqual(len(cells["missed"]) * km2, r["model"]["missed_km2"], places=1)
        self.assertFalse(set(cells["hit"]) & set(cells["burned_at_issue"]))
        self.assertNotIn("cells", self.run_verify(leads_hours=[24])["results"][0])

    def test_nowcast_uses_its_best_guess_threshold(self):
        out = self.run_verify(method="nowcast")
        self.assertAlmostEqual(out["threshold"], sc.median_threshold("nowcast"))
        self.assertLess(out["threshold"], 0.3)
        self.assertEqual(out["results"][0]["status"], "scored")

    def test_pending_and_no_observation(self):
        late = PASSES[3][0] + dt.timedelta(hours=1)
        body = {"bbox": self.src.bbox(), "date": late.strftime("%Y-%m-%dT%H:%M:%SZ"), "leads_hours": [12],
                "additional_params": {"search_buffer_m": 0.0}}
        soon = sc.verify(body, self.src, self.grid, now=late + dt.timedelta(hours=2))
        self.assertEqual(soon["results"][0]["status"], "pending")
        later = sc.verify(body, self.src, self.grid, now=late + dt.timedelta(days=2))
        self.assertEqual(later["results"][0]["status"], "no_observation")

    def test_no_fire_before_ignition(self):
        early = PASSES[0][0] - dt.timedelta(days=1)
        out = self.run_verify(date=early.strftime("%Y-%m-%dT%H:%M:%SZ"))
        self.assertEqual({r["status"] for r in out["results"]}, {"no_fire"})

    def test_overlap_metrics(self):
        m = sc.overlap({1, 2, 3, 4}, {3, 4, 5}, 1.0)
        self.assertEqual((m["new_area_iou"], m["precision"], m["recall"]), (0.4, 0.5, 0.6667))
        self.assertAlmostEqual(m["f1"], 2 * 0.4 / 1.4, places=3)
        self.assertEqual(m["growth_ratio"], 1.333)
        empty = sc.overlap(set(), {1}, 1.0)
        self.assertEqual((empty["new_area_iou"], empty["recall"], empty["precision"]), (0.0, 0.0, None))

    def test_summary_counts_each_fire_once(self):
        rec = lambda fire, iou: {"method": "calibrated", "fire_id": fire, "issued_at": "x", "results": [  # noqa: E731
            {"lead_hours": 24, "status": "scored", "model": {"new_area_iou": iou, "f1": 0, "precision": 0, "recall": 0,
             "growth_ratio": 1.0}, "baseline_dilation": {"new_area_iou": 0.1, "f1": 0, "precision": 0, "recall": 0,
                                                          "growth_ratio": 2.0}}]}
        s = sc.summarize([rec("a", 0.1), rec("a", 0.3), rec("b", 0.6)])[0]
        self.assertEqual((s["fires_scored"], s["forecasts_scored"]), (2, 3))
        self.assertAlmostEqual(s["model_new_area_iou"], (0.2 + 0.6) / 2, places=4)

    def test_bad_request(self):
        from pinn_service.contract import ServiceError

        with self.assertRaises(ServiceError):
            sc.verify({"bbox": self.src.bbox(), "date": "2026-08-05", "leads_hours": [99]}, self.src, self.grid)

    def test_verify_through_the_service_handlers(self):
        try:
            from pinn_service import serve
        except ImportError as exc:
            self.skipTest(f"fastapi unavailable: {exc}")
        original = serve.make_source
        serve.make_source = lambda: self.src
        try:
            body = {"bbox": self.src.bbox(), "date": ISSUE.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "additional_params": {"search_buffer_m": 0.0}}
            status, one = serve.handle_verify(body)
            self.assertEqual(status, 200)
            self.assertEqual(one["type"], "pinn_scorecard")
            status, many = serve.handle_verify([body, body])
            self.assertEqual(status, 200)
            self.assertEqual(len(many["results"]), 2)
            self.assertEqual(many["summary"][0]["forecasts_scored"], 2)
            status, vertex = serve.handle({"instances": [body | {"task": "verify"}]})   # Vertex AI path
            self.assertEqual(status, 200)
            self.assertEqual(vertex["predictions"][0]["type"], "pinn_scorecard")
            status, bad = serve.handle_verify(body | {"method": "unet"})
            self.assertEqual(status, 422)
        finally:
            serve.make_source = original


if __name__ == "__main__":
    unittest.main()
