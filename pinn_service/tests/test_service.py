"""Contract, ingestion and physics tests for the WildfirePINN inference service.

    python -m unittest pinn_service.tests.test_service -v      (from the repo root)

No network: native ingestion is exercised with recorded-format responses served by
a fake fetcher (FIRMS CSV, Open-Meteo JSON, gridMET CSV) built from the synthetic
test fire. ``python -m pinn_service.native_ingestion --smoke`` checks the live sources.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import unittest
import urllib.parse
from pathlib import Path

import numpy as np

from pinn_service import calibrated as cal
from pinn_service import native_ingestion as ni
from pinn_service import physics
from pinn_service.contract import parse_reference_time
from pinn_service.forecaster import signed_distance
from pinn_service.make_test_features import main as make_test_features
from pinn_service.serve import app, handle
from pinn_service.validate_contract import validate

HERE = Path(__file__).resolve().parent.parent
FEATURES = HERE / "test_features"
BBOX = [-122.9, 36.9, -121.5, 38.3]
FIRE_DAY = "2026-10-16"
QUIET_DAY = "2026-10-17"
FIRMS_HEADER = "latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,instrument,confidence,version,bright_ti5,frp,daynight"


class FakeFetch:
    """Serves FIRMS / Open-Meteo / gridMET responses in their documented formats."""

    def __init__(self):
        self.calls: list[str] = []
        self.fail_hosts: set[str] = set()
        day = np.load(FEATURES / f"{FIRE_DAY}.npz")
        lon2, lat2 = np.meshgrid(day["lon"], day["lat"])
        fire = day["active_fire"]
        self.fire_points = [(float(x), float(y), "h" if f >= 1 else "n") for x, y, f in zip(lon2[fire >= 0.5], lat2[fire >= 0.5], fire[fire >= 0.5])]

    def __call__(self, url: str) -> bytes:
        self.calls.append(url)
        host = urllib.parse.urlsplit(url).netloc
        if any(h in host for h in self.fail_hosts):
            raise ConnectionError(f"{host}: simulated outage")
        if "firms" in host:
            return self._firms(url)
        if "open-meteo" in host:
            return self._open_meteo(url)
        if "thredds" in host:
            return b'time,latitude[unit="degrees_north"],longitude[unit="degrees_east"],dead_fuel_moisture_100hr[unit="Percent"]\n' \
                   b"2026-10-15T00:00:00Z,37.17,-122.22,74.0\n2026-10-16T00:00:00Z,37.17,-122.22,70.0\n"  # packed, = 7.4 / 7.0 %
        raise AssertionError(f"unexpected URL {url}")

    def _firms(self, url: str) -> bytes:
        parts = urllib.parse.urlsplit(url).path.split("/")
        source, bbox, days, start = parts[-4], [float(v) for v in parts[-3].split(",")], int(parts[-2]), parts[-1]
        first = dt.date.fromisoformat(start)
        dates = {(first + dt.timedelta(days=i)).isoformat() for i in range(days)}
        rows = [FIRMS_HEADER]
        if source == "VIIRS_SNPP_NRT" and FIRE_DAY in dates:
            for x, y, conf in self.fire_points:
                if bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]:
                    rows.append(f"{y:.5f},{x:.5f},340.1,0.39,0.36,{FIRE_DAY},2000,N,VIIRS,{conf},2.0NRT,290.2,12.5,D")
            rows.append(f"37.30000,-122.40000,301.0,0.39,0.36,{FIRE_DAY},2000,N,VIIRS,l,2.0NRT,280.0,1.1,D")      # low confidence: dropped
            if bbox[0] <= -122.3 <= bbox[2] and bbox[1] <= 37.7 <= bbox[3]:
                rows.append(f"37.70000,-122.30000,360.0,0.39,0.36,{FIRE_DAY},2000,N,VIIRS,h,2.0NRT,300.0,40.0,D")  # 'refinery flare' on non-burnable land
        if source == "VIIRS_NOAA20_NRT" and FIRE_DAY in dates:
            # an earlier overpass (09:30 UTC) that saw only the fire's core: gives the calibrated method a growth interval
            cx = np.mean([p[0] for p in self.fire_points])
            cy = np.mean([p[1] for p in self.fire_points])
            for x, y, conf in self.fire_points:
                if np.hypot((x - cx) * 0.8, y - cy) < 0.008 and bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]:
                    rows.append(f"{y:.5f},{x:.5f},335.0,0.39,0.36,{FIRE_DAY},0930,1,VIIRS,{conf},2.0NRT,288.0,8.0,N")
        if source == "VIIRS_NOAA20_NRT" and "2026-10-14" in dates:
            rows.append(f"37.17000,-122.22000,340.0,0.39,0.36,2026-10-14,2000,1,VIIRS,h,2.0NRT,290.0,9.0,D")  # older than lookback
        return ("\n".join(rows) + "\n").encode()

    def _open_meteo(self, url: str) -> bytes:
        q = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)
        start = dt.date.fromisoformat(q["start_date"][0])
        end = dt.date.fromisoformat(q["end_date"][0])
        hours = int(((end - start).days + 1) * 24)
        times = [(dt.datetime(start.year, start.month, start.day) + dt.timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M") for i in range(hours)]
        body = {
            "latitude": float(q["latitude"][0]), "longitude": float(q["longitude"][0]), "utc_offset_seconds": 0, "timezone": "GMT",
            "hourly_units": {"wind_speed_10m": "m/s", "wind_direction_10m": "°", "relative_humidity_2m": "%", "temperature_2m": "°C"},
            "hourly": {"time": times, "wind_speed_10m": [7.0] * hours, "wind_direction_10m": [320] * hours,
                       "relative_humidity_2m": [14] * hours, "temperature_2m": [31.0] * hours},
        }
        return json.dumps(body).encode()


FAKE: FakeFetch | None = None


def request(**overrides) -> dict:
    body = {"bbox": BBOX, "time_horizon": 6, "model_type": "pinn", "date": FIRE_DAY}
    body.update(overrides)
    return body


def setUpModule():
    global FAKE
    if not (FEATURES / "static_synthetic.npz").exists():
        make_test_features()
    os.environ.update({
        "FEATURE_DIR": str(FEATURES),
        "STATIC_LAYERS": str(FEATURES / "static_synthetic.npz"),
        "FIRMS_MAP_KEY": "TESTKEY",
        "INGESTION": "native",
    })
    FAKE = FakeFetch()
    ni.FETCH = FAKE
    ni._CACHE.clear()


def tearDownModule():
    ni.FETCH = ni._urllib_fetch
    ni._CACHE.clear()


class ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.status, cls.fc = handle(request())

    def test_ok_and_valid(self):
        self.assertEqual(self.status, 200, self.fc)
        errors, _ = validate(self.fc, BBOX, {0, 1, 3, 6})
        self.assertEqual(errors, [])

    def test_all_horizons_present(self):
        self.assertEqual({f["properties"]["horizon_hours"] for f in self.fc["features"]}, {0, 1, 3, 6})

    def test_rev2_properties(self):
        for f in self.fc["features"]:
            p = f["properties"]
            self.assertEqual(p["members"], 1)
            self.assertIn(p["risk_label"], {"ACTIVE_FIRE", "HIGH_RISK", "MODERATE_RISK", "LOW_RISK"})
            self.assertEqual(p["risk_label"] == "ACTIVE_FIRE", p["horizon_hours"] == 0)
            self.assertEqual(p["ensemble_size"], 10)

    def test_metadata(self):
        meta = self.fc["metadata"]
        self.assertEqual(meta["source_model"], "pinn")
        self.assertTrue(meta["timestamp"].endswith("Z"))
        self.assertEqual(meta["reference_time"], "2026-10-16T21:00:00Z")
        self.assertEqual(meta["total_cells"], len(self.fc["features"]))
        self.assertEqual(meta["fires_detected"], 1)
        self.assertTrue(any("gridMET FM100 7.0" in n for n in meta["input_notes"]))

    def test_non_wildland_hot_spot_rejected(self):
        """A high-confidence detection on non-burnable land (refinery, flare) is reported, not forecast."""
        rejected = self.fc["metadata"]["detections_rejected"]
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0]["centre_lon_lat"], [-122.3, 37.7])
        self.assertLess(rejected[0]["burnable_share"], 0.25)
        for f in self.fc["features"]:
            lon, lat = np.mean(f["geometry"]["coordinates"][0][:4], axis=0)
            self.assertFalse(abs(lon + 122.3) < 0.02 and abs(lat - 37.7) < 0.02)

    def test_cells_on_adapter_fixture_lattice(self):
        """Cell edges sit on round 0.01 deg lines, like the U-Net fixture cells."""
        for f in self.fc["features"][:50]:
            ring = np.array(f["geometry"]["coordinates"][0])
            self.assertTrue(np.allclose(ring * 100, np.round(ring * 100), atol=1e-6), ring)

    def test_burned_area_never_shrinks(self):
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
        """Wind from 320 deg blows toward the SE, so the 6 h centroid sits SE of the 0 h centroid."""
        def centroid(h):
            pts = [np.mean(f["geometry"]["coordinates"][0][:4], axis=0) for f in self.fc["features"] if f["properties"]["horizon_hours"] == h]
            return np.mean(pts, axis=0)
        c0, c6 = centroid(0), centroid(6)
        self.assertGreater(c6[0], c0[0])
        self.assertLess(c6[1], c0[1])

    def test_deterministic(self):
        _, again = handle(request())
        strip = lambda fc: [(f["properties"]["grid_id"], f["properties"]["horizon_hours"], f["properties"]["fire_probability"]) for f in fc["features"]]  # noqa: E731
        self.assertEqual(strip(self.fc), strip(again))


class IngestionTests(unittest.TestCase):
    def test_firms_filters_low_confidence_and_old(self):
        ref = parse_reference_time(FIRE_DAY)
        dets = ni.firms_detections(BBOX, ref, 24)
        early = sum(1 for t in dets.time_s if dt.datetime.fromtimestamp(t, dt.timezone.utc).hour == 9)
        self.assertGreater(early, 0)
        self.assertEqual(len(dets) - early, len(FAKE.fire_points) + 1)  # + the non-wildland hot spot (rejected later, not here)
        self.assertTrue(np.all(dets.confidence >= 0.8))
        self.assertTrue(np.allclose(dets.radius_m, np.sqrt(0.39 * 0.36 / np.pi) * 1000))

    def test_key_never_in_errors(self):
        FAKE.fail_hosts = {"firms"}
        try:
            status, body = handle(request())
        finally:
            FAKE.fail_hosts = set()
            ni._CACHE.clear()
        self.assertEqual(status, 503)
        self.assertNotIn("TESTKEY", json.dumps(body))

    def test_missing_key(self):
        saved = os.environ.pop("FIRMS_MAP_KEY")
        try:
            with unittest.mock.patch.object(ni, "PACKAGE_DIR", Path("/nonexistent")), \
                 unittest.mock.patch.object(Path, "cwd", return_value=Path("/nonexistent")):
                status, body = handle(request(date="2026-10-15"))
        finally:
            os.environ["FIRMS_MAP_KEY"] = saved
        self.assertEqual(status, 503)
        self.assertIn("FIRMS", body["detail"])

    def test_gridmet_outage_falls_back_to_emc(self):
        FAKE.fail_hosts = {"thredds"}
        ni._CACHE.clear()
        try:
            status, fc = handle(request(time_horizon=1))
        finally:
            FAKE.fail_hosts = set()
            ni._CACHE.clear()
        self.assertEqual(status, 200)
        self.assertTrue(any("estimated from 24 h RH/T" in n for n in fc["metadata"]["input_notes"]))

    def test_weather_outage_is_503(self):
        FAKE.fail_hosts = {"open-meteo"}
        ni._CACHE.clear()
        try:
            status, body = handle(request())
        finally:
            FAKE.fail_hosts = set()
            ni._CACHE.clear()
        self.assertEqual(status, 503)
        self.assertIn("weather", body["detail"])

    def test_emc_values(self):
        """Simard EMC: hot and dry gives single digits, humid gives teens."""
        self.assertLess(float(ni.emc_percent(31, 14)), 4)
        self.assertGreater(float(ni.emc_percent(15, 80)), 14)

    def test_key_file_skips_comments(self):
        """Regression: firms_key.txt starts with '#' comment lines (as the training CLI expects)."""
        text = "\ufeff# NASA FIRMS map key\n# get one at https://firms.modaps.eosdis.nasa.gov/api/map_key/\n\n  'abcdef0123456789abcdef0123456789'  \n"
        self.assertEqual(ni.key_from_text(text), "abcdef0123456789abcdef0123456789")
        self.assertIsNone(ni.key_from_text("# only comments\n\n"))

    def test_gridmet_parser(self):
        self.assertEqual(ni.parse_gridmet_csv("time,x,y,dead_fuel_moisture_100hr\n2026-10-15,1,2,9.5\n2026-10-16,1,2,8.25\n"), 8.25)

    def test_gridmet_packed_values(self):
        """Live THREDDS CSV returns packed values (149.0 = 14.9 %); fill values are skipped."""
        live = ('time,latitude[unit="degrees_north"],longitude[unit="degrees_east"],dead_fuel_moisture_100hr[unit="Percent"]\n'
                "2026-10-13T00:00:00Z,37.17,-122.22,149.0\n2026-10-14T00:00:00Z,37.17,-122.22,32767.0\n")
        self.assertAlmostEqual(ni.parse_gridmet_csv(live), 14.9)
        self.assertIsNone(ni.unpack_fm100(32767.0))
        self.assertEqual(ni.unpack_fm100(7.5), 7.5)
        self.assertAlmostEqual(ni.unpack_fm100(45.0), 4.5)    # packed dry-season value, not 45 %
        self.assertAlmostEqual(ni.unpack_fm100(60.0), 6.0)


class RequestHandlingTests(unittest.TestCase):
    def test_vertex_wrapper(self):
        status, body = handle({"instances": [request(time_horizon=1)]})
        self.assertEqual(status, 200)
        self.assertEqual({f["properties"]["horizon_hours"] for f in body["predictions"][0]["features"]}, {0, 1})

    def test_quiet_day_is_empty_not_an_error(self):
        status, fc = handle(request(date=QUIET_DAY))
        self.assertEqual(status, 200)
        self.assertEqual(fc["features"], [])
        self.assertEqual(fc["metadata"]["fires_detected"], 0)

    def test_timestamp_and_hour(self):
        self.assertEqual(parse_reference_time("2026-10-16"), dt.datetime(2026, 10, 16, 21, tzinfo=dt.timezone.utc))
        self.assertEqual(parse_reference_time("2026-10-16T18:30:00Z").hour, 18)
        self.assertEqual(parse_reference_time("2026-10-16T11:30:00-07:00").hour, 18)
        _, fc = handle(request(date="2026-10-16T22:15:00Z", time_horizon=0))
        self.assertEqual(fc["metadata"]["reference_time"], "2026-10-16T22:15:00Z")
        _, fc = handle(request(time_horizon=0, additional_params={"hour": 23}))
        self.assertEqual(fc["metadata"]["reference_time"], "2026-10-16T23:00:00Z")

    def test_one_km_bbox_sees_whole_fire(self):
        """The backend may send a single 1 km cell: the fire around it is still modelled, then clipped."""
        x, y, _ = FAKE.fire_points[0]
        lat_step, lon_step = 0.0045, 0.0057  # ~0.5 km
        small = [x - lon_step, y - lat_step, x + lon_step, y + lat_step]
        status, fc = handle(request(bbox=small))
        self.assertEqual(status, 200)
        errors, _ = validate(fc, small)
        self.assertEqual(errors, [])
        self.assertEqual(fc["metadata"]["fires_detected"], 1)
        self.assertGreater(fc["metadata"]["fires"][0]["detections"], 50)
        self.assertGreater(fc["metadata"]["cells_outside_bbox_dropped"], 0)

    def test_team_adapter_defaults(self):
        """ml_adapter/main.py sends time_horizon=24, date=None and an extra 'mock' field."""
        status, fc = handle({"bbox": BBOX, "time_horizon": 24, "model_type": "pinn", "date": None, "mock": False})
        self.assertEqual(status, 200, fc)
        self.assertEqual(fc["metadata"]["horizons_hours"], [0, 1, 3, 6])
        now = dt.datetime.now(dt.timezone.utc)
        ref = dt.datetime.fromisoformat(fc["metadata"]["reference_time"].replace("Z", "+00:00"))
        self.assertLess(abs((now - ref).total_seconds()), 2 * 3600)

    def test_confidence_threshold(self):
        _, fc = handle(request(additional_params={"confidence_threshold": 0.5}))
        self.assertTrue(fc["features"])
        self.assertTrue(all(f["properties"]["fire_probability"] >= 0.5 for f in fc["features"]))

    def test_rejections(self):
        cases = [
            (request(model_type="unet"), 422),
            (request(bbox=[-121.5, 36.9, -122.9, 38.3]), 422),
            (request(time_horizon=99), 422),
            (request(date="yesterday"), 422),
            (request(additional_params={"hour": 25}), 422),
            ({"instances": []}, 400),
            ([1, 2, 3], 400),
        ]
        for body, expected in cases:
            with self.subTest(body=str(body)[:60]):
                status, payload = handle(body)
                self.assertEqual(status, expected)
                self.assertIn("detail", payload)


class CalibratedMethodTests(unittest.TestCase):
    """v0.3.0: growth between two overpasses calibrates alpha; the learned model adjusts it."""

    def test_default_is_calibrated_two_pass(self):
        status, fc = handle(request())
        self.assertEqual(status, 200)
        fire = fc["metadata"]["fires"][0]
        self.assertEqual(fc["metadata"]["method"], "calibrated")
        self.assertIn(fire["calibration_mode"], {"learned", "last-interval"})
        self.assertEqual(fire["calibration_interval"][0][11:16], "09:30")
        self.assertTrue(cal.ALPHAS[0] <= fire["alpha"] <= cal.ALPHAS[-1])
        self.assertEqual(fire["resolution_m"], 180.0)

    def test_learned_model_is_used(self):
        import tempfile

        names = list(cal.features(np.full((5, 5), 101.0), np.zeros((5, 5)), np.eye(5, dtype=bool), np.zeros((5, 5), bool),
                                  np.ones((5, 5)), 180.0, StubEnv(), 0.0, 3 * 3600.0, 0.3))
        model = {"features": names, "mu": [0.0] * len(names), "sd": [1.0] * len(names),
                 "w": [float(np.log(0.5))] + [0.0] * len(names), "alpha_min": 0.01, "alpha_max": 6.0,
                 "fallback_alpha": 0.3, "rmse_log_alpha_LOFO": 0.9, "trained_on": "unit test"}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "learned_alpha.json"
            path.write_text(json.dumps(model))
            with unittest.mock.patch.dict(os.environ, {"LEARNED_ALPHA": str(path)}):
                status, fc = handle(request(time_horizon=1))
        fire = fc["metadata"]["fires"][0]
        self.assertEqual(status, 200)
        self.assertEqual(fire["calibration_mode"], "learned")
        self.assertAlmostEqual(fire["alpha"], 0.5, places=3)          # intercept-only model -> exp(log 0.5)
        self.assertEqual(fire["alpha_log_sigma"], 0.9)
        self.assertEqual(set(fire["alpha_features"]), set(names))

    def test_physics_method_still_available(self):
        status, fc = handle(request(additional_params={"method": "physics"}))
        self.assertEqual(status, 200)
        self.assertEqual(fc["metadata"]["method"], "physics")
        self.assertNotIn("calibration_mode", fc["metadata"]["fires"][0])
        self.assertEqual(handle(request(additional_params={"method": "magic"}))[0], 422)

    def test_ellipse_head_speed(self):
        """With an ellipse the head moves at R; the back far slower."""
        n, h, r, t = 201, 30.0, 0.05, 3600.0
        yy, xx = np.mgrid[0:n, 0:n]
        burned = np.hypot(xx - 100, yy - 100) * h <= 300.0
        e = physics.ellipse_eccentricity(8.0)
        phi = physics.propagate(signed_distance(burned, h), h, np.full((n, n), r), (1.0, 0.0), [t], eccentricity=e)[t]
        row = phi[100] < 0
        head = (np.flatnonzero(row).max() - 100) * h
        back = (100 - np.flatnonzero(row).min()) * h
        self.assertAlmostEqual(head, 300.0 + r * t, delta=2 * h)
        self.assertLess(back, 300.0 + 0.5 * r * t)
        self.assertGreater(e, 0.5)
        self.assertEqual(physics.ellipse_eccentricity(0.0), 0.0)

    def test_active_field_holds_quiet_edge(self):
        n, res = 81, 180.0
        yy, xx = np.mgrid[0:n, 0:n]
        prev = np.hypot(xx - 40, yy - 40) <= 10
        start = prev | ((np.hypot(xx - 40, yy - 40) <= 14) & (xx > 45))      # grew only to the east
        act = cal.active_field(start, prev, res)
        self.assertEqual(act[40, 40 + 16], 1.0)    # in front of the growing east edge
        self.assertEqual(act[40, 40 - 12], 0.0)    # in front of the quiet west edge
        self.assertTrue(np.all(act[start] == 1.0))

    def test_area_matched_alpha_recovers_truth(self):
        n, res, span = 81, 180.0, 6.0
        yy, xx = np.mgrid[0:n, 0:n]
        prev = np.hypot(xx - 40, yy - 40) <= 4
        rate = np.full((n, n), 0.1)
        truth = 0.4
        phi = cal._advance(cal.signed_distance(prev, res), res, rate, 270.0, 3.0, truth * span * 3600.0)
        observed = int(((phi < 0) & ~prev).sum())
        alpha = cal.pick_alpha(cal.area_curve(prev, res, rate, 270.0, 3.0, span, observed), observed)
        self.assertLess(abs(np.log(alpha / truth)), 0.2)

    def test_single_pass_falls_back(self):
        os.environ["INGESTION"] = "npz"
        try:
            status, fc = handle(request())
        finally:
            os.environ["INGESTION"] = "native"
        self.assertEqual(status, 200)
        self.assertEqual(fc["metadata"]["fires"][0]["calibration_mode"], "fallback")

    def test_nowcast_method(self):
        status, fc = handle(request(additional_params={"method": "nowcast"}))
        self.assertEqual(status, 200)
        self.assertEqual(validate(fc, BBOX, {0, 1, 3, 6})[0], [])
        fire = fc["metadata"]["fires"][0]
        self.assertEqual(fc["metadata"]["method"], "nowcast")
        self.assertIn("active-edge", fire["nowcast"])
        self.assertGreater(fire["observed_growth_km2_per_h"], 0)
        probs = [f["properties"]["fire_probability"] for f in fc["features"] if f["properties"]["horizon_hours"] == 6]
        self.assertTrue(any(p == 1.0 for p in probs))                     # the burning fire
        self.assertTrue(any(0 < p <= 0.407 + 1e-9 for p in probs))        # calibrated new growth (max 0.407)

    def test_nowcast_probability_shape(self):
        n, res = 61, 180.0
        yy, xx = np.mgrid[0:n, 0:n]
        start = np.hypot(xx - 30, yy - 30) <= 5
        act = np.ones((n, n))
        cfg = cal.NOWCAST_DEFAULTS
        p = cal.nowcast_probability(start, act, res, 2.0, 6.0, 1.4, cfg["prob_calibration"])
        self.assertTrue(np.all(p[start] == 1.0))
        self.assertGreater(p[30, 36], p[30, 45])                          # nearer the edge is more likely
        self.assertLessEqual(p[~start].max(), 0.407 + 1e-9)
        act[:, :30] = 0                                                    # west side quiet: no growth there
        p = cal.nowcast_probability(start, act, res, 2.0, 6.0, 1.4, cfg["prob_calibration"])
        self.assertEqual(p[30, 20], 0.0)

    def test_nowcast_single_pass_falls_back(self):
        os.environ["INGESTION"] = "npz"
        try:
            status, fc = handle(request(additional_params={"method": "nowcast"}))
        finally:
            os.environ["INGESTION"] = "native"
        self.assertEqual(status, 200)
        self.assertIn("fallback", fc["metadata"]["fires"][0]["nowcast"])

    def test_passes_group_within_20_minutes(self):
        self.assertEqual(cal.pass_times(np.array([0.0, 300.0, 3000.0, 3060.0, 9000.0])), [0.0, 3000.0, 9000.0])


class StubEnv:
    def conditions(self, moment):
        return 5.0, 270.0, 6.0, 8.0


class ReplayModeTests(unittest.TestCase):
    def setUp(self):
        os.environ["INGESTION"] = "npz"

    def tearDown(self):
        os.environ["INGESTION"] = "native"

    def test_replay_matches_contract(self):
        status, fc = handle(request())
        self.assertEqual(status, 200)
        self.assertEqual(validate(fc, BBOX, {0, 1, 3, 6})[0], [])
        self.assertEqual(fc["metadata"]["ingestion"], "npz replay")

    def test_missing_replay_file(self):
        status, _ = handle(request(date="1999-01-01"))
        self.assertEqual(status, 404)


class HttpTests(unittest.TestCase):
    def setUp(self):
        try:
            from fastapi.testclient import TestClient
        except Exception as exc:
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
        rate, unknown = physics.spread_rate([[101]], [[0.0]], [[physics.U_REF]], [[0.0]])
        self.assertAlmostEqual(float(rate[0, 0]), physics.C_REF, places=9)
        self.assertEqual(unknown, [])

    def test_nonburnable_and_unknown(self):
        rate, unknown = physics.spread_rate([[98, 999]], [[0.0, 0.0]], [[5.0, 5.0]], [[5.0, 5.0]])
        self.assertTrue(np.all(rate == 0))
        self.assertEqual(unknown, [999])

    def test_expanding_circle_speed(self):
        n, h, r_m_s, t = 161, 30.0, 0.05, 3600.0
        yy, xx = np.mgrid[0:n, 0:n]
        burned = np.hypot(xx - 80, yy - 80) * h <= 300.0
        phi = physics.propagate(signed_distance(burned, h), h, np.full((n, n), r_m_s), None, [t])[t]
        radius = np.sqrt((phi < 0).sum() * h * h / np.pi)
        self.assertAlmostEqual(radius, 300.0 + r_m_s * t, delta=h)

    def test_heterogeneous_rate_stable(self):
        rng = np.random.default_rng(0)
        n, h = 200, 90.0
        burned = np.zeros((n, n), bool)
        burned[90:110, 90:110] = True
        rate = np.where(rng.random((n, n)) < 0.08, 0.0, 0.2)
        phi = physics.propagate(signed_distance(burned, h), h, rate, None, [21600.0])[21600.0]
        self.assertTrue(np.isfinite(phi).all())
        self.assertFalse((phi[:, 0] < 0).any() or (phi[:, -1] < 0).any() or (phi[0] < 0).any() or (phi[-1] < 0).any())

    def test_tables_match_training_package(self):
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


import unittest.mock  # noqa: E402  (used in IngestionTests)

if __name__ == "__main__":
    unittest.main()
