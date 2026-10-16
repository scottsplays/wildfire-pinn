"""Core checks for the level-set pipeline. No downloads and no full training run."""

import tempfile
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch
from shapely.geometry import box

from wildfire_pinn.data_pipeline import (
    C_REF,
    U_REF,
    _export_size,
    compute_spread_rate,
    filter_detections,
    firms_chunks,
    normalize_acq_time,
    overpass_timestamps,
    parse_firms_csv,
    trim_outliers,
)
from wildfire_pinn.demo import make_demo_case
from wildfire_pinn.geometry import signed_distance, zero_contours
from wildfire_pinn.pinn import (
    LevelSetPINN,
    RealConfig,
    _draw_pool,
    _region_counts,
    _supervision_pools,
    directed_residual,
    downwind_unit_vectors,
    isotropic_residual,
    load_checkpoint,
    outward_normals,
    sample_grid,
    save_checkpoint,
    train_real,
)


class PipelineTests(unittest.TestCase):
    def test_firms_chunks_cover_january_without_gaps(self):
        chunks = firms_chunks("2025-01-07", "2025-01-31")
        self.assertEqual(
            chunks,
            [
                ("2025-01-07", 5),
                ("2025-01-12", 5),
                ("2025-01-17", 5),
                ("2025-01-22", 5),
                ("2025-01-27", 5),
            ],
        )

    def test_landfire_export_matches_the_notebook_size(self):
        self.assertEqual(_export_size((-118.70, 34.00, -118.42, 34.14)), (1039, 520))

    def test_firms_parser_rejects_an_error_page_and_keeps_an_empty_window(self):
        with self.assertRaises(RuntimeError):
            parse_firms_csv("Invalid API key")
        empty = parse_firms_csv("latitude,longitude,acq_date,acq_time,confidence\n")
        self.assertTrue(empty.empty)

    def test_acq_time_and_confidence_filter(self):
        import pandas as pd

        self.assertEqual(normalize_acq_time(["926", "2104.0"]).tolist(), ["0926", "2104"])
        frame = pd.DataFrame({
            "latitude": [34.05, 34.06, 34.07],
            "longitude": [-118.55, -118.56, -118.57],
            "acq_date": ["2025-01-08", "2025-01-08", "2025-01-08"],
            "acq_time": [926, 926, 926],
            "confidence": ["n", "l", "H"],
        })
        kept = filter_detections(frame)
        self.assertEqual(len(kept), 2)
        self.assertTrue((kept["overpass"] == "2025-01-08_0926").all())
        stamp = overpass_timestamps(kept["acq_date"], kept["acq_time"])
        self.assertEqual(stamp.iloc[0].hour, 9)
        self.assertEqual(stamp.iloc[0].minute, 26)

    def test_outlier_trim_drops_a_distant_detection(self):
        rng = np.random.default_rng(0)
        lon = -118.55 + rng.normal(0, 0.004, 40)
        lat = 34.07 + rng.normal(0, 0.004, 40)
        lon = np.append(lon, -118.10)
        lat = np.append(lat, 34.07)
        keep = trim_outliers(np.column_stack([lon, lat]))
        self.assertFalse(keep[-1])
        self.assertGreater(keep.sum(), 30)

    def test_spread_rate_formula_and_nodata(self):
        fuel = np.array([[101.0, 91.0], [101.0, 101.0]])
        slope = np.array([[0.0, 0.0], [45.0, -32768.0]])
        wind = np.full((1, 2, 2), U_REF)
        dry = np.zeros((1, 2, 2))
        rate, stats = compute_spread_rate(fuel, slope, wind, dry)
        self.assertAlmostEqual(float(rate[0, 0, 0]), C_REF, places=5)
        self.assertEqual(float(rate[0, 0, 1]), 0.0)
        self.assertAlmostEqual(float(rate[0, 1, 0]), 2 * C_REF, places=4)
        self.assertEqual(float(rate[0, 1, 1]), 0.0)
        self.assertGreater(stats["burnable_fraction"], 0)
        moist = np.full((1, 2, 2), 15.0)
        damped, _ = compute_spread_rate(fuel, np.zeros_like(slope), wind, moist)
        self.assertEqual(float(damped[0, 0, 0]), 0.0)
        with self.assertRaises(RuntimeError):
            compute_spread_rate(np.array([[1234.0]]), np.zeros((1, 1)), np.zeros((1, 1, 1)), np.zeros((1, 1, 1)))

    def test_kincade_fuel_codes_use_published_class_and_extinction(self):
        # 146 high, 162/201/202 moderate, 143/187 low. Dry, flat, reference wind.
        fuel = np.array([[146.0, 162.0], [201.0, 202.0], [143.0, 187.0]])
        slope = np.zeros_like(fuel)
        wind = np.full((1, 3, 2), U_REF)
        rate, _stats = compute_spread_rate(fuel, slope, wind, np.zeros((1, 3, 2)))
        self.assertAlmostEqual(float(rate[0, 0, 0]), C_REF, places=5)
        self.assertAlmostEqual(float(rate[0, 0, 1]), 0.2 * C_REF, places=5)
        self.assertAlmostEqual(float(rate[0, 1, 0]), 0.2 * C_REF, places=5)
        self.assertAlmostEqual(float(rate[0, 1, 1]), 0.2 * C_REF, places=5)
        self.assertAlmostEqual(float(rate[0, 2, 0]), 0.07 * C_REF, places=5)
        self.assertAlmostEqual(float(rate[0, 2, 1]), 0.07 * C_REF, places=5)
        # Mx is 25 for SB1 and 40 for SH3, so 25% moisture puts one out and leaves the other.
        moist = np.full((1, 3, 2), 25.0)
        damped, _stats = compute_spread_rate(fuel, slope, wind, moist)
        self.assertEqual(float(damped[0, 1, 0]), 0.0)
        self.assertGreater(float(damped[0, 2, 0]), 0.0)

    def test_signed_distance_sign_and_zero_contour(self):
        x = np.arange(0.0, 900.0, 90.0)
        y = np.arange(900.0, 0.0, -90.0)
        sdf = signed_distance(box(180, 270, 540, 630), x, y)
        self.assertLess(sdf.min(), 0)
        self.assertGreater(sdf.max(), 0)
        row = int(np.where(np.isclose(y, 450))[0][0])
        col = int(np.where(np.isclose(x, 360))[0][0])
        self.assertLess(sdf[row, col], 0)
        self.assertGreater(sdf[0, 0], 0)

        axis = np.linspace(-1, 1, 81)
        xx, yy = np.meshgrid(axis, axis)
        field = np.sqrt(xx ** 2 + yy ** 2) - 0.4
        contours = zero_contours(field, axis, axis)
        self.assertTrue(contours)
        radius = np.sqrt(np.sum(contours[0] ** 2, axis=1)).mean()
        self.assertAlmostEqual(radius, 0.4, delta=0.03)

    def test_sampler_recovers_coordinates_when_y_decreases(self):
        x = np.linspace(0, 100, 11)
        y = np.linspace(50, 0, 6)
        easting = np.tile(x, (len(y), 1))
        northing = np.repeat(y[:, None], len(x), axis=1)
        got_x = sample_grid(easting, x, y, [40.0], [25.0])
        got_y = sample_grid(northing, x, y, [40.0], [25.0])
        self.assertAlmostEqual(float(got_x), 40.0, places=4)
        self.assertAlmostEqual(float(got_y), 25.0, places=4)

    def test_isotropic_residual_is_zero_on_the_expanding_circle(self):
        def phi(t, x, y):
            return torch.sqrt(x ** 2 + y ** 2 + 1e-12) - (0.2 + t)

        t = torch.tensor([0.1, 0.2], dtype=torch.float32, requires_grad=True)
        x = torch.tensor([0.4, -0.3], dtype=torch.float32, requires_grad=True)
        y = torch.tensor([0.2, 0.5], dtype=torch.float32, requires_grad=True)
        value = phi(t, x, y)
        phi_t, phi_x, phi_y = torch.autograd.grad(value, (t, x, y), torch.ones_like(value))
        residual, magnitude = isotropic_residual(phi_t, phi_x, phi_y, 1.0)
        self.assertTrue(torch.allclose(residual, torch.zeros_like(residual), atol=1e-4))
        self.assertTrue(torch.allclose(magnitude, torch.ones_like(magnitude), atol=1e-4))

    def test_dimensionless_residual_scales_by_T_over_L(self):
        length, duration, rate, radius = 1000.0, 500.0, 0.2, 50.0

        def phi(tau, x_norm, y_norm):
            return torch.sqrt(x_norm ** 2 + y_norm ** 2 + 1e-12) - radius / length - (rate * duration / length) * tau

        tau = torch.tensor([0.2, 0.4], dtype=torch.float32, requires_grad=True)
        x_norm = torch.tensor([0.1, -0.2], dtype=torch.float32, requires_grad=True)
        y_norm = torch.tensor([0.05, 0.15], dtype=torch.float32, requires_grad=True)
        value = phi(tau, x_norm, y_norm)
        phi_t, phi_x, phi_y = torch.autograd.grad(value, (tau, x_norm, y_norm), torch.ones_like(value))
        spread = torch.full_like(phi_t, rate)
        residual = directed_residual(
            phi_t, phi_x, phi_y, spread,
            torch.zeros_like(phi_t), torch.ones_like(phi_t),
            0.0, duration / length,
        )
        self.assertTrue(torch.allclose(residual, torch.zeros_like(residual), atol=1e-4))
        with_wind = directed_residual(
            phi_t, phi_x, phi_y, spread,
            torch.ones_like(phi_t), torch.zeros_like(phi_t),
            0.6, duration / length,
        )
        self.assertGreater(float(with_wind.abs().mean()), 1e-3)

    def test_eikonal_loss_is_zero_on_a_signed_distance_circle(self):
        from wildfire_pinn.pinn import analytic_signed_distance_eikonal, eikonal_loss_from_gradients

        loss, magnitude = analytic_signed_distance_eikonal()
        self.assertLess(float(loss), 1e-6)
        self.assertTrue(torch.allclose(magnitude, torch.ones_like(magnitude), atol=1e-4))
        flat, flat_magnitude = eikonal_loss_from_gradients(
            torch.zeros(8), torch.zeros(8)
        )
        self.assertAlmostEqual(float(flat), 1.0, places=4)
        self.assertLess(float(flat_magnitude.max()), 1e-5)

    def test_eikonal_pools_stay_inside_the_two_kilometer_band(self):
        from wildfire_pinn.pinn import _eikonal_region_pools, _even_counts

        x = np.arange(0.0, 901.0, 90.0)
        y = np.arange(900.0, -1.0, -90.0)
        geom = box(180, 270, 540, 630)
        length = 900.0
        sdf = signed_distance(geom, x, y) / length
        pools = _eikonal_region_pools(sdf, x, y, torch.device("cpu"), 90.0, 2000.0, length)
        self.assertGreater(pools["near"]["n"], 0)
        self.assertGreater(pools["shell"]["n"], 0)
        self.assertEqual(pools["band"]["n"], pools["near"]["n"] + pools["shell"]["n"])
        self.assertLessEqual(abs(pools["near"]["range_m"][1]), 90.0 + 0.02)
        self.assertGreater(abs(pools["shell"]["range_m"][0]), 90.0)
        self.assertLessEqual(abs(pools["shell"]["range_m"][1]), 2000.0 + 0.02)
        self.assertEqual(_even_counts(4096, 12)[0], 342)
        self.assertEqual(sum(_even_counts(4096, 12)), 4096)

    def test_persistent_burn_keeps_earlier_pixels_inside(self):
        from wildfire_pinn.burned_region import accumulate_footprints, target_checks

        regions = accumulate_footprints([
            [(500.0, 500.0, 250.0)],
            [(1000.0, 500.0, 250.0)],
        ])
        self.assertTrue(regions[0].contains(regions[0].centroid))
        self.assertTrue(regions[1].contains(regions[0].representative_point()))
        self.assertGreater(regions[1].area, regions[0].area)
        x = np.arange(0.0, 1801.0, 90.0)
        y = np.arange(1800.0, -1.0, -90.0)
        rows = target_checks(regions, x, y)
        self.assertEqual(rows[1]["cells_clearly_unburned"], 0)
        self.assertTrue(all(row["zero_contour"] for row in rows))
        self.assertTrue(all(row["sdf_min_m"] < 0 < row["sdf_max_m"] for row in rows))
        self.assertTrue(all(row["inside_negative"] and row["outside_positive"] for row in rows))
        self.assertGreaterEqual(rows[1]["area_km2"], rows[0]["area_km2"])

    def test_wind_from_north_blows_south(self):
        vectors = downwind_unit_vectors(np.zeros((1, 2, 2)))
        self.assertAlmostEqual(float(vectors[0, 0]), 0.0, places=5)
        self.assertAlmostEqual(float(vectors[0, 1]), -1.0, places=5)
        from_east = downwind_unit_vectors(np.full((1, 2, 2), 90.0))
        self.assertAlmostEqual(float(from_east[0, 0]), -1.0, places=5)
        self.assertAlmostEqual(float(from_east[0, 1]), 0.0, places=5)

    def test_outward_normal_of_a_circle(self):
        def phi(t, x, y):
            return torch.sqrt(x ** 2 + y ** 2) - 0.2

        normal_x, normal_y = outward_normals(phi, [0.0], [0.5], [0.0])
        self.assertAlmostEqual(float(normal_x), 1.0, places=4)
        self.assertAlmostEqual(float(normal_y), 0.0, places=4)

    def test_checkpoint_rejects_a_different_width(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.pt"
            save_checkpoint(path, LevelSetPINN(hidden=8, layers=2), {"note": "narrow"})
            wide = LevelSetPINN(hidden=16, layers=2)
            with self.assertRaises(RuntimeError) as raised:
                load_checkpoint(path, wide, torch.device("cpu"))
            self.assertIn("retrain", str(raised.exception).lower())

    def test_perimeter_supervision_covers_inside_boundary_and_outside(self):
        x = np.linspace(0.0, 900.0, 11)
        y = np.linspace(900.0, 0.0, 11)
        geom = box(180, 270, 540, 630)
        sdf = signed_distance(geom, x, y) / 900.0
        pools = _supervision_pools(geom, sdf, x, y, torch.device("cpu"), boundary_count=40)
        n_in, n_on, n_out = _region_counts(30)
        self.assertGreater(n_in, 0)
        self.assertGreater(n_on, 0)
        self.assertGreater(n_out, 0)
        x_in, y_in = _draw_pool(*pools["inside"], n_in)
        x_out, y_out = _draw_pool(*pools["outside"], n_out)
        self.assertTrue(torch.all(sample_grid(sdf, x, y, x_in, y_in) < 0))
        self.assertTrue(torch.all(sample_grid(sdf, x, y, x_out, y_out) > 0))
        bx = pools["boundary"][0].numpy()
        by = pools["boundary"][1].numpy()
        on_edge = (
            np.isclose(bx, 180) | np.isclose(bx, 540) | np.isclose(by, 270) | np.isclose(by, 630)
        )
        self.assertTrue(np.all(on_edge))

    def test_balanced_sdf_batch_is_three_equal_bands(self):
        from wildfire_pinn.pinn import (
            LevelSetPINN,
            _balanced_perimeter_batch,
            _balanced_supervision_pools,
            _draw_pool,
            _region_counts,
            near_front_band_m,
        )

        x = np.arange(0.0, 901.0, 90.0)
        y = np.arange(900.0, -1.0, -90.0)
        geom = box(180, 270, 540, 630)
        length = 900.0
        sdf_m = signed_distance(geom, x, y)
        sdf = sdf_m / length
        band = near_front_band_m(x, y)
        self.assertAlmostEqual(band, 90.0, places=5)
        pool = _balanced_supervision_pools(sdf, x, y, torch.device("cpu"), band, length)
        self.assertGreater(pool["n_interior"], 0)
        self.assertGreater(pool["n_near"], 0)
        self.assertGreater(pool["n_exterior"], 0)
        self.assertLess(pool["range_interior"][1], -band + 1e-3)
        self.assertGreaterEqual(pool["range_near"][0], -band - 1e-3)
        self.assertLessEqual(pool["range_near"][1], band + 1e-3)
        self.assertGreater(pool["range_exterior"][0], band - 1e-3)
        n_in, n_near, n_out = _region_counts(30)
        self.assertEqual(n_in + n_near + n_out, 30)
        model = LevelSetPINN(hidden=8, layers=1)
        loss_in, loss_near, loss_out = _balanced_perimeter_batch(
            model, 0.0, pool, sdf, type("Case", (), {"x": x, "y": y})(),
            30, 450.0, 450.0, length,
        )
        self.assertEqual(loss_in.shape, ())
        self.assertTrue(torch.isfinite(loss_in + loss_near + loss_out))
        clipped_in, clipped_near, clipped_out = _balanced_perimeter_batch(
            model, 0.0, pool, sdf, type("Case", (), {"x": x, "y": y})(),
            30, 450.0, 450.0, length, clip_m=2000.0,
        )
        self.assertTrue(torch.isfinite(clipped_in + clipped_near + clipped_out))
        # A one-cell pool still fills the requested count.
        tiny = {
            "interior": (torch.zeros(1), torch.zeros(1)),
            "near": (torch.zeros(2), torch.ones(2)),
            "exterior": (torch.ones(1), torch.ones(1)),
        }
        drawn = _draw_pool(*tiny["near"], 10)
        self.assertEqual(drawn[0].shape[0], 10)

    def test_real_training_step_is_finite(self):
        case, _elevation = make_demo_case()
        with tempfile.TemporaryDirectory() as folder:
            hp = RealConfig(
                k_train=4, epochs=2, hidden=16, layers=2,
                n_pde=32, n_ic=16, n_obs=8, log_every=1,
            )
            _model, history = train_real(case, Path(folder), hp, retrain=True)
        self.assertEqual(len(history["total"]), 2)
        self.assertTrue(all(np.isfinite(history["total"])))
        self.assertTrue(all(np.isfinite(history["pde"])))

    def test_fronts_geojson_is_lon_lat_and_keeps_a_missing_contour(self):
        import json

        import geopandas as gpd
        import pandas as pd
        from pyproj import Transformer
        from shapely.geometry import Point

        from wildfire_pinn.pinn import FireCase, write_fronts_geojson

        east, north = Transformer.from_crs("EPSG:4326", "EPSG:3310", always_xy=True).transform(-118.55, 34.05)
        times = pd.to_datetime(["2025-01-07 21:00", "2025-01-08 09:00"])
        perimeters = gpd.GeoDataFrame(
            {"dt": times, "geometry": [Point(east, north).buffer(500), Point(east, north).buffer(800)]},
            crs="EPSG:3310",
        )
        case = FireCase(
            x=np.linspace(east - 2000, east + 2000, 5),
            y=np.linspace(north + 2000, north - 2000, 5),
            spread=np.zeros((1, 5, 5), np.float32),
            wind_from_deg=np.zeros((1, 5, 5), np.float32),
            perimeters=perimeters,
            day0=pd.Timestamp("2025-01-07"),
        )
        ring = np.array([
            [east - 400, north],
            [east, north - 400],
            [east + 400, north],
            [east, north + 400],
            [east - 400, north],
        ])
        with tempfile.TemporaryDirectory() as folder:
            path = write_fronts_geojson(
                case, Path(folder) / "fronts.geojson",
                predicted={0: [ring], 1: []},
                k_train=1,
                errors={1: float("nan")},
                case_name="sample",
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["type"], "FeatureCollection")
        sources = [feature["properties"]["source"] for feature in payload["features"]]
        self.assertEqual(sources, ["observed", "predicted", "observed", "predicted"])
        lon, lat = payload["features"][1]["geometry"]["coordinates"][0]
        self.assertAlmostEqual(lon, -118.55, delta=0.05)
        self.assertAlmostEqual(lat, 34.05, delta=0.05)
        self.assertEqual(payload["features"][0]["geometry"]["type"], "Polygon")
        self.assertEqual(payload["features"][0]["properties"]["split"], "train")
        missing = payload["features"][3]
        self.assertIsNone(missing["geometry"])
        self.assertEqual(missing["properties"]["split"], "held_out")
        self.assertIsNone(missing["properties"]["front_error_m"])
        self.assertTrue(missing["properties"]["scored"])

    def test_a_second_case_keeps_its_own_files(self):
        from wildfire_pinn.cli import build_parser, config_from_args
        from wildfire_pinn.config import ProjectConfig

        palisades = ProjectConfig()
        other = ProjectConfig(name="eaton", bbox=(-118.2, 34.1, -117.85, 34.35))
        self.assertEqual(palisades.firms_csv.name, "firms_raw.csv")
        self.assertEqual(palisades.perimeters_path.name, "palisades_perimeters.geojson")
        self.assertEqual(palisades.stack_nc.name, "palisades_stack.nc")
        self.assertEqual(other.firms_csv.name, "eaton_firms.csv")
        self.assertEqual(other.gridmet_nc.name, "eaton_gridmet.nc")
        self.assertEqual(other.output_dir.name, "eaton")
        self.assertNotEqual(palisades.output_dir, other.output_dir)
        args = build_parser().parse_args([
            "--name", "eaton",
            "--bbox=-118.20,34.10,-117.85,34.35",
            "--start", "2025-01-08",
            "--end", "2025-01-15",
            "--crs", "EPSG:3310",
            "check",
        ])
        cfg = config_from_args(args)
        self.assertEqual(cfg.name, "eaton")
        self.assertEqual(cfg.date_start, "2025-01-08")
        self.assertEqual(cfg.date_end, "2025-01-15")
        self.assertEqual(cfg.bbox[0], -118.20)
        self.assertEqual(cfg.target_crs, "EPSG:3310")
        with self.assertRaises(RuntimeError):
            config_from_args(build_parser().parse_args(["--bbox", "1,2,3", "check"]))
        with self.assertRaises(RuntimeError):
            ProjectConfig(name="../secret")

    def test_firms_key_file_skips_comments(self):
        from wildfire_pinn.cli import key_from_file

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "firms_key.txt"
            path.write_text("# comment\n\n  'map-key-value'  \n", encoding="utf-8")
            self.assertEqual(key_from_file(path), "map-key-value")
            path.write_text("# only a comment\n", encoding="utf-8")
            self.assertIsNone(key_from_file(path))
            self.assertIsNone(key_from_file(Path(directory) / "missing.txt"))

    def test_kincade_split_is_chronological_and_palisades_stays_k_train(self):
        from wildfire_pinn.cli import build_parser, config_from_args
        from wildfire_pinn.config import KINCADE_BBOX, ProjectConfig
        from wildfire_pinn.pinn import resolve_time_split

        kincade = config_from_args(build_parser().parse_args(["--name", "kincade", "check"]))
        self.assertEqual(kincade.date_start, "2019-10-23")
        self.assertEqual(kincade.date_end, "2019-11-06")
        self.assertEqual(kincade.split_scheme, "chronological")
        self.assertEqual(kincade.bbox, KINCADE_BBOX)
        self.assertEqual(kincade.firms_csv.name, "kincade_firms.csv")
        west, south, east, north = KINCADE_BBOX
        self.assertLess(west, -122.780053)
        self.assertGreater(east, -122.780053)
        self.assertLess(south, 38.792458)
        self.assertGreater(north, 38.792458)
        palisades = config_from_args(build_parser().parse_args(["check"]))
        self.assertEqual(palisades.name, "palisades")
        self.assertEqual(palisades.split_scheme, "k_train")
        self.assertEqual(palisades.date_start, "2025-01-07")
        self.assertEqual(palisades.firms_csv.name, "firms_raw.csv")
        self.assertNotEqual(kincade.gridmet_nc, ProjectConfig().gridmet_nc)
        legacy = resolve_time_split(14, 10, "k_train")
        self.assertEqual(legacy.train, list(range(10)))
        self.assertEqual(legacy.test, list(range(10, 14)))
        self.assertEqual(legacy.label(10), "held_out")
        ordered = resolve_time_split(25, 10, "chronological", 0.6, 0.2)
        self.assertEqual(ordered.train, list(range(15)))
        self.assertEqual(ordered.validation, list(range(15, 20)))
        self.assertEqual(ordered.test, list(range(20, 25)))
        self.assertEqual(ordered.label(16), "validation")
        self.assertEqual(ordered.label(22), "test")
        self.assertEqual(ordered.train + ordered.validation + ordered.test, list(range(25)))


if __name__ == "__main__":
    unittest.main()
