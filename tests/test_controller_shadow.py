"""Live integration checks using injected point clouds; no physical devices."""
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np

from controller.cli import parser
from controller.config import fingerprint, load_config
from controller.logging import RunLog
from controller.path import subdivide
from controller.runtime import geometry_signature
from controller.shadow import evaluate_shadow, run_shadow
from controller.types import ControllerFault, Observation
from surface_mapper.camera import PointCloudFrame

ROOT = Path(__file__).resolve().parents[1]


class PointCloudSource:
    def __init__(self, grid, *, hole=False, fail=False):
        x, y = np.meshgrid(grid.x_centers_mm(), grid.y_centers_mm())
        keep = np.ones(x.shape, bool)
        if hole:
            keep[9:12, 4:7] = False  # The path's initial interpolation/footprint.
        self.points = np.column_stack((x[keep], y[keep], np.full(keep.sum(), 149.8))) / 1000
        self.device_info = {"name": "injected test source", "serial": "synthetic_test"}
        self.fail, self.index, self.stopped, self.started = fail, 0, False, False

    def start(self):
        self.started = True
        self.origin = time.monotonic()

    def capture(self):
        # Windows monotonic timestamps can have a ~15.6 ms tick. Keep injected
        # host/camera times strictly increasing on that platform too.
        time.sleep(.025)
        self.index += 1
        if self.fail and self.index == 3:
            raise RuntimeError("injected camera disconnect")
        now = time.monotonic_ns()
        return PointCloudFrame(self.index, (now / 1e9 - self.origin) * 1000,
                               now, "test_clock", self.points)

    def stop(self):
        self.stopped = True


class ShadowTests(unittest.TestCase):
    def setup_run(self, directory, **source_options):
        cfg = load_config(ROOT / "configs/controller.example.json")
        cfg = replace(cfg, measurement_frame="printer", height_reference="printer_z",
                      grid=replace(cfg.grid, z_min_mm=140., z_max_mm=160.),
                      predictor=replace(cfg.predictor, max_frame_age_s=.25))
        app = {"frame_name": "printer", "camera": {"width": 640, "height": 480, "fps": 30},
               "grid": cfg.to_dict()["grid"], "camera_to_output": np.eye(4).tolist()}
        cfg = replace(cfg, hardware=replace(cfg.hardware,
                      camera_config_sha256=fingerprint(geometry_signature(app))))
        camera = Path(directory) / "camera.json"
        camera.write_text(json.dumps(app))
        source = PointCloudSource(cfg.grid, **source_options)
        return cfg, camera, source

    def test_camera_prediction_planning_without_transport(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg, camera, source = self.setup_run(directory)
            out = Path(directory) / "run"
            with patch("controller.connection.create_printer_client", side_effect=AssertionError("no printer allowed")):
                result = run_shadow(cfg, camera, subdivide([(110, 120), (115, 120)], 1),
                                    out, frames=12, source=source, progress=None)
            self.assertEqual(result["state"], "READY", result)
            self.assertGreater(result["accepted_proposals"], 0)
            self.assertTrue(result["logging_complete"])
            self.assertFalse(result["motion_allowed"])
            self.assertTrue(source.stopped)
            candidate = json.loads((out / "candidate_preview.json").read_text())["proposal"]
            self.assertAlmostEqual(candidate["moves"][0]["xyz"][2], 154.8, places=4)
            self.assertTrue(all(move["extrusion_mm"] == 0 for move in candidate["moves"]))
            self.assertTrue((out / "latest_forecast.npz").exists())

    def test_missing_depth_support_blocks_proposals(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg, camera, source = self.setup_run(directory, hole=True)
            result = run_shadow(cfg, camera, subdivide([(110, 120), (115, 120)], 1),
                                Path(directory) / "run", frames=10, source=source, progress=None)
            self.assertEqual(result["state"], "BLOCKED", result)
            self.assertEqual(result["accepted_proposals"], 0)
            self.assertGreater(result["rejected_proposals"], 0)
            self.assertFalse(result["motion_allowed"])

    def test_camera_disconnect_closes_and_saves_fault(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg, camera, source = self.setup_run(directory, fail=True)
            result = run_shadow(cfg, camera, subdivide([(110, 120), (115, 120)], 1),
                                Path(directory) / "run", frames=10, source=source, progress=None)
            self.assertEqual(result["state"], "FAULT")
            self.assertIn("disconnect", result["reason"])
            self.assertTrue(source.stopped)
            self.assertTrue(result["logging_complete"])

    def test_changed_transform_fails_before_camera_start(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg, camera, source = self.setup_run(directory)
            app = json.loads(camera.read_text())
            app["camera_to_output"][0][3] = 1.
            camera.write_text(json.dumps(app))
            with self.assertRaisesRegex(ControllerFault, "hash differs"):
                run_shadow(cfg, camera, [(110, 120), (115, 120)], Path(directory) / "run",
                           frames=10, source=source, progress=None)
            self.assertFalse(source.started)

    def test_shadow_has_no_execute_option(self):
        with patch("sys.stderr", new_callable=io.StringIO), self.assertRaises(SystemExit):
            parser().parse_args(["shadow", "--camera-config", "camera.json", "--output", "run", "--execute"])

    def test_small_clock_lead_does_not_request_forecast_in_the_past(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg, camera, _ = self.setup_run(directory)
            shape = (cfg.grid.height, cfg.grid.width)
            host = [0.]
            def observations():
                for i in range(8):
                    observed = 10. + i / 30
                    host[0] = observed - .01
                    yield Observation(i, np.full(shape, 149.8), np.ones(shape, bool),
                        np.full(shape, 4), cfg.geometry_id, observed * 1000, "test_clock",
                        host[0], observed, .1)
            log = RunLog(Path(directory) / "run", cfg)
            result = evaluate_shadow(cfg, subdivide([(110, 120), (115, 120)], 1),
                observations(), geometry_signature(json.loads(camera.read_text())),
                log, clock=lambda: host[0])
            result = log.finish(result)
            self.assertEqual(result["state"], "READY", result)
            self.assertEqual(result["rejected_proposals"], 0)


if __name__ == "__main__":
    unittest.main()
