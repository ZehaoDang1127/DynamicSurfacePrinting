"""Bounded temporal interpolation with deterministic, non-trained test checkpoints."""

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from controller.config import PredictorConfig, from_dict, load_config
from controller.logging import RunLog
from controller.planner import Planner
from controller.predictor_adapter import PredictorAdapter
from controller.runtime import geometry_signature
from controller.shadow import evaluate_shadow
from controller.types import ControllerFault, Observation


ROOT = Path(__file__).resolve().parents[1]


class InterpolationConfigurationTests(unittest.TestCase):
    def test_existing_configs_keep_strict_fixed_policy(self):
        config = load_config(ROOT / "configs/controller.example.json")
        self.assertEqual(config.predictor.temporal_policy, "fixed")
        self.assertEqual(config.predictor.interpolation_error_mm, 0)
        self.assertEqual(config, from_dict(config.to_dict()))

    def test_bridge_requires_model_and_explicit_positive_error(self):
        with self.assertRaisesRegex(ValueError, "requires the ConvLSTM"):
            PredictorConfig(temporal_policy="linear_to_horizon", interpolation_error_mm=.05)
        with self.assertRaisesRegex(ValueError, "positive explicit"):
            PredictorConfig(kind="convlstm", checkpoint="test.pt", temporal_policy="linear_to_horizon")
        for value in (-1, float("nan"), True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                PredictorConfig(interpolation_error_mm=value)
        with self.assertRaisesRegex(ValueError, "temporal_policy"):
            PredictorConfig(temporal_policy="extrapolate")

    def test_interpolation_allowance_is_in_controller_gap_budget(self):
        config = load_config(ROOT / "configs/controller.example.json")
        predictor = replace(config.predictor, kind="convlstm", checkpoint="test.pt",
                            temporal_policy="linear_to_horizon", interpolation_error_mm=.05)
        bridge = replace(config, predictor=predictor)
        self.assertAlmostEqual(bridge.provisional_error_mm, config.provisional_error_mm + .05)
        with self.assertRaisesRegex(ValueError, "interpolation.*gap budget"):
            replace(config, predictor=replace(predictor, interpolation_error_mm=.21))


@unittest.skipIf(torch is None, "optional prediction dependencies are not installed")
class ConvLSTMControllerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        base = load_config(ROOT / "configs/controller.example.json")
        self.config = replace(base, substrate_height_mm=20., grid=replace(base.grid, min_points_per_cell=2),
            predictor=replace(base.predictor, kind="convlstm", history=3, checkpoint=str(self.root / "model.pt"),
                              device="cpu", temporal_policy="linear_to_horizon", interpolation_error_mm=.05))
        self.geometry = geometry_signature({"frame_name": self.config.measurement_frame,
            "camera": {"width": 640, "height": 480, "fps": 30}, "grid": self.config.to_dict()["grid"],
            "camera_to_output": np.eye(4).tolist()})
        self.write_checkpoint()

    def write_checkpoint(self, *, expected_dt_ms=1000 / 30, horizon=30):
        from prediction.config import PredictionConfig
        from prediction.data import Normalization
        from prediction.model import ConvLSTMPredictor
        from prediction.training import CHECKPOINT_SCHEMA_VERSION

        settings = PredictionConfig(history=3, horizon=horizon, expected_dt_ms=expected_dt_ms,
                                    hidden_channels=(2,), min_valid_fraction=.1)
        model = ConvLSTMPredictor(settings)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.head.bias.fill_(.1)
        torch.save({"schema_version": CHECKPOINT_SCHEMA_VERSION, "state_dict": model.state_dict(),
                    "config": settings.to_dict(), "normalization": Normalization(0., 1.).to_dict(),
                    "geometry": self.geometry, "shape": [self.config.grid.height, self.config.grid.width],
                    "train_sessions": ["synthetic-test-only/train"],
                    "validation_sessions": ["synthetic-test-only/validation"],
                    "test_sessions": ["synthetic-test-only/test"]}, self.config.predictor.checkpoint)

    def observation(self, index, height=0.):
        shape = (self.config.grid.height, self.config.grid.width)
        at_s = index / 30
        return Observation(index, np.full(shape, height), np.ones(shape, bool), np.full(shape, 3),
                           self.config.geometry_id, at_s * 1000, "test_camera", at_s, at_s)

    def adapter(self, config=None, heights=(0., 0., 0.)):
        adapter = PredictorAdapter(config or self.config, self.geometry)
        for index, height in enumerate(heights):
            adapter.push(self.observation(index, height))
        self.assertTrue(adapter.ready)
        return adapter

    def test_bridge_matches_measurement_endpoint_and_midpoint(self):
        adapter = self.adapter(heights=(0., .1, .2))
        start = adapter.buffer[-1].observed_s
        current = adapter.forecast(start, start)
        halfway = adapter.forecast(start + .5, start)
        endpoint = adapter.forecast(start + 1., start)
        np.testing.assert_allclose(current.height_mm, .2)
        np.testing.assert_allclose(halfway.height_mm, .15, atol=1e-7)
        np.testing.assert_allclose(endpoint.height_mm, adapter.fixed.prediction_mm, atol=1e-7)
        self.assertEqual(endpoint.model_id, "convlstm+linear_temporal_interpolation")
        self.assertAlmostEqual(endpoint.error_bound_mm, .15)
        self.assertAlmostEqual(adapter.model_horizon_s, 1.)

    def test_no_extrapolation_and_controller_horizon_still_limits_bridge(self):
        wide = replace(self.config, predictor=replace(self.config.predictor, max_horizon_s=2.))
        adapter = self.adapter(wide)
        start = adapter.buffer[-1].observed_s
        with self.assertRaisesRegex(ControllerFault, "extrapolate"):
            adapter.forecast(start + 1.01, start)
        with self.assertRaisesRegex(ControllerFault, "horizon"):
            adapter.forecast(start - .01, start)
        narrow = replace(self.config, predictor=replace(self.config.predictor, max_horizon_s=.5))
        adapter = self.adapter(narrow)
        with self.assertRaisesRegex(ControllerFault, "horizon"):
            adapter.forecast(start + .6, start)

    def test_fixed_policy_stays_strict_and_shadow_refuses_it(self):
        config = replace(self.config, predictor=replace(self.config.predictor, temporal_policy="fixed",
                                                       interpolation_error_mm=0.))
        adapter = self.adapter(config)
        start = adapter.buffer[-1].observed_s
        self.assertEqual(adapter.forecast(start + 1., start).model_id, "convlstm")
        with self.assertRaisesRegex(ControllerFault, "fixed horizon"):
            adapter.forecast(start + .5, start)
        with self.assertRaisesRegex(ControllerFault, "fixed ConvLSTM"):
            evaluate_shadow(config, [[110, 120], [111, 120]], [], self.geometry, None)

    def test_complete_history_and_point_counts_mask_network_and_bridge(self):
        adapter = PredictorAdapter(self.config, self.geometry)
        for index in range(3):
            observation = self.observation(index)
            valid, counts, height = observation.valid.copy(), observation.point_count.copy(), observation.height_mm.copy()
            if index == 0:
                valid[0, 0] = False
            if index == 1:
                height[0, 1] = float("nan")
            if index == 2:
                counts[0, 2] = 1
            adapter.push(replace(observation, valid=valid, point_count=counts, height_mm=height))
        self.assertFalse(adapter.model.buffer[-1][1][0, 2])
        start = adapter.buffer[-1].observed_s
        forecast = adapter.forecast(start + .5, start)
        self.assertFalse(forecast.support[0, :3].any())
        self.assertTrue(np.isnan(forecast.height_mm[0, :3]).all())
        self.assertTrue(forecast.support[1:, :].all())

    def test_geometry_and_checkpoint_cadence_must_match(self):
        wrong_geometry = {**self.geometry, "frame_name": "other_frame"}
        with self.assertRaisesRegex(ValueError, "geometry differs"):
            PredictorAdapter(self.config, wrong_geometry)
        self.write_checkpoint(expected_dt_ms=40., horizon=25)
        with self.assertRaisesRegex(ValueError, "cadence differs"):
            PredictorAdapter(self.config, self.geometry)

    def test_real_checkpoint_bridge_drives_planner_with_extrusion(self):
        adapter = self.adapter()
        start = adapter.buffer[-1].observed_s
        trajectory = Planner(self.config).plan(adapter, (110., 120., 25.), [(111., 120.)],
                                               start, start, 0, extrude=True)
        move = trajectory.moves[0]
        forecast_height = .1 * (move.arrival_s - start)
        self.assertAlmostEqual(move.xyz[2], 25 + forecast_height, places=5)
        self.assertGreater(move.extrusion_mm, 0)
        self.assertLess(move.arrival_s - start, adapter.model_horizon_s)

    def test_explicit_bridge_is_available_to_shadow_without_motion(self):
        log = RunLog(self.root / "shadow", self.config)
        try:
            report = evaluate_shadow(self.config, [[110., 120.], [111., 120.]],
                [self.observation(index) for index in range(3)], self.geometry, log, clock=lambda: 2 / 30)
        finally:
            log.finish({"test_only": True})
        self.assertEqual(report["state"], "READY", report)
        self.assertEqual(report["accepted_proposals"], 1)
        self.assertEqual(report["temporal_policy"], "linear_to_horizon")
        self.assertFalse(report["motion_allowed"])
        self.assertEqual(report["printer_commands_sent"], 0)


if __name__ == "__main__":
    unittest.main()
