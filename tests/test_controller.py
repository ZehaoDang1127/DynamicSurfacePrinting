"""Acceptance tests for controller geometry, timing, extrusion and fault contracts."""

import asyncio
from dataclasses import asdict, replace
from pathlib import Path
import tempfile
import unittest

import numpy as np

from controller.clock_sync import ClockMapper, network_offset_bounds
from controller.config import from_dict, load_config
from controller.cold_run import cold_preview
from controller.path import subdivide
from controller.planner import Planner
from controller.predictor_adapter import PredictorAdapter
from controller.runtime import recording_observations, run_offline
from controller.simulator import MovingPlane
from controller.supervisor import Supervisor
from controller.transport import FakePrinter, gcode_for
from controller.types import ControllerFault, State


ROOT = Path(__file__).resolve().parents[1]


def config():
    return replace(load_config(ROOT / "configs/controller.example.json"), substrate_height_mm=20.0)


def predictor(cfg, scenario="static-plane"):
    source = MovingPlane(cfg, scenario)
    adapter = PredictorAdapter(cfg)
    for i in range(cfg.predictor.history):
        adapter.push(source.observation(i, i/30))
    return adapter


def trajectory(cfg, adapter=None, extrude=False):
    adapter = adapter or predictor(cfg)
    now = adapter.buffer[-1].observed_s
    forecast = adapter.forecast(now, now)
    from controller.planner import sample_map
    z = cfg.surface_z(sample_map(forecast.height_mm, forecast.support, cfg.grid, (110, 120)))+cfg.desired_gap_mm
    return Planner(cfg).plan(adapter, (110., 120., z), [(111, 120), (112, 120)], now, now, 0, extrude=extrude)


class ConfigurationTests(unittest.TestCase):
    def test_unknown_datum_gates_planning_and_cold_motion(self):
        cfg = replace(config(), substrate_height_mm=None)
        with self.assertRaisesRegex(ControllerFault, "substrate_height_mm"):
            trajectory(cfg)
        with self.assertRaisesRegex(ControllerFault, "substrate_height_mm"):
            cold_preview(cfg, np.array([[110, 120], [111, 120]]))

    def test_absolute_height_does_not_add_datum(self):
        cfg = replace(config(), height_reference="printer_z")
        self.assertEqual(cfg.surface_z(23), 23)
        cfg = replace(cfg, substrate_height_mm=None)
        self.assertEqual(cfg.surface_z(23), 23)

    def test_invalid_units_nan_and_unknown_fields_rejected(self):
        raw = config().to_dict()
        for changes in ({"units": "metres"}, {"desired_gap_mm": float("nan")}, {"made_up": 1}):
            with self.assertRaises(ValueError):
                from_dict({**raw, **changes})

    def test_machine_bounds_and_expired_calibration(self):
        cfg = config()
        with self.assertRaises(ControllerFault):
            cfg.require_cold_hardware()
        hardware = replace(cfg.hardware, backend="moonraker", moonraker_url="http://printer:7125", calibration_id="test",
                           camera_config_sha256="abc", expires_utc="2000-01-01T00:00:00Z",
                           fixture_top_z_mm=20, stationary_fixture_verified=True,
                           stop_test_verified=True, exclusive_control_verified=True)
        with self.assertRaisesRegex(ControllerFault, "expired"):
            replace(cfg, hardware=hardware).require_cold_hardware()


class ClockTests(unittest.TestCase):
    def test_affine_offset_drift_and_rtt(self):
        clock = ClockMapper("camera", 1000, 10, scale=1.0001, uncertainty_s=.002)
        self.assertAlmostEqual(clock.map(2000, "camera", 11.003, 1), 11.0001)
        self.assertEqual(network_offset_bounds(1., 1.1, 4.), (2.9, 3.))

    def test_discontinuities_require_new_mapping(self):
        clock = ClockMapper("camera", 0, 0)
        clock.map(10, "camera", .01, 1)
        for args in ((10, "camera", .01, 2), (20, "new-domain", .02, 2), (20, "camera", 1., 2)):
            with self.assertRaises(ControllerFault):
                clock.map(*args)


class PlannerTests(unittest.TestCase):
    def test_static_plane_and_feed_units(self):
        plan = trajectory(config(), extrude=True)
        self.assertEqual(plan.start_xyz, (110., 120., 25.))
        self.assertTrue(all(m.xyz[2] == 25 and m.extrusion_mm > 0 for m in plan.moves))
        script = gcode_for(plan, cold=False)
        self.assertIn("F300.000000", script)
        with self.assertRaises(ControllerFault):
            gcode_for(plan, cold=True)

    def test_vertical_motion_adds_no_filament(self):
        cfg = config()
        stationary = trajectory(cfg, extrude=True)
        moving = trajectory(cfg, predictor(cfg, "moving-plane"), extrude=True)
        self.assertGreater(moving.moves[0].xyz[2], stationary.moves[0].xyz[2])
        self.assertAlmostEqual(moving.moves[0].extrusion_mm, stationary.moves[0].extrusion_mm, places=7)

    def test_slope_adds_surface_relative_length(self):
        cfg = config()
        plan = trajectory(cfg, predictor(cfg, "sloped-plane"), extrude=True)
        self.assertAlmostEqual(plan.moves[0].surface_length_mm, np.hypot(1, .05), places=5)
        self.assertGreater(plan.moves[0].xyz[2], plan.start_xyz[2])

    def test_missing_corridor_cell_rejected_even_with_high_global_coverage(self):
        cfg = config()
        source, adapter = MovingPlane(cfg), PredictorAdapter(cfg)
        for i in range(5):
            obs = source.observation(i, i/30)
            # Footprint cell outside centre-line interpolation still matters.
            mask = obs.valid.copy()
            mask[9, 4] = False
            adapter.push(replace(obs, valid=mask))
        with self.assertRaises(ControllerFault):
            trajectory(cfg, adapter)

    def test_horizon_staleness_and_unreachable_correction(self):
        cfg = config()
        adapter = predictor(cfg)
        with self.assertRaisesRegex(ControllerFault, "stale"):
            adapter.forecast(.3, .3)
        with self.assertRaisesRegex(ControllerFault, "horizon"):
            adapter.forecast(2, .13333333333333333)
        with self.assertRaises(ControllerFault):
            Planner(cfg).plan(adapter, (110, 120, 28), [(111, 120)], 4/30, 4/30, 0)

    def test_workspace_slope_flow_and_z_limits(self):
        cfg = config()
        with self.assertRaisesRegex(ControllerFault, "slope"):
            steep = replace(cfg, limits=replace(cfg.limits, max_slope=.01))
            trajectory(steep, predictor(steep, "sloped-plane"))
        with self.assertRaises(ControllerFault):
            bounded = replace(cfg, limits=replace(cfg.limits, xyz_max_mm=(350, 350, 24)))
            trajectory(bounded)
        limited = replace(cfg, limits=replace(cfg.limits, max_flow_mm3_s=.2))
        plan = trajectory(limited, extrude=True)
        move = plan.moves[0]
        self.assertLessEqual(move.feed_mm_s * limited.bead_area_mm2, .2+1e-8)

    def test_history_resets_on_gap_and_frame_restart(self):
        cfg = config()
        adapter = predictor(cfg)
        source = MovingPlane(cfg)
        self.assertFalse(adapter.push(source.observation(5, 1.0)))
        self.assertEqual(adapter.reset_reason, "frame/timestamp discontinuity")
        self.assertFalse(adapter.push(source.observation(0, 1.01)))
        self.assertEqual(len(adapter.buffer), 1)


class TransportTests(unittest.IsolatedAsyncioTestCase):
    async def test_duplicate_order_and_buffer_bounds(self):
        cfg = config()
        plan = trajectory(cfg)
        printer = FakePrinter(1, cfg.config_id)
        await printer.submit(plan, plan.start_s)
        with self.assertRaisesRegex(ControllerFault, "duplicate"):
            await printer.submit(plan, plan.start_s)
        short = FakePrinter(.1, cfg.config_id)
        with self.assertRaisesRegex(ControllerFault, "buffer"):
            await short.submit(plan, plan.start_s)

    async def test_lost_reply_and_partial_execution_never_retry(self):
        cfg = config()
        plan = trajectory(cfg)
        for kind in ("partial", "lost-reply"):
            printer = FakePrinter(1, cfg.config_id, fail_on_segment=0, failure=kind)
            with self.assertRaises(ControllerFault):
                await printer.submit(plan, plan.start_s)
            self.assertTrue(printer.uncertain)
            self.assertEqual(len(printer.executed), 1)
            with self.assertRaisesRegex(ControllerFault, "uncertain"):
                await printer.submit(plan, plan.start_s)
            supervisor = Supervisor()
            supervisor.transition(State.READY)
            supervisor.transition(State.WARMUP)
            await supervisor.fault("uncertain", printer)
            self.assertEqual(supervisor.state, State.FAULT)
            self.assertTrue(printer.stopped)
            with self.assertRaises(ControllerFault):
                supervisor.transition(State.READY)

    async def test_concurrent_faults_share_one_stop_and_stay_latched(self):
        cfg = config()
        printer = FakePrinter(1, cfg.config_id)
        supervisor = Supervisor()
        supervisor.transition(State.READY)
        supervisor.transition(State.WARMUP)
        await asyncio.gather(supervisor.fault("network failed", printer), supervisor.fault("monitor failed", printer))
        self.assertEqual(supervisor.state, State.FAULT)
        self.assertEqual(supervisor.reason, "network failed")

    async def test_end_to_end_and_saved_run_replay(self):
        cfg = config()
        source = MovingPlane(cfg)
        path = subdivide([(110, 120), (112, 120)], 1)
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "simulation"
            result = await run_offline(cfg, path, out,
                (source.observation(i, i/30) for i in range(80)), truth=source, allow_extrusion=True)
            self.assertTrue(result["path_complete"], result)
            self.assertTrue(result["logging_complete"], result)
            self.assertLess(result["absolute_gap_error_mm"]["max"], .05)
            observations, geometry = recording_observations(cfg, out)
            replay = await run_offline(cfg, path, Path(directory)/"replay", observations, model_geometry=geometry)
            self.assertTrue(replay["path_complete"], replay)

    async def test_dropout_fault_is_latched(self):
        cfg = config()
        source = MovingPlane(cfg)
        path = subdivide([(110, 120), (114, 120)], 1)
        with tempfile.TemporaryDirectory() as directory:
            result = await run_offline(cfg, path, Path(directory)/"run",
                (source.observation(i, i/30, dropout=i >= 10) for i in range(80)), truth=source)
            self.assertEqual(result["state"], "FAULT")
            self.assertFalse(result["path_complete"])


if __name__ == "__main__":
    unittest.main()
