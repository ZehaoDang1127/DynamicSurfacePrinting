"""Live-loop contracts exercised with injected observations and a virtual printer."""
import asyncio
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
import tempfile
import time
from types import SimpleNamespace
import unittest

import numpy as np
try:
    import torch
except ImportError:
    torch = None

from controller.config import fingerprint, load_config
from controller.live_print import CameraHistory, LiveSettings, inspect_inputs, run_live, settings_issues
from controller.planner import Planner
from controller.predictor_adapter import PredictorAdapter
from controller.runtime import geometry_signature
from controller.types import ControllerFault, Observation
from surface_mapper.config import AppConfig, CameraConfig
from surface_mapper.camera import PointCloudFrame

ROOT = Path(__file__).resolve().parents[1]


def fixture():
    cfg = load_config(ROOT / "configs/controller.example.json")
    cfg = replace(cfg, measurement_frame="printer", height_reference="printer_z", timing_uncertainty_s=.1,
        grid=replace(cfg.grid, z_min_mm=15., z_max_mm=25.),
        predictor=replace(cfg.predictor, kind="convlstm", checkpoint="virtual-model.pt", history=3,
            temporal_policy="linear_to_horizon", interpolation_error_mm=.01, max_frame_age_s=1., max_horizon_s=3.))
    app = AppConfig("printer", CameraConfig(serial="virtual-camera"), cfg.grid, np.eye(4))
    cfg = replace(cfg, hardware=replace(cfg.hardware, moonraker_url="http://virtual-printer",
        camera_config_sha256=fingerprint(geometry_signature(app.to_dict())), calibration_id="virtual-test",
        expires_utc=(datetime.now(timezone.utc)+timedelta(days=1)).isoformat(),
        stop_test_verified=True, exclusive_control_verified=True))
    settings = LiveSettings(initial_xyz_mm=(110., 120., 25.), exposure_latency_s=0., camera_clock_uncertainty_s=0.,
        command_start_delay_s=.05, max_schedule_error_s=.1, planning_lead_s=.1, max_segment_s=.5,
        sample_wait_s=.2, poll_s=.01, max_heating_s=1., max_runtime_s=10.,
        calibration_verified=True, model_verified=True, timing_verified=True, deposition_verified=True,
        loaded_and_primed=True, initial_pose_verified=True, path_clearance_verified=True)
    path = np.array([[110., 120.], [111., 120.], [112., 120.]])
    return cfg, app, settings, path


class VirtualPredictor:
    """Use real planning against exact static truth; learned bridge has separate tests."""
    def __init__(self, config, geometry):
        base = replace(config, predictor=replace(config.predictor, kind="persistence", checkpoint=None,
            temporal_policy="fixed", interpolation_error_mm=0.))
        self.inner = PredictorAdapter(base)
        self.model = SimpleNamespace(config=SimpleNamespace(horizon=90, expected_dt_ms=1000/30))

    @property
    def ready(self):
        return self.inner.ready

    @property
    def buffer(self):
        return self.inner.buffer

    def reset(self, reason):
        self.inner.reset(reason)

    def push(self, observation):
        return self.inner.push(observation)

    def forecast(self, target_s, now_s):
        return self.inner.forecast(target_s, now_s)


class VirtualHistory:
    def __init__(self, config, *, stale=False, fail_health=False, rising_surface=False):
        self.config, self.stale, self.fail_health = config, stale, fail_health
        self.index = 0
        self.closed = self.started = False
        self.reads = 0
        self.rising_surface = rising_surface
        self.current_reads = 0

    def start(self):
        self.started = True

    def latest(self, after_frame=None, timeout=5):
        self.reads += 1
        now = time.monotonic() - (10 if self.stale else 0)
        shape = (self.config.grid.height, self.config.grid.width)
        result = []
        for i in range(self.config.predictor.history):
            at = now - (self.config.predictor.history-i-1)/30
            self.index += 1
            result.append(Observation(self.index, np.full(shape, 20.), np.ones(shape, bool), np.ones(shape),
                self.config.geometry_id, at*1000, "virtual-clock", at, at))
        return result

    def health(self, now):
        if self.fail_health and self.reads >= 2:
            raise ControllerFault("injected camera failure during inference")

    def current(self):
        self.current_reads += 1
        now = time.monotonic() - (10 if self.stale else 0)
        shape = (self.config.grid.height, self.config.grid.width)
        height = 24.9 if self.rising_surface and self.current_reads >= 2 else 20.
        return Observation(self.index, np.full(shape, height), np.ones(shape, bool), np.ones(shape),
            self.config.geometry_id, now*1000, "virtual-clock", now, now)

    def close(self):
        self.closed = True


class VirtualPrinter:
    def __init__(self, settings, *, lost_reply=False, zero_pose=False, discover_delay=0., modern=False):
        self.settings = settings
        self.xyz = list(settings.initial_xyz_mm)
        self.temperature, self.target = 25., 0.
        self.scripts = []
        self.stops = self.eventtime = self.moves = 0
        self.closed = False
        self.lost_reply, self.zero_pose = lost_reply, zero_pose
        self.pending = None
        self.accel = 10000.
        self.smoothing_field = "minimum_cruise_ratio" if modern else "max_accel_to_decel"
        self.smoothing = .5 if modern else 5000.
        self.discover_delay = discover_delay
        self.stopped_at = None

    async def discover(self):
        if self.discover_delay:
            await asyncio.sleep(self.discover_delay)
        self.eventtime += 1
        live = list(self.xyz)
        velocity = 0.
        if self.pending:
            a, b, start, duration = self.pending
            fraction = min(max((time.monotonic()-start)/duration, 0), 1)
            live = (np.asarray(a)+(np.asarray(b)-a)*fraction).tolist()
            velocity = 1. if fraction < 1 else 0.
        if self.zero_pose:
            live = [0., 0., 0.]
        values = {"webhooks": {"state": "ready"},
            "toolhead": {"homed_axes": "xyz", "axis_minimum": [0,0,0], "axis_maximum": [350,350,350],
                         "max_velocity": 500, "max_accel": self.accel, "square_corner_velocity": 5,
                         self.smoothing_field: self.smoothing},
            "gcode_move": {"position": [*self.xyz,0], "gcode_position": [*self.xyz,0], "homing_origin": [0,0,0,0],
                "speed_factor": 1, "extrude_factor": 1, "absolute_coordinates": True, "absolute_extrude": False},
            "motion_report": {"live_position": [*live,0], "live_velocity": velocity},
            "extruder": {"temperature": self.temperature, "target": self.target},
            "heater_bed": {"temperature": 25, "target": 0}, "idle_timeout": {"state": "Ready"},
            "virtual_sdcard": {"is_active": False}, "print_stats": {"state": "standby"},
            "bed_mesh": {"profile_name": ""}, "configfile": {"settings": {
                "printer": {"max_z_velocity": 20, "max_z_accel": 500},
                "extruder": {"min_extrude_temp": 170, "max_temp": 300}}}}
        return {"printer": {"state": "ready", "software_version": "virtual", "process_id": 1},
                "query": {"eventtime": self.eventtime, "status": values}, "receipt_s": time.monotonic()}

    async def execute_gcode(self, script):
        self.scripts.append(script)
        match = re.search(r"M104 S([0-9.]+)", script)
        if match:
            self.target = float(match[1])
            self.temperature = self.target or 25.
        accel = re.search(r"SET_VELOCITY_LIMIT ACCEL=([0-9.]+)", script)
        if accel:
            self.accel = float(accel[1])
            self.smoothing = float(re.search(r"(?:ACCEL_TO_DECEL|MINIMUM_CRUISE_RATIO)=([0-9.]+)", script)[1])
        if script.startswith("G1 "):
            self.moves += 1
            target = [float(re.search(rf"{axis}([-0-9.]+)", script)[1]) for axis in "XYZ"]
            feed = float(re.search(r"F([0-9.]+)", script)[1])/60
            distance = float(np.linalg.norm(np.asarray(target)-self.xyz))
            duration = distance/feed + feed/self.accel
            self.pending = (list(self.xyz), target, time.monotonic()+self.settings.command_start_delay_s, duration)
            if self.lost_reply:
                self.xyz = target
                raise ControllerFault("injected ambiguous G-code reply")
            await asyncio.sleep(self.settings.command_start_delay_s+duration)
            self.xyz, self.pending = target, None
        return "ok"

    async def emergency_stop(self):
        self.stops += 1
        self.stopped_at = time.monotonic()
        self.target = 0.
        return "stopped"

    async def close(self):
        self.closed = True


class LiveLoopTests(unittest.IsolatedAsyncioTestCase):
    async def perform(self, *, execute=True, settings_change=None, history_options=None, printer_options=None, planner_factory=Planner):
        cfg, app, settings, path = fixture()
        settings = replace(settings, **(settings_change or {}))
        history = VirtualHistory(cfg, **(history_options or {}))
        printer = VirtualPrinter(settings, **(printer_options or {}))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/"run"
            result = await run_live(cfg, app, path, settings, output, execute=execute, history=history,
                client=printer if execute else None, predictor_factory=VirtualPredictor, planner_factory=planner_factory)
            snapshot = json.loads((output/"last_status_snapshot.json").read_text()) if (output/"last_status_snapshot.json").exists() else None
        return result, history, printer, snapshot

    async def test_complete_path_has_positive_extrusion_one_move_per_barrier(self):
        result, history, printer, _ = await self.perform()
        self.assertEqual(result["state"], "COMPLETE", result)
        self.assertEqual(printer.moves, 2)
        self.assertEqual(printer.stops, 0)
        self.assertTrue(history.closed and printer.closed)
        for script in (s for s in printer.scripts if s.startswith("G1 ")):
            self.assertEqual(script.count("G1 "), 1)
            self.assertTrue(script.endswith("M400"))
            self.assertGreater(float(re.search(r" E([0-9.]+)", script)[1]), 0.)
        self.assertTrue(printer.scripts[-1].startswith("M104 S0\nSET_VELOCITY_LIMIT"))
        self.assertEqual(printer.accel, 10000.)
        self.assertEqual(printer.xyz, [112., 120., 25.])

    async def test_shadow_never_creates_printer_or_extrudes(self):
        result, history, printer, _ = await self.perform(execute=False)
        self.assertEqual(result["state"], "PREVIEW", result)
        self.assertEqual(result["printer_commands_sent"], 0)
        self.assertEqual(printer.scripts, [])
        self.assertTrue(history.closed)

    async def test_missing_readiness_rejects_before_camera_or_writes(self):
        result, history, printer, _ = await self.perform(settings_change={"model_verified": False})
        self.assertEqual(result["state"], "FAULT")
        self.assertIn("model_verified", result["reason"])
        self.assertFalse(history.started)
        self.assertEqual(printer.scripts, [])
        self.assertEqual(printer.stops, 0)

    async def test_lost_reply_never_retries_segment_and_stops(self):
        result, _, printer, _ = await self.perform(printer_options={"lost_reply": True})
        self.assertEqual(result["state"], "FAULT")
        self.assertEqual(printer.moves, 1)
        self.assertEqual(printer.stops, 1)

    async def test_stale_observation_prevents_movement_before_heating(self):
        result, _, printer, _ = await self.perform(history_options={"stale": True})
        self.assertEqual(result["state"], "FAULT")
        self.assertIn("stale", result["reason"])
        self.assertEqual(printer.moves, 0)
        self.assertEqual(printer.stops, 0)
        self.assertEqual(printer.scripts, [])

    async def test_zero_live_position_preserved_and_rejected_before_heating(self):
        result, _, printer, sample = await self.perform(printer_options={"zero_pose": True})
        self.assertEqual(result["state"], "FAULT")
        self.assertEqual(printer.scripts, [])
        self.assertEqual(sample["report"]["query"]["status"]["motion_report"]["live_position"][:3], [0,0,0])

    async def test_monitor_stops_during_slow_planning(self):
        class SlowPlanner(Planner):
            def plan(self, *args, **kwargs):
                time.sleep(.15)
                return super().plan(*args, **kwargs)
        result, _, printer, _ = await self.perform(history_options={"fail_health": True}, planner_factory=SlowPlanner)
        self.assertEqual(result["state"], "FAULT")
        self.assertEqual(printer.moves, 0)
        self.assertEqual(printer.stops, 1)

    async def test_rising_surface_is_detected_while_heating(self):
        result, _, printer, _ = await self.perform(history_options={"rising_surface": True})
        self.assertEqual(result["state"], "FAULT")
        self.assertIn("clearance", result["reason"])
        self.assertEqual(printer.moves, 0)
        self.assertEqual(printer.stops, 1)

    async def test_delayed_status_queries_do_not_use_obsolete_stationary_pose(self):
        result, _, printer, _ = await self.perform(settings_change={"planning_lead_s": .3},
            printer_options={"discover_delay": .08})
        self.assertEqual(result["state"], "COMPLETE", result)
        self.assertEqual(printer.stops, 0)

    async def test_modern_cruise_interface_is_applied_and_restored(self):
        result, _, printer, _ = await self.perform(printer_options={"modern": True})
        self.assertEqual(result["state"], "COMPLETE", result)
        self.assertTrue(any("MINIMUM_CRUISE_RATIO=0.000000000" in s for s in printer.scripts))
        self.assertEqual(printer.smoothing, .5)

    async def test_camera_watchdog_runs_while_network_poll_is_waiting(self):
        cfg, app, settings, path = fixture()
        settings = replace(settings, planning_lead_s=.4)
        history = VirtualHistory(cfg, fail_health=True)
        printer = VirtualPrinter(settings)
        original_discover = printer.discover
        blocked_at = []
        async def slow_discover():
            if history.reads >= 2:
                blocked_at.append(time.monotonic())
                await asyncio.sleep(.4)
            return await original_discover()
        printer.discover = slow_discover
        class SlowPlanner(Planner):
            def plan(self, *args, **kwargs):
                time.sleep(.2)
                return super().plan(*args, **kwargs)
        with tempfile.TemporaryDirectory() as directory:
            result = await run_live(cfg, app, path, settings, Path(directory)/"run", execute=True,
                history=history, client=printer, predictor_factory=VirtualPredictor, planner_factory=SlowPlanner)
        self.assertEqual(result["state"], "FAULT")
        self.assertEqual(printer.stops, 1)
        self.assertEqual(printer.moves, 0)
        if blocked_at:
            self.assertLess(printer.stopped_at - blocked_at[0], .2)

    async def test_close_failure_preserves_report(self):
        cfg, app, settings, path = fixture()
        printer = VirtualPrinter(settings)
        async def fail_close():
            raise OSError("injected close failure")
        printer.close = fail_close
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)/"run"
            result = await run_live(cfg, app, path, replace(settings, model_verified=False), out,
                execute=True, history=VirtualHistory(cfg), client=printer, predictor_factory=VirtualPredictor)
            saved = json.loads((out/"report.json").read_text())
        self.assertEqual(result["cleanup_error"], "injected close failure")
        self.assertEqual(saved["cleanup_error"], "injected close failure")

    @unittest.skipIf(torch is None, "optional PyTorch dependency is not installed")
    async def test_real_convlstm_pointcloud_transform_planner_and_virtual_extrusion(self):
        from prediction.config import PredictionConfig
        from prediction.data import Normalization
        from prediction.model import ConvLSTMPredictor
        from prediction.training import CHECKPOINT_SCHEMA_VERSION
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous_threads)
        cfg, app, settings, path = fixture()
        transform = np.eye(4)
        transform[:3, 3] = [10., 20., 5.]
        app = replace(app, camera_to_output=transform)
        cfg = replace(cfg, hardware=replace(cfg.hardware,
            camera_config_sha256=fingerprint(geometry_signature(app.to_dict()))))
        settings = replace(settings, planning_lead_s=.2, camera_clock_uncertainty_s=.04)
        x, y = np.meshgrid(cfg.grid.x_centers_mm(), cfg.grid.y_centers_mm())
        points = (np.column_stack((x.ravel(), y.ravel(), np.full(x.size, 20.))) - transform[:3, 3]) / 1000
        class Points:
            device_info = {"serial": "virtual-camera"}
            def start(self):
                self.origin = time.perf_counter()
                self.index = 0
            def capture(self):
                self.index += 1
                time.sleep(max(0., self.origin + self.index/30 - time.perf_counter()))
                return PointCloudFrame(self.index, self.index*1000/30, time.monotonic_ns(), "test-camera-clock", points)
            def stop(self):
                pass
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = replace(cfg, predictor=replace(cfg.predictor, checkpoint=str(root/"model.pt"), device="cpu"))
            model_config = PredictionConfig(history=3, horizon=90, hidden_channels=(2,))
            model = ConvLSTMPredictor(model_config)
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.zero_()
            torch.save({"schema_version": CHECKPOINT_SCHEMA_VERSION, "state_dict": model.state_dict(),
                "config": model_config.to_dict(), "normalization": Normalization(20., 1.).to_dict(),
                "geometry": geometry_signature(app.to_dict()), "shape": [cfg.grid.height, cfg.grid.width],
                "train_sessions": ["deterministic-test/train"], "validation_sessions": ["deterministic-test/validation"],
                "test_sessions": ["deterministic-test/test"]}, cfg.predictor.checkpoint)
            history = CameraHistory(app, cfg, settings, source=Points())
            printer = VirtualPrinter(settings)
            result = await run_live(cfg, app, path, settings, root/"run", execute=True, history=history, client=printer)
            self.assertEqual(result["state"], "COMPLETE", result)
            self.assertEqual(printer.moves, 2)
            self.assertTrue((root/"run/convlstm_forecast_000001.npz").is_file())
            with np.load(root/"run/convlstm_forecast_000001.npz") as forecast:
                self.assertTrue(np.allclose(forecast["prediction_mm"], 20.))
            events = [json.loads(line) for line in (root/"run/events.jsonl").read_text().splitlines()]
            frames = [item["frame_id"] for item in events if item["kind"] == "observation"]
            self.assertEqual(frames, sorted(set(frames)))


class LiveInspectionTests(unittest.TestCase):
    def test_gap_budget_and_unknown_measurements_block_hardware(self):
        cfg, app, settings, path = fixture()
        issues = settings_issues(replace(settings, exposure_latency_s=None), replace(cfg, desired_gap_mm=.25), path, hardware=True)
        self.assertTrue(any("exposure_latency_s" in item for item in issues))
        self.assertTrue(any("smaller than nozzle gap" in item for item in issues))

    def test_geometry_mismatch_and_short_model_horizon_are_reported(self):
        cfg, app, settings, path = fixture()
        wrong = replace(app, frame_name="camera")
        result, _, _ = inspect_inputs(cfg, wrong, path, settings, predictor_factory=VirtualPredictor)
        self.assertEqual(result["state"], "BLOCKED")
        class ShortPredictor(VirtualPredictor):
            def __init__(self, *args):
                super().__init__(*args)
                self.model.config.horizon = 1
        result, _, _ = inspect_inputs(cfg, app, path, settings, predictor_factory=ShortPredictor)
        self.assertTrue(any("horizon" in issue for issue in result["issues"]))


if __name__ == "__main__":
    unittest.main()
