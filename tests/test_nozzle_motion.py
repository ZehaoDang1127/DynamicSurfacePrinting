"""Nozzle positioning tests; no network or physical printer is used."""

from copy import deepcopy
from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from controller.config import load_config
from controller.nozzle import build_plan, execute_plan, main, write_preview


ROOT = Path(__file__).resolve().parents[1]


def fixture():
    base = load_config(ROOT / "configs/controller.example.json")
    config = replace(base, hardware=replace(base.hardware, fixture_top_z_mm=20,
                     stationary_fixture_verified=True, stop_test_verified=True, exclusive_control_verified=True))
    raw = {"schema_version": 1, "coordinate_frame": "printer", "start_xyz_mm": [110, 120, 30],
           "travel_z_mm": 35, "xy_speed_mm_s": 5, "z_speed_mm_s": 1, "dwell_s": .5,
           "waypoints": [{"id": "P1", "xyz_mm": [100, 100, 25]},
                         {"id": "P2", "xyz_mm": [140, 100, 25]}], "return_to_start": False}
    return config, raw


class FakeClient:
    def __init__(self, start=(110, 120, 30), fail_at=None):
        self.xyz = list(start)
        self.commands = []
        self.stop_requested = False
        self.closed = False
        self.absolute = False
        self.fail_at = fail_at
        self.eventtime = 0
        self.overrides = {}

    async def discover(self):
        self.eventtime += 1
        position = [*self.xyz, 0]
        values = {
            "webhooks": {"state": "ready"},
            "toolhead": {"homed_axes": "xyz", "axis_minimum": [0, 0, 0, 0],
                         "axis_maximum": [350, 350, 350, 0], "max_velocity": 500, "max_accel": 10000},
            "gcode_move": {"homing_origin": [0, 0, 0, 0], "position": position,
                           "gcode_position": position, "speed_factor": 1, "extrude_factor": 1,
                           "absolute_coordinates": self.absolute},
            "motion_report": {"live_position": position, "live_velocity": 0},
            "extruder": {"target": 0, "temperature": 25}, "heater_bed": {"target": 0, "temperature": 25},
            "idle_timeout": {"state": "Ready"}, "virtual_sdcard": {"is_active": False},
            "bed_mesh": {"profile_name": ""},
            "configfile": {"settings": {"printer": {"max_z_velocity": 20, "max_z_accel": 500}}}}
        for (name, key), value in self.overrides.items():
            values[name][key] = value
        return {"printer": {"state": "ready", "software_version": "fake", "process_id": 1},
                "query": {"status": values, "eventtime": self.eventtime}}

    async def execute_gcode(self, script):
        self.commands.append(script)
        if script == "G90":
            self.absolute = True
        for line in script.splitlines():
            if line.startswith("G1 "):
                self.xyz = [float(re.search(rf"{axis}([-\d.]+)", line)[1]) for axis in "XYZ"]
        if self.fail_at == len(self.commands):
            raise RuntimeError("reply lost after accepting command")
        return "ok"

    async def emergency_stop(self):
        self.stop_requested = True

    async def close(self):
        self.closed = True


class PlanTests(unittest.TestCase):
    def test_lift_travel_approach_and_bounded_requests(self):
        config, raw = fixture()
        plan = build_plan(raw, config)
        previous = np.array(plan["start_xyz_mm"])
        for block in plan["blocks"]:
            current = np.array(block["xyz_mm"])
            delta = current - previous
            if np.linalg.norm(delta[:2]) > 1e-8:
                self.assertAlmostEqual(previous[2], raw["travel_z_mm"])
                self.assertAlmostEqual(current[2], raw["travel_z_mm"])
            else:
                np.testing.assert_allclose(current[:2], previous[:2])
            self.assertLessEqual(block["estimated_duration_s"], config.hardware.request_timeout_s/2 + 1e-8)
            self.assertGreaterEqual(current[2], 25)
            self.assertNotRegex(block["gcode"], r"\b(?:E|M10[49]|G28)\b")
            previous = current
        self.assertEqual([b["waypoint"] for b in plan["blocks"] if b["phase"] == "arrived"], ["P1", "P2"])
        np.testing.assert_allclose(plan["end_xyz_mm"], [140, 100, 35])

    def test_return_to_start(self):
        config, raw = fixture()
        raw["return_to_start"] = True
        self.assertEqual(build_plan(raw, config)["end_xyz_mm"], raw["start_xyz_mm"])

    def test_refuses_unknown_measured_geometry_bounds_speed_and_clearance(self):
        config, raw = fixture()
        for key, value in (("start_xyz_mm", None), ("travel_z_mm", None), ("travel_z_mm", 29),
                           ("xy_speed_mm_s", 100), ("z_speed_mm_s", float("nan")), ("dwell_s", 10)):
            changed = dict(raw, **{key: value})
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                build_plan(changed, config)
        for position in ([100, 100, 24.9], [400, 100, 25], [100, 100, 36]):
            changed = deepcopy(raw)
            changed["waypoints"][0]["xyz_mm"] = position
            with self.assertRaises(ValueError):
                build_plan(changed, config)
        config = replace(config, hardware=replace(config.hardware, fixture_top_z_mm=None))
        with self.assertRaises(ValueError):
            build_plan(raw, config)

    def test_preview_never_creates_a_connection(self):
        config, raw = fixture()
        with tempfile.TemporaryDirectory() as folder, patch("controller.nozzle.create_printer_client") as connect:
            plan = build_plan(raw, config)
            output = Path(folder)/"preview"
            write_preview(output, plan)
            connect.assert_not_called()
            self.assertTrue((output/"motion_preview.gcode").exists())
            with self.assertRaises(FileExistsError):
                write_preview(output, plan)

    def test_run_cli_previews_without_a_connection_or_camera_calibration(self):
        config, raw = fixture()
        with tempfile.TemporaryDirectory() as folder, patch("controller.nozzle.create_printer_client") as connect:
            directory = Path(folder)
            config_file, plan_file = directory/"controller.json", directory/"plan.json"
            config_file.write_text(json.dumps(config.to_dict()), encoding="utf-8")
            plan_file.write_text(json.dumps(raw), encoding="utf-8")
            with redirect_stdout(io.StringIO()) as text:
                code = main(["run", "--config", str(config_file), "--plan", str(plan_file),
                             "--output", str(directory/"preview")])
            self.assertEqual(code, 0)
            self.assertFalse(json.loads(text.getvalue())["motion_sent"])
            connect.assert_not_called()

    def test_status_cli_only_reads_the_printer(self):
        client = FakeClient()
        with patch("controller.nozzle.create_printer_client", return_value=client), redirect_stdout(io.StringIO()) as text:
            self.assertEqual(main(["status", "--config", str(ROOT/"configs/controller.example.json")]), 0)
        self.assertEqual(json.loads(text.getvalue())["printer_xyz_mm"], [110, 120, 30])
        self.assertFalse(client.commands)
        self.assertTrue(client.closed)


class ExecutionTests(unittest.IsolatedAsyncioTestCase):
    async def test_executes_with_no_camera_registration_and_records_arrivals(self):
        config, raw = fixture()
        self.assertIsNone(config.hardware.calibration_id)
        client = FakeClient()
        plan = build_plan(raw, config)
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)/"run"
            write_preview(output, plan)
            result = await execute_plan(config, plan, output, client)
            events = [json.loads(line) for line in (output/"motion_events.jsonl").read_text().splitlines()]
        self.assertEqual(result["state"], "READY")
        self.assertEqual(result["completed_blocks"], len(plan["blocks"]))
        self.assertEqual(client.commands[0], "G90")
        self.assertEqual(client.xyz, plan["end_xyz_mm"])
        self.assertTrue(client.closed)
        self.assertFalse(client.stop_requested)
        self.assertTrue(any(e["kind"] == "submit" and e["phase"] == "arrived" for e in events))

    async def test_lost_reply_is_not_retried_and_requests_stop(self):
        config, raw = fixture()
        client = FakeClient(fail_at=2)
        plan = build_plan(raw, config)
        with tempfile.TemporaryDirectory() as folder:
            write_preview(Path(folder)/"run", plan)
            result = await execute_plan(config, plan, Path(folder)/"run", client)
        self.assertEqual(result["state"], "FAULT")
        self.assertEqual(len(client.commands), 2)
        self.assertTrue(client.stop_requested)
        self.assertTrue(client.closed)
        self.assertEqual(result["completed_blocks"], 0)

    async def test_invalid_printer_state_sends_no_gcode(self):
        config, raw = fixture()
        plan = build_plan(raw, config)
        for override in ({("toolhead", "homed_axes"): "xy"}, {("extruder", "target"): 200},
                         {("gcode_move", "homing_origin"): [0, 0, 1, 0]}):
            client = FakeClient()
            client.overrides = override
            with tempfile.TemporaryDirectory() as folder:
                write_preview(Path(folder)/"run", plan)
                result = await execute_plan(config, plan, Path(folder)/"run", client)
            self.assertEqual(result["state"], "FAULT")
            self.assertFalse(client.commands)
            self.assertFalse(client.stop_requested)

    async def test_mode_change_during_block_stops_before_next_move(self):
        config, raw = fixture()
        class ModeChangingClient(FakeClient):
            async def execute_gcode(self, script):
                reply = await super().execute_gcode(script)
                if len(self.commands) == 2:
                    self.absolute = False
                return reply
        client = ModeChangingClient()
        plan = build_plan(raw, config)
        with tempfile.TemporaryDirectory() as folder:
            write_preview(Path(folder)/"run", plan)
            result = await execute_plan(config, plan, Path(folder)/"run", client)
        self.assertEqual(result["state"], "FAULT")
        self.assertEqual(len(client.commands), 2)
        self.assertTrue(client.stop_requested)


if __name__ == "__main__":
    unittest.main()
