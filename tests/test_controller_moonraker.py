"""Moonraker contract tests using HTTPX's in-process transport; no real printer."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx
import numpy as np

from controller.cold_run import dry_run
from controller.config import load_config
from controller.moonraker import MoonrakerClient, verify_cold_state
from controller.types import ControllerFault


ROOT = Path(__file__).resolve().parents[1]


def config():
    cfg = load_config(ROOT/"configs/controller.example.json")
    hardware = replace(cfg.hardware, backend="moonraker", moonraker_url="http://printer:7125", calibration_id="test-calibration",
                       camera_config_sha256="fixture-test", expires_utc="2100-01-01T00:00:00Z",
                       stationary_fixture_verified=True, stop_test_verified=True,
                       exclusive_control_verified=True, fixture_top_z_mm=20)
    return replace(cfg, substrate_height_mm=20, hardware=hardware)


def status():
    return {
        "webhooks": {"state": "ready"},
        "toolhead": {"homed_axes": "xyz", "axis_minimum": [0, 0, 0, 0],
                     "axis_maximum": [350, 350, 345, 0], "max_velocity": 500,
                     "max_accel": 10000, "square_corner_velocity": 5},
        "gcode_move": {"position": [110, 120, 25, 0], "gcode_position": [110, 120, 25, 0],
                       "homing_origin": [0, 0, 0, 0], "speed_factor": 1, "extrude_factor": 1,
                       "absolute_coordinates": True, "absolute_extrude": False},
        "motion_report": {"live_position": [110, 120, 25, 0], "live_velocity": 0},
        "extruder": {"temperature": 25, "target": 0}, "heater_bed": {"temperature": 25, "target": 0},
        "idle_timeout": {"state": "Ready"}, "virtual_sdcard": {"is_active": False},
        "bed_mesh": {"profile_name": ""},
        "configfile": {"settings": {"printer": {"max_z_velocity": 20, "max_z_accel": 500}}},
    }


class MoonrakerTests(unittest.IsolatedAsyncioTestCase):
    async def test_discovery_sends_no_gcode_and_queries_only_available_objects(self):
        cfg = config()
        requests = []
        values = status()
        def handler(request):
            requests.append(request)
            endpoint = request.url.path
            if endpoint == "/server/info":
                result = {"klippy_state": "ready"}
            elif endpoint == "/printer/info":
                result = {"state": "ready", "software_version": "test", "process_id": 1}
            elif endpoint == "/printer/objects/list":
                result = {"objects": list(values)}
            elif endpoint == "/printer/objects/query":
                queried = json.loads(request.content)["objects"]
                self.assertTrue(set(queried).issubset(values))
                result = {"status": values, "eventtime": 10.0}
            else:
                self.fail(f"unexpected endpoint {endpoint}")
            return httpx.Response(200, json={"result": result})
        async with httpx.AsyncClient(base_url=cfg.hardware.moonraker_url, transport=httpx.MockTransport(handler)) as http:
            client = MoonrakerClient(cfg, http)
            report = await client.discover()
        self.assertFalse(report["motion_schedule_known"])
        self.assertEqual(verify_cold_state(report, cfg, (110, 120, 25)), (110, 120, 25))
        self.assertFalse(any("gcode" in r.url.path or "emergency" in r.url.path for r in requests))

    async def test_preflight_rejects_hot_unhomed_mesh_offsets_and_limits(self):
        cfg = config()
        for obj, key, value in (("extruder", "target", 200), ("toolhead", "homed_axes", "xy"),
                                 ("gcode_move", "speed_factor", 2), ("bed_mesh", "profile_name", "default"),
                                 ("gcode_move", "homing_origin", [0, 0, 1, 0])):
            values = status()
            values[obj][key] = value
            with self.assertRaises(ControllerFault):
                verify_cold_state({"printer": {"state": "ready"}, "query": {"status": values}}, cfg)

    async def test_dedicated_emergency_endpoint_and_invalid_response(self):
        cfg = config()
        endpoints = []
        def handler(request):
            endpoints.append(request.url.path)
            return httpx.Response(200, json={"result": "ok"})
        async with httpx.AsyncClient(base_url=cfg.hardware.moonraker_url, transport=httpx.MockTransport(handler)) as http:
            await MoonrakerClient(cfg, http).emergency_stop()
        self.assertEqual(endpoints, ["/printer/emergency_stop"])
        async with httpx.AsyncClient(base_url=cfg.hardware.moonraker_url, transport=httpx.MockTransport(
                lambda r: httpx.Response(200, json={"error": "bad"}))) as http:
            with self.assertRaises(ControllerFault):
                await MoonrakerClient(cfg, http).discover()

    async def test_cold_preview_has_no_network_side_effects(self):
        cfg = config()
        with tempfile.TemporaryDirectory() as directory, patch("controller.cold_run.create_printer_client") as client:
            report = await dry_run(cfg, np.array([[110, 120], [111, 120]]), Path(directory)/"run")
            self.assertFalse(report["motion_sent"])
            client.assert_not_called()

    async def test_cold_execution_and_request_failure_stop_without_retry(self):
        cfg = config()
        for lost_reply in (False, True):
            values = deepcopy(status())
            scripts = []
            endpoints = []
            eventtime = 0
            def handler(request):
                nonlocal eventtime
                endpoint = request.url.path
                endpoints.append(endpoint)
                if endpoint == "/server/info":
                    result = {"klippy_state": "ready"}
                elif endpoint == "/printer/info":
                    result = {"state": "ready", "software_version": "test", "process_id": 1}
                elif endpoint == "/printer/objects/list":
                    result = {"objects": list(values)}
                elif endpoint == "/printer/objects/query":
                    eventtime += 1
                    result = {"status": values, "eventtime": eventtime}
                elif endpoint == "/printer/gcode/script":
                    script = json.loads(request.content)["script"]
                    scripts.append(script)
                    if "G1 " in script:
                        values["motion_report"]["live_position"] = [111, 120, 25, 0]
                        values["gcode_move"]["position"] = [111, 120, 25, 0]
                        values["gcode_move"]["gcode_position"] = [111, 120, 25, 0]
                        if lost_reply:
                            raise httpx.ReadTimeout("reply lost after execution", request=request)
                    result = "ok"
                elif endpoint == "/printer/emergency_stop":
                    result = "ok"
                else:
                    self.fail(endpoint)
                return httpx.Response(200, json={"result": result})
            async with httpx.AsyncClient(base_url=cfg.hardware.moonraker_url, transport=httpx.MockTransport(handler)) as http:
                client = MoonrakerClient(cfg, http)
                with tempfile.TemporaryDirectory() as directory, patch("controller.cold_run.create_printer_client", return_value=client):
                    report = await dry_run(cfg, np.array([[110, 120], [111, 120]]), Path(directory)/"run", execute=True)
            self.assertEqual(len([s for s in scripts if "G1 " in s]), 1)
            if lost_reply:
                self.assertEqual(report["state"], "FAULT")
                self.assertIn("/printer/emergency_stop", endpoints)
            else:
                self.assertTrue(report["path_complete"], report)
            self.assertFalse(any("M104" in s or "M109" in s or " E" in s for s in scripts))


if __name__ == "__main__":
    unittest.main()
