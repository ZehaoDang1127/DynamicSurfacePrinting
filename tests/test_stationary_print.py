"""Stationary extrusion jobs tested with a virtual Moonraker client only."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

import httpx
import numpy as np

from controller.config import load_config
from controller.moonraker import MoonrakerClient
from controller.stationary import build_job, execute_job, main, template, write_preview
from controller.types import ControllerFault

ROOT = Path(__file__).resolve().parents[1]


def fixture():
    base = load_config(ROOT / "configs/controller.example.json")
    config = replace(base, hardware=replace(base.hardware, backend="moonraker", moonraker_url="http://printer:7125",
        fixture_top_z_mm=20., stationary_fixture_verified=True, stop_test_verified=True,
        exclusive_control_verified=True))
    raw = template()
    raw.update(start_xyz_mm=[110., 120., 30.], travel_z_mm=35., nozzle_z_mm=25.,
               line_start_xy_mm=[110., 120.], line_end_xy_mm=[130., 120.],
               loaded_and_primed=True, path_clearance_verified=True)
    return config, raw


class FakePrinter:
    def __init__(self, job, failure=None):
        self.job, self.failure = job, failure
        self.started, self.closed, self.stopped = False, False, False
        self.uploads, self.starts, self.polls, self.eventtime = 0, 0, 0, 0
        self.script = None

    async def discover(self):
        self.eventtime += 1
        if self.started:
            self.polls += 1
        completed = self.started and self.polls >= 2 and self.failure != "timeout"
        xyz = self.job["expected_end_xyz_mm"] if completed else self.job["resolved_plan"]["start_xyz_mm"]
        values = {"webhooks": {"state": "ready"},
            "toolhead": {"homed_axes": "xyz", "axis_minimum": [0, 0, 0], "axis_maximum": [350, 350, 350],
                         "max_velocity": 500, "max_accel": 10000},
            "gcode_move": {"position": [*xyz, 0], "gcode_position": [*xyz, 0], "homing_origin": [0, 0, 0, 0],
                "speed_factor": 1, "extrude_factor": 1, "absolute_coordinates": True, "absolute_extrude": False},
            "motion_report": {"live_position": [*xyz, 0], "live_velocity": 0},
            "extruder": {"temperature": 220 if self.started else 25, "target": 220 if self.started and not completed else 0},
            "heater_bed": {"temperature": 25, "target": 0}, "idle_timeout": {"state": "Ready"},
            "virtual_sdcard": {"is_active": self.started and not completed, "file_position": 0},
            "print_stats": {"filename": self.job["filename"] if self.started else "old.gcode",
                            "state": "complete" if completed else "printing" if self.started else "standby"},
            "bed_mesh": {"profile_name": ""}, "configfile": {"settings": {
                "printer": {"max_z_velocity": 20, "max_z_accel": 500},
                "extruder": {"min_extrude_temp": 170, "max_temp": 300}}}}
        if self.failure == "preflight":
            values["toolhead"]["homed_axes"] = "xy"
        if self.started and self.failure == "different_job":
            values["print_stats"]["filename"] = "someone_else.gcode"
        if self.started and self.failure == "overheat":
            values["extruder"]["temperature"] = 250
        if self.started and self.failure == "mode_change":
            values["gcode_move"]["absolute_extrude"] = True
        if self.started and self.failure == "off_path":
            values["motion_report"]["live_position"] = [115, 130, 25, 0]
        if self.started and self.failure == "zero_live_position":
            values["motion_report"]["live_position"] = [0, 0, 0, 0]
        if completed and self.failure == "bad_end":
            values["motion_report"]["live_position"] = [110, 120, 34, 0]
        return {"printer": {"state": "ready", "software_version": "test", "process_id": 1},
                "query": {"eventtime": self.eventtime, "status": deepcopy(values)}}

    async def upload_gcode(self, filename, script):
        self.uploads += 1
        if self.failure == "upload":
            raise ControllerFault("upload failed")
        self.script = script
        return {"item": {"root": "gcodes", "path": filename}, "print_started": False,
                "print_queued": self.failure == "queued"}

    async def download_gcode(self, filename):
        return b"changed" if self.failure == "readback" else self.script.encode()

    async def request(self, method, endpoint, body=None):
        assert (method, endpoint, body) == ("POST", "/printer/print/start", {"filename": self.job["filename"]})
        self.starts += 1
        self.started = True
        if self.failure == "lost_start":
            raise ControllerFault("start reply lost")
        return "ok"

    async def emergency_stop(self):
        self.stopped = True

    async def close(self):
        self.closed = True


class BuildTests(unittest.TestCase):
    def test_line_volume_travel_and_no_automatic_coordinate_changes(self):
        cfg, raw = fixture()
        job = build_job(raw, cfg)
        self.assertAlmostEqual(job["filament_feed_mm"], 20 * .08 / (np.pi * (.875 ** 2)))
        lines = job["gcode"].splitlines()
        extrusion = [line for line in lines if re.match(r"G1 .* E", line)]
        self.assertEqual(len(extrusion), 1)
        self.assertIn("F300.000000", extrusion[0])
        moves = [line for line in lines if line.startswith("G1 ")]
        self.assertTrue(moves[0].startswith("G1 Z35."))
        self.assertTrue(moves[-2].startswith("G1 Z35."))
        self.assertNotRegex(job["gcode"], r"(?m)^(?:G28|G92|SET_KINEMATIC_POSITION|SET_GCODE_OFFSET)\b")
        self.assertIn("M104 S0", lines)
        self.assertFalse(job["camera_corrections"])

    def test_missing_height_speed_flow_and_clearance_rejected(self):
        cfg, raw = fixture()
        for changes in ({"nozzle_z_mm": None}, {"start_xyz_mm": None}, {"travel_z_mm": 29},
                        {"xy_speed_mm_s": 100}, {"bead_area_mm2": 10},
                        {"line_end_xy_mm": [400, 120]}, {"nozzle_z_mm": float("nan")}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                build_job(dict(raw, **changes), cfg)

    def test_preview_never_constructs_printer_client(self):
        cfg, raw = fixture()
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            (base / "config.json").write_text(json.dumps(cfg.to_dict()))
            (base / "plan.json").write_text(json.dumps(raw))
            with patch("controller.stationary.create_printer_client") as client, patch("sys.stdout", new_callable=io.StringIO):
                code = main(["run", "--config", str(base / "config.json"), "--plan", str(base / "plan.json"),
                             "--output", str(base / "preview")])
            self.assertEqual(code, 0)
            client.assert_not_called()
            job = json.loads((base / "preview/job.json").read_text())
            self.assertEqual(hashlib.sha256((base / "preview" / job["filename"]).read_bytes()).hexdigest(),
                             job["gcode_sha256"])


class ExecutionTests(unittest.IsolatedAsyncioTestCase):
    async def run_case(self, failure=None):
        cfg, raw = fixture()
        job = build_job(raw, cfg)
        if failure == "timeout":
            job["max_runtime_s"] = .001
        client = FakePrinter(job, failure)
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / "run"
            write_preview(out, job)
            report = await execute_job(cfg, job, out, client, poll_s=.002)
            self.assertTrue((out / "execution.json").exists())
        self.assertTrue(client.closed)
        return report, client

    async def test_success_uploads_checks_starts_once_and_monitors_completion(self):
        report, client = await self.run_case()
        self.assertEqual(report["state"], "COMPLETE", report)
        self.assertEqual((client.uploads, client.starts), (1, 1))
        self.assertFalse(client.stopped)
        self.assertFalse(report["camera_corrections"])

    async def test_failure_before_start_does_not_start_or_stop(self):
        for failure in ("preflight", "upload", "readback"):
            with self.subTest(failure=failure):
                report, client = await self.run_case(failure)
                self.assertEqual(report["state"], "FAULT")
                self.assertEqual(client.starts, 0)
                self.assertFalse(client.stopped)

    async def test_lost_start_response_is_not_retried_and_requests_stop(self):
        report, client = await self.run_case("lost_start")
        self.assertEqual(report["state"], "FAULT")
        self.assertEqual(client.starts, 1)
        self.assertTrue(client.stopped)

    async def test_monitor_faults_stop_without_another_start(self):
        for failure in ("different_job", "overheat", "mode_change", "off_path", "bad_end", "timeout", "queued"):
            with self.subTest(failure=failure):
                report, client = await self.run_case(failure)
                self.assertEqual(report["state"], "FAULT", report)
                self.assertLessEqual(client.starts, 1)
                self.assertTrue(client.stopped)

    async def test_unverified_setup_is_not_enabled_by_skipping_camera_check(self):
        cfg, raw = fixture()
        raw["loaded_and_primed"] = False
        job = build_job(raw, cfg)
        client = FakePrinter(job)
        with tempfile.TemporaryDirectory() as folder, self.assertRaisesRegex(ControllerFault, "primed"):
            await execute_job(cfg, job, folder, client)
        self.assertEqual(client.uploads, 0)

    async def test_faulting_pose_is_recorded_before_emergency_stop(self):
        for failure, expected in (("off_path", [115, 130, 25]), ("zero_live_position", [0, 0, 0])):
            with self.subTest(failure=failure):
                cfg, raw = fixture()
                job = build_job(raw, cfg)
                client = FakePrinter(job, failure)
                with tempfile.TemporaryDirectory() as folder:
                    out = Path(folder) / "run"
                    write_preview(out, job)
                    result = await execute_job(cfg, job, out, client, poll_s=.002)
                    sample = json.loads((out / "last_status_snapshot.json").read_text())
                    self.assertEqual(sample["stage"], "monitor_before_validation")
                    self.assertEqual(sample["path_diagnostic"]["reported_live_xyz_mm"], expected)
                    self.assertGreater(sample["path_diagnostic"]["distance_from_path_mm"], .1)
                    self.assertIn(str([float(value) for value in expected]), result["reason"])
                    self.assertTrue(client.stopped)
                    self.assertEqual(result["state"], "FAULT")


class FileApiTests(unittest.IsolatedAsyncioTestCase):
    async def test_upload_checksum_no_autostart_and_readback(self):
        cfg, raw = fixture()
        job = build_job(raw, cfg)
        requests = []
        def handler(request):
            requests.append(request)
            if request.method == "POST":
                self.assertIn(job["gcode_sha256"].encode(), request.content)
                self.assertRegex(request.content.decode(), 'name="print"\r\n\r\nfalse')
                return httpx.Response(201, json={"result": {"item": {"root": "gcodes", "path": job["filename"]},
                                                           "print_started": False, "print_queued": False}})
            return httpx.Response(200, content=job["gcode"].encode())
        async with httpx.AsyncClient(base_url=cfg.hardware.moonraker_url, transport=httpx.MockTransport(handler)) as http:
            client = MoonrakerClient(cfg, http)
            uploaded = await client.upload_gcode(job["filename"], job["gcode"])
            readback = await client.download_gcode(job["filename"])
            self.assertFalse(uploaded["print_started"])
            self.assertEqual(readback, job["gcode"].encode())
            with self.assertRaises(ValueError):
                await client.upload_gcode("../printer.cfg", "changed")
        self.assertEqual(len(requests), 2)


if __name__ == "__main__":
    unittest.main()
