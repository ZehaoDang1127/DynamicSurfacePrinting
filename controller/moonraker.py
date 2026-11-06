"""Moonraker HTTP discovery and cold characterization, with no motion retries."""

import os
import hashlib
import re
import time
from urllib.parse import urlsplit

import numpy as np

from .transport import SerializedTransport, gcode_for
from .types import ControllerFault


class MoonrakerClient:
    def __init__(self, config, client=None):
        hw = config.hardware
        url = hw.moonraker_url
        if not url:
            raise ValueError("set hardware.moonraker_url first")
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("Moonraker URL must be http(s), with credentials supplied via environment")
        self.config = config
        self.owns_client = client is None
        if client is None:
            try:
                import httpx
            except ImportError as exc:
                raise RuntimeError("install requirements-controller.txt for Moonraker commands") from exc
            key = os.environ.get(hw.api_key_env)
            self.client = httpx.AsyncClient(base_url=url.rstrip("/"), timeout=hw.request_timeout_s,
                                            headers={"X-Api-Key": key} if key else {},
                                            follow_redirects=False, trust_env=False)
        else:
            self.client = client

    async def close(self):
        if self.owns_client:
            await self.client.aclose()

    async def request(self, method, endpoint, body=None):
        try:
            response = await self.client.request(method, endpoint, json=body)
            response.raise_for_status()
            raw = response.json()
        except Exception as exc:
            raise ControllerFault(f"Moonraker {method} {endpoint} failed ({type(exc).__name__}); no motion retry") from exc
        if not isinstance(raw, dict) or "result" not in raw or "error" in raw:
            raise ControllerFault("invalid/error Moonraker response")
        return raw["result"]

    async def discover(self):
        # Only read operations, including the POST query API which does not change state.
        server = await self.request("GET", "/server/info")
        printer = await self.request("GET", "/printer/info")
        available = (await self.request("GET", "/printer/objects/list"))["objects"]
        wanted = {name: None for name in ("webhooks", "toolhead", "gcode_move", "motion_report",
                  "extruder", "heater_bed", "virtual_sdcard", "bed_mesh", "configfile", "idle_timeout",
                  "z_thermal_adjust", "print_stats") if name in available}
        before = time.monotonic()
        status = await self.request("POST", "/printer/objects/query", {"objects": wanted})
        after = time.monotonic()
        return {"server": server, "printer": printer, "objects": available, "query": status,
                "request_sent_s": before, "receipt_s": after, "round_trip_s": after-before,
                "motion_schedule_known": False}

    async def emergency_stop(self):
        # Dedicated endpoint, outside the serialized ordinary G-code writer.
        return await self.request("POST", "/printer/emergency_stop")

    async def execute_gcode(self, script):
        return await self.request("POST", "/printer/gcode/script", {"script": script})

    async def upload_gcode(self, filename, script):
        """Upload a uniquely named job without starting or queuing it."""
        if not re.fullmatch(r"stationary_line_[0-9a-f]{32}\.gcode", filename):
            raise ValueError("stationary upload requires a generated unique job filename")
        data = script.encode("utf-8")
        try:
            response = await self.client.request("POST", "/server/files/upload",
                data={"root": "gcodes", "print": "false", "checksum": hashlib.sha256(data).hexdigest()},
                files={"file": (filename, data, "application/octet-stream")})
            response.raise_for_status()
            raw = response.json()
        except Exception as exc:
            raise ControllerFault(f"Moonraker file upload failed ({type(exc).__name__}); no retry") from exc
        result = raw.get("result", raw) if isinstance(raw, dict) else None
        if not isinstance(result, dict) or "error" in raw or "item" not in result:
            raise ControllerFault("invalid Moonraker file upload response")
        return result

    async def download_gcode(self, filename):
        """Read back the uploaded bytes before starting the job."""
        if not re.fullmatch(r"stationary_line_[0-9a-f]{32}\.gcode", filename):
            raise ValueError("invalid stationary job filename")
        try:
            response = await self.client.request("GET", f"/server/files/gcodes/{filename}")
            response.raise_for_status()
            return response.content
        except Exception as exc:
            raise ControllerFault(f"Moonraker job readback failed ({type(exc).__name__}); no start") from exc


def verify_cold_state(report, config, expected_xyz=None, require_modes=False, allow_owned_motion=False):
    status = report["query"]["status"]
    required = {"webhooks", "toolhead", "gcode_move", "motion_report", "extruder", "heater_bed", "idle_timeout"}
    if not required.issubset(status):
        raise ControllerFault("printer lacks required cold-motion status objects")
    if report["printer"].get("state") != "ready" or status["webhooks"].get("state") != "ready":
        raise ControllerFault("Klipper is not ready")
    tool, gcode, motion = status["toolhead"], status["gcode_move"], status["motion_report"]
    if not set("xyz").issubset(tool.get("homed_axes", "")):
        raise ControllerFault("home and level the printer with the fixture clear before running")
    if status.get("virtual_sdcard", {}).get("is_active", False) or (
            not allow_owned_motion and status["idle_timeout"].get("state") == "Printing"):
        raise ControllerFault("another print/motion owner is active")
    if abs(float(motion["live_velocity"])) > 1e-4:
        raise ControllerFault("printer is still moving")
    for name in ("extruder", "heater_bed"):
        heater = status[name]
        if not np.isfinite([heater["target"], heater["temperature"]]).all() or heater["target"] != 0 or heater["temperature"] > config.hardware.max_cold_temperature_c:
            raise ControllerFault("cold mode requires heaters off and temperatures below configured limit")
    if gcode.get("speed_factor") != 1 or gcode.get("extrude_factor") != 1:
        raise ControllerFault("reset M220 and M221 to 100 before cold characterization")
    # XYZ are first three components on both legacy and current upstream Klipper.
    if not np.isfinite(gcode["homing_origin"][:3]).all() or not np.allclose(gcode["homing_origin"][:3], 0, atol=1e-6):
        raise ControllerFault("clear/verify XYZ G-code offsets before cold characterization")
    if not np.allclose(gcode["position"][:3], gcode["gcode_position"][:3], atol=1e-6):
        raise ControllerFault("G-code frame differs from machine frame")
    if status.get("bed_mesh", {}).get("profile_name"):
        raise ControllerFault("clear bed mesh during controlled setup before cold characterization")
    if status.get("z_thermal_adjust", {}).get("enabled", False):
        raise ControllerFault("disable Z thermal compensation during controlled setup")
    if require_modes and (gcode.get("absolute_coordinates") is not True or gcode.get("absolute_extrude") is not False):
        raise ControllerFault("coordinate/extrusion modes changed during run")
    limits = config.limits
    for coordinate in (tool["axis_minimum"][:3], tool["axis_maximum"][:3], motion["live_position"][:3]):
        if len(coordinate) != 3 or not np.isfinite(coordinate).all():
            raise ControllerFault("invalid printer coordinates")
    if np.any(np.asarray(limits.xyz_min_mm) < tool["axis_minimum"][:3]) or np.any(
            np.asarray(limits.xyz_max_mm) > tool["axis_maximum"][:3]):
        raise ControllerFault("configured workspace exceeds installed travel limits")
    if limits.xyz_speed_mm_s > tool["max_velocity"] or limits.xy_accel_mm_s2 > tool["max_accel"]:
        raise ControllerFault("controller limits exceed active Klipper limits")
    installed = status.get("configfile", {}).get("settings", {}).get("printer", {})
    if any(key not in installed for key in ("max_z_velocity", "max_z_accel")):
        raise ControllerFault("installed Z-motion limits could not be verified")
    if limits.z_speed_mm_s > float(installed["max_z_velocity"]) or limits.z_accel_mm_s2 > float(installed["max_z_accel"]):
        raise ControllerFault("controller Z limits exceed installed Klipper limits")
    position = tuple(float(v) for v in motion["live_position"][:3])
    if expected_xyz is not None and not np.allclose(position, expected_xyz, atol=.02):
        raise ControllerFault("position differs from expected start/end; reconcile before a new session")
    return position


class ColdPrinterTransport(SerializedTransport):
    """One rest-to-rest characterization move plus M400, not continuous tracking."""

    def __init__(self, config, client, log=lambda *a, **kw: None):
        super().__init__(config.hardware.max_batch_s, config.config_id, log)
        self.client = client
        self.config = config

    async def _send(self, trajectory):
        for point in (trajectory.start_xyz, *(m.xyz for m in trajectory.moves)):
            if point[2] < self.config.hardware.fixture_top_z_mm + self.config.hardware.cold_clearance_mm:
                raise ControllerFault("cold path is below the full fixture envelope plus clearance")
        script = gcode_for(trajectory, cold=True, synchronize=True)
        self.log("cold_script", script=script)
        # M400 supplies a deliberate queue drain during timing characterization.
        # Still not independent physical feedback or a timestamped execution API.
        return await self.client.execute_gcode(script)

    async def emergency_stop(self):
        return await self.client.emergency_stop()


ColdMoonrakerTransport = ColdPrinterTransport  # Existing import compatibility.
