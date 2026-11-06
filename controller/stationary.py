"""Preview or execute one stationary, explicitly positioned extrusion line.

The job uses an operator-selected absolute nozzle Z. It does not apply live
camera corrections or qualify the unverified camera registration.
"""

import argparse
import asyncio
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import uuid

import numpy as np

from .config import load_config
from .connection import create_printer_client
from .moonraker import verify_cold_state
from .nozzle import number, xyz_value
from .types import ControllerFault


def template():
    return {"schema_version": 1, "coordinate_frame": "printer",
        "start_xyz_mm": None, "travel_z_mm": 170.,
        "line_start_xy_mm": [170., 83.], "line_end_xy_mm": [190., 83.],
        "nozzle_z_mm": None, "xy_speed_mm_s": 5., "z_speed_mm_s": 1.,
        "nozzle_temperature_c": 220., "bead_area_mm2": .08, "filament_diameter_mm": 1.75,
        "loaded_and_primed": False, "path_clearance_verified": False,
        "max_runtime_s": 300., "fifth_point_check": "not_performed",
        "notes": "Recipe is experimental. Select absolute nozzle Z; marker heights are not bare-sheet heights. No automatic homing, probing or XYZ-offset changes."}


def build_job(raw, config):
    if not isinstance(raw, dict) or raw.get("schema_version") != 1 or raw.get("coordinate_frame") != "printer":
        raise ValueError("stationary plan requires schema 1 and printer coordinates")
    start = xyz_value(raw.get("start_xyz_mm"), "start_xyz_mm")
    travel = number(raw.get("travel_z_mm"), "travel_z_mm")
    z = number(raw.get("nozzle_z_mm"), "nozzle_z_mm")
    a, b = [], []
    for key, destination in (("line_start_xy_mm", a), ("line_end_xy_mm", b)):
        value = raw.get(key)
        if not isinstance(value, list) or len(value) != 2:
            raise ValueError(f"{key} requires two printer XY coordinates")
        destination.extend(number(v, key, minimum=-1e9) for v in value)
    a, b = np.array(a), np.array(b)
    length = float(np.linalg.norm(b - a))
    if not 0 < length <= 50:
        raise ValueError("first stationary line must have length greater than 0 and at most 50 mm")
    xy_speed = number(raw.get("xy_speed_mm_s"), "xy_speed_mm_s", strict=True)
    z_speed = number(raw.get("z_speed_mm_s"), "z_speed_mm_s", strict=True)
    temp = number(raw.get("nozzle_temperature_c"), "nozzle_temperature_c", strict=True)
    area = number(raw.get("bead_area_mm2"), "bead_area_mm2", strict=True)
    diameter = number(raw.get("filament_diameter_mm"), "filament_diameter_mm", strict=True)
    runtime = number(raw.get("max_runtime_s"), "max_runtime_s", strict=True)
    if xy_speed > min(config.limits.xy_speed_mm_s, config.limits.xyz_speed_mm_s) or z_speed > min(
            config.limits.z_speed_mm_s, config.limits.xyz_speed_mm_s):
        raise ValueError("stationary speeds exceed controller limits")
    extrusion = length * area / (math.pi * (diameter / 2) ** 2)
    if xy_speed * area > config.limits.max_flow_mm3_s or extrusion / length > config.limits.max_extrude_ratio:
        raise ValueError("stationary extrusion exceeds flow or extrusion-ratio limits")
    if travel < max(start[2], z):
        raise ValueError("travel Z must be at or above initial and deposition Z")
    coordinates = [start, [*start[:2], travel], [*a, travel], [*a, z], [*b, z], [*b, travel]]
    if any(np.any(np.asarray(p) < config.limits.xyz_min_mm) or np.any(
            np.asarray(p) > config.limits.xyz_max_mm) for p in coordinates):
        raise ValueError("stationary job exceeds controller workspace")
    fixture = config.hardware.fixture_top_z_mm
    if fixture is not None and min(start[2], travel) < fixture + config.hardware.cold_clearance_mm:
        raise ValueError("initial/travel Z must clear the complete fixture plus clearance")
    for flag in ("loaded_and_primed", "path_clearance_verified"):
        if type(raw.get(flag)) is not bool:
            raise ValueError(f"{flag} must be boolean")
    job_id = uuid.uuid4().hex
    state = "stationary_" + job_id
    lines = ["; Stationary experimental extrusion line; explicit absolute nozzle Z",
        "; No live camera correction. Establish/verify the coordinate reference before starting.",
        f"; Required initial XYZ: {start.tolist()}; filament feed: {extrusion:.9f} mm",
        "M400", f"SAVE_GCODE_STATE NAME={state}", "G21", "G90", "M83", "M220 S100", "M221 S100",
        f"G1 Z{travel:.6f} F{z_speed * 60:.6f}", "M400",
        f"G1 X{a[0]:.6f} Y{a[1]:.6f} F{xy_speed * 60:.6f}", "M400",
        f"M109 S{temp:.3f}", f"G1 Z{z:.6f} F{z_speed * 60:.6f}", "M400",
        f"G1 X{b[0]:.6f} Y{b[1]:.6f} Z{z:.6f} E{extrusion:.9f} F{xy_speed * 60:.6f}",
        "M400", "M104 S0", f"G1 Z{travel:.6f} F{z_speed * 60:.6f}", "M400",
        f"G1 X{start[0]:.6f} Y{start[1]:.6f} F{xy_speed * 60:.6f}", "M400",
        f"RESTORE_GCODE_STATE NAME={state} MOVE=0"]
    script = "\n".join(lines) + "\n"
    heater_ready_index = next(i for i, line in enumerate(lines) if line.startswith("M109 ")) + 1
    heater_off_index = lines.index("M104 S0")
    offset = lambda index: len(("\n".join(lines[:index]) + "\n").encode())
    return {"mode": "stationary_line", "config_id": config.config_id, "resolved_plan": raw,
        "filename": f"stationary_line_{job_id}.gcode", "gcode": script,
        "gcode_sha256": hashlib.sha256(script.encode()).hexdigest(),
        "line_length_mm": length, "filament_feed_mm": extrusion,
        "planned_polyline_xyz_mm": [np.asarray(p).tolist() for p in coordinates] + [[*start[:2], travel]],
        "heated_motion_start_byte": offset(heater_ready_index), "heater_off_byte": offset(heater_off_index),
        "body_start_byte": offset(lines.index("M221 S100") + 1),
        "restore_state_byte": offset(len(lines) - 1),
        "expected_end_xyz_mm": [*start[:2], travel], "max_runtime_s": runtime,
        "camera_corrections": False, "independent_metric_validation": False}


def write_preview(output, job):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / job["filename"]).write_bytes(job["gcode"].encode("utf-8"))
    (output / "job.json").write_text(json.dumps(job, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return {"state": "PREVIEW", "printer_commands_sent": 0, "uploaded": False,
            "line_length_mm": job["line_length_mm"], "filament_feed_mm": job["filament_feed_mm"],
            "filename": job["filename"], "output": str(output)}


def require_setup(config, job):
    if config.hardware.backend != "moonraker":
        raise ControllerFault("stationary print jobs currently require the Moonraker backend")
    if not all((config.hardware.stationary_fixture_verified, config.hardware.stop_test_verified,
                config.hardware.exclusive_control_verified)):
        raise ControllerFault("stationary printing requires a stationary fixture, tested stop and exclusive control")
    raw = job["resolved_plan"]
    if not raw["loaded_and_primed"] or not raw["path_clearance_verified"]:
        raise ControllerFault("stationary printing requires primed filament and a verified complete approach/line/return")
    if job["config_id"] != config.config_id:
        raise ControllerFault("stationary preview configuration differs from execution")


def path_diagnostic(position, job):
    """Describe the comparison without treating telemetry as encoder feedback."""
    position = xyz_value(list(position), "reported_live_xyz_mm")
    candidates = []
    polyline = job["planned_polyline_xyz_mm"]
    for index, (a, b) in enumerate(zip(polyline, polyline[1:])):
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        delta = b - a
        fraction = 0. if np.dot(delta, delta) == 0 else np.clip(np.dot(position - a, delta) / np.dot(delta, delta), 0, 1)
        nearest = a + fraction * delta
        candidates.append((float(np.linalg.norm(position - nearest)), index, nearest))
    distance, index, nearest = min(candidates, key=lambda item: item[0])
    return {"reported_live_xyz_mm": position.tolist(), "nearest_path_xyz_mm": nearest.tolist(),
            "nearest_segment_index": index, "distance_from_path_mm": distance,
            "path_tolerance_mm": .1,
            "position_source": "Klipper motion_report.live_position; requested interpolation, not physical feedback"}


def status_snapshot(report, job):
    """Retain the actual sample before a check can fault and request shutdown."""
    values = report.get("query", {}).get("status", {})
    names = ("webhooks", "toolhead", "gcode_move", "motion_report", "extruder", "heater_bed",
             "virtual_sdcard", "print_stats", "bed_mesh", "z_thermal_adjust", "idle_timeout")
    snapshot = {"printer": {key: report.get("printer", {}).get(key) for key in
                            ("state", "state_message", "software_version", "process_id")},
                "eventtime": report.get("query", {}).get("eventtime"),
                "status": {name: values[name] for name in names if name in values}}
    try:
        snapshot["path_diagnostic"] = path_diagnostic(values["motion_report"]["live_position"][:3], job)
    except (KeyError, TypeError, ValueError) as exc:
        snapshot["path_diagnostic_error"] = str(exc)
    # Preserve invalid numeric telemetry as text in the log; the checks still
    # inspect the original report and must not have their fault handling masked
    # by a JSON encoding failure.
    return json.loads(json.dumps(snapshot), parse_constant=lambda value: "NONFINITE:" + value)


def check_job_state(report, config, job, baseline, *, completed=False):
    """Check status without confusing hot printing with the cold preflight."""
    values = report["query"]["status"]
    identity = tuple(report["printer"].get(k) for k in ("software_version", "process_id"))
    if identity != baseline["identity"] or report["query"]["eventtime"] < baseline["eventtime"]:
        raise ControllerFault("printer restarted during stationary job")
    if report["printer"].get("state") != "ready" or values["webhooks"]["state"] != "ready":
        raise ControllerFault("printer is no longer ready")
    if not set("xyz").issubset(values["toolhead"]["homed_axes"]):
        raise ControllerFault("homing reference lost during stationary job")
    gcode = values["gcode_move"]
    if not np.allclose(gcode["homing_origin"][:3], 0, atol=1e-6) or not np.allclose(
            gcode["position"][:3], gcode["gcode_position"][:3], atol=1e-6):
        raise ControllerFault("XYZ coordinate frame changed during stationary job")
    if values.get("bed_mesh", {}).get("profile_name") or values.get("z_thermal_adjust", {}).get("enabled", False):
        raise ControllerFault("height compensation changed during stationary job")
    if gcode["speed_factor"] != 1 or gcode["extrude_factor"] != 1:
        raise ControllerFault("speed/extrusion factor changed during stationary job")
    file_position = values["virtual_sdcard"].get("file_position")
    modes = (gcode.get("absolute_coordinates"), gcode.get("absolute_extrude"))
    if file_position is None or not np.isfinite(file_position) or file_position < 0:
        raise ControllerFault("stationary job lacks usable file-progress status")
    allowed_modes = {(True, False)} if job["body_start_byte"] <= file_position < job["restore_state_byte"] else {
        (True, False), baseline["initial_modes"]}
    if modes not in allowed_modes:
        raise ControllerFault("XYZ/extrusion parsing mode changed during stationary job")
    heater = values["extruder"]
    temp, target = heater["temperature"], heater["target"]
    desired = job["resolved_plan"]["nozzle_temperature_c"]
    if not np.isfinite([temp, target]).all() or temp > min(desired + 15, baseline["max_temperature"]):
        raise ControllerFault("stationary extruder temperature exceeds the job envelope")
    if target not in (0, desired) or values["heater_bed"]["target"] != baseline["bed_target"]:
        raise ControllerFault("heater setpoint changed outside stationary job")
    if file_position is not None and job["heated_motion_start_byte"] <= file_position < job["heater_off_byte"] and target != desired:
        raise ControllerFault("nozzle heating was disabled during deposition/approach")
    position = np.asarray(values["motion_report"]["live_position"][:3], dtype=float)
    if not np.isfinite(position).all() or np.any(position < config.limits.xyz_min_mm) or np.any(
            position > config.limits.xyz_max_mm):
        raise ControllerFault("stationary printer position left configured workspace")
    diagnostic = path_diagnostic(position, job)
    if diagnostic["distance_from_path_mm"] > diagnostic["path_tolerance_mm"]:
        raise ControllerFault("printer position left the submitted stationary path: "
            f"reported XYZ={diagnostic['reported_live_xyz_mm']}, "
            f"distance={diagnostic['distance_from_path_mm']:.6f} mm, "
            f"nearest XYZ={diagnostic['nearest_path_xyz_mm']}")
    if completed:
        if target != 0 or values["virtual_sdcard"]["is_active"] or abs(
                values["motion_report"]["live_velocity"]) > 1e-4:
            raise ControllerFault("completed job has not stopped motion and nozzle heating")
        if not np.allclose(position, job["expected_end_xyz_mm"], atol=.02):
            raise ControllerFault("completed stationary job differs from expected final position")
    return values["print_stats"]


async def execute_job(config, job, output, client=None, *, poll_s=.5):
    require_setup(config, job)
    client = create_printer_client(config) if client is None else client
    output = Path(output)
    possible_start = False
    result = {"state": "FAULT", "uploaded": False, "start_requested": False,
              "physical_validation": False, "camera_corrections": False}
    events = (output / "execution_events.jsonl").open("x", encoding="utf-8")
    def event(kind, **fields):
        events.write(json.dumps({"kind": kind, "host_s": time.monotonic(), **fields}, allow_nan=False) + "\n")
        events.flush()
    def snapshot(stage, report):
        sample = status_snapshot(report, job)
        event("status_snapshot", stage=stage, **sample)
        (output / "last_status_snapshot.json").write_text(
            json.dumps({"stage": stage, **sample}, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    try:
        report = await client.discover()
        snapshot("preflight", report)
        verify_cold_state(report, config, job["resolved_plan"]["start_xyz_mm"])
        values = report["query"]["status"]
        if not {"virtual_sdcard", "print_stats"}.issubset(values):
            raise ControllerFault("stationary file printing requires virtual_sdcard and print_stats")
        if values["print_stats"]["state"] in {"printing", "paused"}:
            raise ControllerFault("another print is loaded or paused")
        installed = values["configfile"]["settings"].get("extruder", {})
        desired = job["resolved_plan"]["nozzle_temperature_c"]
        if not all(k in installed for k in ("min_extrude_temp", "max_temp")) or not (
                float(installed["min_extrude_temp"]) <= desired < float(installed["max_temp"])):
            raise ControllerFault("nozzle setpoint is outside verified installed extrusion-temperature limits")
        baseline = {"identity": tuple(report["printer"].get(k) for k in ("software_version", "process_id")),
                    "eventtime": report["query"]["eventtime"], "bed_target": values["heater_bed"]["target"],
                    "initial_modes": (values["gcode_move"].get("absolute_coordinates"),
                                      values["gcode_move"].get("absolute_extrude")),
                    "max_temperature": float(installed["max_temp"])}
        if any(type(value) is not bool for value in baseline["initial_modes"]):
            raise ControllerFault("initial XYZ/extrusion parsing modes could not be verified")
        event("preflight", expected_start_xyz_mm=job["resolved_plan"]["start_xyz_mm"])
        upload = await client.upload_gcode(job["filename"], job["gcode"])
        result["uploaded"] = True
        event("upload", response=upload, filename=job["filename"])
        if upload.get("print_started") or upload.get("print_queued"):
            possible_start = True
            raise ControllerFault("upload unexpectedly started/queued a print")
        item = upload.get("item", {})
        if item.get("root") != "gcodes" or item.get("path") != job["filename"]:
            raise ControllerFault("uploaded job path differs from preview")
        readback = await client.download_gcode(job["filename"])
        if hashlib.sha256(readback).hexdigest() != job["gcode_sha256"]:
            raise ControllerFault("uploaded G-code differs from reviewed local job")
        current = await client.discover()
        snapshot("before_start", current)
        verify_cold_state(current, config, job["resolved_plan"]["start_xyz_mm"])
        current_values = current["query"]["status"]
        if tuple(current["printer"].get(k) for k in ("software_version", "process_id")) != baseline["identity"] or (
                current["query"]["eventtime"] < baseline["eventtime"] or
                current_values["print_stats"]["state"] in {"printing", "paused"}):
            raise ControllerFault("printer/print ownership changed before job start")
        possible_start = True  # Before POST: a lost response never permits a retry.
        result["start_requested"] = True
        event("start_requested", filename=job["filename"])
        await client.request("POST", "/printer/print/start", {"filename": job["filename"]})
        deadline = time.monotonic() + job["max_runtime_s"]
        while time.monotonic() < deadline:
            report = await client.discover()
            snapshot("monitor_before_validation", report)
            stats = check_job_state(report, config, job, baseline)
            event("job_status", filename=stats.get("filename"), state=stats.get("state"),
                  nozzle_temperature_c=report["query"]["status"]["extruder"]["temperature"])
            if stats.get("filename") != job["filename"]:
                raise ControllerFault("active print filename differs from the submitted stationary job")
            if stats.get("state") == "complete":
                check_job_state(report, config, job, baseline, completed=True)
                result.update(state="COMPLETE", path_complete=True, filename=job["filename"],
                              end_xyz_mm=job["expected_end_xyz_mm"])
                break
            if stats.get("state") != "printing":
                raise ControllerFault("stationary job stopped or paused: " + str(stats.get("message", stats.get("state"))))
            await asyncio.sleep(poll_s)
        else:
            raise ControllerFault("stationary job exceeded its bounded runtime")
    except (Exception, asyncio.CancelledError) as exc:
        result["reason"] = str(exc) or "interrupted"
        try:
            event("fault", reason=result["reason"], start_may_have_executed=possible_start)
        except (OSError, ValueError) as log_error:
            result["logging_error"] = str(log_error)
        if possible_start:
            try:
                await client.emergency_stop()
                result["stop_requested"] = True
            except Exception as stop_error:
                result["stop_error"] = str(stop_error)
    finally:
        events.close()
        await client.close()
        (output / "execution.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


async def status(config, output):
    client = create_printer_client(config)
    try:
        report = await client.discover()
    finally:
        await client.close()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "discovery.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    values = report["query"]["status"]
    return {"state": report["printer"]["state"], "printer_xyz_mm": values["gcode_move"]["position"][:3],
        "gcode_xyz_mm": values["gcode_move"]["gcode_position"][:3],
        "homing_origin_mm": values["gcode_move"]["homing_origin"][:3],
        "homed_axes": values["toolhead"]["homed_axes"], "extruder": values["extruder"],
        "heater_bed": values["heater_bed"], "bed_mesh_profile": values.get("bed_mesh", {}).get("profile_name"),
        "idle_state": values["idle_timeout"]["state"], "print_stats": values.get("print_stats"),
        "printer_commands_sent": 0, "output": str(output)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("init", "status", "run"):
        sub = commands.add_parser(name)
        sub.add_argument("--output", type=Path, required=True)
        if name != "init":
            sub.add_argument("--config", required=True)
        if name == "run":
            sub.add_argument("--plan", required=True)
            sub.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise ValueError("output already exists; choose a new path")
        if args.command == "init":
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(template(), indent=2) + "\n", encoding="utf-8")
            result = {"state": "TEMPLATE", "output": str(args.output)}
        else:
            config = load_config(args.config)
            if args.command == "status":
                result = asyncio.run(status(config, args.output))
            else:
                job = build_job(json.loads(Path(args.plan).read_text(encoding="utf-8-sig")), config)
                result = write_preview(args.output, job)
                if args.execute:
                    result = asyncio.run(execute_job(config, job, args.output))
        print(json.dumps(result, indent=2, allow_nan=False))
        return 2 if result.get("state") == "FAULT" else 0
    except (ControllerFault, ValueError, TypeError, KeyError, OSError, RuntimeError) as exc:
        print(f"stationary: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("stationary: interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
