"""Printer-coordinate nozzle positioning for setup, before camera registration.

This cold, non-extruding tool reuses the existing Moonraker/Klipper clients.
It generates a preview by default. `--execute` is required for live movement.
"""

import argparse
import asyncio
from dataclasses import replace
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

from .config import load_config
from .connection import create_printer_client
from .moonraker import verify_cold_state
from .types import ControllerFault


def number(value, name, minimum=0, strict=False):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a measured finite number")
    if value < minimum or (strict and value == minimum):
        raise ValueError(f"{name} must be {'greater than' if strict else 'at least'} {minimum}")
    return float(value)


def xyz_value(value, name):
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{name} must contain three measured XYZ values in mm")
    return np.array([number(v, name, minimum=-1e9) for v in value])


def build_plan(raw, config):
    """Lift, traverse at clearance Z, approach each point, lift again."""
    if not isinstance(raw, dict):
        raise ValueError("motion plan must be a JSON object")
    if raw.get("schema_version") != 1 or raw.get("coordinate_frame") != "printer":
        raise ValueError("motion plan requires schema_version=1 and coordinate_frame=printer")
    if type(raw.get("return_to_start", False)) is not bool:
        raise ValueError("return_to_start must be boolean")
    fixture = config.hardware.fixture_top_z_mm
    if fixture is None:
        raise ValueError("set hardware.fixture_top_z_mm to the measured full fixture envelope")
    floor = fixture + config.hardware.cold_clearance_mm
    start = xyz_value(raw.get("start_xyz_mm"), "start_xyz_mm")
    travel_z = number(raw.get("travel_z_mm"), "travel_z_mm")
    if start[2] > travel_z:
        raise ValueError("travel_z_mm must be at or above start Z")
    xy_speed = number(raw.get("xy_speed_mm_s"), "xy_speed_mm_s", strict=True)
    z_speed = number(raw.get("z_speed_mm_s"), "z_speed_mm_s", strict=True)
    dwell = number(raw.get("dwell_s", 0), "dwell_s")
    if xy_speed > min(config.limits.xy_speed_mm_s, config.limits.xyz_speed_mm_s) or z_speed > min(
            config.limits.z_speed_mm_s, config.limits.xyz_speed_mm_s):
        raise ValueError("plan speeds exceed controller limits")
    budget = min(config.hardware.max_batch_s, config.hardware.request_timeout_s / 2)
    if dwell > budget:
        raise ValueError("dwell_s exceeds the bounded request time budget")
    waypoints = raw.get("waypoints", [])
    if not isinstance(waypoints, list) or not 1 <= len(waypoints) <= 1000:
        raise ValueError("provide 1 to 1000 labelled waypoints")
    if any(not isinstance(p, dict) for p in waypoints):
        raise ValueError("waypoints must be labelled objects")
    ids = [p.get("id") for p in waypoints]
    if any(not isinstance(v, str) or not v.strip() for v in ids) or len(set(ids)) != len(ids):
        raise ValueError("waypoint IDs must be unique nonempty strings")
    targets = [(p["id"], xyz_value(p.get("xyz_mm"), p["id"])) for p in waypoints]
    coordinates = [start, np.array([*start[:2], travel_z])]
    for _, target in targets:
        coordinates.extend([target, np.array([*target[:2], travel_z])])
    if any(p[2] < floor or p[2] > travel_z or np.any(p < config.limits.xyz_min_mm) or np.any(
            p > config.limits.xyz_max_mm) for p in coordinates):
        raise ValueError("all starts, targets and travel positions must stay inside bounds and above fixture + clearance")
    blocks = []
    current = start.copy()

    def leg(target, phase, waypoint):
        nonlocal current
        delta = target - current
        distance = float(np.linalg.norm(delta))
        if distance < 1e-9:
            return
        vertical = np.linalg.norm(delta[:2]) < 1e-9
        if not vertical and abs(delta[2]) > 1e-9:
            raise ValueError("internal plan error: XY travel must remain at clearance Z")
        speed = z_speed if vertical else xy_speed
        accel = config.limits.z_accel_mm_s2 if vertical else config.limits.xy_accel_mm_s2
        # Maximum distance of a rest-to-rest triangular/trapezoidal move.
        max_distance = accel * budget**2 / 4 if budget <= 2*speed/accel else speed*(budget-speed/accel)
        count = max(1, math.ceil(distance / max_distance))
        if count + len(blocks) > 100000:
            raise ValueError("plan exceeds 100000 bounded blocks")
        origin = current.copy()
        for i in range(1, count+1):
            end = origin + delta * i/count
            piece_distance = distance/count
            duration = (2*math.sqrt(piece_distance/accel) if piece_distance <= speed**2/accel
                        else piece_distance/speed + speed/accel)
            gcode = f"G1 X{end[0]:.6f} Y{end[1]:.6f} Z{end[2]:.6f} F{speed*60:.6f}\nM400"
            blocks.append({"phase": phase, "waypoint": waypoint, "xyz_mm": end.tolist(),
                           "feed_mm_s": speed, "estimated_duration_s": duration, "gcode": gcode})
        current = target.copy()

    leg(np.array([*current[:2], travel_z]), "initial_lift", None)
    for identity, target in targets:
        leg(np.array([*target[:2], travel_z]), "travel", identity)
        leg(target, "approach", identity)
        # Explicit arrival record even when approach has zero length.
        blocks.append({"phase": "arrived", "waypoint": identity, "xyz_mm": target.tolist(),
                       "estimated_duration_s": dwell,
                       "gcode": f"G4 P{math.ceil(dwell*1000)}\nM400" if dwell else "M400"})
        leg(np.array([*target[:2], travel_z]), "lift", identity)
    if raw.get("return_to_start", False):
        leg(np.array([*start[:2], travel_z]), "return_travel", None)
        leg(start, "return_approach", None)
    return {"schema_version": 1, "coordinate_frame": "printer", "start_xyz_mm": start.tolist(),
            "end_xyz_mm": current.tolist(), "travel_z_mm": travel_z, "fixture_clearance_z_mm": floor,
            "no_extrusion": True, "blocks": blocks,
            "estimated_duration_s": sum(b["estimated_duration_s"] for b in blocks)}


def write_preview(output, plan, synthetic=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    plan = dict(plan, synthetic_demo=synthetic)
    (output / "motion_preview.json").write_text(json.dumps(plan, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    header = ("; SYNTHETIC DEMO: coordinates are invented; not a measured hardware job\n" if synthetic else
              "; Cold nozzle positioning; manually home first and verify the recorded start position\n")
    script = header + "; XYZ in mm; G-code feed is mm/min; no homing/heating/extrusion commands\nG90\n"
    script += "\n".join(b["gcode"] for b in plan["blocks"]) + "\n"
    (output / "motion_preview.gcode").write_text(script, encoding="utf-8")


def require_positioning_setup(config):
    hw = config.hardware
    if not all((hw.stationary_fixture_verified, hw.stop_test_verified, hw.exclusive_control_verified)):
        raise ControllerFault("positioning requires a stationary fixture, tested stop and exclusive control")
    # Camera registration is intentionally not a prerequisite for moves already
    # expressed in printer XYZ: these moves are needed to establish registration.
    if hw.fixture_top_z_mm is None:
        raise ControllerFault("positioning requires the measured fixture envelope")


async def execute_plan(config, plan, output, client=None):
    require_positioning_setup(config)
    client = create_printer_client(config) if client is None else client
    output = Path(output)
    completed = 0
    motion_possible = False
    result = {"state": "FAULT", "motion_sent": False, "completed_blocks": 0}
    events = (output / "motion_events.jsonl").open("x", encoding="utf-8")

    def event(kind, **fields):
        events.write(json.dumps({"kind": kind, "monotonic_s": time.monotonic(), **fields}, allow_nan=False)+"\n")
        events.flush()

    try:
        report = await client.discover()
        verify_cold_state(report, config, plan["start_xyz_mm"])
        identity = tuple(report["printer"].get(k) for k in ("software_version", "process_id"))
        eventtime = report["query"]["eventtime"]
        event("preflight", start_xyz_mm=plan["start_xyz_mm"])
        await client.execute_gcode("G90")
        expected = plan["start_xyz_mm"]
        for block in plan["blocks"]:
            before = await client.discover()
            verify_cold_state(before, config, expected, allow_owned_motion=True)
            if not before["query"]["status"]["gcode_move"].get("absolute_coordinates"):
                raise ControllerFault("absolute-coordinate mode changed during positioning")
            if tuple(before["printer"].get(k) for k in ("software_version", "process_id")) != identity or before["query"]["eventtime"] < eventtime:
                raise ControllerFault("printer restarted during positioning")
            eventtime = before["query"]["eventtime"]
            motion_possible = True
            event("submit", block_index=completed, **block)
            await client.execute_gcode(block["gcode"])  # Exactly once; no automatic retries.
            after = await client.discover()
            verify_cold_state(after, config, block["xyz_mm"], allow_owned_motion=True)
            if not after["query"]["status"]["gcode_move"].get("absolute_coordinates"):
                raise ControllerFault("absolute-coordinate mode changed while a block was in flight")
            if tuple(after["printer"].get(k) for k in ("software_version", "process_id")) != identity or after["query"]["eventtime"] < eventtime:
                raise ControllerFault("printer restarted while a positioning block was in flight")
            eventtime = after["query"]["eventtime"]
            expected = block["xyz_mm"]
            completed += 1
            event("queue_drained", block_index=completed-1, xyz_mm=expected, physical_feedback=False)
        result = {"state": "READY", "motion_sent": motion_possible, "completed_blocks": completed,
                  "path_complete": True, "end_xyz_mm": plan["end_xyz_mm"], "physical_validation": False}
    except (Exception, asyncio.CancelledError) as exc:
        result.update(reason=str(exc) or "interrupted", motion_sent=motion_possible, completed_blocks=completed)
        event("fault", **result)
        if motion_possible:
            try:
                await client.emergency_stop()
                result["stop_requested"] = True
            except Exception as stop_error:
                result["stop_error"] = str(stop_error)
    finally:
        events.close()
        await client.close()
        (output / "execution.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("status", "run", "demo"):
        sub = commands.add_parser(name)
        sub.add_argument("--config", default="configs/controller.example.json")
        if name != "status":
            sub.add_argument("--output", type=Path, required=True)
        if name == "run":
            sub.add_argument("--plan", type=Path, required=True)
            sub.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        if args.command == "status":
            async def status():
                client = create_printer_client(config)
                try:
                    report = await client.discover()
                    values = report["query"]["status"]
                    return {"gcode_xyz_mm": values["gcode_move"]["gcode_position"][:3],
                            "printer_xyz_mm": values["gcode_move"]["position"][:3],
                            "homing_origin_mm": values["gcode_move"]["homing_origin"][:3],
                            "homed_axes": values["toolhead"]["homed_axes"], "motion_sent": False}
                finally:
                    await client.close()
            result = asyncio.run(status())
        else:
            if args.command == "demo":
                config = replace(config, hardware=replace(config.hardware, fixture_top_z_mm=20))
                raw = {"schema_version": 1, "coordinate_frame": "printer", "start_xyz_mm": [110, 120, 30],
                       "travel_z_mm": 35, "xy_speed_mm_s": 5, "z_speed_mm_s": 1, "dwell_s": .5,
                       "waypoints": [{"id": f"P{i+1}", "xyz_mm": [x, y, 25]}
                                     for i, (x, y) in enumerate([(100, 100), (140, 100), (140, 140), (100, 140)])],
                       "return_to_start": False}
            else:
                raw = json.loads(args.plan.read_text(encoding="utf-8-sig"))
            plan = build_plan(raw, config)
            write_preview(args.output, plan, synthetic=args.command == "demo")
            if args.command == "run" and args.execute:
                result = asyncio.run(execute_plan(config, plan, args.output))
            else:
                result = {"state": "PREVIEW", "motion_sent": False, "blocks": len(plan["blocks"]),
                          "estimated_duration_s": plan["estimated_duration_s"], "output": str(args.output)}
        print(json.dumps(result, indent=2, allow_nan=False))
        return 2 if result.get("state") == "FAULT" else 0
    except (ControllerFault, ValueError, TypeError, KeyError, OSError, RuntimeError) as exc:
        print(f"nozzle: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("nozzle: interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
