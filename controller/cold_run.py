"""Elevated stationary-fixture characterization; no live surface corrections."""

from dataclasses import asdict, replace
import asyncio
import json
import time

import numpy as np

from .logging import RunLog
from .connection import create_printer_client
from .moonraker import ColdPrinterTransport, verify_cold_state
from .planner import motion_profile
from .supervisor import Supervisor
from .types import ControllerFault, Move, State, Trajectory


def cold_preview(config, path_xy):
    config.require_cold_hardware()
    z = config.substrate_height_mm + config.desired_gap_mm
    if z < config.hardware.fixture_top_z_mm + config.hardware.cold_clearance_mm:
        raise ControllerFault("desired cold height must clear the entire fixture envelope plus cold_clearance_mm")
    xyz = np.column_stack((path_xy, np.full(len(path_xy), z)))
    if not np.isfinite(xyz).all() or np.any(xyz < config.limits.xyz_min_mm) or np.any(xyz > config.limits.xyz_max_mm):
        raise ControllerFault("cold pattern exceeds machine workspace")
    moves = []
    for a, b in zip(xyz, xyz[1:]):
        duration, feed, _ = motion_profile(a, b, config.limits)
        if duration > config.hardware.max_batch_s:
            raise ControllerFault("one cold characterization move exceeds max_batch_s")
        moves.append(Move(tuple(b), 0, duration, feed, 0, float(np.linalg.norm(b[:2]-a[:2])), 0))
    return tuple(xyz[0]), moves


async def dry_run(config, path_xy, output, execute=False):
    # Validate and create a concrete reviewable path before connection or writes.
    initial_xyz, moves = cold_preview(config, path_xy)
    log = RunLog(output, config)
    preview = {"start_xyz_mm": initial_xyz, "moves": [asdict(m) for m in moves],
               "no_extrusion": True, "stationary_fixture": True,
               "timing": "estimated rest-to-rest; M400 characterization boundaries"}
    (log.path / "cold_preview.json").write_text(json.dumps(preview, indent=2), encoding="utf-8")
    lines = ["; COLD CHARACTERIZATION: manually position at the reviewed start first", "G90", "M83", "M220 S100", "M221 S100"]
    from .transport import gcode_for
    for i, move in enumerate(moves):
        trajectory = Trajectory(i, 0, i-1 if i else None, config.config_id, initial_xyz, 0,
                                (replace(move, arrival_s=move.duration_s),), (), 0)
        lines.append(gcode_for(trajectory, synchronize=True))
    (log.path / "cold_preview.gcode").write_text("\n".join(lines)+"\n", encoding="utf-8")
    if not execute:
        return log.finish({"mode": "cold_preview", "state": "READY", "motion_sent": False,
                           "start_xyz_mm": initial_xyz, "planned_moves": len(moves)})
    supervisor = Supervisor(log.event)
    client = None
    transport = None
    motion_possible = False
    completed = 0
    stop_event = asyncio.Event()
    monitor = None
    monitor_reason = None
    eventtime = None
    baseline_limits = None
    printer_identity = None
    async def watch():
        nonlocal monitor_reason, eventtime
        while not stop_event.is_set():
            await asyncio.sleep(.2)
            try:
                report = await client.discover()
                status = report["query"]["status"]
                current_eventtime = report["query"]["eventtime"]
                if eventtime is not None and current_eventtime < eventtime:
                    raise ControllerFault("printer eventtime restarted")
                eventtime = current_eventtime
                tool, gcode = status["toolhead"], status["gcode_move"]
                if report["printer"].get("state") != "ready" or status["webhooks"]["state"] != "ready":
                    raise ControllerFault("printer restarted/shutdown during cold motion")
                if tuple(report["printer"].get(k) for k in ("software_version", "process_id")) != printer_identity:
                    raise ControllerFault("printer session identity changed during cold motion")
                if gcode["speed_factor"] != 1 or gcode["extrude_factor"] != 1 or not gcode["absolute_coordinates"] or gcode["absolute_extrude"]:
                    raise ControllerFault("coordinate modes or multipliers changed during run")
                if not np.allclose(gcode["homing_origin"][:3], 0, atol=1e-6):
                    raise ControllerFault("coordinate offsets changed during run")
                active = tuple(tool[k] for k in ("max_velocity", "max_accel", "square_corner_velocity"))
                if active != baseline_limits:
                    raise ControllerFault("Klipper motion limits changed during run")
                if status.get("virtual_sdcard", {}).get("is_active"):
                    raise ControllerFault("a competing print started")
                if status.get("bed_mesh", {}).get("profile_name") or status.get("z_thermal_adjust", {}).get("enabled"):
                    raise ControllerFault("height compensation changed during run")
                for heater in (status["extruder"], status["heater_bed"]):
                    if heater["target"] != 0 or heater["temperature"] > config.hardware.max_cold_temperature_c:
                        raise ControllerFault("heater state changed during cold run")
                log.event("telemetry", status=status, eventtime=current_eventtime,
                          receipt_s=report["receipt_s"])
            except Exception as exc:
                monitor_reason = str(exc)
                stop_event.set()
                await supervisor.fault(exc, transport)
                return
    try:
        client = create_printer_client(config)
        transport = ColdPrinterTransport(config, client, log.event)
        report = await client.discover()
        verify_cold_state(report, config, initial_xyz)
        log.event("inventory", report=report)
        supervisor.transition(State.READY)
        supervisor.transition(State.WARMUP)
        supervisor.arm({"cold_hardware_gates": True, "stationary_fixture": True, "reviewed_path": True})
        # Setup explicitly establishes the generated move contract; no homing/heating commands.
        motion_possible = True
        await client.execute_gcode("G90\nM83\nM220 S100\nM221 S100")
        report = await client.discover()
        verify_cold_state(report, config, initial_xyz, require_modes=True, allow_owned_motion=True)
        baseline_limits = tuple(report["query"]["status"]["toolhead"][k] for k in
                                ("max_velocity", "max_accel", "square_corner_velocity"))
        eventtime = report["query"]["eventtime"]
        printer_identity = tuple(report["printer"].get(k) for k in ("software_version", "process_id"))
        supervisor.transition(State.RUNNING)
        monitor = asyncio.create_task(watch())
        xyz = initial_xyz
        for i, move in enumerate(moves):
            if stop_event.is_set():
                raise ControllerFault(monitor_reason or "stop requested")
            config.require_cold_hardware()  # Expiry remains a gate throughout the run.
            before = await client.discover()
            verify_cold_state(before, config, xyz, require_modes=True, allow_owned_motion=True)
            now = time.monotonic()
            if transport.committed_xyz is not None:
                # Only after M400 and verified stationary telemetry: a new characterization boundary.
                transport.committed_end_s = now
                log.event("stationary_boundary", at_s=now, xyz=xyz,
                          physical_feedback=False)
            timed = Move(move.xyz, now+move.duration_s, move.duration_s, move.feed_mm_s, 0, move.surface_length_mm, 0)
            trajectory = Trajectory(i, 0, transport.last_id, config.config_id, xyz, now, (timed,), (),
                                    now+config.hardware.request_timeout_s)
            log.trajectory(trajectory)
            sent_s = time.monotonic()
            await transport.submit(trajectory, now)
            if stop_event.is_set():
                raise ControllerFault(monitor_reason or "stop requested")
            after = await client.discover()
            verify_cold_state(after, config, move.xyz, require_modes=True, allow_owned_motion=True)
            log.event("queue_drained", segment_id=i, submitted_s=sent_s, reply_s=time.monotonic(),
                      expected_duration_s=move.duration_s, physical_completion=False)
            xyz = move.xyz
            completed += 1
        stop_event.set()
        if monitor:
            await monitor
        if supervisor.state == State.FAULT:
            raise ControllerFault(monitor_reason or supervisor.reason)
        supervisor.transition(State.STOPPING)
        supervisor.transition(State.READY)
    except (Exception, asyncio.CancelledError) as exc:
        stop_event.set()
        await supervisor.fault(exc or "operator interrupted", transport if motion_possible else None)
    finally:
        stop_event.set()
        if monitor and not monitor.done():
            monitor.cancel()
            try:
                await monitor
            except asyncio.CancelledError:
                pass
        if client:
            await client.close()
    return log.finish({"mode": "cold_characterization", "state": supervisor.state.value,
                       "reason": supervisor.reason, "completed_moves": completed,
                       "path_complete": completed == len(moves), "motion_sent": motion_possible,
                       "physical_validation": False})
