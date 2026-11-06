"""Experimental camera/ConvLSTM/XYZ-E loop with one queue-drained segment at a time.

This is soft real-time, stop-and-replan control. Moonraker does not provide an
absolute execution timestamp; declared measured timing bounds remain necessary.
No automatic approach, homing, recovery, or lifting is performed.
"""

import argparse
import asyncio
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import io
from pathlib import Path
import threading
import time
import uuid

import numpy as np

from surface_mapper.camera import RealSenseD405
from surface_mapper.config import load_config as load_camera_config
from surface_mapper.heightmap import HeightMapProjector
from .clock_sync import ClockMapper
from .config import fingerprint, load_config
from .connection import create_printer_client
from .logging import RunLog
from .moonraker import verify_cold_state
from .path import load_path
from .planner import Planner, corridor_cells, motion_profile, sample_map
from .predictor_adapter import PredictorAdapter
from .runtime import check_geometry
from .transport import SerializedTransport, gcode_for
from .types import ControllerFault, Observation


@dataclass(frozen=True)
class LiveSettings:
    initial_xyz_mm: tuple | None = None
    exposure_latency_s: float | None = None
    camera_clock_uncertainty_s: float | None = None
    command_start_delay_s: float | None = None
    max_schedule_error_s: float | None = None
    planning_lead_s: float = .1
    max_segment_s: float = .5
    nozzle_temperature_c: float = 220.
    temperature_tolerance_c: float = 5.
    max_heating_s: float = 120.
    max_runtime_s: float = 180.
    sample_wait_s: float = 5.
    poll_s: float = .1
    telemetry_timeout_s: float = .5
    max_segments: int = 100
    calibration_verified: bool = False
    model_verified: bool = False
    timing_verified: bool = False
    deposition_verified: bool = False
    loaded_and_primed: bool = False
    initial_pose_verified: bool = False
    path_clearance_verified: bool = False


def template():
    """Unknown measurements stay null; factual readiness never defaults true."""
    return asdict(LiveSettings())


def load_settings(path):
    raw = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(raw, dict):
        raise ValueError("live settings must be a JSON object")
    try:
        return LiveSettings(**raw)
    except TypeError as exc:
        raise ValueError(f"invalid live settings: {exc}") from exc


def settings_issues(settings, config, path_xy, *, hardware=False):
    issues = []
    for name in ("exposure_latency_s", "camera_clock_uncertainty_s", "command_start_delay_s", "max_schedule_error_s"):
        value = getattr(settings, name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            issues.append(f"{name} requires a measured finite nonnegative value")
    for name in ("planning_lead_s", "max_segment_s", "nozzle_temperature_c", "temperature_tolerance_c",
                 "max_heating_s", "max_runtime_s", "sample_wait_s", "poll_s", "telemetry_timeout_s"):
        value = getattr(settings, name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            issues.append(f"{name} must be finite and positive")
    if type(settings.max_segments) is not int or settings.max_segments < 1:
        issues.append("max_segments must be a positive integer")
    path_xy = np.asarray(path_xy, dtype=float)
    if path_xy.ndim != 2 or path_xy.shape[1] != 2 or len(path_xy) < 2 or not np.isfinite(path_xy).all():
        issues.append("path requires at least two finite XY points")
    elif type(settings.max_segments) is int and len(path_xy) - 1 > settings.max_segments:
        issues.append("path exceeds max_segments; no partial path is automatically selected")
    xyz = settings.initial_xyz_mm
    if not isinstance(xyz, (list, tuple)) or len(xyz) != 3 or any(isinstance(v, bool) or not isinstance(v, (int, float))
            or not math.isfinite(v) for v in xyz):
        issues.append("initial_xyz_mm requires three measured printer coordinates")
    elif path_xy.ndim == 2 and len(path_xy) and not np.allclose(xyz[:2], path_xy[0], atol=1e-6):
        issues.append("initial XYZ must already be at the first path XY; no automatic approach")
    if config.predictor.kind != "convlstm" or getattr(config.predictor, "temporal_policy", "fixed") != "linear_to_horizon":
        issues.append("live segmented planning requires convlstm with explicit linear_to_horizon temporal policy")
    if config.measurement_frame != "printer" or config.height_reference != "printer_z":
        issues.append("live printing requires absolute printer-frame height maps")
    if hardware:
        for name in ("calibration_verified", "model_verified", "timing_verified", "deposition_verified",
                     "loaded_and_primed", "initial_pose_verified", "path_clearance_verified"):
            if getattr(settings, name) is not True:
                issues.append(f"{name} has not been established")
        hw = config.hardware
        if hw.backend != "moonraker" or not hw.moonraker_url:
            issues.append("a Moonraker connection is required")
        if not hw.calibration_id or not hw.camera_config_sha256:
            issues.append("versioned calibration and camera geometry hash are required")
        if not hw.stop_test_verified or not hw.exclusive_control_verified:
            issues.append("tested stop and exclusive printer control are required")
        try:
            expiry = datetime.fromisoformat(hw.expires_utc.replace("Z", "+00:00"))
            if expiry.tzinfo is None or expiry <= datetime.now(timezone.utc):
                raise ValueError()
        except (AttributeError, TypeError, ValueError):
            issues.append("hardware calibration expiry is missing, invalid or expired")
        if config.allowed_gap_error_mm >= config.desired_gap_mm or config.provisional_error_mm >= config.desired_gap_mm:
            issues.append("allowed and provisional error must be smaller than nozzle gap; diagnostic millimetre errors cannot qualify a submillimetre gap")
        if isinstance(settings.max_schedule_error_s, (int, float)) and config.timing_uncertainty_s < settings.max_schedule_error_s:
            issues.append("controller timing_uncertainty_s must include max_schedule_error_s")
        if isinstance(settings.max_segment_s, (int, float)) and isinstance(settings.command_start_delay_s, (int, float)):
            if settings.max_segment_s + settings.command_start_delay_s >= hw.request_timeout_s:
                issues.append("request_timeout_s must cover one segment plus command start delay")
    return issues


class CameraHistory:
    """Always keep the latest full history; inference never processes a FIFO backlog."""
    def __init__(self, app, config, settings, source=None):
        self.app, self.config, self.settings = app, config, settings
        self.source = source if source is not None else RealSenseD405(app.camera)
        self.projector = HeightMapProjector(app.grid, app.camera_to_output)
        self.frames = deque(maxlen=config.predictor.history)
        self.condition = threading.Condition()
        self.stop_event = threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()

    def _run(self):
        try:
            self.source.start()
            if self.app.camera.serial and self.source.device_info.get("serial") != self.app.camera.serial:
                raise ControllerFault("opened camera serial differs from calibrated camera")
            mapper = None
            while not self.stop_event.is_set():
                frame = self.source.capture()
                height = self.projector.project(frame.points_camera_m)
                receipt = frame.host_monotonic_ns / 1e9
                corrected = receipt - self.settings.exposure_latency_s
                if mapper is None:
                    mapper = ClockMapper(frame.timestamp_domain, frame.camera_timestamp_ms, corrected,
                        uncertainty_s=self.settings.camera_clock_uncertainty_s,
                        # ClockMapper adds these two tolerances. Do not silently
                        # double the measured uncertainty used by the planner.
                        max_residual_s=1e-6)
                observed = mapper.map(frame.camera_timestamp_ms, frame.timestamp_domain, corrected, frame.frame_number)
                item = Observation(frame.frame_number, height.height_mm, height.valid, height.point_count,
                    self.config.geometry_id, frame.camera_timestamp_ms, frame.timestamp_domain, receipt, observed,
                    self.settings.camera_clock_uncertainty_s)
                with self.condition:
                    self.frames.append(item)
                    self.condition.notify_all()
        except Exception as exc:
            with self.condition:
                self.error = exc
                self.condition.notify_all()
        finally:
            try:
                self.source.stop()
            except Exception as exc:
                self.error = self.error or exc

    def latest(self, after_frame=None, timeout=5):
        deadline = time.monotonic() + timeout
        with self.condition:
            while True:
                if self.error:
                    raise ControllerFault(f"camera acquisition failed: {self.error}")
                if len(self.frames) == self.frames.maxlen and (after_frame is None or self.frames[-1].frame_id > after_frame):
                    return list(self.frames)
                remaining = deadline - time.monotonic()
                if remaining <= 0 or self.stop_event.is_set():
                    raise ControllerFault("timed out waiting for a fresh complete camera history")
                self.condition.wait(min(remaining, .1))

    def health(self, now):
        with self.condition:
            if self.error:
                raise ControllerFault(f"camera acquisition failed: {self.error}")
            if self.frames and now - self.frames[-1].observed_s > self.config.predictor.max_frame_age_s:
                raise ControllerFault("camera stopped supplying fresh observations")

    def current(self):
        with self.condition:
            if self.error:
                raise ControllerFault(f"camera acquisition failed: {self.error}")
            if not self.frames:
                raise ControllerFault("no current depth observation is available")
            return self.frames[-1]

    def close(self):
        self.stop_event.set()
        with self.condition:
            self.condition.notify_all()
        if self.thread.ident is not None:
            self.thread.join(timeout=10)
            if self.thread.is_alive():
                raise ControllerFault("camera worker failed to stop")


def _json_safe(value):
    return json.loads(json.dumps(value), parse_constant=lambda item: "NONFINITE:" + item)


def _atomic_bytes(path, data):
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def _save_clearance(path, observed):
    buffer = io.BytesIO()
    np.savez_compressed(buffer, height_mm=observed.height_mm, valid=observed.valid,
        point_count=observed.point_count, frame_id=observed.frame_id,
        observed_s=observed.observed_s, uncertainty_s=observed.uncertainty_s)
    _atomic_bytes(path, buffer.getvalue())


def _save_forecast(path, fixed, policy, frame_ids):
    buffer = io.BytesIO()
    np.savez_compressed(buffer, prediction_mm=fixed.prediction_mm,
        source_timestamp_ms=fixed.source_timestamp_ms, target_timestamp_ms=fixed.target_timestamp_ms,
        observation_valid=fixed.observation_valid, inference_ms=fixed.inference_ms,
        temporal_policy=policy, source_frame_ids=frame_ids)
    _atomic_bytes(path, buffer.getvalue())


def _segment_distance(position, a, b):
    a, b, position = np.asarray(a), np.asarray(b), np.asarray(position)
    delta = b - a
    d = float(delta @ delta)
    fraction = 0 if d == 0 else float(np.clip((position - a) @ delta / d, 0, 1))
    return float(np.linalg.norm(position - (a + fraction * delta)))


def smoothing_contract(tool):
    """Select only a status-confirmed Klipper acceleration/cruise interface."""
    if "minimum_cruise_ratio" in tool:
        field, parameter = "minimum_cruise_ratio", "MINIMUM_CRUISE_RATIO"
    elif "max_accel_to_decel" in tool:
        field, parameter = "max_accel_to_decel", "ACCEL_TO_DECEL"
    else:
        raise ControllerFault("installed Klipper lacks a recognized cruise/acceleration status contract")
    value = tool[field]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ControllerFault("installed cruise/acceleration setting is invalid")
    if (field == "minimum_cruise_ratio" and value >= 1) or (field == "max_accel_to_decel" and value <= 0):
        raise ControllerFault("installed cruise/acceleration setting is outside the firmware contract")
    return field, parameter, float(value)


def segment_acceleration(trajectory):
    move = trajectory.moves[0]
    distance = float(np.linalg.norm(np.asarray(move.xyz) - trajectory.start_xyz))
    ramp_s = move.duration_s - distance / move.feed_mm_s
    if ramp_s <= 0:
        raise ControllerFault("planned segment has no valid acceleration phase")
    return move.feed_mm_s / ramp_s


def verify_observed_clearance(observation, config, xyz, now):
    """Check measured support/gap even while heating or waiting for inference.

    This is sampled camera feedback, not a guarantee between frames or an
    independent physical clearance sensor.
    """
    age = now - observation.observed_s
    if age < -observation.uncertainty_s or age + observation.uncertainty_s > config.predictor.max_frame_age_s:
        raise ControllerFault("current clearance observation is stale")
    support = (observation.valid & (observation.point_count >= config.grid.min_points_per_cell)
               & (observation.height_mm >= config.grid.z_min_mm) & (observation.height_mm <= config.grid.z_max_mm))
    footprint = corridor_cells(config.grid, xyz[:2], xyz[:2], config.footprint_radius_mm)
    if not support[footprint].all():
        raise ControllerFault("current nozzle footprint has missing measured depth support")
    surface = config.surface_z(sample_map(observation.height_mm, support, config.grid, xyz[:2]))
    highest = config.surface_z(float(np.max(observation.height_mm[footprint])))
    budget = config.calibration_error_mm + config.mechanical_error_mm
    if xyz[2] - highest <= budget:
        raise ControllerFault("current measured surface violates nozzle-footprint clearance")
    gap = xyz[2] - surface
    if abs(gap - config.desired_gap_mm) + budget > config.allowed_gap_error_mm:
        raise ControllerFault("current measured nozzle gap left the allowed envelope")
    return {"observed_frame_id": observation.frame_id, "observed_gap_mm": gap,
            "highest_footprint_clearance_mm": xyz[2] - highest, "measurement_budget_mm": budget}


def verify_hot_state(report, config, settings, baseline, *, expected_xyz=None, segment=None, require_temperature=False):
    """No coordinate fallback: an unavailable/zero live pose is a rejected sample."""
    values = report["query"]["status"]
    needed = {"webhooks", "toolhead", "gcode_move", "motion_report", "extruder", "heater_bed", "virtual_sdcard", "print_stats"}
    if not needed.issubset(values):
        raise ControllerFault("required live printer status objects are missing")
    if report["printer"].get("state") != "ready" or values["webhooks"].get("state") != "ready":
        raise ControllerFault("printer is not ready")
    identity = tuple(report["printer"].get(k) for k in ("software_version", "process_id"))
    if identity != baseline["identity"] or report["query"]["eventtime"] < baseline["eventtime"]:
        raise ControllerFault("printer session changed or restarted")
    if not set("xyz").issubset(values["toolhead"].get("homed_axes", "")):
        raise ControllerFault("printer homing reference was lost")
    if values["virtual_sdcard"].get("is_active") or values["print_stats"].get("state") in {"printing", "paused"}:
        raise ControllerFault("competing or paused print is present")
    tool, gcode, motion = values["toolhead"], values["gcode_move"], values["motion_report"]
    if tuple(tool.get(k) for k in ("max_velocity", "max_accel", "square_corner_velocity")) != baseline["limits"]:
        raise ControllerFault("active printer motion limits changed")
    if tool.get(baseline["smoothing_field"]) != baseline["smoothing_value"]:
        raise ControllerFault("active printer cruise/acceleration setting changed")
    if gcode.get("absolute_coordinates") is not True or gcode.get("absolute_extrude") is not False:
        raise ControllerFault("XYZ/E parsing modes changed")
    if gcode.get("speed_factor") != 1 or gcode.get("extrude_factor") != 1:
        raise ControllerFault("speed or extrusion multiplier changed")
    if not np.allclose(gcode["homing_origin"][:3], 0, atol=1e-6) or not np.allclose(
            gcode["position"][:3], gcode["gcode_position"][:3], atol=1e-6):
        raise ControllerFault("XYZ coordinate offsets changed")
    if values.get("bed_mesh", {}).get("profile_name") or values.get("z_thermal_adjust", {}).get("enabled"):
        raise ControllerFault("height compensation changed")
    position = np.asarray(motion["live_position"][:3], dtype=float)
    if position.shape != (3,) or not np.isfinite(position).all() or np.any(position < config.limits.xyz_min_mm) or np.any(position > config.limits.xyz_max_mm):
        raise ControllerFault("reported live XYZ is invalid or outside workspace")
    if segment is not None and _segment_distance(position, segment.start_xyz, segment.moves[-1].xyz) > .1:
        raise ControllerFault(f"reported live XYZ left submitted segment: {position.tolist()}")
    if expected_xyz is not None:
        if not np.isfinite(motion["live_velocity"]) or abs(motion["live_velocity"]) > 1e-4:
            raise ControllerFault("printer has not reached a stationary boundary")
        if not np.allclose(position, expected_xyz, atol=.02) or not np.allclose(gcode["position"][:3], expected_xyz, atol=.02):
            raise ControllerFault("reported stationary position differs from confirmed endpoint")
    heater = values["extruder"]
    temp, target = heater["temperature"], heater["target"]
    if not np.isfinite([temp, target]).all() or target != settings.nozzle_temperature_c or temp > min(
            settings.nozzle_temperature_c + 15, baseline["max_temp"]):
        raise ControllerFault("extruder temperature/setpoint left run envelope")
    if require_temperature and (abs(temp - target) > settings.temperature_tolerance_c or temp < baseline["min_extrude_temp"]):
        raise ControllerFault("extruder is outside deposition temperature range")
    if values["heater_bed"].get("target") != baseline["bed_target"]:
        raise ControllerFault("bed heater setpoint changed")
    return tuple(float(v) for v in position)


class LiveTransport(SerializedTransport):
    def __init__(self, config, client, log):
        super().__init__(config.hardware.max_batch_s, config.config_id, log)
        self.cold = False
        self.client = client

    async def _send(self, trajectory):
        if len(trajectory.moves) != 1:
            raise ControllerFault("live transport allows exactly one in-flight segment")
        return await self.client.execute_gcode(gcode_for(trajectory, cold=False, synchronize=True))

    async def emergency_stop(self):
        return await self.client.emergency_stop()


def _plan(predictor, planner, observations, settings, xyz, next_xy, segment_id, predecessor_id, clock, *, extrude):
    predictor.reset("new stationary boundary; newest complete history")
    for observation in observations:
        predictor.push(observation)
    if not predictor.ready:
        raise ControllerFault("latest camera history is discontinuous or lacks model support")
    now = clock()
    start = now + settings.planning_lead_s + settings.command_start_delay_s
    trajectory = planner.plan(predictor, xyz, [next_xy], start, now, segment_id, predecessor_id, extrude=extrude)
    if trajectory.moves[0].duration_s > settings.max_segment_s:
        raise ControllerFault("planned move exceeds maximum segment duration")
    if clock() > trajectory.start_s - settings.command_start_delay_s:
        raise ControllerFault("prediction/planning missed the submission deadline")
    return trajectory


def inspect_inputs(config, app, path_xy, settings, *, hardware=False, predictor_factory=PredictorAdapter):
    issues = settings_issues(settings, config, path_xy, hardware=hardware)
    predictor, geometry, horizon = None, None, None
    try:
        geometry = check_geometry(config, app.to_dict())
        if hardware and not app.camera.serial:
            issues.append("hardware execution requires a pinned camera serial")
        predictor = predictor_factory(config, geometry)
        horizon = predictor.model.config.horizon * predictor.model.config.expected_dt_ms / 1000.
        if abs(predictor.model.config.expected_dt_ms - 1000. / app.camera.fps) > 1e-6:
            issues.append("model training cadence differs from camera FPS")
        if not issues:
            a, b = path_xy[:2]
            earliest = motion_profile((*a, settings.initial_xyz_mm[2]), (*b, settings.initial_xyz_mm[2]), config.limits)[0]
            required = earliest + settings.planning_lead_s + settings.command_start_delay_s + settings.exposure_latency_s
            if required >= horizon or required >= config.predictor.max_horizon_s:
                issues.append("model forecast horizon is shorter than even the first XY segment plus declared latency")
    except (OSError, ValueError, RuntimeError, ImportError, AttributeError) as exc:
        issues.append(str(exc))
    return {"state": "BLOCKED" if issues else "READY", "issues": issues, "model_horizon_s": horizon,
            "hardware_timing_estimated": True, "physical_validation": False}, predictor, geometry


async def run_live(config, app, path_xy, settings, output, *, execute=False, history=None, client=None,
                   predictor_factory=PredictorAdapter, planner_factory=Planner, clock=time.monotonic):
    """Injectable core. Shadow runs the identical predictor/planner without a printer."""
    path_xy = np.asarray(path_xy, dtype=float)
    log = RunLog(output, config)
    report = {"mode": "segmented_live_print" if execute else "segmented_live_shadow", "state": "FAULT",
              "path_complete": False, "completed_segments": 0, "printer_commands_sent": 0,
              "hardware_timing_estimated": True, "physical_validation": False,
              "temporal_contract": "ConvLSTM endpoint forecast with bounded linear temporal interpolation"}
    stop_event = asyncio.Event()
    status_lock = asyncio.Lock()
    stop_lock = asyncio.Lock()
    monitor = None
    watchdog = None
    active = None
    xyz = None
    heated = False
    deposition_ready = False
    monitor_fault = None
    baseline = None
    transport = None
    history_started = False
    started = clock()
    last_frame = None
    last_logged_frame = -1
    motion_generation = 0

    async def snapshot(stage, value):
        saved = _json_safe({"stage": stage, "report": value, "active_segment": asdict(active) if active else None})
        log.event("status_snapshot", **saved)
        await asyncio.to_thread(_atomic_bytes, log.path / "last_status_snapshot.json",
            (json.dumps(saved, indent=2, allow_nan=False) + "\n").encode("utf-8"))

    async def request_stop(reason):
        nonlocal monitor_fault
        monitor_fault = monitor_fault or str(reason)
        stop_event.set()
        async with stop_lock:
            if report.get("stop_attempted"):
                return
            report["stop_attempted"] = True
            try:
                await asyncio.wait_for(client.emergency_stop(), timeout=settings.telemetry_timeout_s)
                report["stop_requested"] = True
            except Exception as stop_error:
                report["stop_error"] = str(stop_error) or type(stop_error).__name__

    async def read_status(stage, *, stationary=None, monitor_auto=False):
        async with status_lock:
            for _ in range(3):
                generation = motion_generation
                expected = xyz if monitor_auto and active is None else None if monitor_auto else stationary
                value = await asyncio.wait_for(client.discover(), timeout=settings.telemetry_timeout_s)
                await snapshot(stage, value)
                observed = history.current()
                await asyncio.to_thread(_save_clearance, log.path / "last_clearance_observation.npz", observed)
                # A query begun before a submission/endpoint transition can
                # contain an old pose. Retain it, then obtain a new sample.
                if generation != motion_generation:
                    log.event("status_generation_changed", stage=stage, old_generation=generation,
                              new_generation=motion_generation)
                    continue
                verify_hot_state(value, config, settings, baseline, expected_xyz=expected,
                                 segment=active, require_temperature=deposition_ready)
                clearance = verify_observed_clearance(observed, config,
                    value["query"]["status"]["motion_report"]["live_position"][:3], clock())
                log.event("measured_clearance", stage=stage, **clearance)
                baseline["eventtime"] = value["query"]["eventtime"]
                return value
            raise ControllerFault("telemetry repeatedly crossed a motion-state transition")

    async def watch():
        nonlocal monitor_fault
        try:
            while not stop_event.is_set():
                await asyncio.sleep(settings.poll_s)
                if stop_event.is_set():
                    break
                await read_status("monitor", monitor_auto=True)
        except (Exception, asyncio.CancelledError) as exc:
            if isinstance(exc, asyncio.CancelledError):
                raise
            await request_stop(str(exc) or type(exc).__name__)

    async def watch_camera():
        try:
            while not stop_event.is_set():
                await asyncio.sleep(min(settings.poll_s, .05))
                if stop_event.is_set():
                    return
                if clock() - started > settings.max_runtime_s:
                    raise ControllerFault("live run exceeded bounded runtime")
                history.health(clock())
                if log.error or log.dropped:
                    raise ControllerFault("run logging failed or dropped records")
                if active is None:
                    verify_observed_clearance(history.current(), config, xyz, clock())
        except (Exception, asyncio.CancelledError) as exc:
            if isinstance(exc, asyncio.CancelledError):
                raise
            await request_stop(str(exc) or type(exc).__name__)

    def check_running():
        if stop_event.is_set():
            raise ControllerFault(monitor_fault or "run stopped")
        if clock() - started > settings.max_runtime_s:
            raise ControllerFault("live run exceeded bounded runtime")
        if log.error or log.dropped:
            raise ControllerFault("run logging failed or dropped records")

    try:
        (log.path / "live_settings.json").write_text(json.dumps(asdict(settings), indent=2) + "\n", encoding="utf-8")
        (log.path / "camera_config.json").write_text(json.dumps(app.to_dict(), indent=2) + "\n", encoding="utf-8")
        log.event("original_path", xy_mm=path_xy.tolist(), execute=execute)
        check, predictor, geometry = inspect_inputs(config, app, path_xy, settings, hardware=execute,
                                                    predictor_factory=predictor_factory)
        (log.path / "checks.json").write_text(json.dumps(check, indent=2) + "\n", encoding="utf-8")
        if check["issues"]:
            raise ControllerFault("; ".join(check["issues"]))
        if Path(config.predictor.checkpoint).is_file():
            log.event("checkpoint", sha256=hashlib.sha256(Path(config.predictor.checkpoint).read_bytes()).hexdigest())
        log.event("camera_geometry", geometry=geometry, sha256=fingerprint(geometry))
        planner = planner_factory(config)
        xyz = tuple(settings.initial_xyz_mm)
        history = history if history is not None else CameraHistory(app, config, settings)
        history.start()
        history_started = True
        if execute:
            client = create_printer_client(config) if client is None else client
            transport = LiveTransport(config, client, log.event)
            value = await asyncio.wait_for(client.discover(), timeout=settings.telemetry_timeout_s)
            await snapshot("cold_preflight", value)
            verify_cold_state(value, config, xyz)
            values = value["query"]["status"]
            if values.get("print_stats", {}).get("state") in {"printing", "paused"}:
                raise ControllerFault("another print is loaded or paused")
            installed = values["configfile"]["settings"].get("extruder", {})
            minimum, maximum = float(installed["min_extrude_temp"]), float(installed["max_temp"])
            if not minimum <= settings.nozzle_temperature_c < maximum:
                raise ControllerFault("nozzle target is outside installed extrusion-temperature limits")
            baseline = {"identity": tuple(value["printer"].get(k) for k in ("software_version", "process_id")),
                "eventtime": value["query"]["eventtime"], "bed_target": values["heater_bed"]["target"],
                "max_temp": maximum, "min_extrude_temp": minimum,
                "limits": tuple(values["toolhead"].get(k) for k in ("max_velocity", "max_accel", "square_corner_velocity"))}
            field, parameter, smoothing = smoothing_contract(values["toolhead"])
            baseline.update(smoothing_field=field, smoothing_parameter=parameter, smoothing_value=smoothing,
                original_smoothing=smoothing, original_limits=baseline["limits"])
            # Verify an operational camera before issuing even a heating command.
            await asyncio.to_thread(history.latest, None, settings.sample_wait_s)
            verify_observed_clearance(history.current(), config, xyz, clock())
            heated = True  # Before write: an ambiguous response must still stop.
            report["printer_commands_sent"] += 1
            await client.execute_gcode(f"G90\nM83\nM220 S100\nM221 S100\nM104 S{settings.nozzle_temperature_c:.3f}")
            monitor = asyncio.create_task(watch())
            watchdog = asyncio.create_task(watch_camera())
            heat_deadline = clock() + settings.max_heating_s
            while True:
                check_running()
                value = await read_status("heating", stationary=xyz)
                if abs(value["query"]["status"]["extruder"]["temperature"] - settings.nozzle_temperature_c) <= settings.temperature_tolerance_c:
                    break
                if clock() >= heat_deadline:
                    raise ControllerFault("nozzle did not reach temperature before deadline")
                await asyncio.sleep(settings.poll_s)
            deposition_ready = True
        for index, next_xy in enumerate(path_xy[1:]):
            check_running()
            if execute:
                issues = settings_issues(settings, config, path_xy, hardware=True)
                if issues:
                    raise ControllerFault("; ".join(issues))
                await read_status("stationary_boundary", stationary=xyz)
            observations = await asyncio.to_thread(history.latest, last_frame, settings.sample_wait_s)
            last_frame = observations[-1].frame_id
            for observation in observations:
                if observation.frame_id > last_logged_frame:
                    log.observation(observation)
                    last_logged_frame = observation.frame_id
            predecessor = transport.last_id if execute else index - 1 if index else None
            proposal = await asyncio.to_thread(_plan, predictor, planner, observations, settings, xyz, next_xy,
                                               index, predecessor, clock, extrude=execute)
            check_running()
            log.trajectory(proposal)
            fixed = getattr(predictor, "fixed", None)
            if fixed is not None:
                await asyncio.to_thread(_save_forecast, log.path / f"convlstm_forecast_{index:06d}.npz",
                    fixed, config.predictor.temporal_policy, proposal.source_frame_ids)
            send_at = proposal.start_s - settings.command_start_delay_s
            if clock() > send_at:
                raise ControllerFault("trajectory expired after planning; no command sent")
            if execute:
                # Klipper's global scalar acceleration otherwise differs from
                # the projected scalar used by Planner.motion_profile. Apply
                # and read back the exact scalar before the G1; disable cruise
                # smoothing through the installed firmware's reported interface.
                accel = float(f"{segment_acceleration(proposal):.9f}")
                if accel > baseline["original_limits"][1] + 1e-6:
                    raise ControllerFault("planned scalar acceleration exceeds preflight active limit")
                smoothing = 0. if baseline["smoothing_field"] == "minimum_cruise_ratio" else accel
                async with status_lock:
                    check_running()
                    report["printer_commands_sent"] += 1
                    await client.execute_gcode(f"SET_VELOCITY_LIMIT ACCEL={accel:.9f} "
                        f"{baseline['smoothing_parameter']}={smoothing:.9f}")
                    changed = await asyncio.wait_for(client.discover(), timeout=settings.telemetry_timeout_s)
                    await snapshot("segment_timing_parameters", changed)
                    changed_tool = changed["query"]["status"]["toolhead"]
                    if changed_tool.get("max_accel") != accel or changed_tool.get(baseline["smoothing_field"]) != smoothing:
                        raise ControllerFault("Klipper did not apply the planned acceleration/cruise parameters")
                    baseline["limits"] = (baseline["limits"][0], accel, baseline["limits"][2])
                    baseline["smoothing_value"] = smoothing
                    verify_hot_state(changed, config, settings, baseline, expected_xyz=xyz, require_temperature=True)
                await asyncio.sleep(max(0., send_at - clock()))
                check_running()
                if clock() > send_at + settings.max_schedule_error_s or clock() > proposal.start_s:
                    raise ControllerFault("submission missed measured scheduling envelope")
                if transport.committed_xyz is not None:
                    if not np.allclose(transport.committed_xyz, xyz, atol=1e-6):
                        raise ControllerFault("last confirmed endpoint no longer joins proposed move")
                    # Only after M400 + read_status(stationary_boundary) above.
                    transport.committed_end_s = proposal.start_s
                    log.event("new_stationary_time_boundary", at_s=proposal.start_s, xyz=xyz)
                active = proposal
                motion_generation += 1
                report["printer_commands_sent"] += 1
                await transport.submit(proposal, clock())
                check_running()
                reply_at = clock()
                log.event("queue_drained", expected_end_s=proposal.end_s, reply_s=reply_at, physical_feedback=False)
                if abs(reply_at - proposal.end_s) > settings.max_schedule_error_s:
                    raise ControllerFault("M400 reply time left measured scheduling envelope")
                await read_status("confirmed_endpoint", stationary=proposal.moves[-1].xyz)
            xyz = proposal.moves[-1].xyz
            active = None
            motion_generation += 1
            report["completed_segments"] += 1
        if execute:
            stop_event.set()
            await monitor
            await watchdog
            if monitor_fault:
                raise ControllerFault(monitor_fault)
            report["printer_commands_sent"] += 1
            await client.execute_gcode("M104 S0\n"
                f"SET_VELOCITY_LIMIT ACCEL={baseline['original_limits'][1]:.9f} "
                f"{baseline['smoothing_parameter']}={baseline['original_smoothing']:.9f}")
            final_status = await asyncio.wait_for(client.discover(), timeout=settings.telemetry_timeout_s)
            await snapshot("normal_heater_off", final_status)
            if final_status["query"]["status"]["extruder"]["target"] != 0:
                raise ControllerFault("heater-off command was not confirmed")
            final_tool = final_status["query"]["status"]["toolhead"]
            if final_tool.get("max_accel") != baseline["original_limits"][1] or final_tool.get(baseline["smoothing_field"]) != baseline["original_smoothing"]:
                raise ControllerFault("normal-completion acceleration restoration was not confirmed")
        report.update(state="COMPLETE" if execute else "PREVIEW", path_complete=True,
                      final_xyz_mm=xyz, reason=None, automatic_lift=False)
    except (Exception, asyncio.CancelledError) as exc:
        report["reason"] = str(exc) or "interrupted"
        log.event("fault", reason=report["reason"])
        if execute and heated and client is not None:
            await request_stop(report["reason"])
    finally:
        stop_event.set()
        if monitor and not monitor.done():
            monitor.cancel()
            try:
                await monitor
            except asyncio.CancelledError:
                pass
        if watchdog and not watchdog.done():
            watchdog.cancel()
            try:
                await watchdog
            except asyncio.CancelledError:
                pass
        if history_started:
            try:
                await asyncio.to_thread(history.close)
            except Exception as exc:
                report.update(state="FAULT", reason=str(exc))
        if client is not None:
            try:
                await client.close()
            except Exception as exc:
                report.update(state="FAULT", cleanup_error=str(exc))
        report = await asyncio.to_thread(log.finish, report)
        if not report["logging_complete"]:
            report.update(state="FAULT", reason="live run logging was incomplete")
            (log.path / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("check", "shadow", "run"):
        sub = commands.add_parser(name)
        for key in ("config", "camera-config", "path", "settings", "output"):
            sub.add_argument("--" + key, required=True)
        if name == "run":
            sub.add_argument("--execute", action="store_true", help="send experimental segmented XYZ/E commands after factual readiness checks")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        # Keep CPU inference bounded on Pi; this does not promise a frame rate.
        import torch
        torch.set_num_threads(4)
        app = load_camera_config(args.camera_config)
        settings = load_settings(args.settings)
        path = load_path(args.path, config.path_spacing_mm)
        if args.command == "check":
            report, _, _ = inspect_inputs(config, app, path, settings, hardware=True)
            output = Path(args.output)
            output.mkdir(parents=True, exist_ok=False)
            (output / "checks.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        else:
            report = asyncio.run(run_live(config, app, path, settings, args.output,
                                          execute=args.command == "run" and args.execute))
        print(json.dumps(report, indent=2, allow_nan=False))
        return 2 if report["state"] in {"FAULT", "BLOCKED"} else 0
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        print(f"live_print: {exc}")
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
