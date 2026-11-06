"""Bounded acquisition and receding-horizon offline execution.

The planner commits only its earliest move. Proposals after that point are
discarded and recomputed from new observations; committed endpoints are fixed.
"""

from dataclasses import asdict, replace
from pathlib import Path
import json
import queue
import threading
import time

import numpy as np

from surface_mapper.config import load_config as load_camera_config
from surface_mapper.camera import RealSenseD405
from surface_mapper.heightmap import HeightMapProjector

from .clock_sync import ClockMapper
from .config import fingerprint
from .logging import RunLog
from .planner import Planner, corridor_cells, profile_fraction, sample_map
from .predictor_adapter import PredictorAdapter
from .simulator import MovingPlane
from .supervisor import Supervisor
from .transport import FakePrinter
from .types import ControllerFault, Observation, State


def geometry_signature(app_dict):
    # Identical stream/geometry fields to prediction.data, without importing PyTorch.
    from surface_mapper.config import AppConfig, CameraConfig, GridConfig
    app = AppConfig(app_dict["frame_name"], CameraConfig(**app_dict["camera"]),
                    GridConfig(**app_dict["grid"]), app_dict["camera_to_output"])
    signature = app.to_dict()
    signature["camera"] = {k: v for k, v in signature["camera"].items()
                           if k not in {"serial", "warmup_frames", "timeout_ms"}}
    return signature


def check_geometry(config, app_dict):
    geometry = geometry_signature(app_dict)
    if geometry["frame_name"] != config.measurement_frame or geometry["grid"] != asdict(config.grid):
        raise ControllerFault("camera/recording frame and grid differ from controller; camera-depth maps cannot drive printer Z")
    expected = config.hardware.camera_config_sha256
    if expected and fingerprint(geometry) != expected:
        raise ControllerFault("camera transform/stream hash differs from calibrated run configuration")
    return geometry


def recording_observations(config, session_path):
    directory = Path(session_path)
    if (directory / "events.jsonl").exists():
        saved_config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        if saved_config["grid"] != asdict(config.grid) or saved_config["geometry_id"] != config.geometry_id or saved_config["height_reference"] != config.height_reference:
            raise ControllerFault("controller recording geometry/height convention differs")
        report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
        if not report["logging_complete"]:
            raise ControllerFault("controller recording has logging drops/errors")
        events = []
        with (directory / "events.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                raw = json.loads(line)
                if raw["kind"] == "observation":
                    events.append(raw)
        if not events:
            raise ControllerFault("controller recording has no observations")
        def iterator():
            for raw in events:
                file = (directory / raw["file"]).resolve()
                if file.parent != directory.resolve():
                    raise ControllerFault("recording file leaves session directory")
                with np.load(file, allow_pickle=False) as data:
                    observation = Observation(raw["frame_id"], data["height_mm"], data["valid"], data["point_count"],
                                              raw["geometry_id"], raw["camera_time_ms"], raw["timestamp_domain"],
                                              raw["receipt_s"], raw["observed_s"], raw["uncertainty_s"])
                yield observation
        return iterator(), None
    from prediction.data import load_session
    session = load_session(directory)
    geometry = check_geometry(config, session.manifest["config"])
    origin = session.frames[0].camera_timestamp_ms
    def iterator():
        for frame in session.frames:
            with np.load(frame.path, allow_pickle=False) as data:
                # Virtual camera timeline for offline comparisons, not calibrated USB latency.
                at_s = (frame.camera_timestamp_ms-origin) / 1000
                observation = Observation(frame.frame_number, data["height_mm"], data["valid"], data["point_count"],
                                          config.geometry_id, frame.camera_timestamp_ms, frame.timestamp_domain, at_s, at_s)
            yield observation
    return iterator(), geometry


async def run_offline(config, path_xy, output, observations, truth=None, model_geometry=None,
                      fail_on_segment=None, failure="lost-reply", allow_extrusion=False):
    log = RunLog(output, config)
    log.event("original_path", xy_mm=np.asarray(path_xy).tolist(), virtual_only=True)
    supervisor = Supervisor(log.event)
    transport = FakePrinter(config.predictor.max_horizon_s, config.config_id, log.event,
                            fail_on_segment, failure)
    planner = Planner(config)
    gap_errors, horizons, planning_ms, frame_ages = [], [], [], []
    next_index, segment, pending, start_xyz, start_s = 1, 0, None, None, None
    completed = False
    frame_count = 0
    last_now = None
    gcode = ["G90", "M83"]
    from .transport import gcode_for
    def virtual_reference(trajectory):
        if truth is None:
            return
        # Independent analytical truth is used for evaluation only, never supplied to planner.
        xyz, t = trajectory.start_xyz, trajectory.start_s
        for move in trajectory.moves:
            distance = float(np.linalg.norm(np.asarray(move.xyz)-xyz))
            for elapsed in np.linspace(0, move.duration_s, 21):
                fraction = profile_fraction(elapsed, move.duration_s, distance, move.feed_mm_s)
                position = np.asarray(xyz) + fraction*(np.asarray(move.xyz)-xyz)
                measured_gap = position[2] - config.surface_z(float(truth.height(position[:2], t+elapsed)))
                gap_errors.append(measured_gap-config.desired_gap_mm)
                log.event("reference", observed_s=t+elapsed, xyz=position.tolist(), gap_mm=measured_gap,
                          reference_kind="analytical_simulation")
            xyz, t = move.xyz, move.arrival_s
    try:
        config.surface_z(0.0)
        predictor = PredictorAdapter(config, model_geometry)
        supervisor.transition(State.READY)
        supervisor.transition(State.WARMUP)
        for observation in observations:
            now = observation.observed_s
            if last_now is not None and now <= last_now:
                raise ControllerFault("observation clock reversed or repeated")
            last_now = now
            log.observation(observation)
            frame_count += 1
            # At a completion event, plan using only observations already acquired before it.
            while pending is not None and pending.end_s <= now:
                virtual_reference(pending)
                start_xyz, start_s = pending.moves[-1].xyz, pending.end_s
                pending = None
                if next_index >= len(path_xy):
                    completed = True
                    break
                started = time.perf_counter()
                proposal = planner.plan(predictor, start_xyz,
                    path_xy[next_index:next_index+config.proposal_moves], start_s, start_s, segment,
                    transport.last_id, allow_extrusion)
                planning_ms.append((time.perf_counter()-started)*1000)
                log.trajectory(proposal)
                pending = replace(proposal, moves=proposal.moves[:1])
                await transport.submit(pending, start_s)
                gcode.append(gcode_for(pending, cold=not allow_extrusion))
                horizons.append(pending.end_s - predictor.buffer[-1].observed_s)
                frame_ages.append(start_s - predictor.buffer[-1].observed_s)
                next_index += 1
                segment += 1
            if completed:
                break
            predictor.push(observation)
            if predictor.reset_reason and supervisor.state == State.RUNNING:
                raise ControllerFault(predictor.reset_reason)
            if supervisor.state == State.RUNNING and pending is not None:
                supported = predictor.forecast(now, now)
                corridor = corridor_cells(config.grid, pending.start_xyz[:2], pending.moves[-1].xyz[:2],
                                          config.footprint_radius_mm)
                if not supported.support[corridor].all():
                    raise ControllerFault("lost observation support over committed path corridor")
            if supervisor.state == State.WARMUP and predictor.ready:
                latest = predictor.forecast(now, now)
                initial_surface = sample_map(latest.height_mm, latest.support, config.grid, path_xy[0])
                start_xyz = (*path_xy[0], config.surface_z(initial_surface)+config.desired_gap_mm)
                start_s = now
                started = time.perf_counter()
                proposal = planner.plan(predictor, start_xyz,
                    path_xy[next_index:next_index+config.proposal_moves], start_s, now, segment,
                    None, allow_extrusion)
                planning_ms.append((time.perf_counter()-started)*1000)
                log.trajectory(proposal)
                supervisor.arm({"history": True, "path_validated": True, "virtual_printer": True})
                supervisor.transition(State.RUNNING)
                pending = replace(proposal, moves=proposal.moves[:1])
                await transport.submit(pending, now)
                gcode.append(gcode_for(pending, cold=not allow_extrusion))
                horizons.append(pending.end_s - predictor.buffer[-1].observed_s)
                frame_ages.append(0.0)
                next_index += 1
                segment += 1
            elif pending is not None and truth is None:
                # Recorded camera maps are a sensor comparison, not independent physical gap truth.
                move = pending.moves[0]
                elapsed = now - pending.start_s
                if 0 <= elapsed <= move.duration_s:
                    distance = float(np.linalg.norm(np.asarray(move.xyz)-pending.start_xyz))
                    fraction = profile_fraction(elapsed, move.duration_s, distance, move.feed_mm_s)
                    position = np.asarray(pending.start_xyz) + fraction*(np.asarray(move.xyz)-pending.start_xyz)
                    measured = sample_map(observation.height_mm, observation.valid, config.grid, position[:2])
                    gap_errors.append(position[2]-config.surface_z(measured)-config.desired_gap_mm)
        if not completed:
            raise ControllerFault("input ended before path completion; no automatic extension of observations")
        supervisor.transition(State.STOPPING)
        supervisor.transition(State.READY)
    except Exception as exc:
        await supervisor.fault(exc, transport)
    def statistics(values):
        if len(values) == 0:
            return None
        return {"median": float(np.median(values)), "p95": float(np.percentile(values, 95)),
                "p99": float(np.percentile(values, 99)), "max": float(np.max(values))}
    report = {"mode": "simulation" if truth else "replay", "state": supervisor.state.value,
              "reason": supervisor.reason, "path_complete": completed, "frames": frame_count,
              "submitted_segments": len(transport.executed), "uncertain_motion": transport.uncertain,
              "reference_kind": "analytical_simulation" if truth else "recorded_camera_comparison",
              "absolute_gap_error_mm": statistics(np.abs(gap_errors)),
              "forecast_horizon_s": statistics(horizons), "planning_ms": statistics(planning_ms),
              "frame_age_s": statistics(frame_ages), "motion_schedule": "virtual rest-to-rest estimate",
              "physical_validation": False}
    (log.path / "preview.gcode").write_text("; VIRTUAL PREVIEW: not a calibrated hardware job\n" + "\n".join(gcode)+"\n", encoding="utf-8")
    report = log.finish(report)
    (log.path / "SUMMARY.md").write_text(
        f"# Controller {report['mode']}\n\nState: {report['state']}. Path complete: {completed}.\n\n"
        f"Submitted segments: {report['submitted_segments']}.\n\n"
        f"Reference: {report['reference_kind']}; physical validation: false.\n\n"
        f"Gap error statistics (mm): `{report['absolute_gap_error_mm']}`.\n\n"
        f"Fault: {report['reason'] or 'none'}. Logging drops: {report['logging_drops']}.\n", encoding="utf-8")
    return report


class AcquisitionWorker:
    """The newest bounded complete history survives planner/disk delays."""
    def __init__(self, source, projector, capacity):
        self.source, self.projector = source, projector
        self.queue = queue.Queue(maxsize=capacity)
        self.stop_event = threading.Event()
        self.dropped = 0
        self.error = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        try:
            self.source.start()
            while not self.stop_event.is_set():
                frame = self.source.capture()
                mapped = self.projector.project(frame.points_camera_m)
                try:
                    self.queue.put_nowait((frame, mapped))
                except queue.Full:
                    try:
                        self.queue.get_nowait()
                        self.dropped += 1
                    except queue.Empty:
                        pass  # Consumer drained between the full check and eviction.
                    self.queue.put_nowait((frame, mapped))
        except Exception as exc:
            self.error = exc
        finally:
            self.source.stop()

    def start(self):
        self.thread.start()

    def get(self, timeout=1):
        while True:
            try:
                return self.queue.get(timeout=timeout)
            except queue.Empty:
                if self.error:
                    raise self.error
                if not self.thread.is_alive():
                    raise ControllerFault("camera acquisition stopped")

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=10)
        if self.thread.is_alive():
            raise ControllerFault("camera worker did not stop within timeout")


def observe(config, camera_config_path, output, frames=300):
    app = load_camera_config(camera_config_path)
    geometry = check_geometry(config, app.to_dict())
    source = RealSenseD405(app.camera)
    worker = AcquisitionWorker(source, HeightMapProjector(app.grid, app.camera_to_output), config.predictor.history+2)
    log = RunLog(output, config)
    log.event("camera_geometry", geometry=geometry, sha256=fingerprint(geometry))
    worker.start()
    mapper = None
    captured = 0
    reason = None
    try:
        for _ in range(frames):
            frame, height = worker.get()
            receipt = frame.host_monotonic_ns / 1e9
            if mapper is None:
                # Diagnostic anchor only. Unknown USB/exposure latency remains uncalibrated.
                mapper = ClockMapper(frame.timestamp_domain, frame.camera_timestamp_ms, receipt,
                                     uncertainty_s=.1, max_residual_s=.1)
            mapped_s = mapper.map(frame.camera_timestamp_ms, frame.timestamp_domain, receipt, frame.frame_number)
            obs = Observation(frame.frame_number, height.height_mm, height.valid, height.point_count,
                              config.geometry_id, frame.camera_timestamp_ms, frame.timestamp_domain,
                              receipt, mapped_s, mapper.uncertainty_s)
            log.observation(obs)
            captured += 1
    except Exception as exc:
        reason = str(exc)
    finally:
        try:
            worker.close()
        except Exception as exc:
            reason = str(exc)
    return log.finish({"mode": "observe", "frames": captured, "reason": reason,
                       "acquisition_drops": worker.dropped, "camera_config_sha256": fingerprint(geometry),
                       "clock_mapping_calibrated": False, "motion_allowed": False})
