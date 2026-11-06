"""Live camera -> prediction -> candidate planning, without a printer transport.

Each proposal independently starts at the first path point with a hypothetical
nozzle at the measured surface plus the configured gap. No proposal is executed
or used as a claim about the real nozzle position.
"""

from collections import Counter
from dataclasses import asdict
import json
import time

import numpy as np

from surface_mapper.camera import RealSenseD405
from surface_mapper.config import load_config as load_camera_config
from surface_mapper.heightmap import HeightMapProjector

from .clock_sync import ClockMapper
from .config import fingerprint
from .logging import RunLog
from .planner import Planner, sample_map
from .predictor_adapter import PredictorAdapter
from .runtime import AcquisitionWorker, check_geometry
from .types import ControllerFault, Observation


def evaluate_shadow(config, path_xy, observations, geometry, log, *, clock=time.monotonic,
                    progress=None):
    """Evaluate causal proposals against fresh observations; never submit moves."""
    if config.predictor.kind == "convlstm" and config.predictor.temporal_policy == "fixed":
        raise ControllerFault("shadow arrival planning cannot use a single fixed ConvLSTM horizon; "
                              "select explicit linear_to_horizon interpolation with its separate error allowance")
    predictor, planner = PredictorAdapter(config, geometry), Planner(config)
    path_xy = np.asarray(path_xy, dtype=float)
    if path_xy.ndim != 2 or path_xy.shape[1] != 2 or len(path_xy) < 2 or not np.isfinite(path_xy).all():
        raise ValueError("shadow path needs at least two finite XY points")
    coverage, frame_ages, processing_ms, horizons = [], [], [], []
    rejected = Counter()
    forecasts = proposals = captured = resets = 0
    last_forecast = last_proposal = None
    reason = None
    try:
        for obs in observations:
            started = time.perf_counter()
            captured += 1
            log.observation(obs)
            coverage.append(float(obs.valid.mean()))
            if not predictor.push(obs):
                if predictor.reset_reason:
                    resets += 1
                    log.event("shadow_history_reset", reason=predictor.reset_reason)
                continue
            now = clock()
            # Receipt-anchored camera time can lead the current host tick within
            # its uncertainty. Keep actual `now` for age checks, while avoiding a
            # request to predict before the newest observation.
            planning_s = max(now, obs.observed_s)
            frame_ages.append(now - obs.observed_s)
            try:
                forecast = predictor.forecast(planning_s, now)
                forecasts += 1
                last_forecast = forecast
                surface = sample_map(forecast.height_mm, forecast.support, config.grid, path_xy[0])
                start_xyz = (*path_xy[0], config.surface_z(surface) + config.desired_gap_mm)
                proposal = planner.plan(predictor, start_xyz,
                    path_xy[1:1 + config.proposal_moves], planning_s, now, proposals, extrude=False)
                proposals += 1
                last_proposal = proposal
                log.trajectory(proposal)
                horizons.append(proposal.end_s - obs.observed_s)
            except ControllerFault as exc:
                rejected[str(exc)] += 1
                log.event("shadow_rejected", frame_id=obs.frame_id, reason=str(exc))
            processing_ms.append((time.perf_counter() - started) * 1000)
            if progress and (captured == config.predictor.history or captured % 30 == 0):
                progress(f"frames={captured} coverage={coverage[-1]:.1%} "
                         f"forecasts={forecasts} proposals={proposals} rejected={sum(rejected.values())}")
    except Exception as exc:
        reason = str(exc)
    if last_forecast is not None:
        np.savez_compressed(log.path / "latest_forecast.npz",
            height_mm=last_forecast.height_mm, support=last_forecast.support,
            source_frame_ids=last_forecast.source_frame_ids,
            source_s=last_forecast.source_s, target_s=last_forecast.target_s)
    if last_proposal is not None:
        (log.path / "candidate_preview.json").write_text(json.dumps({
            "mode": "shadow_candidate_only", "motion_allowed": False,
            "hypothetical_start": True, "proposal": asdict(last_proposal),
        }, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    def stats(values):
        if not values:
            return None
        return {"median": float(np.median(values)), "p95": float(np.percentile(values, 95)),
                "max": float(np.max(values))}
    return {
        "mode": "live_shadow", "state": "FAULT" if reason else "READY" if proposals else "BLOCKED",
        "reason": reason or (None if proposals else "no accepted planning proposal; inspect rejection reasons and coverage"),
        "frames": captured, "forecasts": forecasts, "accepted_proposals": proposals,
        "rejected_proposals": sum(rejected.values()), "rejection_reasons": dict(rejected),
        "history_resets": resets, "coverage_fraction": stats(coverage),
        "host_frame_age_s": stats(frame_ages), "prediction_and_planning_ms": stats(processing_ms),
        "forecast_horizon_s": stats(horizons),
        "motion_allowed": False, "extrusion_allowed": False, "printer_commands_sent": 0,
        "proposal_scope": "independent lookahead from hypothetical first-path-point start; no path execution",
        "lookahead_moves": min(config.proposal_moves, len(path_xy) - 1),
        "path_points": len(path_xy), "clock_mapping_calibrated": False,
        "physical_validation": False, "error_bounds_measured": False,
        "predictor_kind": config.predictor.kind,
        "temporal_policy": config.predictor.temporal_policy,
        "interpolation_error_mm": config.predictor.interpolation_error_mm,
    }


def run_shadow(config, camera_config_path, path_xy, output, frames=300, *, source=None, progress=print):
    if type(frames) is not int or frames < 1:
        raise ValueError("shadow frames must be positive")
    if config.predictor.kind == "convlstm" and config.predictor.temporal_policy == "fixed":
        raise ControllerFault("shadow arrival planning cannot use a single fixed ConvLSTM horizon; "
                              "select explicit linear_to_horizon interpolation with its separate error allowance")
    app = load_camera_config(camera_config_path)
    geometry = check_geometry(config, app.to_dict())
    source = source if source is not None else RealSenseD405(app.camera)
    worker = AcquisitionWorker(source, HeightMapProjector(app.grid, app.camera_to_output),
                               config.predictor.history + 2)
    log = RunLog(output, config)
    (log.path / "camera_config.json").write_text(json.dumps(app.to_dict(), indent=2) + "\n", encoding="utf-8")
    log.event("camera_geometry", geometry=geometry, sha256=fingerprint(geometry))
    log.event("original_path", xy_mm=np.asarray(path_xy).tolist(), virtual_only=True)
    def observations():
        mapper = None
        for _ in range(frames):
            frame, height = worker.get()
            receipt = frame.host_monotonic_ns / 1e9
            if mapper is None:
                # Receipt anchor is diagnostic: exposure and USB delay are unknown.
                mapper = ClockMapper(frame.timestamp_domain, frame.camera_timestamp_ms, receipt,
                                     uncertainty_s=.1, max_residual_s=.1)
                log.event("camera_device", **source.device_info)
            at_s = mapper.map(frame.camera_timestamp_ms, frame.timestamp_domain, receipt, frame.frame_number)
            yield Observation(frame.frame_number, height.height_mm, height.valid, height.point_count,
                              config.geometry_id, frame.camera_timestamp_ms, frame.timestamp_domain,
                              receipt, at_s, mapper.uncertainty_s)
    worker.start()
    report = None
    try:
        report = evaluate_shadow(config, path_xy, observations(), geometry, log, progress=progress)
    except BaseException as exc:
        report = {"mode": "live_shadow", "state": "FAULT", "reason": str(exc) or type(exc).__name__,
                  "motion_allowed": False, "printer_commands_sent": 0, "physical_validation": False}
        raise
    finally:
        try:
            worker.close()
        except Exception as exc:
            report = {**(report or {}), "state": "FAULT", "reason": str(exc)}
        report = log.finish({**(report or {}), "acquisition_drops": worker.dropped,
                             "camera_config_sha256": fingerprint(geometry)})
    if not report["logging_complete"]:
        report.update(state="FAULT", reason="shadow logging was incomplete")
        (log.path / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
