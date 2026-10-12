"""Untrained constant-acceleration capture, replay, and evaluation commands."""

from pathlib import Path
import time

import numpy as np

from surface_mapper.camera import RealSenseD405, SyntheticSurfaceSource
from surface_mapper.config import load_config
from surface_mapper.heightmap import HeightMapProjector
from surface_mapper.recording import SessionWriter

from .artifacts import ForecastWriter, prepare_output, write_json
from .baselines import BaselineConfig, StreamingBaselinePredictor, fit_kinematic_baselines
from .data import geometry_signature, load_session
from .metrics import ErrorStats
from .reporting import write_visual_report


METHODS = ('persistence', 'constant_velocity', 'constant_acceleration')


def _settings(fps, history=5, horizon_ms=None, max_gap_ms=100.0, min_valid_fraction=0.1):
    return BaselineConfig(history=history,
                          horizon_ms=1000.0 / fps if horizon_ms is None else horizon_ms,
                          max_gap_ms=max_gap_ms, min_valid_fraction=min_valid_fraction)


def _read_frame(record):
    with np.load(record.path, allow_pickle=False) as data:
        height = np.asarray(data['height_mm'], dtype=np.float32)
        valid = np.asarray(data['valid'], dtype=bool) & np.isfinite(height)
    return height, valid


def baseline_predict(session_path, output_dir, *, history=5, horizon_ms=None,
                     max_gap_ms=100.0, min_valid_fraction=0.1):
    session = load_session(session_path)
    config = _settings(session.geometry['camera']['fps'], history, horizon_ms,
                       max_gap_ms, min_valid_fraction)
    predictor = StreamingBaselinePredictor(config, session.geometry, session.shape)
    writer = ForecastWriter(output_dir, predictor, 'constant_acceleration recorded replay')
    writer.manifest['source_session'] = str(session.path)
    complete = False
    try:
        for frame in session.frames:
            height, valid = _read_frame(frame)
            forecast = predictor.push(height, valid, frame.camera_timestamp_ms,
                                      frame.timestamp_domain, frame.frame_number)
            if forecast is not None:
                writer.write(forecast, frame.frame_number)
        complete = True
    finally:
        writer.close(complete)
    return writer.manifest


def baseline_live(camera_config, output_dir, *, source_name='realsense', frames=300,
                  history=5, horizon_ms=None, max_gap_ms=100.0, min_valid_fraction=0.1,
                  record_input=False, save_depth=False):
    if frames < 0:
        raise ValueError('frames cannot be negative')
    if save_depth and not record_input:
        raise ValueError('--save-depth requires --record-input')
    if source_name not in ('realsense', 'synthetic'):
        raise ValueError('source must be realsense or synthetic')
    camera = load_config(camera_config)
    config = _settings(camera.camera.fps, history, horizon_ms, max_gap_ms, min_valid_fraction)
    if source_name == 'synthetic' and not np.allclose(camera.camera_to_output, np.eye(4)):
        raise ValueError('synthetic source requires the identity camera transform')
    predictor = StreamingBaselinePredictor(config, geometry_signature(camera.to_dict()),
                                           (camera.grid.height, camera.grid.width))
    source = RealSenseD405(camera.camera) if source_name == 'realsense' else SyntheticSurfaceSource(camera.grid, camera.camera.fps)
    projector = HeightMapProjector(camera.grid, camera.camera_to_output)
    writer = ForecastWriter(output_dir, predictor, f'constant_acceleration live {source_name}')
    writer.manifest.update(
        observation_session='observations' if record_input else None,
        latency_note=('Host receive latency excludes exposure, USB, camera buffering, and output file writes. '
                      'Positive lead is only an upper bound; input/output recording can reduce capture rate.'),
    )
    recorder = None
    previous_record = None
    captured = 0
    complete = False
    try:
        source.start()
        if record_input:
            recorder = SessionWriter(writer.output / 'observations', camera, source.device_info, save_depth)
            recorder.start()
        print(f'constant acceleration: history={history}, horizon={config.horizon_ms:.3f} ms, '
              f'grid={camera.grid.height}x{camera.grid.width}, frame={camera.frame_name}', flush=True)
        while frames == 0 or captured < frames:
            started = time.perf_counter()
            frame = source.capture()
            if recorder is not None and previous_record is not None:
                if (frame.frame_number <= previous_record.frame_number
                        or frame.camera_timestamp_ms <= previous_record.camera_timestamp_ms
                        or frame.timestamp_domain != previous_record.timestamp_domain):
                    raise RuntimeError('camera clock or frame counter restarted; begin a new input recording')
            heightmap = projector.project(frame.points_camera_m)
            forecast = predictor.push(heightmap.height_mm, heightmap.valid, frame.camera_timestamp_ms,
                                      frame.timestamp_domain, frame.frame_number)
            captured += 1
            # Measure availability of the prediction before disk IO. These times
            # cannot establish camera exposure-to-result latency without clock mapping.
            host_latency_ms = (time.monotonic_ns() - frame.host_monotonic_ns) / 1e6
            if forecast is not None:
                lead = config.horizon_ms - host_latency_ms
                writer.write(forecast, frame.frame_number,
                             host_receive_to_prediction_ms=host_latency_ms,
                             nominal_host_lead_ms_upper_bound=lead,
                             deadline_definitely_missed=lead <= 0)
                if writer.count == 1 or writer.count % 30 == 0:
                    print(f'forecast={writer.count} support={forecast.prediction_valid.mean():.1%} '
                          f'target={forecast.target_timestamp_ms:.3f} ms '
                          f'fit={forecast.inference_ms:.2f} ms host_lead_upper_bound={lead:.2f} ms', flush=True)
            elif captured % 30 == 0:
                print(f'captured={captured} waiting for usable history; '
                      f'coverage={heightmap.valid_fraction:.1%} resets={predictor.reset_count}', flush=True)
            if recorder is not None:
                recorder.write(frame, heightmap)
                previous_record = frame
            if source_name == 'synthetic':
                time.sleep(max(0, 1.0 / camera.camera.fps - (time.perf_counter() - started)))
        complete = True
    except KeyboardInterrupt:
        complete = True
        print('constant-acceleration capture stopped by user', flush=True)
    finally:
        try:
            source.stop()
        finally:
            try:
                if complete and recorder is not None:
                    recorder.finish()
            finally:
                writer.manifest['captured_frames'] = captured
                writer.close(complete)
    return writer.manifest


def _group():
    return {region: {method: ErrorStats() for method in METHODS}
            for region in ('all', 'moving', 'stationary')}


def _results(group):
    return {region: {method: stats.result() for method, stats in methods.items()}
            for region, methods in group.items()}


def baseline_evaluate(session_path, output_dir, *, history=5, horizon_ms=None,
                      max_gap_ms=100.0, min_valid_fraction=0.1, roi=None,
                      motion_threshold_mm=0.5, target_tolerance_ms=None):
    session = load_session(session_path)
    fps = session.geometry['camera']['fps']
    config = _settings(fps, history, horizon_ms, max_gap_ms, min_valid_fraction)
    tolerance = 0.2 * 1000.0 / fps if target_tolerance_ms is None else target_tolerance_ms
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError('target tolerance must be finite and nonnegative')
    if not np.isfinite(motion_threshold_mm) or motion_threshold_mm < 0:
        raise ValueError('motion threshold must be finite and nonnegative')
    height, width = session.shape
    roi_mask = np.ones(session.shape, dtype=bool)
    if roi is not None:
        x0, y0, x1, y1 = roi
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError('ROI must be x0 y0 x1 y1 within the grid, upper bounds exclusive')
        roi_mask[:] = False
        roi_mask[y0:y1, x0:x1] = True
    output = prepare_output(output_dir)
    predictor = StreamingBaselinePredictor(config, session.geometry, session.shape)
    timestamps = np.array([frame.camera_timestamp_ms for frame in session.frames])
    group = _group()
    common_count = target_count = forecast_count = matched_count = scored_count = 0
    offsets, timings = [], []
    for index, frame in enumerate(session.frames):
        values, mask = _read_frame(frame)
        forecast = predictor.push(values, mask, frame.camera_timestamp_ms,
                                  frame.timestamp_domain, frame.frame_number)
        if forecast is None:
            continue
        forecast_count += 1
        # Future timestamps choose evaluation ground truth only. The predictor
        # has already run and never receives a future observation or timestamp.
        insertion = int(np.searchsorted(timestamps, forecast.target_timestamp_ms))
        candidates = [j for j in (insertion - 1, insertion) if index < j < len(timestamps)]
        if not candidates:
            continue
        target_index = min(candidates, key=lambda j: abs(timestamps[j] - forecast.target_timestamp_ms))
        offset = float(timestamps[target_index] - forecast.target_timestamp_ms)
        if abs(offset) > tolerance + 1e-6:
            continue
        matched_count += 1
        target, target_valid = _read_frame(session.frames[target_index])
        target_valid &= roi_mask
        estimates = fit_kinematic_baselines(
            np.stack([item[0] for item in predictor.buffer]),
            np.stack([item[1] for item in predictor.buffer]),
            [item[2] for item in predictor.buffer], config.horizon_ms,
        )
        common = target_valid.copy()
        for estimate in estimates.values():
            common &= estimate.prediction_valid
        target_count += int(target_valid.sum())
        common_count += int(common.sum())
        offsets.append(offset)
        timings.append(forecast.inference_ms)
        if not common.any():
            continue
        scored_count += 1
        motion = np.abs(target - values) > motion_threshold_mm
        for region, support in (('all', common), ('moving', common & motion), ('stationary', common & ~motion)):
            for method, estimate in estimates.items():
                group[region][method].update(estimate.prediction_mm[support] - target[support])
        if scored_count <= 5:
            arrays = {f'{method}_mm': estimate.prediction_mm for method, estimate in estimates.items()}
            arrays.update({f'{method}_valid': estimate.prediction_valid for method, estimate in estimates.items()})
            np.savez_compressed(output / f'example_{scored_count - 1:04d}.npz',
                                prediction_mm=forecast.prediction_mm, prediction_valid=forecast.prediction_valid,
                                target_mm=target, target_valid=target_valid, comparison_valid=common,
                                error_mm=np.where(common, forecast.prediction_mm - target, np.nan),
                                source_timestamp_ms=forecast.source_timestamp_ms,
                                nominal_target_timestamp_ms=forecast.target_timestamp_ms,
                                observed_target_timestamp_ms=timestamps[target_index], **arrays)
    if scored_count == 0:
        raise ValueError('no scorable forecasts; check history, gaps, target timing tolerance, and common valid coverage')
    metrics = _results(group)
    report = {
        'schema_version': 1, 'predictor': 'constant_acceleration',
        'source_session': str(session.path), 'config': config.to_dict(), 'geometry': session.geometry,
        'windows': scored_count, 'forecast_windows': forecast_count,
        'timing_matched_windows': matched_count, 'unmatched_target_windows': forecast_count - matched_count,
        'zero_support_windows': matched_count - scored_count,
        'buffer_resets': predictor.reset_count,
        'roi_xyxy': list(roi) if roi is not None else None,
        'motion_threshold_mm': motion_threshold_mm, 'target_tolerance_ms': tolerance,
        'comparison_support': 'target and every history observation valid, with finite fits for all methods; optional ROI',
        'target_valid_pixels': target_count, 'comparison_valid_pixels': common_count,
        'comparison_coverage': common_count / target_count if target_count else 0,
        'metrics': metrics, 'per_session': {str(session.path): metrics},
        'inference_ms': {'median': float(np.median(timings)), 'p95': float(np.percentile(timings, 95)),
                         'includes': 'NumPy fits only; no acquisition, projection, target scoring, or recording'},
        'observed_target_offset_from_nominal_ms': {'mean': float(np.mean(offsets)),
                                                  'max_absolute': float(np.max(np.abs(offsets)))},
        'notes': ['Linear and quadratic fits use all history observations and actual timestamps.',
                  'Future observations are used for scoring only.',
                  'Depth noise can be amplified by acceleration extrapolation.',
                  'Agreement with sensor targets is not independent physical accuracy.'],
    }
    write_json(output / 'metrics.json', report)
    write_visual_report(output, report)
    return report
