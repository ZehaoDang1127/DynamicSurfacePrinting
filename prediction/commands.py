"""Replay, capture integration and a self-contained synthetic workflow."""

from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np

from surface_mapper.camera import PointCloudFrame, RealSenseD405, SyntheticSurfaceSource
from surface_mapper.config import AppConfig, CameraConfig, GridConfig, load_config
from surface_mapper.heightmap import HeightMap, HeightMapProjector
from surface_mapper.recording import SessionWriter

from .artifacts import ForecastWriter
from .config import PredictionConfig
from .data import geometry_signature, load_session
from .evaluation import evaluate, prepare_output, write_json
from .inference import StreamingPredictor
from .training import train


def predict_session(checkpoint_path, session_path, output_dir, device='auto') -> dict:
    predictor = StreamingPredictor(checkpoint_path, device)
    session = load_session(session_path)
    if session.geometry != predictor.geometry:
        raise ValueError('recording geometry differs from checkpoint')
    writer = ForecastWriter(output_dir, predictor, 'recorded replay; not wall-clock paced')
    writer.manifest.update(checkpoint=str(Path(checkpoint_path).resolve()),
                           source_session=str(session.path.resolve()))
    complete = False
    try:
        for entry in session.manifest['frames']:
            with np.load(session.path / entry['file'], allow_pickle=False) as frame:
                result = predictor.push(frame['height_mm'], frame['valid'],
                                        float(frame['camera_timestamp_ms']),
                                        entry['timestamp_domain'], int(frame['frame_number']))
            if result is not None:
                writer.write(result, entry['frame_number'])
        complete = True
    finally:
        writer.close(complete)
    return writer.manifest


def live(checkpoint_path, camera_config, output_dir, source_name='realsense', frames=300,
         device='auto') -> dict:
    if frames < 0:
        raise ValueError('frames cannot be negative')
    predictor = StreamingPredictor(checkpoint_path, device)
    config = load_config(camera_config)
    if geometry_signature(config.to_dict()) != predictor.geometry:
        raise ValueError('camera configuration geometry differs from checkpoint')
    configured_dt = 1000.0 / config.camera.fps
    if abs(configured_dt - predictor.config.expected_dt_ms) > 1e-6:
        raise ValueError('camera cadence differs from checkpoint training cadence')
    if source_name == 'synthetic' and not np.allclose(config.camera_to_output, np.eye(4)):
        raise ValueError('synthetic live source requires the identity camera transform')
    source = RealSenseD405(config.camera) if source_name == 'realsense' else SyntheticSurfaceSource(config.grid, config.camera.fps)
    projector = HeightMapProjector(config.grid, config.camera_to_output)
    writer = ForecastWriter(output_dir, predictor, f'live {source_name}; prediction only')
    writer.manifest.update(checkpoint=str(Path(checkpoint_path).resolve()),
                           latency_note='Host receive latency excludes exposure, USB and camera buffering. Positive lead is only an upper bound.')
    complete = False
    captured = 0
    try:
        source.start()
        while frames == 0 or captured < frames:
            started = time.perf_counter()
            frame = source.capture()
            heightmap = projector.project(frame.points_camera_m)
            result = predictor.push(heightmap.height_mm, heightmap.valid,
                                    frame.camera_timestamp_ms, frame.timestamp_domain, frame.frame_number)
            captured += 1
            if result is not None:
                host_latency_ms = (time.monotonic_ns() - frame.host_monotonic_ns) / 1e6
                lead = predictor.config.horizon * predictor.config.expected_dt_ms - host_latency_ms
                writer.write(result, frame.frame_number,
                             host_receive_to_prediction_ms=host_latency_ms,
                             nominal_host_lead_ms_upper_bound=lead,
                             deadline_definitely_missed=lead <= 0)
                if writer.count == 1 or writer.count % 30 == 0:
                    print(f'forecast={writer.count} target={result.target_timestamp_ms:.3f} ms '
                          f'inference={result.inference_ms:.2f} ms host_lead_upper_bound={lead:.2f} ms', flush=True)
            if source_name == 'synthetic':
                time.sleep(max(0, 1.0 / config.camera.fps - (time.perf_counter() - started)))
        complete = True
    except KeyboardInterrupt:
        complete = True
        print('live prediction stopped by user', flush=True)
    finally:
        try:
            source.stop()
        finally:
            writer.manifest['captured_frames'] = captured
            writer.close(complete)
    return writer.manifest


def demo(output_dir, epochs=2, device='cpu') -> dict:
    """Varied synthetic trials exercise plumbing, not physical generalization."""
    config = replace(PredictionConfig(), history=6, hidden_channels=(8, 8),
                     batch_size=8, epochs=epochs, patience=max(epochs, 1))
    output = prepare_output(output_dir).resolve()
    grid = GridConfig(-16, 16, -12, 12, 2, 200, 300)
    capture_config = AppConfig('camera', CameraConfig(), grid, np.eye(4))
    write_json(output / 'camera.json', capture_config.to_dict())
    write_json(output / 'convlstm.json', config.to_dict())
    names = ['train_01', 'train_02', 'train_03', 'validation_01', 'test_01']
    x, y = np.meshgrid(grid.x_centers_mm(), grid.y_centers_mm())
    for trial, name in enumerate(names):
        rng = np.random.default_rng(100 + trial)
        writer = SessionWriter(output / name, capture_config,
                               {'name': 'synthetic prediction workflow trial', 'serial': name})
        writer.start()
        for index in range(72):
            t = index / 30.0
            phase = 0.7 * trial
            frequency = 0.35 + 0.08 * trial
            height = (250 + (2 + 0.3 * trial) * np.sin(2*np.pi*frequency*t + phase)
                      + 1.2 * np.sin(x / 7 + t * 2 + phase) * np.cos(y / 8)
                      + rng.normal(0, 0.03, x.shape)).astype(np.float32)
            mask = rng.random(x.shape) > 0.02
            height[~mask] = np.nan
            frame = PointCloudFrame(index, t*1000, index*33_333_333, 'synthetic',
                                    np.empty((0, 3), dtype=np.float32))
            heightmap = HeightMap(height, mask, mask.astype(np.uint32), mask.size, int(mask.sum()))
            writer.write(frame, heightmap)
        writer.finish()
    splits = {'train': names[:3], 'validation': [names[3]], 'test': [names[4]]}
    write_json(output / 'splits.json', splits)
    print('Synthetic recordings ready; training compact smoke-test model.', flush=True)
    training = train(output / 'splits.json', config, output / 'model', device)
    metrics = evaluate(output / 'model' / 'best.pt', output / 'splits.json',
                       output / 'evaluation', device=device)
    replay = predict_session(output / 'model' / 'best.pt', output / names[-1],
                             output / 'predictions', device)
    report = {'purpose': 'Software smoke test only; no claim of physical prediction accuracy.',
              'training': training, 'test_metrics': metrics['metrics']['all'],
              'replay_forecasts': replay['forecast_count']}
    write_json(output / 'demo_report.json', report)
    return report
