"""Replay an unlabeled recording through the experimental observer.

The recorder's valid mask must already describe the surface, not the tool.
No future frames are read when constructing a prediction window.
"""
from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import numpy as np
import torch

from .artifacts import prepare_output, write_json
from .data import _load_arrays, load_session
from .occlusion import load_checkpoint
from .occlusion_data import encode_history


def replay(checkpoint_path, session_path, output_dir, device='auto'):
    model, config, normalization, checkpoint = load_checkpoint(checkpoint_path, device)
    session = load_session(session_path)
    if session.geometry != checkpoint['geometry'] or list(session.shape) != list(checkpoint['shape']):
        raise ValueError('recording geometry or shape differs from checkpoint')
    output = prepare_output(output_dir)
    history = deque(maxlen=config['history'])
    results = []
    resets = 0
    complete = False
    manifest = {
        'schema_version': 1, 'complete': False,
        'checkpoint': str(Path(checkpoint_path).resolve()), 'session': str(session.path),
        'geometry': session.geometry, 'config': config, 'forecast_count': 0,
        'timestamp_domain': session.frames[0].timestamp_domain,
        'note': ('Offline replay; validity must already exclude tool measurements. '
                 'Masks describe observations, not forecast confidence. Forecast times are nominal.'),
    }
    write_json(output / 'manifest.json', manifest)
    try:
        for frame in session.frames:
            if history:
                dt = frame.camera_timestamp_ms - history[-1][2]
                if abs(dt - config['expected_dt_ms']) > config['expected_dt_ms'] * config['timing_tolerance'] + 1e-6:
                    history.clear()
                    resets += 1
            height, valid = _load_arrays(frame.path, session.shape)
            history.append((height, valid, frame.camera_timestamp_ms))
            if len(history) < config['history']:
                continue
            inputs = encode_history(np.stack([item[0] for item in history]),
                                    np.stack([item[1] for item in history]),
                                    [item[2] for item in history], normalization,
                                    config['age_scale_ms'])
            with torch.inference_mode():
                tensor = torch.from_numpy(inputs[None]).to(next(model.parameters()).device)
                predicted = model(tensor, config['horizon'])[0, :, 0].cpu().numpy()
            forecast = normalization.denormalize(predicted).astype(np.float32)
            if not np.isfinite(forecast).all():
                raise ValueError('model produced nonfinite predictions')
            filename = f'prediction_{len(results):06d}.npz'
            np.savez_compressed(output / filename, prediction_mm=forecast,
                                observation_valid=valid,
                                history_observed=inputs[-1, 3] == 0,
                                source_timestamp_ms=np.float64(frame.camera_timestamp_ms),
                                nominal_target_timestamps_ms=frame.camera_timestamp_ms +
                                np.arange(config['horizon'] + 1) * config['expected_dt_ms'])
            results.append({'file': filename, 'source_frame_number': frame.frame_number,
                            'source_timestamp_ms': frame.camera_timestamp_ms})
        complete = True
    finally:
        manifest.update(complete=complete, forecast_count=len(results), history_resets=resets,
                        forecasts=results)
        write_json(output / 'manifest.json', manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='auto')
    args = parser.parse_args(argv)
    result = replay(args.checkpoint, args.session, args.output, args.device)
    print(f"Wrote {result['forecast_count']} forecasts to {args.output}")


if __name__ == '__main__':
    main()
