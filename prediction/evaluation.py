"""Physical-unit errors and fair persistence/velocity/acceleration comparisons."""

from pathlib import Path
import time

import numpy as np
import torch

from .data import HeightSequenceDataset, load_splits
from .artifacts import prepare_output, write_json
from .baselines import fit_kinematic_baselines
from .metrics import ErrorStats
from .training import load_checkpoint


def baselines(sample: dict, horizon_ms: float) -> dict[str, np.ndarray]:
    """Forecast from physical-unit history, retaining the legacy two-frame API.

    Samples without history keys retain the original persistence/velocity
    outputs and cannot supply acceleration. Normal dataset samples always
    contain history and use the shared least-squares kinematic implementation.
    """
    if 'history_mm' in sample:
        estimates = fit_kinematic_baselines(sample['history_mm'], sample['history_valid'],
                                           sample['history_timestamps_ms'], horizon_ms)
        return {name: estimate.prediction_mm for name, estimate in estimates.items()}
    last = sample['last_mm']
    velocity = last + (last - sample['previous_mm']) * (horizon_ms / sample['dt_ms'])
    return {'persistence': last, 'constant_velocity': velocity}


def _new_group(method_names):
    return {region: {model: ErrorStats() for model in method_names}
            for region in ('all', 'moving', 'stationary')}


def _results(group):
    return {region: {model: stats.result() for model, stats in methods.items()}
            for region, methods in group.items()}


def evaluate(checkpoint_path, split_path, output_dir, split='test', device='auto',
             roi=None, motion_threshold_mm: float = 0.5) -> dict:
    if not np.isfinite(motion_threshold_mm) or motion_threshold_mm < 0:
        raise ValueError('motion threshold must be finite and nonnegative')
    model, config, normalization, checkpoint = load_checkpoint(checkpoint_path, device)
    splits = load_splits(split_path)
    sessions = splits[split]
    if any(session.geometry != checkpoint['geometry'] for session in sessions):
        raise ValueError('evaluation geometry differs from checkpoint')
    forbidden = {Path(p).resolve() for p in checkpoint.get('train_sessions', [])}
    if split == 'test':
        forbidden.update(Path(p).resolve() for p in checkpoint.get('validation_sessions', []))
    if split != 'train' and any(s.path.resolve() in forbidden for s in sessions):
        raise ValueError('held-out evaluation overlaps checkpoint training/validation sessions')
    dataset = HeightSequenceDataset(sessions, config, normalization, cache_frames=True)
    if not len(dataset):
        raise ValueError('no eligible evaluation windows; check cadence, coverage, and history')
    height, width = sessions[0].shape
    roi_mask = np.ones((height, width), dtype=bool)
    if roi is not None:
        x0, y0, x1, y1 = roi
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError('ROI must be x0 y0 x1 y1 within the grid, upper bounds exclusive')
        roi_mask[:] = False
        roi_mask[y0:y1, x0:x1] = True
    output = prepare_output(output_dir)
    method_names = ['convlstm', 'persistence', 'constant_velocity']
    if config.history >= 3:
        method_names.append('constant_acceleration')
    total, per_session, dense = _new_group(method_names), {}, ErrorStats()
    common_count = target_count = 0
    timings, timestamp_offsets = [], []
    example_indices = set(np.linspace(0, len(dataset) - 1, min(5, len(dataset)), dtype=int).tolist())
    device_obj = next(model.parameters()).device
    for index in range(len(dataset)):
        sample = dataset[index]
        started = time.perf_counter()
        with torch.inference_mode():
            predicted = model(torch.from_numpy(sample['inputs'][None]).to(device_obj))[0, 0].cpu().numpy()
        prediction = normalization.denormalize(predicted)
        timings.append((time.perf_counter() - started) * 1000)
        if not np.isfinite(prediction).all():
            raise RuntimeError('model produced non-finite depth predictions')
        target = normalization.denormalize(sample['target'][0])
        target_valid = sample['target_valid'][0] & roi_mask
        baseline_estimates = fit_kinematic_baselines(
            sample['history_mm'], sample['history_valid'],
            sample['history_timestamps_ms'], config.horizon * config.expected_dt_ms)
        common = target_valid.copy()
        for estimate in baseline_estimates.values():
            common &= estimate.prediction_valid
        target_count += int(target_valid.sum())
        common_count += int(common.sum())
        dense.update(prediction[target_valid] - target[target_valid])
        motion = np.abs(target - sample['last_mm']) > motion_threshold_mm
        methods = {'convlstm': prediction, **{
            name: estimate.prediction_mm for name, estimate in baseline_estimates.items()
        }}
        group = per_session.setdefault(sample['session'], _new_group(method_names))
        for region, mask in (('all', common), ('moving', common & motion), ('stationary', common & ~motion)):
            for method, values in methods.items():
                errors = values[mask] - target[mask]
                total[region][method].update(errors)
                group[region][method].update(errors)
        timestamp_offsets.append(sample['target_dt_ms'] - config.horizon * config.expected_dt_ms)
        if index in example_indices:
            baseline_examples = {}
            for name, estimate in baseline_estimates.items():
                baseline_examples[f'{name}_mm'] = np.where(
                    estimate.prediction_valid, estimate.prediction_mm, np.nan)
                baseline_examples[f'{name}_valid'] = estimate.prediction_valid
                baseline_examples[f'{name}_error_mm'] = np.where(
                    common, estimate.prediction_mm - target, np.nan)
            np.savez_compressed(output / f'example_{index:04d}.npz', prediction_mm=prediction,
                                target_mm=target, target_valid=target_valid, comparison_valid=common,
                                **baseline_examples,
                                error_mm=np.where(target_valid, prediction - target, np.nan),
                                source_session=sample['session'],
                                source_timestamp_ms=sample['last_timestamp_ms'],
                                observed_target_timestamp_ms=sample['target_timestamp_ms'])
    report = {
        'schema_version': 1, 'checkpoint': str(Path(checkpoint_path).resolve()),
        'split': split, 'config': config.to_dict(), 'geometry': checkpoint['geometry'],
        'windows': len(dataset), 'roi_xyxy': list(roi) if roi is not None else None,
        'motion_threshold_mm': motion_threshold_mm,
        'comparison_support': 'target and every input in the fitted history finite and valid; '
                              'all baseline forecasts finite; optional ROI',
        'target_valid_pixels': target_count, 'comparison_valid_pixels': common_count,
        'comparison_coverage': common_count / target_count if target_count else 0,
        'metrics': _results(total),
        'per_session': {name: _results(group) for name, group in per_session.items()},
        'convlstm_all_valid_targets': dense.result(),
        'inference_ms': {'median': float(np.median(timings)), 'p95': float(np.percentile(timings, 95)),
                         'includes': 'tensor transfer, forward pass, CPU result and denormalization; no acquisition'},
        'observed_target_offset_from_nominal_ms': {
            'mean': float(np.mean(timestamp_offsets)),
            'max_absolute': float(np.max(np.abs(timestamp_offsets))),
        },
        'notes': ['Motion strata use observed target change for evaluation only.',
                  'Kinematic baselines fit actual input timestamps and forecast the same nominal horizon as ConvLSTM.',
                  'Dense predictions do not constitute calibrated confidence or observed validity.',
                  'Physical accuracy is limited by the recorded sensor targets.'],
    }
    write_json(output / 'metrics.json', report)
    from .reporting import write_visual_report
    write_visual_report(output, report)
    return report
