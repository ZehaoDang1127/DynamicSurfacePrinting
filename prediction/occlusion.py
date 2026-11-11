"""Train and evaluate the experimental occlusion-aware forecasting model.

All reported errors use recorded, valid measurements as ground truth. Synthetic
occlusions are reproducible and cannot establish performance on real occlusions.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from .artifacts import prepare_output, write_json
from .data import Normalization, fit_normalization, load_splits
from .metrics import ErrorStats
from .occlusion_data import (LARGE_OCCLUSION_DEFAULTS, OCCLUSION_SHAPES, OcclusionDataset,
                             validate_large_occlusion)
from .occlusion_model import INPUT_CHANNELS, RECURRENCES, OcclusionUNetConvLSTM, validate_input_channels
from .training import resolve_device, _save_checkpoint


DEFAULT_CONFIG = {
    'history': 10, 'horizon': 4, 'expected_dt_ms': 1000 / 30,
    'timing_tolerance': .2, 'min_valid_fraction': .1,
    'age_scale_ms': 1000., 'occlusion_fraction': .25, 'occlusion_shape': 'rectangle',
    'gaussian_step_interval': 1, **LARGE_OCCLUSION_DEFAULTS,
    'base_channels': 16, 'kernel_size': 3, 'input_channels': list(INPUT_CHANNELS),
    'recurrence': 'dynamics_correction', 'epochs': 30, 'patience': 5,
    'batch_size': 4, 'learning_rate': .001, 'grad_clip': 1.,
    'hidden_weight': 1., 'seed': 42, 'cache_frames': False, 'train_window_stride': 1,
}
SCHEMA_VERSION = 1


def read_config(value: dict) -> dict:
    if not isinstance(value, dict):
        raise ValueError('configuration must be a JSON object')
    unknown = set(value) - set(DEFAULT_CONFIG)
    if unknown:
        raise ValueError('unknown configuration fields: ' + ', '.join(sorted(unknown)))
    config = {**DEFAULT_CONFIG, **value}
    for key in ('history', 'horizon', 'base_channels', 'kernel_size', 'epochs', 'patience', 'batch_size',
                'train_window_stride', 'gaussian_step_interval'):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if type(config['cache_frames']) is not bool:
        raise ValueError('cache_frames must be a boolean')
    if config['kernel_size'] % 2 != 1:
        raise ValueError('kernel_size must be odd')
    if type(config['seed']) is not int or not 0 <= config['seed'] < 2**32 - 3:
        raise ValueError('seed must be an integer between 0 and 2**32 - 4')
    for key in ('expected_dt_ms', 'timing_tolerance', 'min_valid_fraction', 'age_scale_ms',
                'occlusion_fraction', 'learning_rate', 'grad_clip', 'hidden_weight'):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f'{key} must be finite and numeric')
    for key in ('expected_dt_ms', 'age_scale_ms', 'learning_rate', 'grad_clip'):
        if config[key] <= 0:
            raise ValueError(f'{key} must be positive')
    if not 0 <= config['timing_tolerance'] < 1:
        raise ValueError('timing_tolerance must be in [0, 1)')
    if not 0 < config['min_valid_fraction'] <= 1:
        raise ValueError('min_valid_fraction must be in (0, 1]')
    if not 0 <= config['occlusion_fraction'] <= 1:
        raise ValueError('occlusion_fraction must be in [0, 1]')
    if config['occlusion_shape'] not in OCCLUSION_SHAPES:
        raise ValueError('occlusion_shape must be one of ' + ', '.join(OCCLUSION_SHAPES))
    config.update(validate_large_occlusion(config))
    if config['hidden_weight'] < 0:
        raise ValueError('hidden_weight must be nonnegative')
    config['input_channels'] = list(validate_input_channels(config['input_channels']))
    if config['recurrence'] not in RECURRENCES:
        raise ValueError('recurrence must be one of ' + ', '.join(RECURRENCES))
    return config


def _model(config):
    return OcclusionUNetConvLSTM(base_channels=config['base_channels'], kernel_size=config['kernel_size'],
                                 input_channels=tuple(config['input_channels']),
                                 recurrence=config['recurrence'])


def _collate(samples):
    return {key: torch.stack([torch.as_tensor(sample[key]) for sample in samples])
            for key in ('inputs', 'targets', 'target_valid', 'hidden_origin')}


def _masked_sum(prediction, target, mask):
    # Index before subtracting to ensure unobserved NaNs cannot propagate.
    count = int(mask.sum().item())
    if not count:
        return prediction.new_zeros(()), 0
    errors = prediction[mask] - target[mask]
    if not torch.isfinite(errors).all():
        raise ValueError('nonfinite prediction or target at a valid pixel')
    return errors.abs().sum(), count


def _loss_terms(prediction, target, valid, hidden):
    if prediction.shape != target.shape or target.shape != valid.shape:
        raise ValueError('model output, targets and target validity shapes must match')
    if hidden.shape != (target.shape[0], *target.shape[2:]):
        raise ValueError('hidden_origin must have shape [batch, 1, height, width]')
    hidden = hidden[:, None].expand_as(valid)
    return {
        'current': _masked_sum(prediction[:, :1], target[:, :1], valid[:, :1]),
        'future': _masked_sum(prediction[:, 1:], target[:, 1:], valid[:, 1:]),
        'hidden': _masked_sum(prediction, target, valid & hidden),
    }


def _epoch(model, loader, device, config, scale, optimizer=None):
    training = optimizer is not None
    model.train(training)
    totals = {name: [0., 0] for name in ('current', 'future', 'hidden')}
    with torch.set_grad_enabled(training):
        for batch in loader:
            inputs = batch['inputs'].to(device=device, dtype=torch.float32)
            target = batch['targets'].to(device=device, dtype=torch.float32)
            valid = batch['target_valid'].to(device=device, dtype=torch.bool)
            hidden = batch['hidden_origin'].to(device=device, dtype=torch.bool)
            if training:
                optimizer.zero_grad(set_to_none=True)
            prediction = model(inputs, rollout_steps=config['horizon'])
            terms = _loss_terms(prediction, target, valid, hidden)
            losses = [terms[name][0] / terms[name][1] for name in ('current', 'future') if terms[name][1]]
            if terms['hidden'][1] and config['hidden_weight']:
                losses.append(config['hidden_weight'] * terms['hidden'][0] / terms['hidden'][1])
            if not losses:
                continue
            if training:
                sum(losses).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'], error_if_nonfinite=True)
                optimizer.step()
            for name, (total, count) in terms.items():
                totals[name][0] += float(total.detach()) * scale
                totals[name][1] += count
    if not totals['current'][1] or not totals['future'][1]:
        raise ValueError('dataset needs valid current and future target pixels')
    result = {name: {'mae_mm': total / count if count else None, 'valid_pixels': count}
              for name, (total, count) in totals.items()}
    result['objective_mm'] = result['current']['mae_mm'] + result['future']['mae_mm']
    if result['hidden']['valid_pixels']:
        result['objective_mm'] += config['hidden_weight'] * result['hidden']['mae_mm']
    return result


def train(split_path, config, output_dir, device='auto', initialize_from=None):
    """Train on session-level splits; optionally start from pretrained weights.

    ``initialize_from`` copies model weights only. Normalization is always fitted
    on the target training sessions and the optimizer starts fresh.
    """
    config = read_config(config)
    output = Path(output_dir).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f'output must be a new or empty directory: {output}')
    device = resolve_device(device)
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['seed'])
    splits = load_splits(split_path)
    normalization = fit_normalization(splits['train'])
    datasets = {name: OcclusionDataset(splits[name], config, normalization, seed=config['seed'] + offset,
                                      cache_frames=config['cache_frames'],
                                      window_stride=config['train_window_stride'] if name == 'train' else 1)
                for offset, name in enumerate(('train', 'validation'))}
    if any(not len(dataset) for dataset in datasets.values()):
        raise ValueError('training and validation need at least one eligible window')
    generator = torch.Generator().manual_seed(config['seed'])
    loaders = {name: DataLoader(dataset, batch_size=config['batch_size'], shuffle=name == 'train',
                                generator=generator if name == 'train' else None,
                                num_workers=0, collate_fn=_collate)
               for name, dataset in datasets.items()}
    model = _model(config)
    initialization = None if initialize_from is None else _initialize_weights(model, config, initialize_from)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    output = prepare_output(output)
    provenance = {f'{name}_sessions': [str(s.path.resolve()) for s in splits[name]]
                  for name in ('train', 'validation', 'test')}
    checkpoint = {'schema_version': SCHEMA_VERSION, 'model_type': 'occlusion_unet_convlstm',
                  'config': config, 'normalization': normalization.to_dict(),
                  'geometry': splits['train'][0].geometry, 'shape': list(splits['train'][0].shape),
                  'split_path': str(Path(split_path).resolve()), **provenance}
    if initialization is not None:
        checkpoint['initialization'] = initialization
    metadata = {**checkpoint, 'status': 'running', 'started_utc': datetime.now(timezone.utc).isoformat(),
                'device': str(device), 'dataset_summaries': {k: d.summary() for k, d in datasets.items()},
                'test_evaluated': False, 'checkpoint': str(output / 'best.pt')}
    write_json(output / 'run.json', metadata)
    history, best, stale = [], math.inf, 0
    try:
        for epoch in range(1, config['epochs'] + 1):
            train_result = _epoch(model, loaders['train'], device, config, normalization.scale_mm, optimizer)
            val_result = _epoch(model, loaders['validation'], device, config, normalization.scale_mm)
            improved = val_result['objective_mm'] < best
            history.append({'epoch': epoch, 'train': train_result, 'validation': val_result, 'is_best': improved})
            if improved:
                best, stale = val_result['objective_mm'], 0
                _save_checkpoint(output / 'best.pt', {**checkpoint, 'epoch': epoch,
                    'validation_objective_mm': best,
                    'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()}})
            else:
                stale += 1
            write_json(output / 'history.json', history)
            print(f'epoch {epoch}/{config["epochs"]}: validation objective {val_result["objective_mm"]:.6f} mm', flush=True)
            if stale >= config['patience']:
                break
        metadata.update(status='complete', epochs_completed=len(history), best_validation_objective_mm=best,
                        best_epoch=next(row['epoch'] for row in history if row['is_best'] and
                                        row['validation']['objective_mm'] == best))
    except BaseException as error:
        metadata.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                        error=str(error), epochs_completed=len(history))
        raise
    finally:
        metadata['finished_utc'] = datetime.now(timezone.utc).isoformat()
        write_json(output / 'run.json', metadata)
    return metadata


def _initialize_weights(model, config, path):
    """Copy weights from a compatible occlusion checkpoint and record their source.

    Only ``base_channels``, ``kernel_size``, ``input_channels`` and ``recurrence``
    determine parameter shapes; history,
    horizon, frame rate, grid size and geometry may differ between source and target.
    Pretraining provides an initialization only and does not qualify the source
    task or geometry for the target setup.
    """
    source_path = Path(path).expanduser().resolve()
    source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    source_model, source_config, _, source = load_checkpoint(source_path, 'cpu')
    differing = [k for k in ('base_channels', 'kernel_size', 'input_channels', 'recurrence')
                 if source_config[k] != config[k]]
    if differing:
        raise ValueError('initialization architecture differs from target: ' + ', '.join(differing))
    model.load_state_dict(source_model.state_dict(), strict=True)
    return {'kind': 'pretrained_weights_only', 'source_checkpoint': str(source_path),
            'source_sha256': source_sha256, 'source_epoch': source.get('epoch'),
            'source_config': source_config, 'source_geometry': source['geometry'],
            'source_shape': list(source['shape']), 'source_train_sessions': source['train_sessions'],
            'optimizer_state_transferred': False,
            'normalization': 'fitted exclusively on target training sessions'}


def load_checkpoint(path, device='auto'):
    device = resolve_device(device)
    checkpoint = torch.load(Path(path), map_location='cpu', weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get('schema_version') != SCHEMA_VERSION or checkpoint.get('model_type') != 'occlusion_unet_convlstm':
        raise ValueError('unsupported occlusion checkpoint')
    for key in ('config', 'normalization', 'geometry', 'shape', 'state_dict', 'train_sessions', 'validation_sessions', 'test_sessions'):
        if key not in checkpoint:
            raise ValueError(f'checkpoint missing {key}')
    if not isinstance(checkpoint['geometry'], dict):
        raise ValueError('checkpoint geometry must be an object')
    shape = checkpoint['shape']
    if not isinstance(shape, (list, tuple)) or len(shape) != 2 or any(type(n) is not int or n < 1 for n in shape):
        raise ValueError('checkpoint shape must contain two positive integers')
    for key in ('train_sessions', 'validation_sessions', 'test_sessions'):
        if not isinstance(checkpoint[key], list) or not checkpoint[key] or not all(isinstance(p, str) for p in checkpoint[key]):
            raise ValueError(f'checkpoint {key} must be a nonempty list of paths')
    config = read_config(checkpoint['config'])
    normalization = Normalization.from_dict(checkpoint['normalization'])
    model = _model(config)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    if not all(torch.isfinite(t).all() for t in model.state_dict().values()):
        raise ValueError('checkpoint contains nonfinite model parameters')
    return model.to(device).eval(), config, normalization, checkpoint


MASK_KEYS = ('occlusion_shape', 'occlusion_fraction', 'gaussian_step_interval', *LARGE_OCCLUSION_DEFAULTS)


def read_mask_config(path):
    """Read a JSON object that replaces the checkpoint's synthetic-mask settings for evaluation."""
    if path is None:
        return None
    overrides = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if not isinstance(overrides, dict) or not overrides:
        raise ValueError('mask configuration must be a nonempty JSON object')
    unknown = set(overrides) - set(MASK_KEYS)
    if unknown:
        raise ValueError('mask configuration may only set ' + ', '.join(MASK_KEYS)
                         + '; unknown: ' + ', '.join(sorted(unknown)))
    return overrides


def evaluate(checkpoint_path, split_path, output_dir, split='test', device='auto', mask_config=None):
    """Score a checkpoint on one split.

    ``mask_config`` (path to a JSON object) replaces only the synthetic-mask
    settings, so models trained with different masks can be compared under the
    same test masks. Window selection and the mask seed are unchanged.
    """
    if split not in ('train', 'validation', 'test'):
        raise ValueError('split must be train, validation or test')
    model, config, normalization, checkpoint = load_checkpoint(checkpoint_path, device)
    mask_overrides = read_mask_config(mask_config)
    evaluation_config = read_config({**config, **(mask_overrides or {})})
    splits = load_splits(split_path)
    sessions = splits[split]
    if any(s.geometry != checkpoint['geometry'] or list(s.shape) != list(checkpoint['shape']) for s in sessions):
        raise ValueError('evaluation geometry or shape differs from checkpoint')
    forbidden = {Path(p).resolve() for p in checkpoint['train_sessions']}
    if split == 'test':
        forbidden.update(Path(p).resolve() for p in checkpoint['validation_sessions'])
    if split != 'train' and any(s.path.resolve() in forbidden for s in sessions):
        raise ValueError('held-out evaluation overlaps checkpoint training/validation sessions')
    seed = config['seed'] + {'train': 0, 'validation': 1, 'test': 2}[split]
    dataset = OcclusionDataset(sessions, evaluation_config, normalization, seed=seed,
                               cache_frames=config['cache_frames'], window_stride=1)
    if not len(dataset):
        raise ValueError('no eligible evaluation windows')
    output = prepare_output(output_dir)
    horizon = config['horizon']
    # Each model is also scored on the common baseline-supported pixels.
    groups = [{region: {method: ErrorStats() for method in ('model', 'baseline', 'model_on_baseline_support')}
               for region in ('all', 'visible', 'hidden')} for _ in range(horizon + 1)]
    support = [{region: {'region_pixels': 0, 'target_valid': 0, 'baseline_supported': 0, 'never_seen': 0}
                for region in ('all', 'visible', 'hidden')} for _ in range(horizon + 1)]
    model_device = next(model.parameters()).device
    for sample in dataset:
        inputs = sample['inputs']
        with torch.inference_mode():
            predicted = model(torch.as_tensor(inputs[None], device=model_device, dtype=torch.float32),
                              rollout_steps=horizon)[0].cpu().numpy()
        target = sample['targets']
        valid = sample['target_valid'].astype(bool)
        hidden = inputs[-1, 1:2] <= .5
        if predicted.shape != target.shape or valid.shape != target.shape or hidden.shape != target.shape[1:]:
            raise ValueError('model and dataset shapes do not match')
        if not np.isfinite(predicted).all():
            raise ValueError('model produced nonfinite predictions')
        # Last measured value within the supplied history; never invent a value
        # for pixels with no observation in that history.
        last = np.zeros_like(inputs[0, :1])
        seen = np.zeros_like(last, dtype=bool)
        for frame in inputs:
            observed = frame[1:2] > .5
            last[observed] = frame[0:1][observed]
            seen |= observed
        for lead in range(horizon + 1):
            for region, region_mask in (('all', np.ones_like(hidden)), ('visible', ~hidden), ('hidden', hidden)):
                mask = valid[lead] & region_mask
                common = mask & seen
                support[lead][region]['region_pixels'] += int(region_mask.sum())
                support[lead][region]['target_valid'] += int(mask.sum())
                support[lead][region]['baseline_supported'] += int(common.sum())
                support[lead][region]['never_seen'] += int((mask & ~seen).sum())
                groups[lead][region]['model'].update((predicted[lead][mask] - target[lead][mask]) * normalization.scale_mm)
                groups[lead][region]['model_on_baseline_support'].update((predicted[lead][common] - target[lead][common]) * normalization.scale_mm)
                groups[lead][region]['baseline'].update((last[common] - target[lead][common]) * normalization.scale_mm)
    results = []
    for lead in range(horizon + 1):
        regions = {}
        for region in ('all', 'visible', 'hidden'):
            counts = support[lead][region]
            regions[region] = {**counts,
                'target_coverage': counts['target_valid'] / counts['region_pixels'] if counts['region_pixels'] else 0.,
                'baseline_coverage': counts['baseline_supported'] / counts['target_valid'] if counts['target_valid'] else 0.,
                'never_seen_fraction': counts['never_seen'] / counts['target_valid'] if counts['target_valid'] else 0.,
                **{name: stats.result() for name, stats in groups[lead][region].items()}}
        results.append({'lead_steps': lead, 'nominal_lead_ms': lead * config['expected_dt_ms'], 'regions': regions})
    report = {'schema_version': 1, 'checkpoint': str(Path(checkpoint_path).resolve()),
              'split': split, 'split_path': str(Path(split_path).resolve()), 'config': config,
              'geometry': checkpoint['geometry'], 'windows': len(dataset), 'dataset_summary': dataset.summary(),
              'mask_seed': seed, 'mask_override': mask_overrides,
              'evaluation_masks': {key: evaluation_config[key] for key in MASK_KEYS},
              'leads': results,
              'notes': ['Hidden and visible regions are defined by observed validity at the current input frame, including natural missing cells.',
                        'Lead times are nominal; actual timestamps may vary within the configured timing tolerance.',
                        'Ground truth exists only at valid recorded pixels.',
                        'Baseline carries the last observation within the input window; never-seen pixels are unsupported.',
                        'Synthetic occlusion results do not establish real-occlusion accuracy.']}
    write_json(output / 'metrics.json', report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    fit = sub.add_parser('train', help='train using session-level splits; test is held out')
    fit.add_argument('--splits', required=True, type=Path)
    fit.add_argument('--config', required=True, type=Path, help='JSON object; omitted fields use defaults')
    fit.add_argument('--output', required=True, type=Path)
    fit.add_argument('--device', default='auto')
    fit.add_argument('--initialize-from', type=Path, default=None,
                     help='occlusion checkpoint whose weights initialize training (fine-tuning)')
    score = sub.add_parser('evaluate', help='score current and future predictions against observed targets')
    score.add_argument('--checkpoint', required=True, type=Path)
    score.add_argument('--splits', required=True, type=Path)
    score.add_argument('--output', required=True, type=Path)
    score.add_argument('--split', choices=('train', 'validation', 'test'), default='test')
    score.add_argument('--device', default='auto')
    score.add_argument('--mask-config', type=Path, default=None,
                       help='JSON object replacing the synthetic-mask settings for this evaluation')
    inspect = sub.add_parser('inspect', help='show checkpoint metadata (no model weights)')
    inspect.add_argument('--checkpoint', required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == 'train':
        config = json.loads(args.config.read_text(encoding='utf-8-sig'))
        result = train(args.splits, config, args.output, args.device, args.initialize_from)
    elif args.command == 'evaluate':
        result = evaluate(args.checkpoint, args.splits, args.output, args.split, args.device, args.mask_config)
    else:
        _, _, _, checkpoint = load_checkpoint(args.checkpoint, 'cpu')
        result = {key: value for key, value in checkpoint.items() if key != 'state_dict'}
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
