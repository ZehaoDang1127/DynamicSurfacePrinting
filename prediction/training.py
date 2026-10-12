"""Train, validate, and load reproducible one-map ConvLSTM predictors."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import PredictionConfig
from .data import HeightSequenceDataset, Normalization, fit_normalization, load_splits
from .model import ConvLSTMPredictor, masked_mae


CHECKPOINT_SCHEMA_VERSION = 1


def resolve_device(device: str | torch.device = 'auto') -> torch.device:
    """Choose CUDA when available; an explicit unavailable device is an error."""
    if str(device) == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    resolved = torch.device(device)
    if resolved.type not in ('cpu', 'cuda'):
        raise ValueError('device must be auto, cpu, cuda, or cuda:<index>')
    if resolved.type == 'cuda':
        if not torch.cuda.is_available():
            raise ValueError('CUDA was requested but is unavailable in this PyTorch installation')
        if resolved.index is not None and resolved.index >= torch.cuda.device_count():
            raise ValueError(f'CUDA device index is unavailable: {resolved.index}')
    return resolved


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f'.{path.name}.tmp')
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def _save_checkpoint(path: Path, checkpoint: dict) -> None:
    temporary = path.with_name(f'.{path.name}.tmp')
    torch.save(checkpoint, temporary)
    os.replace(temporary, path)


def _collate_samples(samples: list[dict]) -> dict[str, torch.Tensor]:
    # Leave arbitrary session metadata outside PyTorch's default collator.
    return {
        key: torch.stack([torch.as_tensor(sample[key]) for sample in samples])
        for key in ('inputs', 'target', 'target_valid')
    }


def _run_epoch(
    model: ConvLSTMPredictor,
    loader: DataLoader,
    device: torch.device,
    scale_mm: float,
    optimizer: torch.optim.Optimizer | None = None,
    grad_clip: float = 1.0,
) -> tuple[float, int]:
    model.train(optimizer is not None)
    absolute_error_sum = 0.0
    pixel_count = 0
    with torch.set_grad_enabled(optimizer is not None):
        for batch in loader:
            valid = batch['target_valid'].to(device=device, dtype=torch.bool)
            count = int(valid.sum().item())
            if count == 0:
                continue
            inputs = batch['inputs'].to(device=device, dtype=torch.float32)
            target = batch['target'].to(device=device, dtype=torch.float32)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            prediction = model(inputs)
            loss = masked_mae(prediction, target, valid)
            if optimizer is not None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip, error_if_nonfinite=True)
                optimizer.step()
            absolute_error_sum += float(loss.detach().item()) * count * scale_mm
            pixel_count += count
    if not pixel_count:
        raise ValueError('dataset contains no valid target pixels')
    return absolute_error_sum / pixel_count, pixel_count


def train(
    split_path: str | Path,
    config: PredictionConfig,
    output_dir: str | Path,
    device: str | torch.device = 'auto',
    *, cache_frames: bool = False, window_stride: int = 1,
) -> dict:
    """Fit on training sessions, select on validation, and never evaluate test.

    The best checkpoint records the exact local session paths and geometry.
    Normalization is fitted exclusively on the training recordings, before
    window construction. All epochs are bounded by config.epochs.
    """
    output = Path(output_dir).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f'training output must be a new or empty directory: {output}')
    target_device = resolve_device(device)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    splits = load_splits(split_path)
    normalization = fit_normalization(splits['train'])
    train_dataset = HeightSequenceDataset(splits['train'], config, normalization,
                                         cache_frames=cache_frames, window_stride=window_stride)
    validation_dataset = HeightSequenceDataset(splits['validation'], config, normalization,
                                              cache_frames=cache_frames)
    if len(train_dataset) == 0 or len(validation_dataset) == 0:
        raise ValueError('training and validation each need at least one eligible history/target window')
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        generator=generator, num_workers=0, collate_fn=_collate_samples,
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=0, collate_fn=_collate_samples,
    )
    model = ConvLSTMPredictor(config).to(target_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    output.mkdir(parents=True, exist_ok=True)
    best_path = output / 'best.pt'
    history_path = output / 'history.json'
    metadata_path = output / 'run.json'
    provenance = {
        f'{name}_sessions': [str(session.path.resolve()) for session in splits[name]]
        for name in ('train', 'validation', 'test')
    }
    geometry = splits['train'][0].geometry
    shape = list(splits['train'][0].shape)
    metadata = {
        'schema_version': 1,
        'started_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'running',
        'split_path': str(Path(split_path).resolve()),
        'config': config.to_dict(),
        'normalization': normalization.to_dict(),
        'geometry': geometry,
        'shape': shape,
        'device': str(target_device),
        'torch_version': str(torch.__version__),
        'train_windows': len(train_dataset),
        'train_window_stride': window_stride,
        'cache_frames': cache_frames,
        'validation_windows': len(validation_dataset),
        'checkpoint': str(best_path),
        'history': str(history_path),
        'run_metadata': str(metadata_path),
        'test_evaluated': False,
        'provenance': 'Trained locally; session paths identify data sources. Normalization uses training sessions only.',
        **provenance,
    }
    _write_json(metadata_path, metadata)
    history: list[dict] = []
    best_validation = math.inf
    best_epoch = 0
    stale_epochs = 0
    try:
        for epoch in range(1, config.epochs + 1):
            train_mae, train_pixels = _run_epoch(
                model, train_loader, target_device, normalization.scale_mm, optimizer, config.grad_clip,
            )
            validation_mae, validation_pixels = _run_epoch(
                model, validation_loader, target_device, normalization.scale_mm,
            )
            improved = validation_mae < best_validation
            history.append({
                'epoch': epoch,
                'train_mae_mm': train_mae,
                'validation_mae_mm': validation_mae,
                'train_valid_pixels': train_pixels,
                'validation_valid_pixels': validation_pixels,
                'is_best': improved,
            })
            if improved:
                best_validation = validation_mae
                best_epoch = epoch
                stale_epochs = 0
                checkpoint = {
                    'schema_version': CHECKPOINT_SCHEMA_VERSION,
                    'state_dict': {name: value.detach().cpu() for name, value in model.state_dict().items()},
                    'config': config.to_dict(),
                    'normalization': normalization.to_dict(),
                    'geometry': geometry,
                    'shape': shape,
                    'epoch': epoch,
                    'best_validation_mae_mm': best_validation,
                    'model_variant': 'Stacked ConvLSTM without peepholes; independent windows; linear 1x1 head',
                    'torch_version': str(torch.__version__),
                    'split_path': str(Path(split_path).resolve()),
                    **provenance,
                }
                _save_checkpoint(best_path, checkpoint)
            else:
                stale_epochs += 1
            _write_json(history_path, history)
            print(
                f'epoch {epoch}/{config.epochs}: train MAE {train_mae:.6f} mm, '
                f'validation MAE {validation_mae:.6f} mm',
                flush=True,
            )
            if stale_epochs >= config.patience:
                break
        metadata.update({
            'status': 'complete',
            'epochs_completed': len(history),
            'best_epoch': best_epoch,
            'best_validation_mae_mm': best_validation,
            'early_stopped': len(history) < config.epochs,
        })
    except KeyboardInterrupt:
        metadata.update({'status': 'interrupted', 'epochs_completed': len(history)})
        raise
    except Exception as error:
        metadata.update({'status': 'failed', 'error': str(error), 'epochs_completed': len(history)})
        raise
    finally:
        metadata['finished_utc'] = datetime.now(timezone.utc).isoformat()
        _write_json(metadata_path, metadata)
    return metadata


def load_checkpoint(
    path: str | Path, device: str | torch.device = 'auto',
) -> tuple[ConvLSTMPredictor, PredictionConfig, Normalization, dict]:
    """Load this project's tensor/primitive checkpoint with restricted unpickling.

    Checkpoint session paths document training provenance; they are metadata
    only and are never opened here. No arbitrary model objects are unpickled.
    """
    target_device = resolve_device(device)
    checkpoint = torch.load(Path(path), map_location='cpu', weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get('schema_version') != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError('unsupported prediction checkpoint schema')
    required = (
        'state_dict', 'config', 'normalization', 'geometry', 'shape',
        'train_sessions', 'validation_sessions', 'test_sessions',
    )
    missing = [key for key in required if key not in checkpoint]
    if missing:
        raise ValueError(f'prediction checkpoint is missing fields: {", ".join(missing)}')
    if not isinstance(checkpoint['geometry'], dict):
        raise ValueError('checkpoint geometry must be a dictionary')
    shape = checkpoint['shape']
    if not isinstance(shape, (list, tuple)) or len(shape) != 2 or any(type(n) is not int or n < 1 for n in shape):
        raise ValueError('checkpoint shape must contain two positive integers')
    for key in ('train_sessions', 'validation_sessions', 'test_sessions'):
        if not isinstance(checkpoint[key], list) or not checkpoint[key] or not all(isinstance(p, str) for p in checkpoint[key]):
            raise ValueError(f'checkpoint {key} must contain session path strings')
    config = PredictionConfig.from_dict(checkpoint['config'])
    normalization = Normalization.from_dict(checkpoint['normalization'])
    model = ConvLSTMPredictor(config)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    if not all(torch.isfinite(tensor).all() for tensor in model.state_dict().values()):
        raise ValueError('checkpoint contains nonfinite model parameters')
    model.to(target_device)
    model.eval()
    return model, config, normalization, checkpoint
