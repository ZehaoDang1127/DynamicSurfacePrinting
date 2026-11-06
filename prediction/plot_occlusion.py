"""Plot one held-out synthetic-occlusion example from a trained checkpoint.

Run as ``python -m prediction.plot_occlusion --checkpoint ... --splits ...
--output ...png``. The selected example is the midpoint eligible window of the
first test session, independent of model errors. Colors show camera depth,
not calibrated printer height. This figure is not an aggregate score.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import torch

from .data import load_splits
from .occlusion import load_checkpoint
from .occlusion_data import OcclusionDataset


def plot(checkpoint_path: Path, split_path: Path, output: Path) -> None:
    model, config, norm, checkpoint = load_checkpoint(checkpoint_path, 'cpu')
    sessions = load_splits(split_path)['test']
    forbidden = {Path(p).resolve() for key in ('train_sessions', 'validation_sessions')
                 for p in checkpoint[key]}
    if any(session.path.resolve() in forbidden for session in sessions):
        raise ValueError('test sessions overlap checkpoint training or validation')
    if any(session.geometry != checkpoint['geometry'] or list(session.shape) != list(checkpoint['shape'])
           for session in sessions):
        raise ValueError('test session geometry differs from checkpoint')
    dataset = OcclusionDataset(sessions, config, norm, seed=config['seed'] + 2,
                               cache_frames=False, window_stride=1)
    first_session_indices = [i for i, (session_index, _) in enumerate(dataset.windows) if session_index == 0]
    if not first_session_indices:
        raise ValueError('first test session has no eligible windows')
    index = first_session_indices[len(first_session_indices) // 2]
    session_index, start = dataset.windows[index]
    sample = dataset[index]
    inputs = sample['inputs']
    valid = sample['target_valid'][0, 0].astype(bool)
    hidden = sample['hidden_origin'][0].astype(bool)
    observed = inputs[-1, 1] > .5
    with torch.inference_mode():
        tensor = torch.as_tensor(inputs[None], dtype=torch.float32)
        model_depth = norm.denormalize(model(tensor, rollout_steps=config['horizon'])[0, 0, 0].numpy())
    target = norm.denormalize(sample['targets'][0, 0])
    input_depth = norm.denormalize(inputs[-1, 0])
    last = np.zeros(valid.shape, dtype=np.float32)
    seen = np.zeros(valid.shape, dtype=bool)
    for frame in inputs:
        mask = frame[1] > .5
        last[mask] = frame[0][mask]
        seen |= mask
    baseline = norm.denormalize(last)
    common = valid & hidden & seen
    if not common.any():
        raise ValueError('example has no masked cells supported by baseline')
    values = target[valid]
    vmin, vmax = np.percentile(values, [1, 99])
    if vmax <= vmin:
        vmin, vmax = float(values.min()) - 1, float(values.max()) + 1
    cmap = plt.get_cmap('viridis').copy()
    cmap.set_bad('#dddddd')
    images = [(target, valid, 'Clean recorded target'),
              (input_depth, observed, 'Input with synthetic mask'),
              (model_depth, valid, 'ConvLSTM–U-Net reconstruction'),
              (baseline, valid & seen, 'Last-observed baseline')]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for ax, (values, support, title) in zip(axes.flat, images):
        image = ax.imshow(np.ma.array(values, mask=~support), origin='upper',
                          interpolation='nearest', vmin=vmin, vmax=vmax, cmap=cmap)
        # Pink pixels mark the synthetic hidden area on the input only.
        if title.startswith('Input'):
            overlay = np.ma.array(np.ones(hidden.shape), mask=~hidden)
            ax.imshow(overlay, cmap=ListedColormap(['#ec468f']), vmin=0, vmax=1,
                      origin='upper', interpolation='nearest')
        if hidden.any() and (~hidden).any():
            ax.contour(hidden.astype(float), levels=[.5], colors=['#f04a9a'], linewidths=1.2)
        ax.set_title(title)
        ax.set_xlabel('Camera X grid cell (4 mm)')
        ax.set_ylabel('Camera Y grid cell (4 mm)')
    model_error = np.abs(model_depth[common] - target[common]).mean()
    baseline_error = np.abs(baseline[common] - target[common]).mean()
    fig.colorbar(image, ax=axes, shrink=.8, label='Camera-axis depth (mm; 1st–99th percentile)')
    fig.suptitle('Synthetic occlusion: one held-out current-frame example', fontsize=14)
    fig.text(.5, -.015,
             f'{sessions[session_index].path.name}, window start {start}; pink = synthetic hidden cells; '
             f'gray = unavailable. Hidden cells supported by baseline: {int(common.sum())}; '
             f'MAE model {model_error:.2f} mm, baseline {baseline_error:.2f} mm. One example only.',
             ha='center', fontsize=9, wrap=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {output}: {sessions[session_index].path.name}, start={start}, '
          f'hidden_common={int(common.sum())}, model_mae={model_error:.3f} mm, baseline_mae={baseline_error:.3f} mm')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--splits', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    plot(args.checkpoint, args.splits, args.output)


if __name__ == '__main__':
    main()
