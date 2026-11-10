"""Compare predicted surface change with the change implied by the commanded servo motion.

Prediction accuracy against RGB-D frames does not show that a model forecasts
motion: if the surface barely moves over the horizon, carrying the last
estimate forward scores almost as well. This tool scores the *change* each
model predicts, delta_pred(k) = prediction(lead k) - prediction(lead 0), against

* delta_cmd(k): the change implied by the designed servo trajectory, through a
  per-cell cubic map from commanded angle to camera depth fitted on the
  training recordings only (with one response lag fitted on the same data), and
* delta_meas(k): the measured change between the clean reference frames
  (synthetic test) or the separate repeat recording (real print head).

Persistence (delta = 0) is the baseline. Statistics are reported per lead for
hidden and visible cells, both per cell and as the per-window mean over the
region, which suppresses the roughly 2 mm per-frame depth noise.

Commanded angles are software commands at host receipt, not measured servo
angles; the angle-to-depth map is an empirical fit, so delta_cmd is a reference
for the motion the rig was designed to produce, not ground truth.

Subcommands
  calibrate   fit the lag and angle-to-depth map on the training split
  synthetic   run a checkpoint on the held-out test windows with synthetic masks
  real-head   score saved real print-head predictions (evaluate_real_occlusion_repeats.py --steps ...)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'acquisition' / 'servo_d405'))

DEGREE = 3
CENTER_DEG, HALF_RANGE_DEG = 150.0, 120.0


# ----------------------------------------------------------------------------- commands

def read_recording_commands(recording: Path):
    """Return ({depth_frame_number: elapsed_s}, trajectory[N, 2]) for one recording folder."""
    with (recording / 'frames.csv').open(newline='', encoding='utf-8') as handle:
        elapsed = {int(row['depth_frame_number']): float(row['elapsed_s']) for row in csv.DictReader(handle)}
    with (recording / 'trajectory.csv').open(newline='', encoding='utf-8') as handle:
        trajectory = np.array([[float(row['elapsed_s']), float(row['commanded_angle_deg'])]
                               for row in csv.DictReader(handle)])
    return elapsed, trajectory


def designed_trajectory(pattern: str):
    """Rebuild a designed trajectory (patterns 21-25 have no trajectory.csv beside their maps)."""
    import servo_patterns
    return np.array(servo_patterns.build_trajectory(pattern).samples, dtype=float)


def angle_at(trajectory, elapsed_s, lag_s):
    return np.interp(np.asarray(elapsed_s, float) - lag_s, trajectory[:, 0], trajectory[:, 1])


def design(theta):
    u = (np.asarray(theta, float) - CENTER_DEG) / HALF_RANGE_DEG
    return np.stack([u ** k for k in range(DEGREE + 1)], axis=-1)


def session_arrays(session):
    """Heights [T, cells], validity [T, cells] and frame numbers of a loaded map session."""
    heights, valid = [], []
    for frame in session.frames:
        with np.load(frame.path) as f:
            heights.append(f['height_mm'].astype(np.float64).ravel())
            valid.append(f['valid'].astype(bool).ravel())
    return np.array(heights), np.array(valid), np.array([f.frame_number for f in session.frames])


def session_elapsed(session, recordings: Path):
    elapsed, trajectory = read_recording_commands(recordings / session.path.name)
    return np.array([elapsed[f.frame_number] for f in session.frames]), trajectory


# ----------------------------------------------------------------------------- calibration

def fit_map(theta, heights, valid, min_samples=200):
    """Per-cell least-squares cubic height(theta); returns coefficients, usable mask, R^2."""
    X = design(theta)
    cells = heights.shape[1]
    coef = np.zeros((cells, DEGREE + 1))
    usable = np.zeros(cells, bool)
    r2 = np.full(cells, np.nan)
    for c in range(cells):
        m = valid[:, c]
        if m.sum() < min_samples:
            continue
        coef[c] = np.linalg.lstsq(X[m], heights[m, c], rcond=None)[0]
        residual = heights[m, c] - X[m] @ coef[c]
        var = heights[m, c].var()
        r2[c] = 1 - residual.var() / var if var > 0 else np.nan
        usable[c] = True
    return coef, usable, r2


def residual_rms(theta, heights, valid, coef, usable):
    prediction = design(theta) @ coef.T
    m = valid & usable[None]
    return float(np.sqrt(np.mean((heights - prediction)[m] ** 2)))


def calibrate(split_path: Path, recordings: Path, output: Path, lags_ms):
    from prediction.data import load_splits
    sessions = load_splits(split_path)['train']
    data = []
    for session in sessions:
        heights, valid, _ = session_arrays(session)
        elapsed, trajectory = session_elapsed(session, recordings)
        data.append((heights, valid, elapsed, trajectory))
    heights = np.concatenate([d[0] for d in data])
    valid = np.concatenate([d[1] for d in data])
    scores = {}
    for lag_ms in lags_ms:
        theta = np.concatenate([angle_at(d[3], d[2], lag_ms / 1000) for d in data])
        coef, usable, _ = fit_map(theta, heights, valid)
        scores[float(lag_ms)] = residual_rms(theta, heights, valid, coef, usable)
    lag_ms = min(scores, key=scores.get)
    theta = np.concatenate([angle_at(d[3], d[2], lag_ms / 1000) for d in data])
    coef, usable, r2 = fit_map(theta, heights, valid)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / 'calibration.npz', coefficients=coef, usable=usable, r2=r2,
                        lag_ms=lag_ms, degree=DEGREE, center_deg=CENTER_DEG, half_range_deg=HALF_RANGE_DEG)
    report = {'lag_ms': lag_ms, 'residual_rms_mm_by_lag_ms': scores,
              'usable_cells': int(usable.sum()), 'r2_median': float(np.nanmedian(r2)),
              'r2_p90': float(np.nanpercentile(r2, 90)),
              'train_sessions': [str(s.path) for s in sessions],
              'note': 'Per-cell cubic camera depth vs commanded angle, training recordings only.'}
    (output / 'calibration.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return report


def load_calibration(path: Path):
    with np.load(path) as f:
        return f['coefficients'], f['usable'], float(f['lag_ms'])


# ----------------------------------------------------------------------------- statistics

class DeltaStats:
    """Accumulates per-cell and per-window region-mean deltas for one lead and region."""

    def __init__(self):
        self.cell = {'pred': [], 'cmd': [], 'pred_m': [], 'meas_m': []}
        self.area = {'pred': [], 'cmd': [], 'meas': []}

    def add(self, pred, cmd, meas, region, meas_valid, min_cells=20):
        cells = region & np.isfinite(cmd)
        if cells.sum() < min_cells:
            return
        self.cell['pred'].append(pred[cells]); self.cell['cmd'].append(cmd[cells])
        both = cells & meas_valid
        self.cell['pred_m'].append(pred[both]); self.cell['meas_m'].append(meas[both])
        self.area['pred'].append(pred[cells].mean()); self.area['cmd'].append(cmd[cells].mean())
        self.area['meas'].append(meas[both].mean() if both.sum() >= min_cells else np.nan)

    @staticmethod
    def compare(pred, ref, active_mm):
        pred, ref = np.asarray(pred, float), np.asarray(ref, float)
        keep = np.isfinite(pred) & np.isfinite(ref)
        pred, ref = pred[keep], ref[keep]
        if pred.size < 3 or ref.std() == 0:
            return None
        out = {'n': int(pred.size), 'reference_abs_median_mm': float(np.median(np.abs(ref))),
               'reference_abs_p95_mm': float(np.percentile(np.abs(ref), 95)),
               'predicted_abs_median_mm': float(np.median(np.abs(pred))),
               'pearson_r': float(np.corrcoef(pred, ref)[0, 1]),
               'gain': float(np.polyfit(ref, pred, 1)[0]),
               'mae_model_mm': float(np.mean(np.abs(pred - ref))),
               'mae_persistence_mm': float(np.mean(np.abs(ref)))}
        active = np.abs(ref) > active_mm
        if active.sum() >= 3:
            out['active'] = {'threshold_mm': active_mm, 'n': int(active.sum()),
                             'sign_agreement': float(np.mean(np.sign(pred[active]) == np.sign(ref[active]))),
                             'mae_model_mm': float(np.mean(np.abs(pred[active] - ref[active]))),
                             'mae_persistence_mm': float(np.mean(np.abs(ref[active])))}
        return out

    def result(self, cell_active_mm=1.0, area_active_mm=0.5):
        if not self.area['pred']:
            return None
        cat = {k: np.concatenate(v) for k, v in self.cell.items()}
        return {'windows': len(self.area['pred']),
                'area_mean_vs_command': self.compare(self.area['pred'], self.area['cmd'], area_active_mm),
                'area_mean_vs_measured': self.compare(self.area['pred'], self.area['meas'], area_active_mm),
                'measured_vs_command_area_mean': self.compare(self.area['meas'], self.area['cmd'], area_active_mm),
                'per_cell_vs_command': self.compare(cat['pred'], cat['cmd'], cell_active_mm),
                'per_cell_vs_measured': self.compare(cat['pred_m'], cat['meas_m'], cell_active_mm)}


# ----------------------------------------------------------------------------- synthetic test

def synthetic(checkpoint, split_path, recordings, calibration, output, steps, device, mask_config):
    import torch
    from prediction.data import load_splits
    from prediction.occlusion import load_checkpoint, read_config, read_mask_config
    from prediction.occlusion_data import OcclusionDataset

    coef, usable, lag_ms = load_calibration(calibration)
    model, config, normalization, ckpt = load_checkpoint(checkpoint, device)
    overrides = read_mask_config(mask_config) if mask_config else None
    evaluation_config = read_config({**config, **(overrides or {})})
    rollout = max(steps)
    evaluation_config['horizon'] = rollout          # targets must reach the longest scored lead
    sessions = load_splits(split_path)['test']
    trained = {Path(p).resolve() for p in ckpt['train_sessions'] + ckpt['validation_sessions']}
    if any(s.path.resolve() in trained for s in sessions):
        raise ValueError('test sessions overlap the checkpoint training/validation sessions')
    timing = {s.path: session_elapsed(s, recordings) for s in sessions}
    dataset = OcclusionDataset(sessions, evaluation_config, normalization, seed=config['seed'] + 2,
                               cache_frames=True, window_stride=1)
    stats = {k: {r: DeltaStats() for r in ('hidden', 'visible')} for k in steps}
    model_device = next(model.parameters()).device
    for index in range(len(dataset)):
        session_index, start = dataset.windows[index]
        session = dataset.sessions[session_index]
        elapsed, trajectory = timing[session.path]
        origin = start + config['history'] - 1
        sample = dataset[index]
        with torch.inference_mode():
            predicted = model(torch.as_tensor(sample['inputs'][None], device=model_device, dtype=torch.float32),
                              rollout_steps=rollout)[0, :, 0].cpu().numpy()
        predicted = normalization.denormalize(predicted).reshape(rollout + 1, -1)
        target = normalization.denormalize(sample['targets'][:, 0]).reshape(rollout + 1, -1)
        valid = sample['target_valid'][:, 0].reshape(rollout + 1, -1)
        hidden = sample['hidden_origin'][0].ravel()
        observed = sample['inputs'][-1, 1].ravel() > .5
        theta = angle_at(trajectory, elapsed[[origin + k for k in (0, *steps)]], lag_ms / 1000)
        surface = design(theta) @ coef.T
        surface[:, ~usable] = np.nan
        for j, k in enumerate(steps, start=1):
            delta_pred = predicted[k] - predicted[0]
            delta_cmd = surface[j] - surface[0]
            delta_meas = target[k] - target[0]
            meas_valid = valid[k] & valid[0]
            stats[k]['hidden'].add(delta_pred, delta_cmd, delta_meas, hidden, meas_valid)
            stats[k]['visible'].add(delta_pred, delta_cmd, delta_meas, observed, meas_valid)
    report = {'checkpoint': str(Path(checkpoint).resolve()), 'trained_horizon_steps': config['horizon'],
              'mask_override': overrides, 'calibration': str(Path(calibration).resolve()), 'lag_ms': lag_ms,
              'windows': len(dataset), 'dataset_summary': dataset.summary(),
              'leads': {str(k): {'nominal_ms': k * config['expected_dt_ms'],
                                 **{r: s.result() for r, s in stats[k].items()}} for k in steps},
              'notes': ['delta = lead k minus lead 0 of the same window; persistence predicts zero change.',
                        'Measured deltas come from the clean recorded frames and include sensor noise.',
                        'Leads beyond the trained horizon are extrapolated rollouts.']}
    output.mkdir(parents=True, exist_ok=True)
    (output / 'motion_delta.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return report


# ----------------------------------------------------------------------------- real print head

def real_head(predictions: Path, prepared: Path, calibration: Path, output: Path):
    coef, usable, lag_ms = load_calibration(calibration)
    per_pattern, overall = {}, {}
    for path in sorted(predictions.glob('*.npz')):
        with np.load(path) as f:
            if 'prediction_future_mm' not in f.files:
                raise ValueError(f'{path.name}: rerun evaluate_real_occlusion_repeats.py with --steps')
            origins, current = f['origin_index'], f['prediction_mm']
            steps, future = [int(k) for k in f['prediction_steps']], f['prediction_future_mm']
        with np.load(prepared / path.name) as f:
            elapsed, hidden = f['blocked_elapsed_s'], f['conservative_hidden'].ravel()
            visible_valid = f['blocked_valid']
            repeat_h = f['repeat_height_mm'].reshape(len(f['repeat_height_mm']), -1)
            repeat_v = f['repeat_valid'].reshape(len(repeat_h), -1)
            match = f['reference_index']
        trajectory = designed_trajectory(path.stem)
        stats = {k: {r: DeltaStats() for r in ('hidden', 'visible')} for k in steps}
        current = current.reshape(len(current), -1)
        for w, origin in enumerate(origins):
            for j, k in enumerate(steps):
                if origin + k >= len(elapsed) or match[origin] < 0 or match[origin + k] < 0:
                    continue
                theta = angle_at(trajectory, elapsed[[origin, origin + k]], lag_ms / 1000)
                surface = design(theta) @ coef.T
                surface[:, ~usable] = np.nan
                delta_pred = future[w, j].ravel() - current[w]
                a, b = match[origin], match[origin + k]
                delta_meas = repeat_h[b] - repeat_h[a]
                meas_valid = repeat_v[a] & repeat_v[b]
                observed = visible_valid[origin].ravel() & ~hidden
                for region, mask in (('hidden', hidden), ('visible', observed)):
                    stats[k][region].add(delta_pred, surface[1] - surface[0], delta_meas, mask, meas_valid)
                    overall.setdefault(k, {}).setdefault(region, DeltaStats()).add(
                        delta_pred, surface[1] - surface[0], delta_meas, mask, meas_valid)
        per_pattern[path.stem] = {str(k): {r: s.result() for r, s in stats[k].items()} for k in steps}
    report = {'predictions': str(predictions.resolve()), 'prepared': str(prepared.resolve()),
              'lag_ms': lag_ms,
              'overall': {str(k): {r: s.result() for r, s in v.items()} for k, v in overall.items()},
              'patterns': per_pattern,
              'notes': ['Hidden = conservative stationary head mask; trajectories rebuilt from the pattern definitions.',
                        'Measured deltas come from the separate, unregistered repeat recording and are approximate.']}
    output.mkdir(parents=True, exist_ok=True)
    (output / 'motion_delta_real_head.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    cal = sub.add_parser('calibrate')
    cal.add_argument('--splits', type=Path, required=True)
    cal.add_argument('--recordings', type=Path, required=True,
                     help='folder holding the 20 recording folders with frames.csv and trajectory.csv')
    cal.add_argument('--output', type=Path, required=True)
    cal.add_argument('--lags-ms', type=float, nargs='+', default=list(range(0, 401, 20)))
    syn = sub.add_parser('synthetic')
    syn.add_argument('--checkpoint', type=Path, required=True)
    syn.add_argument('--splits', type=Path, required=True)
    syn.add_argument('--recordings', type=Path, required=True)
    syn.add_argument('--calibration', type=Path, required=True)
    syn.add_argument('--output', type=Path, required=True)
    syn.add_argument('--steps', type=int, nargs='+', default=[6, 15, 30])
    syn.add_argument('--device', default='auto')
    syn.add_argument('--mask-config', type=Path, default=None)
    real = sub.add_parser('real-head')
    real.add_argument('--predictions', type=Path, required=True)
    real.add_argument('--prepared', type=Path, required=True)
    real.add_argument('--calibration', type=Path, required=True)
    real.add_argument('--output', type=Path, required=True)
    a = parser.parse_args(argv)
    if a.command == 'calibrate':
        result = calibrate(a.splits, a.recordings, a.output, a.lags_ms)
    elif a.command == 'synthetic':
        if min(a.steps) < 1:
            parser.error('steps must be positive')
        result = synthetic(a.checkpoint, a.splits, a.recordings, a.calibration, a.output,
                           sorted(set(a.steps)), a.device, a.mask_config)
        result = {k: result[k] for k in ('windows', 'lag_ms')}
    else:
        result = real_head(a.predictions, a.prepared, a.calibration, a.output)
        result = {'lag_ms': result['lag_ms'], 'patterns': list(result['patterns'])}
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
