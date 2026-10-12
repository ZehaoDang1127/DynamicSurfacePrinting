"""Reproducible forecasting benchmark on EPFL's textureless cloth sequences.

The supplied per-image reconstruction splits are deliberately not used: future
prediction needs whole-sequence holdouts. No raw data is modified. Filename
indices and the publisher's 5 FPS cadence define nominal (not measured) times.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np

from surface_mapper.config import AppConfig, CameraConfig, GridConfig
from surface_mapper.heightmap import HeightMapProjector

from .artifacts import write_json
from .config import PredictionConfig
from .data import HeightSequenceDataset, load_splits
from .baselines import fit_kinematic_baselines


DATASET_URL = 'https://www.epfl.ch/labs/cvlab/data/texless-defsurf-data/'
VALIDATION = ('Lc_top_edge_a', 'Lc_top_edge_b', 'Ll_bottom_edge')
TEST = ('Lc_left_edge', 'Ld_bottom_edge', 'Lr_top_edge_2')
METHODS = ('convlstm', 'persistence', 'constant_velocity', 'constant_acceleration')


def dataset_root(path):
    path = Path(path).resolve()
    if not (path / 'camera_intrinsics.txt').is_file():
        path = path / 'textureless_deformable_surfaces'
    if not (path / 'camera_intrinsics.txt').is_file():
        raise ValueError('dataset root must contain camera_intrinsics.txt')
    return path


def depth_points(depth, intrinsics):
    """Unproject finite positive metric depth; zero denotes missing/background."""
    depth = np.asarray(depth)
    if depth.ndim != 2:
        raise ValueError('depth must be a 2D map')
    intrinsics = np.asarray(intrinsics, dtype=np.float64)
    if intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all():
        raise ValueError('intrinsics must be a finite 3x3 matrix')
    if intrinsics[0, 0] <= 0 or intrinsics[1, 1] <= 0:
        raise ValueError('focal lengths must be positive')
    yy, xx = np.nonzero(np.isfinite(depth) & (depth > 0))
    z = depth[yy, xx].astype(np.float64)
    return np.column_stack(((xx - intrinsics[0, 2]) * z / intrinsics[0, 0],
                            (yy - intrinsics[1, 2]) * z / intrinsics[1, 1], z))


def prepare(raw_path, output_dir):
    root = dataset_root(raw_path)
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'prepare output must be empty: {output}')
    output.mkdir(parents=True, exist_ok=True)
    intrinsics = np.loadtxt(root / 'camera_intrinsics.txt')
    # 32x32 camera-frame metric grid: fixed XY cells, not fixed image pixels.
    config = AppConfig('tds_camera', CameraConfig(width=224, height=224, fps=5, warmup_frames=0),
                       GridConfig(-320, 320, -320, 320, 20, 400, 2000,
                                  min_points_per_cell=3), np.eye(4))
    projector = HeightMapProjector(config.grid, config.camera_to_output)
    sequences = sorted((root / 'cloth' / 'depth_maps').iterdir())
    splits = {name: [] for name in ('train', 'validation', 'test')}
    audit = []
    for sequence in sequences:
        if not sequence.is_dir():
            continue
        name = sequence.name
        split = 'validation' if name in VALIDATION else 'test' if name in TEST else 'train'
        destination = output / 'sessions' / name
        destination.mkdir(parents=True)
        files = sorted(sequence.glob('depth_*.npz'))
        if not files:
            raise ValueError(f'empty sequence: {sequence}')
        records, ids, valid_fractions, positive_ranges = [], [], [], []
        for file in files:
            number = int(file.stem.split('_')[-1])
            with np.load(file, allow_pickle=False) as archive:
                depth = archive['depth']
            if depth.shape != (224, 224):
                raise ValueError(f'unexpected depth shape: {file}')
            points = depth_points(depth, intrinsics)
            heightmap = projector.project(points)
            timestamp = number * 200.0
            filename = f'frame_{number:06d}.npz'
            np.savez_compressed(destination / filename, height_mm=heightmap.height_mm,
                                valid=heightmap.valid, point_count=heightmap.point_count,
                                frame_number=np.int64(number), camera_timestamp_ms=np.float64(timestamp),
                                host_monotonic_ns=np.int64(number * 200_000_000),
                                timestamp_domain='inferred_filename_5fps')
            records.append({'file': filename, 'frame_number': number,
                            'camera_timestamp_ms': timestamp, 'host_monotonic_ns': number * 200_000_000,
                            'timestamp_domain': 'inferred_filename_5fps',
                            'valid_fraction': heightmap.valid_fraction})
            ids.append(number)
            valid_fractions.append(heightmap.valid_fraction)
            if points.size:
                positive_ranges.append((float(points[:, 2].min()), float(points[:, 2].max())))
        manifest = {'schema_version': 1, 'complete': True, 'config': config.to_dict(),
                    'device': {'dataset': DATASET_URL, 'source_sequence': str(sequence),
                               'depth_units': 'metres in source; millimetres in height_mm',
                               'time_basis': 'filename frame index times 200ms; no measured timestamps'},
                    'grid': {'shape': [config.grid.height, config.grid.width],
                             'x_centers_mm': config.grid.x_centers_mm().tolist(),
                             'y_centers_mm': config.grid.y_centers_mm().tolist()},
                    'frame_count': len(records), 'frames': records}
        write_json(destination / 'manifest.json', manifest)
        splits[split].append(f'sessions/{name}')
        audit.append({'sequence': name, 'split': split, 'frames': len(records),
                      'missing_index_gaps': int(np.count_nonzero(np.diff(ids) != 1)),
                      'first_frame': ids[0], 'last_frame': ids[-1],
                      'mean_grid_valid_fraction': float(np.mean(valid_fractions)),
                      'source_depth_range_m': [min(v[0] for v in positive_ranges),
                                               max(v[1] for v in positive_ranges)]})
        print(f'prepared {name}: {len(records)} frames, {split}, '
              f'coverage {np.mean(valid_fractions):.1%}', flush=True)
    if not all(splits.values()) or not set(VALIDATION + TEST).issubset({s.name for s in sequences}):
        raise ValueError('expected cloth holdout sequences are missing')
    write_json(output / 'splits.json', splits)
    write_json(output / 'data_audit.json', {'source': str(root), 'publisher': DATASET_URL,
               'selection': 'All 18 rectangular-cloth sequences; closest available substrate analogue.',
               'source_intrinsics': intrinsics.tolist(), 'config': config.to_dict(),
               'sequences': audit, 'total_frames': sum(v['frames'] for v in audit),
               'notes': ['Whole recordings assigned before creating temporal windows.',
                         'Consecutive Lc_top_edge_a/b segments are grouped together in validation.',
                         'Original per-frame reconstruction test lists are not temporal forecasting splits.',
                         'No interpolation, hole filling, or temporal smoothing.',
                         'Nominal timing inferred from frame indices and published 5 FPS.',
                         'Source depths are interpreted as metres, consistent with Kinect and mesh Z values.',
                         'Heights are camera Z; no measured printer calibration is available.']})
    return output / 'splits.json'


def paired_blocks(checkpoint_path, splits_path, output_dir, *, block_frames=25):
    """Paired moving-pixel comparison, with temporal block bootstrap CIs.

    Resample non-overlapping 5-second source-time blocks within each held-out
    recording. Intervals describe these recordings, not independent new trials.
    """
    import torch
    from .training import load_checkpoint
    model, config, normalization, _ = load_checkpoint(checkpoint_path, 'auto')
    sessions = load_splits(splits_path)['test']
    dataset = HeightSequenceDataset(sessions, config, normalization, cache_frames=True)
    device = next(model.parameters()).device
    records, blocks = [], {}
    roi = np.zeros(sessions[0].shape, dtype=bool)
    roi[11:21, 11:21] = True  # Cell centres between -90 and +90mm on each axis.
    with torch.inference_mode():
        for i in range(len(dataset)):
            sample = dataset[i]
            prediction = normalization.denormalize(
                model(torch.from_numpy(sample['inputs'][None]).to(device))[0, 0].cpu().numpy())
            target = normalization.denormalize(sample['target'][0])
            estimates = fit_kinematic_baselines(sample['history_mm'], sample['history_valid'],
                sample['history_timestamps_ms'], config.horizon * config.expected_dt_ms)
            common = sample['target_valid'][0].copy()
            for estimate in estimates.values():
                common &= estimate.prediction_valid
            methods = {'convlstm': prediction, **{k: v.prediction_mm for k, v in estimates.items()}}
            masks = {'all': common, 'moving': common & (np.abs(target - sample['last_mm']) > 5),
                     'central_roi': common & roi}
            row = {'sequence': Path(sample['session']).name,
                   'target_timestamp_ms': sample['target_timestamp_ms']}
            source_index = int(round(sample['last_timestamp_ms'] / 200))
            block_key = (row['sequence'], source_index // block_frames)
            block = blocks.setdefault(block_key, {region: {'count': 0, **{m: 0.0 for m in METHODS}}
                                                 for region in masks})
            for region, mask in masks.items():
                count = int(mask.sum())
                row[f'{region}_pixels'] = count
                block[region]['count'] += count
                for method, values in methods.items():
                    absolute_sum = float(np.abs(values[mask] - target[mask]).sum(dtype=np.float64))
                    row[f'{region}_{method}_absolute_sum_mm'] = absolute_sum
                    block[region][method] += absolute_sum
            records.append(row)
    output = Path(output_dir)
    with (output / 'per_window_errors.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    rng = np.random.default_rng(42)
    groups = [[value for key, value in blocks.items() if key[0] == session.path.name] for session in sessions]
    summary = {}
    for region in ('all', 'moving', 'central_roi'):
        sums = np.array([[b[region]['count'], *[b[region][m] for m in METHODS]] for b in blocks.values()])
        totals = sums.sum(axis=0)
        if not totals[0]:
            summary[region] = {'valid_pixels': 0, 'metrics': {}, 'paired_vs_persistence': {}}
            continue
        metrics = {m: float(totals[k + 1] / totals[0]) for k, m in enumerate(METHODS)}
        draws = []
        for _ in range(2000):
            sampled = []
            for group in groups:
                values = np.array([[b[region]['count'], *[b[region][m] for m in METHODS]] for b in group])
                sampled.append(values[rng.integers(0, len(values), len(values))].sum(axis=0))
            total = np.sum(sampled, axis=0)
            if total[0]:
                draws.append(total[1:] / total[0])
        draws = np.array(draws)
        baseline_index = METHODS.index('persistence')
        comparisons = {}
        for index, method in enumerate(METHODS):
            if method == 'persistence':
                continue
            difference = draws[:, baseline_index] - draws[:, index]
            comparisons[method] = {'mae_reduction_mm': metrics['persistence'] - metrics[method],
                'mae_reduction_percent': 100 * (1 - metrics[method] / metrics['persistence']),
                'conditional_95pct_ci_mm': np.percentile(difference, [2.5, 97.5]).tolist()}
        summary[region] = {'valid_pixels': int(totals[0]), 'metrics': metrics,
                           'paired_vs_persistence': comparisons}
    result = {'horizon_ms': config.horizon * config.expected_dt_ms, 'windows': len(records),
              'temporal_blocks': len(blocks), 'block_frames': block_frames,
              'motion_threshold_mm': 5, 'central_roi_xy_mm': [-100, -100, 100, 100],
              'bootstrap_note': '2000 within-recording resamples of nonoverlapping 5s blocks; '
                                'conditional on only three held-out recordings, not population-level evidence.',
              'regions': summary}
    write_json(output / 'paired_comparison.json', result)
    return result


def benchmark(prepared_dir, output_dir, *, horizons=(1, 2, 3), epochs=40, device='auto'):
    import torch
    from .training import train
    from .evaluation import evaluate
    prepared, output = Path(prepared_dir).resolve(), Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'benchmark output must be empty: {output}')
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    # No test-data-driven hyperparameter changes. Each horizon is a direct model.
    config = PredictionConfig(history=5, expected_dt_ms=200, timing_tolerance=0.01,
                              min_valid_fraction=0.1, hidden_channels=(16, 16), batch_size=32,
                              epochs=epochs, patience=10, learning_rate=0.001, seed=42)
    results = []
    for horizon in horizons:
        horizon_dir = output / f'horizon_{horizon * 200}ms'
        horizon_dir.mkdir()
        current = replace(config, horizon=horizon)
        write_json(horizon_dir / 'config.json', current.to_dict())
        print(f'Training direct {horizon * 200}ms ConvLSTM', flush=True)
        started = time.perf_counter()
        training = train(prepared / 'splits.json', current, horizon_dir / 'model', device,
                         cache_frames=True, window_stride=5)
        print(f'Evaluating held-out {horizon * 200}ms forecasts', flush=True)
        report = evaluate(horizon_dir / 'model/best.pt', prepared / 'splits.json',
                          horizon_dir / 'evaluation', device=device, motion_threshold_mm=5)
        paired = paired_blocks(horizon_dir / 'model/best.pt', prepared / 'splits.json',
                               horizon_dir / 'evaluation')
        results.append({'horizon_ms': horizon * 200, 'training': training,
                        'evaluation': report, 'paired': paired,
                        'elapsed_seconds': time.perf_counter() - started})
        write_json(output / 'benchmark.json', {'data_audit': str(prepared / 'data_audit.json'),
                                             'results': results})
        print(json.dumps({'horizon_ms': horizon * 200, 'MAE_mm': paired['regions']['all']['metrics']}),
              flush=True)
    write_summary(output, prepared, results)
    return results


def write_summary(output, prepared, results):
    audit = json.loads((prepared / 'data_audit.json').read_text(encoding='utf-8'))
    lines = ['# Textureless cloth forecasting experiment', '',
             'All 18 cloth sequences (15,799 frames) are converted to camera-frame metric heightmaps. '
             'Twelve complete sequences train the model, three select checkpoints, and three are held out. '
             'The consecutive a/b segments stay together in validation.', '',
             '| Horizon | ConvLSTM MAE | Persistence MAE | Velocity MAE | Acceleration MAE | Common coverage |',
             '|---|---:|---:|---:|---:|---:|']
    rows = []
    for result in results:
        metrics = result['evaluation']['metrics']['all']
        row = {'horizon_ms': result['horizon_ms'],
               **{m + '_mae_mm': metrics[m]['mae_mm'] for m in METHODS},
               'comparison_coverage': result['evaluation']['comparison_coverage']}
        rows.append(row)
        lines.append(f"| {row['horizon_ms']} ms | " + ' | '.join(f"{metrics[m]['mae_mm']:.3f} mm" for m in METHODS)
                     + f" | {row['comparison_coverage']:.1%} |")
    lines += ['', '## Evaluation design', '',
              '- Rectangular cloth is the closest available analogue of a deforming substrate. RGB, normals, '
              'garments, and sparse fitted meshes are unnecessary for depth-to-future-depth forecasting.',
              '- Test sequences: ' + ', '.join(TEST) + '.',
              '- Validation sequences: ' + ', '.join(VALIDATION) + '.',
              '- Consecutive Lc_top_edge_a/b segments are kept together to avoid capture leakage.',
              '- Source depth in metres is unprojected using supplied intrinsics and aggregated with a mean '
              'onto a fixed 32 x 32 camera XY grid spanning -320 to 320 mm at 20 mm spacing. '
              'At least three source points support a cell. Zero, negative, nonfinite, and missing depths are invalid.',
              '- Timestamps are inferred from original frame indices at the published 5 FPS. '
              'Any window crossing a missing index is rejected; no temporal interpolation occurs.',
              '- All methods use five observations covering 800 ms. ConvLSTM has two 16-channel recurrent '
              'layers and the existing absolute-height linear head. It is trained separately for each horizon '
              'with masked MAE, Adam, batch 32, seed 42, training stride 5, and validation-selected checkpoints. '
              'Normalization uses training frames only. Validation and test score every eligible window.',
              '- Baselines use the same history and forecast horizon: last measurement, linear least-squares '
              'fit, and quadratic least-squares fit. No clipping or fallback is applied.',
              '- Comparisons use exactly the same finite target and complete-history cells. '
              'Moving cells change by more than 5 mm between the last input and target, determined offline only. '
              'The central 200 x 200 mm ROI is also reported.',
              '- Exact MAE/RMSE and sampled P95, per-recording scores, inference timings, '
              'per-window error sums, and conditional temporal-block bootstrap intervals are saved.', '',
              '## Interpretation', '']
    for result in results:
        p = result['paired']['regions']['all']['paired_vs_persistence']
        lines.append(f"At {result['horizon_ms']} ms, ConvLSTM reduces MAE by "
                     f"{p['convlstm']['mae_reduction_percent']:.1f}% and constant acceleration by "
                     f"{p['constant_acceleration']['mae_reduction_percent']:.1f}% relative to persistence "
                     '(negative reductions indicate worse predictions).')
        ci = p['convlstm']['conditional_95pct_ci_mm']
        lines.append(f"The ConvLSTM MAE reduction is {p['convlstm']['mae_reduction_mm']:.3f} mm "
                     f"with a conditional block-bootstrap 95% interval of {ci[0]:.3f} to {ci[1]:.3f} mm. "
                     f"Common comparison coverage is {result['evaluation']['comparison_coverage']:.1%} "
                     f"over {result['evaluation']['windows']} test windows.")
        if ci[0] <= 0 <= ci[1]:
            lines.append('The interval includes zero: this horizon does not show a clear overall benefit '
                         'for ConvLSTM over the latest observation on these recordings.')
        elif ci[0] > 0:
            lines.append('The interval is positive: these recordings support a modest overall '
                         'ConvLSTM improvement at this horizon.')
        else:
            lines.append('The interval is negative: ConvLSTM performs worse than the latest observation '
                         'at this horizon on these recordings.')
    lines += ['', '## Stationary-region tradeoff', '',
              'Stationary here means an observed last-to-target change of at most 5 mm. '
              'This classification is available for offline evaluation only.', '',
              '| Horizon | ConvLSTM MAE | Persistence MAE | Acceleration MAE |', '|---|---:|---:|---:|']
    for result in results:
        stationary = result['evaluation']['metrics']['stationary']
        lines.append(f"| {result['horizon_ms']} ms | " + ' | '.join(
            f"{stationary[m]['mae_mm']:.3f} mm" for m in ('convlstm', 'persistence', 'constant_acceleration')) + ' |')
    lines += ['', '## Results by held-out recording', '',
              '| Horizon | Recording | ConvLSTM MAE | Persistence MAE | Acceleration MAE |',
              '|---|---|---:|---:|---:|']
    for result in results:
        for session, regions in result['evaluation']['per_session'].items():
            lines.append(f"| {result['horizon_ms']} ms | {Path(session).name} | " + ' | '.join(
                f"{regions['all'][m]['mae_mm']:.3f} mm" for m in ('convlstm', 'persistence', 'constant_acceleration')) + ' |')
    lines += ['', 'These results establish only offline forecast performance for the tested model and recordings. '
              'Three held-out sessions and one model seed do not establish broad generalization. '
              'The 20 mm grid averages away fine spatial detail. Kinect targets contain noise, lateral motion '
              'and occlusions. Camera Z is not calibrated printer Z, and 5 FPS cannot validate forecasts '
              'shorter than 200 ms. Printer usefulness still requires D405 data, measured latency, '
              'toolpath/gap error, independent geometry, and matched printing experiments.', '',
              f'[Dataset publisher]({DATASET_URL})', '', '## Prepared sequence inventory', '',
              '| Sequence | Split | Frames | Missing-index gaps |', '|---|---|---:|---:|']
    lines += [f"| {s['sequence']} | {s['split']} | {s['frames']} | {s['missing_index_gaps']} |"
              for s in audit['sequences']]
    (output / 'SUMMARY.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    with (output / 'summary.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    command = commands.add_parser('prepare')
    command.add_argument('--raw', required=True, type=Path)
    command.add_argument('--output', required=True, type=Path)
    command = commands.add_parser('benchmark')
    command.add_argument('--prepared', required=True, type=Path)
    command.add_argument('--output', required=True, type=Path)
    command.add_argument('--epochs', type=int, default=40)
    command.add_argument('--horizons', nargs='+', type=int, default=[1, 2, 3])
    command.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.raw, args.output)
    else:
        benchmark(args.prepared, args.output, epochs=args.epochs, horizons=args.horizons, device=args.device)


if __name__ == '__main__':
    main()
