"""Write a learning curve and matched comparison for a fuller cloth run."""

import argparse
import csv
import json
from pathlib import Path

from .artifacts import write_json


def write_training_report(run_directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    root = Path(run_directory).resolve()
    run = json.loads((root / 'model/run.json').read_text(encoding='utf-8'))
    history = json.loads((root / 'model/history.json').read_text(encoding='utf-8'))
    protocol = json.loads((root / 'protocol.json').read_text(encoding='utf-8'))
    evaluation = json.loads((root / 'evaluation/metrics.json').read_text(encoding='utf-8'))
    previous_root = Path(protocol['prior_experiment'])
    previous = json.loads((previous_root / 'evaluation/metrics.json').read_text(encoding='utf-8'))
    previous_run = json.loads((previous_root / 'model/run.json').read_text(encoding='utf-8'))
    if run['status'] != 'complete':
        raise ValueError('training must finish before writing a final report')
    for key in ('windows', 'comparison_valid_pixels', 'target_valid_pixels', 'geometry'):
        if evaluation[key] != previous[key]:
            raise ValueError(f'previous/new evaluation differ on {key}')
    for name in ('history', 'horizon', 'expected_dt_ms', 'min_valid_fraction', 'timing_tolerance'):
        if evaluation['config'][name] != previous['config'][name]:
            raise ValueError(f'previous/new forecasting setup differs on {name}')
    current_metrics, earlier_metrics = evaluation['metrics']['all'], previous['metrics']['all']
    comparison = [
        {'method': 'ConvLSTM full', **current_metrics['convlstm']},
        {'method': 'ConvLSTM earlier', **earlier_metrics['convlstm']},
        {'method': 'Latest measurement', **current_metrics['persistence']},
        {'method': 'Constant velocity', **current_metrics['constant_velocity']},
        {'method': 'Constant acceleration', **current_metrics['constant_acceleration']},
    ]
    for name in ('persistence', 'constant_velocity', 'constant_acceleration'):
        if abs(current_metrics[name]['mae_mm'] - earlier_metrics[name]['mae_mm']) > 1e-9:
            raise ValueError(f'baseline scores differ for {name}')
    reference = current_metrics['persistence']['mae_mm']
    current = current_metrics['convlstm']['mae_mm']
    earlier = earlier_metrics['convlstm']['mae_mm']
    reductions = {'vs_latest_measurement_percent': 100 * (1 - current / reference),
                  'vs_earlier_convlstm_percent': 100 * (1 - current / earlier)}
    write_json(root / 'comparison.json', {'training': run, 'protocol': protocol,
               'comparison': comparison, 'mae_reduction': reductions,
               'evaluation': evaluation, 'test_reuse_note': protocol['test_reuse_note']})
    with (root / 'comparison.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['method', 'mae_mm', 'rmse_mm', 'p95_absolute_error_mm'])
        writer.writeheader()
        writer.writerows({k: row[k] for k in writer.fieldnames} for row in comparison)

    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={'width_ratios': [1.1, 1]})
    axes[0].plot([r['epoch'] for r in history], [r['train_mae_mm'] for r in history],
                 label='Training', color='#2368b5', linewidth=2)
    axes[0].plot([r['epoch'] for r in history], [r['validation_mae_mm'] for r in history],
                 label='Validation', color='#db8125', linewidth=2)
    axes[0].axvline(run['best_epoch'], color='#586471', linestyle=':', label='Selected checkpoint')
    axes[0].set(title='Learning curve', xlabel='Epoch', ylabel='All-target MAE (mm)')
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=.2)
    values = [r['mae_mm'] for r in comparison]
    axes[1].barh([r['method'] for r in comparison], values,
                 color=['#2368b5', '#88afd5', '#43505c', '#198b7c', '#db8125'])
    axes[1].invert_yaxis()
    axes[1].set(title='200 ms test comparison', xlabel='Common-support MAE (mm)')
    axes[1].set_xlim(0, max(values) * 1.18)
    for index, value in enumerate(values):
        axes[1].text(value + max(values) * .02, index, f'{value:.2f}', va='center')
    axes[1].grid(axis='x', alpha=.2)
    fig.suptitle('Cloth-only ConvLSTM training: 32 / 64 channels, every eligible window',
                 fontsize=14, fontweight='bold')
    fig.text(.5, .012, 'Checkpoint selected on validation | Same test recordings as earlier benchmark; comparison is exploratory',
             ha='center', fontsize=9, color='#586471')
    fig.tight_layout(rect=(0, .04, 1, .93))
    fig.savefig(root / 'training_results.png', dpi=180)
    fig.savefig(root / 'training_results.pdf')
    plt.close(fig)

    lines = ['# Cloth-only ConvLSTM training', '',
             f"Training completed after {run['epochs_completed']} epochs; selected epoch {run['best_epoch']}.", '',
             f"- Training: {protocol['frames_by_split']['train']} frames across 12 sequences; {run['train_windows']} windows.",
             f"- Validation: {protocol['frames_by_split']['validation']} frames across 3 sequences; {run['validation_windows']} windows.",
             f"- Test: {protocol['frames_by_split']['test']} frames across 3 sequences; {evaluation['windows']} windows.",
             '- Consecutive Lc_top_edge_a/b segments remain together in validation.',
             '- Five camera-Z heightmaps and masks predict 200 ms ahead at nominal 5 FPS.',
             '- Two recurrent layers (32/64 channels), absolute-height head, masked MAE, Adam, seed 42.',
             '- Every eligible training window is used; normalization is fitted only on training frames.',
             '- Up to 50 epochs, with eight-epoch validation patience. Training starts from scratch.', '',
             '## Validation selection', '',
             f"Best validation MAE: **{run['best_validation_mae_mm']:.3f} mm** "
             f"(earlier smaller model: {previous_run['best_validation_mae_mm']:.3f} mm).", '',
             '## Matched test comparison', '',
             '| Method | MAE (mm) | RMSE (mm) | P95 absolute error (mm) |', '|---|---:|---:|---:|']
    lines += [f"| {r['method']} | {r['mae_mm']:.3f} | {r['rmse_mm']:.3f} | {r['p95_absolute_error_mm']:.3f} |"
              for r in comparison]
    lines += ['', f"MAE reduction relative to the latest observation: {reductions['vs_latest_measurement_percent']:.1f}%. "
              f"Relative to the earlier ConvLSTM: {reductions['vs_earlier_convlstm_percent']:.1f}% "
              '(negative values mean worse performance).', '',
              f"Common comparison coverage: {evaluation['comparison_coverage']:.1%}. All methods use the same cells. "
              'Whole-image improvements do not establish improved printer gap accuracy.', '',
              '## Motion strata', '', '| Region | New ConvLSTM MAE | Latest measurement MAE |', '|---|---:|---:|']
    for region in ('moving', 'stationary'):
        scores = evaluation['metrics'][region]
        lines.append(f"| {region} | {scores['convlstm']['mae_mm']:.3f} mm | {scores['persistence']['mae_mm']:.3f} mm |")
    lines += ['', 'The moving/stationary distinction uses observed target changes for evaluation only (>5 mm).', '',
              '## Artifacts', '',
              '- [Selected checkpoint](model/best.pt)', '- [Training metadata](model/run.json)',
              '- [Epoch history](model/history.json)', '- [Learning curve and comparison](training_results.png)',
              '- [Future-height examples](evaluation/report.html)', '- [Full test metrics](evaluation/metrics.json)',
              '- [Configuration](config.json)', '- [Frozen sequence split paths](splits.json)', '',
              '## Interpretation limits', '', protocol['test_reuse_note'], '',
              'The larger model and denser training both change, so this run cannot isolate their separate effects. '
              'The 20 mm camera XY grid is coarse; timestamps are inferred from frame indices. '
              'This is one seed on Kinect cloth data. D405 data, camera-to-printer calibration, '
              'actual latency, finer maps, and independent nozzle-gap measurements are needed for printer use.']
    (root / 'SUMMARY.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return reductions


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    print(json.dumps(write_training_report(parser.parse_args().run), indent=2))
