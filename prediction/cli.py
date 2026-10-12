"""Commands for session inspection, learning, evaluation and inference."""

import argparse
import json
from pathlib import Path

from .config import load_prediction_config


def build_parser():
    parser = argparse.ArgumentParser(description='Predict future surface maps with kinematic baselines or ConvLSTM.')
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('inspect', 'train'):
        command = commands.add_parser(name)
        command.add_argument('--splits', type=Path, required=True)
        command.add_argument('--config', type=Path, default=Path('configs/convlstm.json'))
        if name == 'train':
            command.add_argument('--output', type=Path, required=True)
            command.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
            command.add_argument('--cache-frames', action='store_true',
                                 help='cache loaded training/validation maps in RAM')
            command.add_argument('--window-stride', type=int, default=1,
                                 help='training window start stride; validation always uses stride 1')
    command = commands.add_parser('evaluate')
    command.add_argument('--splits', type=Path, required=True)
    command.add_argument('--split', choices=('train', 'validation', 'test'), default='test')
    command.add_argument('--roi', nargs=4, type=int, metavar=('X0', 'Y0', 'X1', 'Y1'))
    command.add_argument('--motion-threshold-mm', type=float, default=0.5)
    for name in ('predict', 'live'):
        sub = commands.add_parser(name)
        if name == 'predict':
            sub.add_argument('--session', type=Path, required=True)
        else:
            sub.add_argument('--camera-config', type=Path, default=Path('configs/d405_camera_frame.json'))
            sub.add_argument('--source', choices=('realsense', 'synthetic'), default='realsense')
            sub.add_argument('--frames', type=int, default=300, help='0 runs until Ctrl+C')
    for name in ('evaluate', 'predict', 'live'):
        sub = commands.choices[name]
        sub.add_argument('--checkpoint', type=Path, required=True)
        sub.add_argument('--output', type=Path, required=True)
        sub.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    command = commands.add_parser('demo')
    command.add_argument('--output', type=Path, required=True)
    command.add_argument('--epochs', type=int, default=2)
    command.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='cpu')
    for name in ('baseline-live', 'baseline-predict', 'baseline-evaluate'):
        sub = commands.add_parser(name, help='NumPy constant-acceleration baseline; no checkpoint')
        sub.add_argument('--output', type=Path, required=True)
        sub.add_argument('--history', type=int, default=5)
        sub.add_argument('--horizon-ms', type=float, default=None, help='default: one configured camera interval')
        sub.add_argument('--max-gap-ms', type=float, default=100.0)
        sub.add_argument('--min-valid-fraction', type=float, default=0.1)
        if name == 'baseline-live':
            sub.add_argument('--camera-config', type=Path, default=Path('configs/d405_camera_frame.json'))
            sub.add_argument('--source', choices=('realsense', 'synthetic'), default='realsense')
            sub.add_argument('--frames', type=int, default=300, help='0 runs until Ctrl+C')
            sub.add_argument('--record-input', action='store_true', help='save incoming maps under output/observations')
            sub.add_argument('--save-depth', action='store_true', help='include raw depth; requires --record-input')
        else:
            sub.add_argument('--session', type=Path, required=True)
        if name == 'baseline-evaluate':
            sub.add_argument('--roi', nargs=4, type=int, metavar=('X0', 'Y0', 'X1', 'Y1'))
            sub.add_argument('--motion-threshold-mm', type=float, default=0.5)
            sub.add_argument('--target-tolerance-ms', type=float, default=None,
                             help='maximum observed/nominal target offset; default 20%% of camera interval')
    for name, sub in commands.choices.items():
        if name == 'inspect' or name.startswith('baseline-'):
            continue
        sub.add_argument('--threads', type=int, default=4, help='CPU intra-op PyTorch threads')
    return parser


def run(args):
    if args.command == 'inspect':
        from .data import HeightSequenceDataset, fit_normalization, load_splits
        config = load_prediction_config(args.config)
        splits = load_splits(args.splits)
        normalization = fit_normalization(splits['train'])
        report = {'normalization': normalization.to_dict(), 'splits': {}}
        for name, sessions in splits.items():
            dataset = HeightSequenceDataset(sessions, config, normalization)
            report['splits'][name] = {**dataset.summary(), 'eligible_windows': len(dataset),
                                      'shape': list(sessions[0].shape)}
        print(json.dumps(report, indent=2))
        return 0
    if args.command.startswith('baseline-'):
        from .baseline_commands import baseline_live, baseline_predict, baseline_evaluate
        settings = dict(history=args.history, horizon_ms=args.horizon_ms,
                        max_gap_ms=args.max_gap_ms, min_valid_fraction=args.min_valid_fraction)
        if args.command == 'baseline-live':
            result = baseline_live(args.camera_config, args.output, source_name=args.source,
                                   frames=args.frames, record_input=args.record_input,
                                   save_depth=args.save_depth, **settings)
        elif args.command == 'baseline-predict':
            result = baseline_predict(args.session, args.output, **settings)
        else:
            result = baseline_evaluate(args.session, args.output, roi=args.roi,
                                       motion_threshold_mm=args.motion_threshold_mm,
                                       target_tolerance_ms=args.target_tolerance_ms, **settings)
            print(json.dumps({'output': str(args.output), 'windows': result['windows'],
                              'comparison_coverage': result['comparison_coverage'],
                              'metrics': result['metrics']['all']}, indent=2))
            return 0
        print(json.dumps({'output': str(args.output), 'complete': result['complete'],
                          'forecasts': result['forecast_count'], 'buffer_resets': result['buffer_resets']}, indent=2))
        return 0
    import torch
    if args.threads < 1:
        raise ValueError('--threads must be positive')
    torch.set_num_threads(args.threads)
    if args.command == 'train':
        from .training import train
        result = train(args.splits, load_prediction_config(args.config), args.output, args.device,
                       cache_frames=args.cache_frames, window_stride=args.window_stride)
    elif args.command == 'evaluate':
        from .evaluation import evaluate
        result = evaluate(args.checkpoint, args.splits, args.output, args.split,
                          args.device, args.roi, args.motion_threshold_mm)
        print(json.dumps(result['metrics']['all'], indent=2))
        return 0
    elif args.command == 'predict':
        from .commands import predict_session
        result = predict_session(args.checkpoint, args.session, args.output, args.device)
    elif args.command == 'live':
        from .commands import live
        result = live(args.checkpoint, args.camera_config, args.output, args.source, args.frames, args.device)
    else:
        from .commands import demo
        result = demo(args.output, args.epochs, args.device)
    print(json.dumps(result, indent=2))
    return 0


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run(args)
    except ImportError as exc:
        requirements = 'requirements.txt' if args.command.startswith('baseline-') else 'requirements-prediction.txt'
        parser.exit(2, f'error: {exc}. Install dependencies with python -m pip install -r {requirements}\n')
    except (ValueError, OSError, RuntimeError, KeyError, TypeError) as exc:
        parser.exit(2, f'error: {exc}\n')


if __name__ == '__main__':
    raise SystemExit(main())
