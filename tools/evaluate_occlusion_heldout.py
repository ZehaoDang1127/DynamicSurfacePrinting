"""Evaluate a checkpoint's original held-out split without rereading training frames.

Use only with the original split manifest used at training. The training run
already validated cross-split frame identities; this command checks the saved
session paths still match before loading the held-out sessions and delegates
all scoring and seed selection to the standard occlusion evaluator.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import patch

from prediction import occlusion
from prediction.data import load_session


def evaluate(checkpoint, split_path, output, device='auto', mask_config=None):
    split_path = Path(split_path).resolve()
    _, _, _, ckpt = occlusion.load_checkpoint(checkpoint, 'cpu')
    if Path(ckpt['split_path']).resolve() != split_path:
        raise ValueError('split path differs from the original training split')
    raw = json.loads(split_path.read_text(encoding='utf-8-sig'))
    if set(raw) != {'train', 'validation', 'test'}:
        raise ValueError('split must contain train, validation, and test')
    resolved = {}
    for name in ('train', 'validation', 'test'):
        if not isinstance(raw[name], list) or not raw[name] or not all(isinstance(p, str) and p for p in raw[name]):
            raise ValueError(f'{name} split must be a nonempty list of paths')
        resolved[name] = [(split_path.parent / p).resolve() for p in raw[name]]
        expected = [Path(p).resolve() for p in ckpt[f'{name}_sessions']]
        if resolved[name] != expected:
            raise ValueError(f'{name} session list differs from checkpoint provenance')
    paths = sum(resolved.values(), [])
    if len(set(paths)) != len(paths):
        raise ValueError('duplicate session in split')
    sessions = [load_session(path) for path in resolved['test']]
    with patch.object(occlusion, 'load_splits', return_value={'test': sessions}):
        return occlusion.evaluate(checkpoint, split_path, output, 'test', device, mask_config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--splits', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--mask-config', type=Path, default=None,
                        help='JSON object replacing the synthetic-mask settings (e.g. configs/masks/*.json)')
    args = parser.parse_args()
    result = evaluate(args.checkpoint, args.splits, args.output, args.device, args.mask_config)
    print(json.dumps({'windows': result['windows'], 'mask_seed': result['mask_seed'],
                      'output': str(args.output)}, indent=2))
