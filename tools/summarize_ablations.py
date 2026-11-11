"""Tabulate hidden-cell errors of occlusion models evaluated with run_ablations.ps1.

Each model is identified by an evaluation prefix: ``<prefix>_paper_masks``,
``<prefix>_large_masks`` and, optionally, ``<prefix>_real_head`` folders that each
contain ``metrics.json``. Prefixes are discovered under ``--root`` and can be added
with ``--model NAME=PREFIX`` (for example the round-2 primary model).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

COLUMNS = (
    ('paper_masks', 'Paper masks, all hidden (current / 200 ms)'),
    ('paper_support', 'Paper masks, baseline-supported hidden (current / 200 ms)'),
    ('large_masks', 'Large masks 70-90%, all hidden (current / 200 ms)'),
    ('real_head', 'Real head, hidden vs. repeat (current / 200 ms)'),
)


def _read(path: Path):
    metrics = path / 'metrics.json'
    return json.loads(metrics.read_text(encoding='utf-8')) if metrics.is_file() else None


def _synthetic(report, method):
    if report is None:
        return None
    first, last = report['leads'][0], report['leads'][-1]
    return tuple(lead['regions']['hidden'][method]['mae_mm'] for lead in (first, last))


def summarize(prefixes: dict[str, Path]) -> dict:
    rows = {}
    for name, prefix in prefixes.items():
        paper = _read(prefix.with_name(prefix.name + '_paper_masks'))
        large = _read(prefix.with_name(prefix.name + '_large_masks'))
        real = _read(prefix.with_name(prefix.name + '_real_head'))
        rows[name] = {
            'prefix': str(prefix),
            'paper_masks': _synthetic(paper, 'model'),
            'paper_support': _synthetic(paper, 'model_on_baseline_support'),
            'large_masks': _synthetic(large, 'model'),
            'real_head': None if real is None else (real['overall']['hidden']['mae_mm'],
                                                    real['overall']['future_hidden']['mae_mm']),
        }
    return rows


def markdown(rows: dict) -> str:
    def cell(pair):
        return 'n/a' if pair is None or None in pair else f'{pair[0]:.2f} / {pair[1]:.2f}'
    lines = ['| Model | ' + ' | '.join(title for _, title in COLUMNS) + ' |',
             '| --- |' + ' ---: |' * len(COLUMNS)]
    lines += ['| ' + name + ' | ' + ' | '.join(cell(row[key]) for key, _ in COLUMNS) + ' |'
              for name, row in rows.items()]
    return '\n'.join(lines) + '\n\nHidden-cell MAE in mm of camera-axis depth.\n'


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, help='folder searched for <name>_paper_masks evaluations')
    parser.add_argument('--model', action='append', default=[], metavar='NAME=PREFIX',
                        help='additional evaluation prefix, e.g. primary=output/eval_round2_20261108/mixed_100ep')
    parser.add_argument('--output', type=Path, help='write the table (.md) and values (.json) with this stem')
    args = parser.parse_args(argv)
    prefixes = {}
    for item in args.model:
        name, sep, prefix = item.partition('=')
        if not sep or not name or not prefix:
            parser.error('--model must be NAME=PREFIX')
        prefixes[name] = Path(prefix)
    if args.root is not None:
        for folder in sorted(args.root.glob('*_paper_masks')):
            prefixes.setdefault(folder.name[:-len('_paper_masks')], folder.with_name(folder.name[:-len('_paper_masks')]))
    if not prefixes:
        parser.error('no evaluations found; pass --root or --model')
    rows = summarize(prefixes)
    table = markdown(rows)
    print(table)
    if args.output is not None:
        args.output.with_suffix('.md').write_text(table, encoding='utf-8')
        args.output.with_suffix('.json').write_text(json.dumps(rows, indent=2) + '\n', encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
