"""Export the data behind the paper figures to CSV for the MATLAB scripts.

Reads the lab run folders under output/ (history.json, metrics.json) and writes
  data/training_history.csv  run, epoch, train_objective_mm, validation_objective_mm, is_best
  data/hidden_mae_vs_lead.csv model, masks, lead_step, lead_ms, hidden_mae_mm, hidden_rmse_mm, valid_pixels

Usage: python tools/matlab/paper_figures/export_figure_data.py --output-root output
"""
import argparse
import csv
import json
from pathlib import Path

HISTORIES = {
    'standard_scratch': 'servo_30fps_convlstm_100ep_20261107',
    'standard_pretrained': 'servo_30fps_finetune_20261107',
    'mixed_scratch': 'servo_30fps_mixed_100ep_20261108',
    'mixed_pretrained': 'servo_30fps_mixed_finetune_20261108',
    'cloth_pretraining': 'cloth_pretrain_20261107',
}
METRICS = {  # (model, masks) -> run folder; test windows from the stop-start and biased-oscillation recordings
    ('mixed_scratch', 'standard'): 'eval_round2_20261108/mixed_100ep_paper_masks',
    ('mixed_pretrained', 'standard'): 'eval_round2_20261108/mixed_finetune_paper_masks',
    ('standard_scratch', 'standard'): 'servo_30fps_100ep_synthetic_eval_20261107',
    ('standard_pretrained', 'standard'): 'servo_30fps_finetune_synthetic_eval_20261107',
    ('mixed_scratch', 'large'): 'eval_round2_20261108/mixed_100ep_large_masks',
    ('mixed_pretrained', 'large'): 'eval_round2_20261108/mixed_finetune_large_masks',
    ('standard_scratch', 'large'): 'eval_round2_20261108/100ep_large_masks',
    ('standard_pretrained', 'large'): 'eval_round2_20261108/finetune_large_masks',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--output-root', default='output')
    parser.add_argument('--out', default=str(Path(__file__).with_name('data')))
    args = parser.parse_args()
    root, out = Path(args.output_root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    with open(out / 'training_history.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['run', 'epoch', 'train_objective_mm', 'validation_objective_mm', 'is_best'])
        for run, folder in HISTORIES.items():
            history = json.loads((root / folder / 'history.json').read_text())
            best = min(row['validation']['objective_mm'] for row in history)
            for row in history:
                v = row['validation']['objective_mm']
                w.writerow([run, row['epoch'], f"{row['train']['objective_mm']:.6f}", f'{v:.6f}', int(v == best)])

    with open(out / 'hidden_mae_vs_lead.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['model', 'masks', 'lead_step', 'lead_ms', 'hidden_mae_mm', 'hidden_rmse_mm', 'valid_pixels'])
        for (model, masks), folder in METRICS.items():
            metrics = json.loads((root / folder / 'metrics.json').read_text())
            if metrics['windows'] != 1740 or metrics['split'] != 'test':
                raise ValueError(f'{folder}: expected 1,740 test windows')
            for lead in metrics['leads']:
                m = lead['regions']['hidden']['model']
                w.writerow([model, masks, lead['lead_steps'], f"{lead['nominal_lead_ms']:.3f}",
                            f"{m['mae_mm']:.6f}", f"{m['rmse_mm']:.6f}", m['valid_pixels']])


if __name__ == '__main__':
    main()
