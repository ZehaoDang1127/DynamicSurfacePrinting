"""Plot per-epoch training and validation objectives for the occlusion models.

Reads ``history.json`` from each run folder under ``output/`` and writes a
2 x 3 small-multiple PDF (four 100-epoch servo models plus cloth pretraining) (full text width) and a one-column PDF for the primary
model. The objective is the training loss: current MAE + future MAE
+ hidden_weight x hidden MAE, in millimeters (see ``prediction/occlusion.py``).

Usage:
    python tools/plot_training_curves.py --output-root output --out docs/occlusion_forecasting/figures
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator  # noqa: E402

RUNS = [  # (run folder, title, gridspec slot)
    ('servo_30fps_convlstm_100ep_20261107', '(a) Standard masks, scratch', (0, 0)),
    ('servo_30fps_finetune_20261107', '(b) Standard masks, cloth-pretrained', (0, 1)),
    ('servo_30fps_mixed_100ep_20261108', '(c) Mixed masks, scratch (primary)', (1, 0)),
    ('servo_30fps_mixed_finetune_20261108', '(d) Mixed masks, cloth-pretrained', (1, 1)),
    ('cloth_pretrain_20261107', '(e) Cloth pretraining (EPFL, 5 fps)', (slice(0, 2), 2)),
]
PRIMARY = 'servo_30fps_mixed_100ep_20261108'
TRAIN_COLOR = '#2a78d6'
VAL_COLOR = '#eb6834'
INK = '#0b0b0b'
MUTED = '#52514e'
GRID = '#e4e3df'


def load(root, name):
    history = json.loads((Path(root) / name / 'history.json').read_text())
    epochs = [row['epoch'] for row in history]
    train = [row['train']['objective_mm'] for row in history]
    val = [row['validation']['objective_mm'] for row in history]
    best = min(range(len(val)), key=val.__getitem__)
    return epochs, train, val, best


def style():
    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'mathtext.fontset': 'stix', 'font.size': 7, 'axes.titlesize': 7, 'axes.labelsize': 7,
        'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'legend.fontsize': 6.5,
        'axes.edgecolor': MUTED, 'axes.linewidth': 0.5, 'xtick.color': MUTED, 'ytick.color': MUTED,
        'xtick.major.width': 0.5, 'ytick.major.width': 0.5, 'xtick.major.size': 2.5,
        'ytick.major.size': 2.5, 'axes.labelcolor': INK, 'text.color': INK, 'pdf.fonttype': 42,
    })


def panel(ax, epochs, train, val, best, title, ticks):
    ax.plot(epochs, train, color=TRAIN_COLOR, lw=1.0, ls=(0, (3, 1.5)), label='Training')
    ax.plot(epochs, val, color=VAL_COLOR, lw=1.0, label='Validation')
    ax.axvline(epochs[best], color=MUTED, lw=0.5, ls=':', zorder=0)
    ax.plot(epochs[best], val[best], 'o', ms=3.5, mfc=VAL_COLOR, mec='white', mew=0.6, zorder=3)
    ax.annotate(f'best: epoch {epochs[best]}\n{val[best]:.2f} mm', xy=(epochs[best], val[best]),
                xytext=(0.97, 0.93), textcoords='axes fraction', ha='right', va='top', fontsize=6.5,
                color=INK, zorder=4, bbox=dict(boxstyle='square,pad=0.15', fc='white', ec='none'))
    ax.set_yscale('log')
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:g}'))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(ticks[0] * 0.92, max(max(train), max(val)) * 1.1)
    ax.set_xlim(0, epochs[-1] + 1)
    ax.grid(axis='y', color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.set_title(title, loc='left', pad=3)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--output-root', default='output')
    parser.add_argument('--out', default='docs/occlusion_forecasting/figures')
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    style()
    servo_ticks = [8, 10, 15, 20, 30, 50]
    cloth_ticks = [30, 40, 50, 70, 90]

    fig = plt.figure(figsize=(7.05, 3.6))
    grid = fig.add_gridspec(2, 3)
    axes = {}
    for name, title, (row, col) in RUNS:
        ax = axes[name] = fig.add_subplot(grid[row, col])
        epochs, train, val, best = load(args.output_root, name)
        panel(ax, epochs, train, val, best, title, cloth_ticks if name.startswith('cloth') else servo_ticks)
        if row != 0:
            ax.set_xlabel('Epoch')
        if col == 0:
            ax.set_ylabel('Objective (mm, log scale)')
        print(f'{name}: {len(epochs)} epochs, best {epochs[best]} ({val[best]:.3f} mm), '
              f'final train {train[-1]:.2f} / val {val[-1]:.2f} mm')
    handles, labels = axes[PRIMARY].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=1.0, w_pad=1.0)
    fig.savefig(out / 'training_curves.pdf', bbox_inches='tight')
    fig.savefig(out / 'training_curves.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.45, 2.1))
    epochs, train, val, best = load(args.output_root, PRIMARY)
    panel(ax, epochs, train, val, best, '', servo_ticks)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Objective (mm, log scale)')
    ax.legend(loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    fig.savefig(out / 'training_curves_primary.pdf', bbox_inches='tight')
    fig.savefig(out / 'training_curves_primary.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
