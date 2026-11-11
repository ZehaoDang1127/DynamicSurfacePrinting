# Architecture ablations

Nov 11, 2026 · @Zehao

These runs test whether the two design choices described in the paper contribute to the
primary model's hidden-cell accuracy: the visibility channels in the input (observation mask,
time since last seen, never-observed flag) and the split of the recurrence into a dynamics cell
and a correction cell. Without them, the paper can describe these components but cannot
attribute the occlusion results to them. The main confound so far is training data: the same
architecture failed under head-scale masks when trained with standard masks only (36.31 mm)
and succeeded with mixed masks (3.28 mm).

## Variants

Every config is the primary model's `configs/occlusion_d405_30fps_mixed_100ep.json` with one
field added. Data, masks, seed, epochs, loss and evaluation are unchanged.

| Run | Config | Change | Question | Parameters |
| --- | --- | --- | --- | ---: |
| `primary_repeat` | `configs/occlusion_d405_30fps_mixed_100ep.json` | none | How much do two identical runs differ? | 708,753 |
| `depth_mask` | `configs/ablations/mixed_100ep_depth_mask.json` | encoder sees depth and mask only | Do observation age and the never-observed flag help? | 708,465 |
| `stacked_cells` | `configs/ablations/mixed_100ep_stacked_cells.json` | two stacked ConvLSTM layers instead of dynamics + correction | Does the predict–correct split help at equal size? | 708,753 |
| `depth_only` | `configs/ablations/mixed_100ep_depth_only.json` | encoder sees depth only | Does the encoder need to know which cells are measured? | 708,321 |
| `depth_mask_age` | `configs/ablations/mixed_100ep_depth_mask_age.json` | drops the never-observed flag | Does the flag add anything beyond age? | 708,609 |
| `single_cell` | `configs/ablations/mixed_100ep_single_cell.json` | one ConvLSTM cell | Is a second recurrent cell needed at all? | 413,585 |

The first three runs are the priority. `primary_repeat` is needed because GPU training is not
repeatable here (two identical one-epoch runs scored 25.5 and 42.7 mm). An ablation that
differs from the primary model by less than the primary model differs from its own repeat
cannot be called better or worse.

What the switches change (`prediction/occlusion_model.py`):

- `input_channels` removes channels from the encoder input only. The data still carry all
  four channels, depth placeholders are still zeroed with the mask, and frame-level gating
  still uses the mask. In `depth_only`, a hidden cell therefore looks like a cell at the mean
  training depth.
- `recurrence: "stacked"` feeds the encoded frame into a lower ConvLSTM layer, whose hidden
  state feeds an upper layer; the decoder reads the upper layer. `"single"` uses one cell.
  Both receive zero input during rollout and on frames without observed cells, as the
  dynamics cell does. `"stacked"` has exactly the primary model's parameter count.

The default (`input_channels` all four, `recurrence: "dynamics_correction"`) is the paper
model. Existing checkpoints load unchanged.

## Running

On the lab computer, with the environment from
[TRAINING_100EP_AND_MIXED_MASKS.md](TRAINING_100EP_AND_MIXED_MASKS.md):

```powershell
cd C:\Users\haim_\Desktop\DynamicSurfacePrinting
git fetch origin
git checkout <this branch, or main after merging>
$py = ".\.venv-train\Scripts\python.exe"
& $py -m unittest tests.test_occlusion_model tests.test_occlusion_pipeline
powercfg /change standby-timeout-ac 0
powershell -ExecutionPolicy Bypass -File tools\run_ablations.ps1
```

Each run takes about 35 minutes at the round-2 speed (about 20 s per epoch on the RTX 5090),
plus a few minutes of evaluation, so all six take roughly 4 hours. To run only the priority
set:

```powershell
powershell -ExecutionPolicy Bypass -File tools\run_ablations.ps1 -Runs depth_mask,stacked_cells,primary_repeat
```

For every run, the script trains into `output\ablations_<date>\<run>`, then evaluates the test
split with the paper masks (`<run>_paper_masks`), large stationary masks (`<run>_large_masks`)
and the real print head (`<run>_real_head`). These use the same seeds and masks as round 2, so
the numbers are directly comparable with the primary model's
`output\eval_round2_20261108\mixed_100ep_*` folders, which the summary includes when present.
At the end, `output\ablations_<date>\summary.md` lists hidden-cell MAE (current and 200 ms) for
each model. The script can be restarted: completed runs and evaluations are skipped. Delete an
interrupted run folder before restarting.

Bring back `output\ablations_<date>` (the `best.pt` files are about 3 MB each).

## Reading the results

- Compare each ablation with both `primary` and `primary_repeat`. The spread between those two
  is the minimum difference worth reporting.
- The large-mask column is the most informative for occlusion: it has true labels and most
  hidden cells are never observed in the window.
- If an ablation matches the primary model within that spread, report that the component
  was not shown to matter in this setting, rather than dropping the result.
- If differences are close to the spread, train the deciding runs with two more seeds
  (change `seed` in a copy of the config) before claiming a difference. The evaluation mask
  seed is derived from the training seed, so other seeds are also scored on different masks;
  compare averages over seeds, not single runs.
