# Forecasting on textureless deformable cloth

This experiment uses the rectangular cloth subset of the [EPFL Texture-less
Deformable Surfaces Dataset](https://www.epfl.ch/labs/cvlab/data/texless-defsurf-data/).
It compares the existing ConvLSTM against persistence, constant velocity, and
constant acceleration at nominal 200, 400, and 600 ms horizons.

The source is Kinect depth, not D405 printer-frame data. Filename indices and
the published 5 FPS cadence provide nominal times; there are no measured capture
timestamps in the supplied NPZ archives. Missing-index windows are rejected.
Source depths are interpreted as metres, consistent with Kinect depths and the
supplied mesh Z values. The importer unprojects them using the supplied camera
intrinsics, then uses the project's existing projector to aggregate camera Z on
a fixed metric XY grid. It preserves missing-depth masks and changes no raw data.

All 18 cloth sequences are useful; garments have unrelated body motion, paper
has fewer examples, and sparse fitted meshes add an unnecessary tracking step.
The image-by-image reconstruction test lists supplied with the dataset are
unsuitable for forecasting because overlapping temporal windows would leak
between splits. The benchmark uses 12 complete training recordings, three
validation recordings, and three test recordings instead.
The consecutively numbered `Lc_top_edge_a/b` segments stay together in
validation because they may be parts of the same capture.

```powershell
.venv-tds/Scripts/python.exe -m prediction.tds prepare `
  --raw textureless_deformable_surfaces `
  --output output/tds_cloth/prepared

.venv-tds/Scripts/python.exe -m prediction.tds benchmark `
  --prepared output/tds_cloth/prepared `
  --output output/tds_cloth/benchmark `
  --epochs 40 --horizons 1 2 3 --device cuda

.venv-tds/Scripts/python.exe -m prediction.tds_plot `
  --benchmark output/tds_cloth/benchmark
```

Use a new or empty output directory for preparation and benchmarking. CPU is
supported with `--device cpu`. A separate experiment environment keeps the
original environment intact; the completed local experiment uses PyTorch's
official CUDA 12.8 wheel. `requirements-tds-benchmark.txt` lists the Python
dependencies. Install a CUDA-enabled PyTorch build appropriate to your GPU
before running with `--device cuda`.

Each model uses five input maps and validity masks, two 16-channel ConvLSTM
layers, and the existing absolute-height head. Training normalization is fitted
only on training data. Training samples start every five frames to reduce
overlap and cost; validation and test evaluate every eligible window. The best
checkpoint is selected using validation MAE. No hyperparameters use test errors.
All methods use the same five-frame history and nominal horizon. Constant
acceleration extrapolates a least-squares quadratic fit without clipping.

Read `SUMMARY.md` and `summary.csv` for the comparison. Each horizon contains a
checkpoint, training history, JSON metrics, a visual report, five NPZ examples,
per-recording metrics, per-window error sums, and paired temporal-block bootstrap
intervals. Metrics cover the common complete-history support, moving cells
(more than 5 mm change), and a central 200 x 200 mm ROI. Positive MAE reductions
relative to persistence mean useful forecasting at that horizon. Bootstrap
intervals resample five-second blocks within the three held-out recordings;
they do not quantify generalization to independent new trials.

The 32 x 32 grid has 20 mm spacing and averages away fine surface detail. These
results can screen predictor choices but cannot establish nozzle-gap accuracy
or printing quality. D405 measurements, printer calibration, actual pipeline
latency, finer grids, toolpath errors, and matched printing trials remain needed.

## Fuller cloth training

`configs/convlstm_tds_cloth_full.json` uses the same five input frames, 200 ms
forecast and fixed metric grid, with 32/64 recurrent channels. Train on every
eligible window and retain the validation-selected checkpoint:

```powershell
.venv-tds/Scripts/python.exe -m prediction train `
  --splits output/tds_cloth/prepared/splits.json `
  --config configs/convlstm_tds_cloth_full.json `
  --output output/tds_cloth/cloth_full_model `
  --device cuda --threads 4 --cache-frames --window-stride 1

.venv-tds/Scripts/python.exe -m prediction evaluate `
  --checkpoint output/tds_cloth/cloth_full_model/best.pt `
  --splits output/tds_cloth/prepared/splits.json `
  --split test --output output/tds_cloth/cloth_full_evaluation `
  --device cuda --threads 4 --motion-threshold-mm 5
```

`--cache-frames` retains loaded maps in RAM to avoid reading overlapping windows
from disk every epoch. `--window-stride` controls training only; validation
always scores every eligible window. Training is from scratch, with a maximum
of 50 epochs and early stopping after eight epochs without a better validation
MAE. `best.pt`, `history.json`, and `run.json` contain the selected weights,
learning curve, and exact run settings.

The October 12 run is saved under
`output/tds_cloth/convlstm_full_20261012_200ms`, with snapshots of its config and
split paths. It uses the same test sequences as the earlier benchmark.
Comparisons with that benchmark are exploratory because these test recordings
have already been examined; they are excluded from parameter updates,
normalization, and checkpoint selection.

Regenerate the saved October 12 run's learning curve and matched comparison
after evaluation with:

```powershell
.venv-tds/Scripts/python.exe -m prediction.tds_train_report `
  --run output/tds_cloth/convlstm_full_20261012_200ms
```
