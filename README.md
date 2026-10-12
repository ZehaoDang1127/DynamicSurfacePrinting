# Dynamic Surface Printer: ConvLSTM and baselines

Python code for recording surface height maps, forecasting a future map with
ConvLSTM or kinematic baselines, and evaluating those forecasts. This README
covers the software setup and commands. Run every command from the repository
root. Command blocks use PowerShell.

## Install

Use Python 3.10 or newer. Create an environment and install the dependencies
for the workflows you need:

```powershell
git clone https://github.com/ZehaoDang1127/DynamicSurfacePrinting.git
Set-Location DynamicSurfacePrinting
python -m venv .venv
$python = ".\.venv\Scripts\python.exe"

# Capture and NumPy baselines.
& $python -m pip install -r requirements.txt

# ConvLSTM (includes capture dependencies).
& $python -m pip install -r requirements-prediction.txt

# Cloth benchmark plots (includes prediction dependencies).
& $python -m pip install -r requirements-tds-benchmark.txt
```

In an existing checkout, start from its root and set `$python` to the environment
you want to use. Select the existing local benchmark environment with
`$python = ".\.venv-tds\Scripts\python.exe"`.

`requirements.txt` installs NumPy and, on Windows, `pyrealsense2`.
Synthetic capture and all kinematic baselines only need NumPy at runtime.
ConvLSTM additionally needs PyTorch; benchmark plotting additionally needs
Matplotlib. A physical RealSense camera also requires its runtime/driver.

Use `--device cpu` for CPU execution, `--device auto` to select CUDA when
available, or `--device cuda` to require a CUDA-enabled PyTorch installation.
The current local benchmark environment uses a CUDA 12.8 PyTorch build. To
install that build in a compatible environment:

```powershell
& $python -m pip install --upgrade "torch>=2.6,<3" --index-url https://download.pytorch.org/whl/cu128
& $python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

## Source layout

| Path | Purpose |
| --- | --- |
| `surface_mapper/` | Camera/synthetic acquisition, height-map projection, session recording |
| `prediction/model.py` | Stacked ConvLSTM cells and masked MAE loss |
| `prediction/data.py` | Session loading, split validation, normalization, temporal windows |
| `prediction/training.py` | Training, validation, early stopping, checkpoints |
| `prediction/inference.py`, `prediction/commands.py` | Streaming inference, replay, synthetic demo |
| `prediction/baselines.py`, `prediction/baseline_commands.py` | Persistence, velocity, acceleration, baseline workflows |
| `prediction/evaluation.py`, `prediction/metrics.py`, `prediction/reporting.py` | Shared-support metrics and HTML reports |
| `prediction/artifacts.py` | Forecast maps, manifests, and timing logs |
| `prediction/tds.py`, `prediction/tds_plot.py` | Cloth dataset preparation, benchmark, plots |
| `prediction/tds_train_report.py` | Matched comparison with an earlier cloth training run |
| `visualize_ca_live.m` | MATLAB browser for recorded baseline forecasts and errors |
| `configs/` | Camera, ConvLSTM, and example split configurations |
| `tests/` | Unit and integration tests |

Use a new or empty output directory for each capture, training, evaluation,
replay, or benchmark run. Outputs go under `output/` in these examples.

## Verify everything without a camera

```powershell
& $python -m unittest discover -s tests -v
& $python -m surface_mapper --source synthetic --frames 30 --output output/synthetic_capture
& $python -m prediction demo --output output/prediction_demo --epochs 2 --device cpu
```

The demo generates independent train/validation/test recordings, trains a
small ConvLSTM, evaluates it against baselines, and replays the test session.
It uses six input frames and 8/8 hidden channels. See
`output/prediction_demo/demo_report.json` and
`output/prediction_demo/evaluation/report.html`.

Exercise live ConvLSTM inference with the demo's matching camera configuration:

```powershell
& $python -m prediction live `
  --checkpoint output/prediction_demo/model/best.pt `
  --camera-config output/prediction_demo/camera.json `
  --source synthetic --frames 30 `
  --output output/demo_live --device cpu
```

## Record height-map sessions

```powershell
& $python -m surface_mapper `
  --config configs/d405_camera_frame.json `
  --source realsense --frames 300 `
  --output output/train_trial_01 --save-depth
```

Repeat with distinct output names for other recordings used in your split file.
Use `--frames 0` to capture until Ctrl+C, `--source synthetic` for generated
input, and omit `--save-depth` when raw depth is unnecessary.

A completed session contains `manifest.json`, `frame_*.npz`, and
`latest_height_mm.csv`. Prediction reads `height_mm`, `valid`, frame numbers,
and camera timestamps. Invalid heights are NaN. Raw depth and manual labels
are unnecessary for prediction; future recorded maps supply training targets.

Recordings used by one model must share grid geometry, camera cadence, and
coordinate transform. The default camera configuration uses an identity
transform, so heights represent camera Z in millimetres.

## Run the baselines

The baseline CLI produces constant-acceleration forecasts without a checkpoint.
Evaluation compares persistence (the latest map), a linear least-squares
velocity fit, and a quadratic least-squares acceleration fit. Fits use actual
observation timestamps.

Run, replay, and evaluate a synthetic baseline trial:

```powershell
& $python -m prediction baseline-live `
  --camera-config configs/d405_camera_frame.json `
  --source synthetic --frames 30 --history 5 `
  --output output/ca_synthetic --record-input

& $python -m prediction baseline-predict `
  --session output/ca_synthetic/observations `
  --history 5 --output output/ca_replay

& $python -m prediction baseline-evaluate `
  --session output/ca_synthetic/observations `
  --history 5 --output output/ca_evaluation
```

Run the predictor with a RealSense camera and save its inputs:

```powershell
& $python -m prediction baseline-live `
  --camera-config configs/d405_camera_frame.json `
  --source realsense --frames 0 --history 5 `
  --output output/ca_live --record-input --save-depth

# Run after stopping capture with Ctrl+C.
& $python -m prediction baseline-evaluate `
  --session output/ca_live/observations `
  --history 5 --output output/ca_live_evaluation
```

The default horizon is one configured camera interval, about 33.3 ms at 30 FPS.
Set `--horizon-ms 66.6666666667` for a two-interval forecast. All three baseline
commands accept `--history`, `--horizon-ms`, `--max-gap-ms` (default 100), and
`--min-valid-fraction` (default 0.1). History must be at least three frames.
Unsupported fitted cells remain NaN.

`baseline-evaluate` also accepts `--target-tolerance-ms`,
`--motion-threshold-mm`, and `--roi X0 Y0 X1 Y1`. Its default target matching
tolerance is 20% of a camera interval. ROI bounds are grid indices with the
upper bounds excluded. Reports are `metrics.json`, `report.html`, and
`example_*.npz`.

### Optional MATLAB baseline viewer

After a `baseline-live` run with `--record-input`, run this from MATLAB with
the repository as the current folder:

```matlab
visualize_ca_live(fullfile(pwd, 'output', 'ca_synthetic'))
% Or inspect the real-camera example above:
visualize_ca_live(fullfile(pwd, 'output', 'ca_live'))
```

The viewer shows forecasts, matched future observations, errors, velocity,
acceleration, support, and timing. Use its slider or arrow keys to browse.
It reads NPZ files directly without a Python bridge or extra toolbox.
Calling `visualize_ca_live` without an argument selects `output/ca_live`.

## Train and evaluate ConvLSTM

Create the split file:

```powershell
Copy-Item configs/splits.example.json configs/splits.json
```

Edit `configs/splits.json` to identify your completed recordings. Paths resolve
relative to the split file, so this example refers to the repository's `output/`:

```json
{
  "train": ["../output/train_trial_01", "../output/train_trial_02"],
  "validation": ["../output/validation_trial_01"],
  "test": ["../output/test_trial_01"]
}
```

Keep whole recordings in exactly one split. Normalization uses training frames,
checkpoint selection uses validation MAE, and testing supplies the final
comparison.

```powershell
& $python -m prediction inspect `
  --splits configs/splits.json --config configs/convlstm.json

& $python -m prediction train `
  --splits configs/splits.json --config configs/convlstm.json `
  --output output/convlstm_model --device auto --threads 4

& $python -m prediction evaluate `
  --checkpoint output/convlstm_model/best.pt `
  --splits configs/splits.json --split test `
  --output output/convlstm_evaluation --device auto --threads 4

& $python -m prediction predict `
  --checkpoint output/convlstm_model/best.pt `
  --session output/test_trial_01 `
  --output output/convlstm_replay --device auto

& $python -m prediction live `
  --checkpoint output/convlstm_model/best.pt `
  --camera-config configs/d405_camera_frame.json `
  --source realsense --frames 300 `
  --output output/convlstm_live --device auto
```

The checkpoint and live/replay inputs must have matching geometry. Temporal
gaps, timestamp-domain changes, or insufficient coverage reset streaming
history. Training excludes windows with unacceptable intervals.

`configs/convlstm.json` uses ten height/mask input frames, 32/64 recurrent
channels, 3x3 kernels, and a linear 1x1 output head. Each window starts with zero
hidden states. Training uses masked MAE, Adam, batch size 8, learning rate 0.001,
gradient clipping at 1.0, seed 42, up to 50 epochs, and validation patience 8.
The nominal target is `latest_timestamp_ms + horizon * expected_dt_ms`.
Default `horizon=1` and `expected_dt_ms=33.333333333333336` predict one map about
33.3 ms ahead. Change the configuration before training for another cadence
or horizon.

Add `--cache-frames --window-stride 1` to training to cache maps in RAM and use
every eligible start position. Larger strides subsample training windows;
validation still uses stride 1. Evaluation accepts `--roi X0 Y0 X1 Y1` and
`--motion-threshold-mm` (default 0.5).

| Workflow | Files written |
| --- | --- |
| Training | `best.pt`, `history.json`, `run.json` |
| Evaluation | `metrics.json`, `report.html`, `example_*.npz` |
| Replay/live prediction | `prediction_*.npz`, `forecasts.jsonl`, `manifest.json` |

Evaluation reports MAE, RMSE, P95 absolute error, comparison coverage, and
all/moving/stationary results per session. ConvLSTM and baselines share valid
target and complete-history support. Two-frame checkpoints compare persistence
and velocity; acceleration requires at least three history frames.
The additional `convlstm_all_valid_targets` metric uses all valid targets.

## Prepare and benchmark the cloth dataset

Place the extracted dataset in `textureless_deformable_surfaces/`. The adapter
accepts either the directory containing `camera_intrinsics.txt` or its parent
containing a nested `textureless_deformable_surfaces/` directory. It reads
`cloth/depth_maps/<sequence>/depth_*.npz` with a `depth` array. Raw dataset
files are local inputs excluded from Git.

```powershell
& $python -m prediction.tds prepare `
  --raw textureless_deformable_surfaces `
  --output output/tds_cloth/prepared

& $python -m prediction.tds benchmark `
  --prepared output/tds_cloth/prepared `
  --output output/tds_cloth/benchmark `
  --epochs 40 --horizons 1 2 3 --device auto

& $python -m prediction.tds_plot `
  --benchmark output/tds_cloth/benchmark
```

Preparation produces camera-Z maps on a 32x32 metric grid with 20 mm cells.
Times are inferred from frame indices at nominal 5 FPS; windows crossing missing
indices are rejected. Whole sequences define the split:
`Lc_top_edge_a`, `Lc_top_edge_b`, and `Ll_bottom_edge` are validation;
`Lc_left_edge`, `Ld_bottom_edge`, and `Lr_top_edge_2` are test; remaining cloth
sequences are training. The consecutive top-edge segments stay together.

The benchmark trains separate models for 200/400/600 ms horizons with five
input maps, 16/16 channels, batch size 32, and training stride 5. All three
baselines use the same comparison support. Outputs include `benchmark.json`,
`SUMMARY.md`, `summary.csv`, per-horizon model/evaluation directories,
five-second temporal-block bootstrap comparisons, and central ROI scores.
Plotting adds `prediction_vs_baselines.png`, `prediction_vs_baselines.pdf`,
`training_curves.png`, and `report.html`.

Train the fuller 32/64-channel model on every eligible window at 200 ms:

```powershell
& $python -m prediction inspect `
  --splits output/tds_cloth/prepared/splits.json `
  --config configs/convlstm_tds_cloth_full.json

& $python -m prediction train `
  --splits output/tds_cloth/prepared/splits.json `
  --config configs/convlstm_tds_cloth_full.json `
  --output output/tds_cloth/cloth_full_model `
  --device auto --threads 4 --cache-frames --window-stride 1

& $python -m prediction evaluate `
  --checkpoint output/tds_cloth/cloth_full_model/best.pt `
  --splits output/tds_cloth/prepared/splits.json --split test `
  --output output/tds_cloth/cloth_full_evaluation `
  --device auto --threads 4 --motion-threshold-mm 5
```

Regenerate the existing local fuller-run comparison with:

```powershell
& $python -m prediction.tds_train_report `
  --run output/tds_cloth/convlstm_full_20261012_200ms
```

This optional report requires a completed run containing `model/run.json`,
`model/history.json`, `evaluation/metrics.json`, and `protocol.json`.
The protocol must supply `prior_experiment` (an earlier run directory with
`model/run.json` and `evaluation/metrics.json`), `frames_by_split` with
`train`/`validation`/`test` counts, and `test_reuse_note`. It verifies matching
forecast settings and comparison support, then writes `comparison.json`,
`comparison.csv`, `training_results.png`, `training_results.pdf`, and
`SUMMARY.md`. The saved experiment inputs remain local.

## CLI reference

```powershell
& $python -m surface_mapper --help
& $python -m prediction --help
& $python -m prediction train --help
& $python -m prediction evaluate --help
& $python -m prediction baseline-live --help
& $python -m prediction baseline-evaluate --help
& $python -m prediction.tds prepare --help
& $python -m prediction.tds benchmark --help
& $python -m prediction.tds_plot --help
& $python -m prediction.tds_train_report --help
```

Prediction source, configuration examples, dependencies, tests, the MATLAB
viewer, and this README are versioned. Virtual environments, recordings,
downloaded datasets, checkpoints, temporary files, and generated experiment
outputs remain local.
