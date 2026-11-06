# Experimental occlusion-aware ConvLSTM + U-Net

This offline pipeline trains a current surface estimator and an open-loop,
multi-step predictor from **clean**, timestamped height-map sessions. It is a
first implementation of the observation and forecasting problem in the
[research plan](OCCLUSION_AWARE_FORECASTING_PLAN.md). Its synthetic masks are
grid-space approximations, not measured tool silhouettes. No trained weights or
physical performance claims are supplied.

## Prepare data

Record several independent sessions with the nozzle out of view, following the
[recording protocol](RECORDING_PROTOCOL.md). Use a consistent grid, coordinate
transform, and cadence. Each finished session needs the recorder's
`manifest.json` and `frame_*.npz` files containing `height_mm`, `valid`, and
timestamps. The valid mask must identify the target surface. Camera-frame Z is
not printer-frame height; obtain a measured transform for printer-frame use.

Split whole sessions before training, as in `configs/splits.example.json`.
Provide nonempty `train`, `validation`, and `test` lists, with paths relative to
the split JSON. Do not copy or divide a recording across splits. The loader
checks geometry and duplicate paths and frame identities, but cannot detect all
duplicated recordings or correlated physical trials. Collect different motions
and conditions for meaningful validation and keep test sessions held out.

Training masks copies of clean input histories with a moving rectangle by default; the
original valid measurements remain labels. Natural sensor holes do not become
labels. Real occluded recordings alone cannot supervise hidden geometry: the
nozzle can return valid depth and the current recorder does not remove it. An
independent, synchronized reference is needed to evaluate those hidden cells.

## Train, evaluate, and replay

### D405 servo recordings (5 FPS, retired)

The first 5 FPS servo collection (`DataCollection/`, 3 November 2026) and its
Gaussian-mask model were deleted on 6 November 2026 after the 30 FPS collection in
`Dataset/recordings/servo_patterns_20_30fps_20261105` replaced them. `tools/convert_servo_depth.py`
and `configs/occlusion_servo_gaussian_5fps.json` remain for PNG-format recordings.

### Training with genuine 30 FPS recordings (current)

`configs/occlusion_gaussian_30fps.json` provides 25 input frames spanning
approximately 800 ms, followed by six predicted frames through approximately
200 ms at nominal 33.33 ms intervals. Gaussian masks move every six frames and
training windows use stride 6; validation and test use stride 1. At the same
grid spacing this preserves the previous 5 FPS mask movement in cells per
second. Prepare clean, independent 30 FPS height-map sessions and update a copy
of the example split file before training:

```powershell
Copy-Item configs/splits_d405.example.json configs/splits_d405_30fps.json
# Edit the copied split file to point to completed 30 FPS recordings.
python -m prediction.occlusion train --splits configs/splits_d405_30fps.json --config configs/occlusion_gaussian_30fps.json --output output/occlusion_30fps --device auto
python -m prediction.occlusion evaluate --checkpoint output/occlusion_30fps/best.pt --splits configs/splits_d405_30fps.json --split test --output output/occlusion_30fps_eval --device auto
```

These settings require retraining on genuine 30 FPS recordings; changing a
configuration cannot make the existing 5 FPS checkpoint a 30 FPS predictor.
For a new 20-pattern raw D405 dataset in the same format, use
`python -m tools.convert_servo_depth --source <raw-dataset> --output <new-output>`.
The converter retains the recorded stream's nominal FPS and timestamps, checks
for consistent stream metadata, and creates a recording-level split. Its
scene-specific crop must be reviewed for the new setup. Timestamp validation
still rejects dropped or irregular frames; a nominal stream setting does not
establish that every frame was captured at 30 FPS. This offline setup does not
establish that acquisition and inference meet a 30 Hz real-time deadline.

From the repository root in PowerShell:

```powershell
python -m pip install -r requirements-prediction.txt
Copy-Item configs/splits.example.json configs/my_splits.json
Copy-Item configs/occlusion_convlstm_unet.example.json configs/my_occlusion.json
# Edit the split paths and settings to match the recordings.
python -m prediction.occlusion train --splits configs/my_splits.json --config configs/my_occlusion.json --output output/occlusion_model --device auto
python -m prediction.occlusion evaluate --checkpoint output/occlusion_model/best.pt --splits configs/my_splits.json --split test --output output/occlusion_evaluation --device auto
python -m prediction.occlusion_inference --checkpoint output/occlusion_model/best.pt --session output/test_trial_01 --output output/occlusion_replay --device auto
```

Use new or empty output directories. The split file must point to completed
sessions before training can start. Change `expected_dt_ms` to match acquisition;
windows crossing gaps outside `timing_tolerance` or insufficient clean coverage
are rejected. `history` is the number of observed input frames; `horizon` is the
number of successive future maps. The first model output is the current
reconstruction, followed by one map per future step. Lead times are nominal;
actual sample times can differ within the allowed tolerance. Normalization uses
only valid training-session measurements. The checkpoint includes the model,
normalization, configuration, geometry, and split provenance. `run.json` and
`history.json` record training; `best.pt` is selected by validation objective.

For larger datasets, set `cache_frames` to `true` to keep frame arrays in RAM
after their first read, and set `train_window_stride` to a positive integer to
reduce overlapping training windows. Validation and evaluation always use
stride 1. Caching can require substantial RAM for large recordings.

`metrics.json` reports MAE, RMSE, counts, and coverage for each lead, separately
for cells visible or hidden at the forecast origin. Last-valid-height persistence
is scored where any observation exists in the input history. The model is also
scored on that same support for a fair comparison; never-seen baseline cells are
reported and have no invented height. The replay command uses only past frames
and does not require future labels. Its NPZ files contain `prediction_mm` with
shape `[horizon + 1, height, width]`, observation masks, and nominal timestamps
in the recording's clock domain. These masks do not represent model confidence.
Replay assumes the recording's valid mask already excludes any toolhead points.

## Model and limitations

Input channels are normalized observed height, observation validity, clipped
time since last observation, and a never-observed flag. Age is computed within
each history window. Missing heights are finite placeholders, not measurements.
A U-Net encoder supplies spatial features to ConvLSTM dynamics and correction;
the decoder uses skip features from the latest history frame with observations.
If a frame is entirely unobserved, the model propagates without correction.
During forecasting the latent dynamics advance without future observations,
while the last available skip features remain fixed. This is an architectural
assumption to test, not a guarantee of accuracy under long occlusions.

The loss combines independently normalized current and future valid-pixel MAE
with an optional extra loss on clean-valid cells hidden at the origin;
`hidden_weight` controls that extra term. Masks are deterministic for a given
seed, session path, and window, including across epochs. Set `occlusion_shape`
to `gaussian` to instead use a binary contour of an anisotropic Gaussian; its
orientation and aspect ratio vary between windows, and its center moves smoothly
through each history. The brightest Gaussian cells are hidden; measured height
values are not blurred or replaced by Gaussian intensities. For either shape,
the masked area is approximately `occlusion_fraction` of the grid;
use different held-out sessions and seeds, and later add measured trajectories,
silhouettes, and sensor artifacts before drawing conclusions about the rig.

`gaussian_step_interval` controls how many input frames elapse per synthetic
Gaussian center movement and defaults to 1 for existing checkpoints. For 30 FPS,
set it to 6 to approximate the grid-cell speed of an interval of 1 at 5 FPS,
assuming the same grid spacing. This setting affects Gaussian masks only.

This implementation does not segment the nozzle, quantify uncertainty, query
future toolpaths, model interventions, or control the printer. Its checkpoint
format and offline replay are separate from the existing single-horizon
ConvLSTM and live controller. Validate physical error, timing, occlusion masks,
and the full closed loop before using predictions to make motion decisions.
