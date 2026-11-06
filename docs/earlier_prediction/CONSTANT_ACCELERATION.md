# Constant acceleration prediction from the D405

Start with this baseline before training a ConvLSTM. It estimates the recent
vertical velocity and acceleration independently at each fixed XY grid cell,
then predicts that cell's height at a future camera timestamp. It requires
NumPy, and `pyrealsense2` for a physical D405; it needs no training, checkpoint,
PyTorch, GPU, or manual labels.

## Run live and record your surface

Connect the D405, verify its depth stream in RealSense Viewer, then close
Viewer so Python can use the camera. Keep the camera fixed relative to the
surface setup. From PowerShell:

```powershell
Set-Location "D:\HAIM_Lab\Dynamic_Surface_Printer"
$py = ".\.venv\Scripts\python.exe"

& $py -m prediction baseline-live `
  --camera-config configs/d405_camera_frame.json `
  --source realsense `
  --history 5 `
  --frames 0 `
  --output output/ca_live_01 `
  --record-input `
  --save-depth
```

Stop with **Ctrl+C** and use a new output directory for each run. The command
records incoming height maps, masks, timestamps, and raw depth under
`output/ca_live_01/observations`, while saving forecasts alongside them. Omit
`--save-depth` if raw depth is unnecessary; it requires `--record-input`.

After five acceptable frames, the buffer can emit predictions. By default the
forecast horizon is one configured camera interval: approximately 33.3 ms at
30 FPS. The fitted motion uses the actual camera timestamps, including uneven
intervals. `--horizon-ms 66.6666666667` instead predicts approximately 66.7 ms
after the latest observation. A longer horizon increases extrapolation error;
it does not make capture faster.

The console reports the first forecast and every 30 forecasts. Live mode
saves maps and timing logs; it has no graphical live viewer and issues no
printer commands. Capture, projection, fitting, and disk writing all consume
time, so measure the full pipeline on your computer before relying on a
forecast deadline.

To check the workflow without hardware:

```powershell
& $py -m prediction baseline-live --camera-config configs/d405_camera_frame.json --source synthetic --frames 60 --output output/ca_synthetic --record-input
```

## What the model calculates

For each grid cell, fit a quadratic by least squares over the recent valid
height measurements. Express sample times in seconds relative to the newest
camera timestamp:

```text
z(s) = z_fit + v * s + 0.5 * a * s^2
predicted_z = z_fit + v * tau + 0.5 * a * tau^2
```

Here `z_fit` is the fitted height at the newest timestamp in millimetres,
`v` is velocity in mm/s, `a` is acceleration in mm/s², and `tau` is the requested
forecast horizon in seconds. With more than three samples, the fitted current
height can differ from the latest noisy observation.

For exactly three equally spaced observations and a one-interval horizon,
this reduces to:

```text
predicted_z_next = 3 * z_latest - 3 * z_previous + z_two_frames_ago
```

The default five-frame fit is intended to reduce sensitivity to raw second
differences. Acceleration estimates can still amplify depth noise, especially
over a short history. More history smooths noise but may miss rapid changes in
acceleration. Compare histories and horizons on separate recorded trials;
constant acceleration need not outperform persistence or constant velocity.

The fit describes height changes at fixed grid cells. It does not track a
material point moving sideways, and it cannot infer a newly occluded surface.
Keep the grid, camera mounting, and transform consistent. The example
`configs/d405_camera_frame.json` uses camera coordinates, so the output is
camera Z, positive away from the camera. Printer Z requires a measured
camera-to-printer transform as described in the [README](README_2026-10_full_workflows.md).

## History, missing depth, and output files

| Option | Default | Meaning |
| --- | --- | --- |
| `--history` | `5` | Recent observations to fit; at least `3` |
| `--horizon-ms` | One configured camera interval | Time after the latest observation to forecast |
| `--max-gap-ms` | `100` | Largest acceptable gap between consecutive observations |
| `--min-valid-fraction` | `0.1` | Minimum valid measurement coverage for an accepted frame |
| `--record-input` | Off | Live mode: save input observations for replay/evaluation |
| `--save-depth` | Off | Live mode: also save raw depth; requires `--record-input` |

A timestamp discontinuity, timestamp-domain change, excessive gap, or
insufficient frame coverage resets the buffer. It must refill before another
forecast. Every cell must be finite and valid in every fitted history frame;
unsupported predictions are `NaN` and have `prediction_valid = False`.

The output directory contains `prediction_*.npz`, `forecasts.jsonl`, and a
`manifest.json`. Forecast arrays include:

| Array | Meaning |
| --- | --- |
| `prediction_mm` | Predicted height at the nominal target camera timestamp |
| `velocity_mm_s` | Fitted velocity at the newest input timestamp |
| `acceleration_mm_s2` | Fitted constant acceleration |
| `prediction_valid` | All required history measurements support this fit |
| `observation_valid` | Valid cells in the latest measured input |
| `history_observed` | Cells measured at least once within the history |

These masks describe measurement support, not calibrated confidence or future
visibility. In particular, `history_observed` is less restrictive than
`prediction_valid`.

The timing log records the source and nominal target timestamps and prediction
timing. Camera time and host monotonic time may use different clocks; never
subtract them directly to estimate capture latency. Inference time alone does
not include all acquisition, disk, or eventual actuator delay.

## Replay and compare baselines

Replay the observations recorded during the live run:

```powershell
& $py -m prediction baseline-predict `
  --session output/ca_live_01/observations `
  --history 5 `
  --output output/ca_replay_01

& $py -m prediction baseline-evaluate `
  --session output/ca_live_01/observations `
  --history 5 `
  --output output/ca_eval_01
```

You can also use an existing `surface_mapper` recording, such as
`--session output/d405_surface_test`. Replay runs as fast as processing allows;
it does not recreate the original wall-clock capture rate. No dataset split
file or model checkpoint is required for either command. Use the same history,
horizon, gap, and coverage settings when comparing replay with live operation.

Evaluation compares persistence, a least-squares constant-velocity fit, and
the constant-acceleration fit. Every method is scored on the same cells: valid
throughout the history and in the matched future measurement. Coverage is
reported alongside errors so a small supported area is not mistaken for
accuracy over the entire surface.

The target is the nearest future recorded measurement to the nominal forecast
time within the allowed tolerance. The default tolerance is 20% of the
recording's configured camera interval; override it with
`--target-tolerance-ms`. Evaluation reports the actual target-time offset.
An unmatched forecast is excluded rather than compared against an arbitrary
later frame. These measurements contain camera noise; they are not an
independent measurement of the physical surface.

Limit scoring to the printing region and inspect performance while the surface
is moving, for example:

```powershell
& $py -m prediction baseline-evaluate --session output/ca_live_01/observations --history 5 --output output/ca_eval_roi --roi 20 15 80 60 --motion-threshold-mm 0.5
```

`--roi x0 y0 x1 y1` uses grid indices, with upper bounds excluded. The example
selects columns 20–79 and rows 15–59. Moving cells change by more than the
motion threshold between the latest input and observed target. This label is
used only for evaluation; no future measurements enter the prediction fit.

Once this baseline is characterized on your surface, use the
[ConvLSTM guide](PREDICTION.md) to train a model and compare its added value.
