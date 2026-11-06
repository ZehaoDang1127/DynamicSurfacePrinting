# ConvLSTM height-map prediction

To start predicting without training, use the
[constant-acceleration workflow](CONSTANT_ACCELERATION.md). It runs live from
the D405, optionally records its inputs, and compares acceleration against
persistence and constant velocity with NumPy alone. The remainder of this
guide covers the trained ConvLSTM workflow.

The predictor learns `recent height maps -> one future height map` from the
timestamped sessions written by `surface_mapper`. No manual frame labels are
needed: the future recorded measurement supplies the target. Predictions use
the recording's coordinate frame and millimetres. They are sensor-based
estimates, not independently measured surface truth.

## Run the complete workflow

From the repository root, install `requirements-prediction.txt`. PyTorch is an
optional dependency for ConvLSTM; acquisition and constant-acceleration
prediction remain usable without it.
Use `--device cpu` for CPU-only operation or `--device auto` for automatic
device selection.

```powershell
python -m pip install -r requirements-prediction.txt
python -m prediction demo --output output/prediction_demo --epochs 2 --device cpu
```

The demo generates varied, independent synthetic sessions on a small grid,
trains, evaluates against baselines, and replays predictions. It verifies the
workflow with a short training run; its scores do not establish performance on
physical surfaces. Use a new or empty output directory for each run.

For real data, follow [the recording protocol](../occlusion_forecasting/RECORDING_PROTOCOL.md), copy
`configs/splits.example.json` to `configs/splits.json`, and replace the example
paths with your recording directories. Relative session paths resolve from the
split file's directory. The keys are `train`, `validation`, and `test`; keep
each recording in exactly one split. Copying a recording under another name
does not make it independent.

```powershell
python -m prediction inspect --splits configs/splits.json --config configs/convlstm.json
python -m prediction train --splits configs/splits.json --config configs/convlstm.json --output output/model --device auto
python -m prediction evaluate --checkpoint output/model/best.pt --splits configs/splits.json --split test --output output/evaluation
python -m prediction predict --checkpoint output/model/best.pt --session output/test_trial_01 --output output/predictions
```

Inspection validates session geometry and reports session/frame counts,
eligible windows, and windows rejected for timing or coverage. Training computes normalization from training data
only, uses masked mean absolute error, clips gradients, and selects the best
checkpoint by validation loss. `best.pt` stores model weights, prediction
configuration, normalization, and geometry for consistent inference. Keep the
test set out of model and hyperparameter selection.

## Input preparation and timing

Each recording provides `manifest.json` and `frame_*.npz` files. The predictor
uses `height_mm`, `valid`, and camera timestamps; raw `depth_m` is not required.
All sessions must share the same grid, transform, and coordinate convention.
For the example 75-by-100 grid, default tensor shapes are:

```text
inputs:       [batch, 10, 2, 75, 100]
target:       [batch,     1, 75, 100]
target_valid: [batch,     1, 75, 100]
```

The two input channels are normalized height and validity. The height channel
is `(height_mm - training_mean_mm) / training_scale_mm`, with invalid cells
filled with zero after normalization. The mask distinguishes this placeholder
from an observed value. Normalization is global, not recomputed per frame, so
vertical displacement remains visible. Targets are scored only where the
recorded mask is valid and the height is finite.

The default window is frames 0 through 9 as input and frame 10 as target.
`horizon = 3` instead selects frame 12 as the single target; it does not produce
three intermediate maps. Change the horizon before training and train a model
for that horizon. The nominal forecast time is:

```text
target_camera_timestamp_ms = latest_camera_timestamp_ms
                            + horizon * expected_dt_ms
```

The default cadence is 33.333 ms with a relative interval tolerance of 0.2
(approximately 26.7 to 40.0 ms). Windows crossing unacceptable gaps or containing
insufficient valid depth are rejected. Changing camera FPS requires a matching
prediction config. Allowed jitter means the observed target can differ slightly
from the nominal forecast time; evaluation reports that offset.

## Architecture and equations

Two ConvLSTM layers have 32 and 64 hidden channels, 3-by-3 convolution kernels,
and padding that preserves height and width. Each timestep updates both layers.
A linear 1-by-1 convolution maps the final hidden state to one normalized height
channel. Each independent window starts from zero hidden and cell states,
during both training and inference.

For one layer, concatenate input and previous hidden state along channels.
One convolution calculates four groups of gate activations:

```text
[a_i, a_f, a_o, a_g] = Conv2d(concat(X_t, H_(t-1)))
i_t = sigmoid(a_i)                 # amount of new information written
f_t = sigmoid(a_f)                 # amount of existing memory retained
g_t = tanh(a_g)                    # candidate memory update
o_t = sigmoid(a_o)                 # amount of memory exposed
C_t = f_t * C_(t-1) + i_t * g_t     # * is elementwise multiplication
H_t = o_t * tanh(C_t)
prediction = Conv1x1(H_last_layer_at_final_input)
```

This is a simplified adaptation of
[Shi et al. (2015), Sections 3.1–3.2](https://papers.nips.cc/paper/5955-convolutional-lstm-network-a-machine-learning-approach-for-precipitation-nowcasting.pdf).
The paper's Equation 3 includes cell-to-gate peephole terms, which this
implementation omits. Its encoding/forecasting network generates a sequence;
this implementation uses a direct output head for one selected future map.

## Configuration

`configs/convlstm.json` contains the training and sequence settings:

| Setting | Default | Meaning |
| --- | --- | --- |
| `history` | 10 | Input frames; at least 2; acceleration comparison requires 3 |
| `horizon` | 1 | Frame steps from last input to target |
| `expected_dt_ms` | 33.333333333333336 | Expected camera interval |
| `timing_tolerance` | 0.2 | Allowed relative interval deviation |
| `min_valid_fraction` | 0.1 | Minimum measurement coverage per frame |
| `hidden_channels` | `[32, 64]` | Hidden channels per recurrent layer |
| `kernel_size` | 3 | Odd convolution kernel width and height |
| `batch_size` | 8 | Training batch size; reduce to fit memory |
| `learning_rate` | 0.001 | Adam learning rate |
| `epochs` | 50 | Maximum training epochs |
| `patience` | 8 | Early-stopping patience in epochs |
| `grad_clip` | 1.0 | Gradient norm limit |
| `seed` | 42 | Random seed |

These are starting settings. Use validation recordings to adjust history,
capacity, or learning rate. Longer history may help slower dynamics but
increases memory and computation. Repeat experiments with independent trials;
a seed alone does not establish repeatability across devices or datasets.

## Evaluate the forecast

Evaluation writes `metrics.json`, `example_*.npz` maps, and a standalone
`report.html`. Open the HTML report in a browser to compare measured future,
predicted, and signed-error maps using its example slider; measured and
predicted maps share a height scale. It reports MAE, RMSE, and 95th-percentile
absolute error in millimetres, overall and per session, for all, moving, and
stationary cells. MAE and RMSE use all scored errors. The percentile uses at
most 100,000 sampled errors per method/region for large evaluations and is
marked as sampled in the JSON report. Moving cells have an
observed target change above `--motion-threshold-mm` (default 0.5 mm) relative
to the latest input. This is an evaluation label, not a future model input.

```powershell
python -m prediction evaluate --checkpoint output/model/best.pt --splits configs/splits.json --split test --output output/evaluation_roi --roi 20 15 80 60 --motion-threshold-mm 0.5
```

ROI arguments are grid indices `x0 y0 x1 y1`, with upper bounds excluded, not
physical coordinates. The example selects columns 20–79 and rows 15–59.
All methods share the reported common support: valid target and every history
input, restricted to the ROI if requested.

The baselines are persistence (copy the latest map), constant velocity, and
constant acceleration. The motion fits use actual observation timestamps and
the nominal forecast horizon. Constant acceleration fits a quadratic over the
input history and is included when the checkpoint uses at least three history
frames; two-frame checkpoints still work and omit that comparison. See the
[baseline guide](CONSTANT_ACCELERATION.md) for the acceleration equations and
its sensitivity to depth noise. Coverage is reported because a small
well-observed subset can hide poor coverage. The additional `convlstm_all_valid_targets`
metric scores the model on all valid targets and has different support.

Check moving-region scores and example error maps before deciding that the
model predicts motion. It may lose to a simple baseline, especially after a
short run or with insufficient varied recordings. Targets still contain camera
noise; these metrics are not independent measurements of physical accuracy.

## Replay and live inference

`predict` reads a saved session through the same causal history buffer used for
live capture. Run live prediction after training on matching geometry:

```powershell
python -m prediction live --checkpoint output/model/best.pt --camera-config configs/d405_camera_frame.json --source realsense --frames 300 --output output/live
```

Use `--source synthetic` to exercise the live path without a camera, with a
checkpoint whose geometry matches the selected camera configuration. The
small-grid demo checkpoint requires its matching demo geometry; it cannot be
used with the default 75-by-100 grid. After running the demo:

```powershell
python -m prediction live --checkpoint output/prediction_demo/model/best.pt --camera-config output/prediction_demo/camera.json --source synthetic --frames 300 --output output/demo_live --device cpu
```

Replay and live prediction write `prediction_*.npz`, a `forecasts.jsonl` timing
log, and a `manifest.json` with forecast and buffer-reset counts. The arrays
include `prediction_mm`, `observation_valid`, and `history_observed`. Replay
runs as fast as processing allows rather than following wall-clock timing.

The buffer needs `history` acceptable frames before emitting a forecast. Gaps,
timestamp-domain changes, and insufficient coverage reset the buffer.
Predictions are dense estimates; `observation_valid` describes the latest
measured input and `history_observed` describes support anywhere in the input
history. Neither mask measures future confidence or proves accuracy in
occluded cells.

Forecasts carry source and nominal target camera timestamps and inference
duration. This duration measures prediction work, not complete sensor-to-
actuator delay. Camera time and host monotonic time can belong to different
clocks: do not subtract them directly to estimate capture latency. Map the
clocks and measure acquisition, processing, communication, and queued motion
delays before choosing a printer-control horizon. A 33.3 ms forecast is useful
for a deadline only if the complete system supplies it before that deadline.
This project issues no printer commands.
