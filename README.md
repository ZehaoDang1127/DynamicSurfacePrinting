# Dynamic Surface Printer

Occlusion-aware surface estimation and forecasting for printing on a moving surface.
A RealSense D405 observes a deforming surface. Depth inside a fixed region of interest
(ROI) is converted to a regular height map with a validity mask. A **ConvLSTM + U-Net**
then reconstructs the current surface, including cells hidden by the print head, and
forecasts the next six frames (about 200 ms at 30 FPS).

```text
D405 RGB-D ──► ROI crop + depth gate ──► 23×45 height map + valid mask ──► ConvLSTM + U-Net ──► current map + 6 future maps
                                              ▲                                   (hidden cells filled)
                         print-head mask ─────┘  (synthetic Gaussian masks in training,
                                                  measured head box + depth envelope at test)
```

Status (6 Nov 2026): the offline pipeline is trained and evaluated on 30 FPS servo
recordings. Height values are camera-axis depth in a camera-XY grid, not printer-frame
height. Physical accuracy behind a real nozzle has not yet been validated (see
[Current results](#current-results)).

## Repository layout

```text
Dynamic_Surface_Printer/
├── Dataset/                       Raw D405 recordings (read-only)
│   ├── recordings/servo_patterns_20_30fps_20261105/   20 clean servo patterns, 30 FPS → train/val/test
│   ├── Test/                      Patterns 21–25 with the print head in view (real occlusion)
│   └── Test_groundTruth/          Same patterns recorded ~45 min earlier, head out of view (repeat reference)
├── acquisition/servo_d405/        Recorder and servo-pattern generator used to capture Dataset/
├── surface_mapper/                Depth → gridded height map with validity mask; session recording
├── prediction/                    Models and training
│   ├── occlusion_model.py         U-Net encoder/decoder + ConvLSTM dynamics (current + multi-step output)
│   ├── occlusion_data.py          Windows, normalisation, synthetic occlusion masks
│   ├── occlusion.py               CLI: train / evaluate / inspect
│   ├── occlusion_inference.py     Causal replay of a recorded session
│   ├── plot_occlusion.py          Held-out visualisation
│   └── model.py, baselines.py, tds*.py, …   earlier single-horizon ConvLSTM, kinematic baselines, cloth benchmark
├── tools/                         Pipeline scripts (run with python -m tools.<name>)
│   ├── convert_servo_30fps.py     ROI crop + depth gate → height maps + recording-level splits
│   ├── evaluate_occlusion_heldout.py      Synthetic-mask evaluation on the held-out split
│   ├── prepare_real_occlusion_repeats.py  Real head mask + time pairing of Test vs Test_groundTruth
│   ├── evaluate_real_occlusion_repeats.py, visualize_real_occlusion_repeat.py
│   ├── estimate_pair_registration.py      Exploratory 3-D registration of a repeat pair
│   ├── head_mask/                 RGB head-silhouette analysis and synthetic blocking (2 Nov pilot)
│   └── matlab/, build_pi_bundle.py, convert_servo_depth.py, …
├── configs/                       JSON configs; occlusion_d405_30fps_20261105.json = trained model
├── output/                        Main-line results, one folder per pipeline stage → output/README.md
├── docs/                          Method, protocols and subsystem guides → docs/README.md
├── tests/                         unittest suite
├── controller/  CameraCalibration/  pi_support/   Printer control, camera–printer calibration, Pi launcher
├── archive/                       Earlier and side experiments → archive/README.md
└── requirements*.txt  setup_pi.sh  run_pi.sh
```

## Environments

Two local environments are used (Windows, PowerShell, run from the repository root):

| Environment | Use | Install |
| --- | --- | --- |
| `.venv` | Conversion and real-head preparation (OpenCV) | `pip install -r requirements-calibration.txt -r requirements-prediction.txt` |
| `.venv-tds` | Training/evaluation on CUDA (PyTorch cu128) | `pip install -r requirements-tds-benchmark.txt` + CUDA torch (below) |

```powershell
& .\.venv-tds\Scripts\python.exe -m pip install --upgrade "torch>=2.6,<3" --index-url https://download.pytorch.org/whl/cu128
```

## Pipeline

Use a **new, empty** output folder for every run. Replace `<date>` with the run date.

### 1. Record (Raspberry Pi + D405 + goBILDA servo)

`acquisition/servo_d405/record_servo_d405.py` records lossless FFV1 RGB/depth, camera
timestamps and commanded servo positions. Follow the [recording protocol](docs/occlusion_forecasting/RECORDING_PROTOCOL.md)
and [acquisition README](acquisition/servo_d405/README.md). Store finished collections in
`Dataset/`.

### 2. ROI crop → height maps

```powershell
.\.venv\Scripts\python.exe -m tools.convert_servo_30fps `
  --source Dataset\recordings\servo_patterns_20_30fps_20261105 `
  --output output\servo_30fps_camera_maps_<date>
```

The script crops native depth pixels x 245–420 and y 205–315 (the ROI), keeps depth in
230–450 mm, and bins it into a 23 × 45 grid of 4 mm camera-XY cells. It checks every
decoded frame against the recorded SHA-256, and holds out whole patterns
(validation: 15, 19; test: 18, 20). The ROI and depth gate are specific to this scene and
are not a verified segmentation.

### 3. Train the ConvLSTM + U-Net

```powershell
.\.venv-tds\Scripts\python.exe -m prediction.occlusion train `
  --splits output\servo_30fps_camera_maps_20261105\splits.json `
  --config configs\occlusion_d405_30fps_20261105.json `
  --output output\servo_30fps_convlstm_<date> --device cuda
```

Inputs: 25 frames (about 800 ms) of normalized height, validity, time since last
observation and a never-observed flag. Outputs: the current reconstruction plus 6 future
maps. During training, moving anisotropic Gaussian masks hide about 20% of the grid in
the input history. The loss is masked MAE with an extra term (`hidden_weight` = 2) on
hidden cells. See the [pipeline guide](docs/occlusion_forecasting/OCCLUSION_CONVLSTM_UNET.md).

### 4. Evaluate with synthetic occlusion (held-out patterns)

```powershell
.\.venv-tds\Scripts\python.exe -m tools.evaluate_occlusion_heldout `
  --checkpoint output\servo_30fps_convlstm_20261105\best.pt `
  --splits output\servo_30fps_camera_maps_20261105\splits.json `
  --output output\servo_30fps_internal_synthetic_eval_<date> --device cuda
```

### 5. Evaluate with the real print head

The head mask is a hand-inspected RGB box padded by 20 px. It is projected through a
depth envelope taken only from the training data and the factory depth-to-color
calibration. See [real-head evaluation](docs/occlusion_forecasting/REAL_OCCLUSION_REPEAT_EVALUATION.md).

```powershell
.\.venv\Scripts\python.exe -m tools.prepare_real_occlusion_repeats `
  --blocked Dataset\Test --repeat-reference Dataset\Test_groundTruth `
  --training-splits output\servo_30fps_camera_maps_20261105\splits.json `
  --output output\real_occlusion_prior_repeats_<date>

.\.venv-tds\Scripts\python.exe -m tools.evaluate_real_occlusion_repeats `
  --prepared output\real_occlusion_prior_repeats_<date> `
  --checkpoint output\servo_30fps_convlstm_20261105\best.pt `
  --output output\real_occlusion_repeat_evaluation_<date> --device cuda

.\.venv-tds\Scripts\python.exe -m tools.visualize_real_occlusion_repeat `
  --prepared output\real_occlusion_prior_repeats_<date>\21_amplitude_beats.npz `
  --predictions output\real_occlusion_repeat_evaluation_<date>\21_amplitude_beats.npz `
  --output output\real_occlusion_repeat_evaluation_<date>\example.png
```

### 6. Replay on any recorded session

```powershell
.\.venv-tds\Scripts\python.exe -m prediction.occlusion_inference `
  --checkpoint output\servo_30fps_convlstm_20261105\best.pt `
  --session output\servo_30fps_camera_maps_20261105\18_stop_start `
  --output output\replay_<date> --device cuda
```

## Current results

Model: `output/servo_30fps_convlstm_20261105/best.pt` (epoch 26 of 30). It was trained on
16 patterns, validated on 2 and tested on 2.

| Test | Region | Horizon | Model MAE | Persistence MAE (same support) |
| --- | --- | --- | ---: | ---: |
| Synthetic masks, held-out patterns 18 & 20 | hidden cells | current | 2.59 mm | 3.60 mm |
| Synthetic masks, held-out patterns 18 & 20 | hidden cells | 200 ms | 3.47 mm | 4.41 mm |
| Real head, patterns 21–25 vs. separate repeat run | head-mask region | current | 5.79 mm* | n/a |
| Real head, patterns 21–25 vs. separate repeat run | head-mask region | 200 ms | 5.21 mm* | n/a |

\*These are differences from an unregistered repeat recording made about 45 minutes
earlier, not ground-truth errors. On cells visible in both runs, the two recordings
already differ by 5.49 mm MAE. Details are in
[`output/servo_30fps_convlstm_20261105/RESULTS.md`](output/servo_30fps_convlstm_20261105/RESULTS.md)
and [`output/real_occlusion_repeat_evaluation_20261106/TEST_RESULTS.md`](output/real_occlusion_repeat_evaluation_20261106/TEST_RESULTS.md).

**Open issues:** validated camera–printer registration; synchronized independent
reference behind the head; a measured per-frame nozzle mask instead of a padded box;
real-time 30 Hz inference.

## Tests

```powershell
.\.venv-tds\Scripts\python.exe -m unittest discover -s tests -v
```

## Other parts of the project

- Printer control (SV08 / Klipper / Moonraker), stationary print trials:
  [docs/printer_control](docs/printer_control/) and the `controller/` package.
- Camera–printer calibration: [docs/geometry_calibration](docs/geometry_calibration/) and `CameraCalibration/`.
- Raspberry Pi deployment: [docs/raspberry_pi](docs/raspberry_pi/), `setup_pi.sh`, `run_pi.sh`.
- Earlier prediction work (single-horizon ConvLSTM, constant-acceleration baselines,
  cloth benchmark, transfer learning): [docs/earlier_prediction](docs/earlier_prediction/). The full
  pre-reorganization README is kept there as `README_2026-10_full_workflows.md`.
- Original September project outline: [docs/project](docs/project/).

Raw recordings, checkpoints, `output/` and `archive/` stay local and are excluded from Git.
