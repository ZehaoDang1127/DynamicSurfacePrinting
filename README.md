# Occlusion-Aware Forecasting of Surface Motion for Adaptive 3D Printing in Dynamic Environments (Pending NAMRC 55)

Occlusion-aware surface estimation and forecasting for printing on a moving surface.
A RealSense D405 observes a deforming surface. Depth inside a fixed region of interest
(ROI) is converted to a regular height map with a validity mask. A **ConvLSTM + U-Net**
then reconstructs the current surface, including cells hidden by the print head, and
forecasts the next six frames (about 200 ms at 30 FPS).

```text
D405 RGB-D ──► ROI crop + depth gate ──► 23×45 height map + valid mask ──► ConvLSTM + U-Net ──► current map + 6 future maps
                                              ▲                                   (hidden cells filled)
                         print-head mask ─────┘  (synthetic 20 % + head-scale 40–90 % masks in training,
                                                  measured head box + depth envelope at test)
```

Status (8 Nov 2026): the offline pipeline is trained and evaluated on 30 FPS servo
recordings. The primary model is trained for 100 epochs with mixed (standard + head-scale)
synthetic masks. Height maps are camera-axis depth on a camera-XY grid; real-head results
are also re-expressed as printer height through a provisional four-dot registration.
Physical accuracy behind a real nozzle has not yet been validated (see
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
├── configs/                       JSON configs; occlusion_d405_30fps_mixed_100ep.json = primary model
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
  --config configs\occlusion_d405_30fps_mixed_100ep.json `
  --output output\servo_30fps_mixed_100ep_<date> --device cuda
```

The cloth-pretrained variant adds `--initialize-from output\cloth_pretrain_20261107\best.pt`
with `configs\occlusion_d405_30fps_mixed_finetune.json`. The first paper model used
`configs\occlusion_d405_30fps_20261105.json` (30 epochs, standard masks only). See
[the training guide](docs/occlusion_forecasting/TRAINING_100EP_AND_MIXED_MASKS.md).

Inputs: 25 frames (about 800 ms) of normalized height, validity, time since last
observation and a never-observed flag. Outputs: the current reconstruction plus 6 future
maps. During training, half the windows get moving anisotropic Gaussian masks hiding
about 20% of the grid; the other half get head-scale masks hiding 40–90%, stationary
half the time. The loss is masked MAE with an extra term (`hidden_weight` = 2) on
hidden cells. See the [pipeline guide](docs/occlusion_forecasting/OCCLUSION_CONVLSTM_UNET.md).

### 4. Evaluate with synthetic occlusion (held-out patterns)

```powershell
.\.venv-tds\Scripts\python.exe -m tools.evaluate_occlusion_heldout `
  --checkpoint output\servo_30fps_convlstm_20261105\best.pt `
  --splits output\servo_30fps_camera_maps_20261105\splits.json `
  --output output\servo_30fps_internal_synthetic_eval_<date> --device cuda
```

Add `--mask-config configs\masks\large_stationary.json` for the head-scale stress test
(70–90% of the grid hidden, stationary).

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

# Re-express the difference as printer height (needs scipy and a registration transform)
.\.venv-tds\Scripts\python.exe -m tools.rescore_real_head_printer `
  --transform output\registration_20261108\transform.json `
  --prepared output\real_occlusion_prior_repeats_<date> `
  --model mixed100=output\real_occlusion_repeat_evaluation_<date> `
  --output output\printer_frame_rescore_<date>
```

### 6. Replay on any recorded session

```powershell
.\.venv-tds\Scripts\python.exe -m prediction.occlusion_inference `
  --checkpoint output\servo_30fps_convlstm_20261105\best.pt `
  --session output\servo_30fps_camera_maps_20261105\18_stop_start `
  --output output\replay_<date> --device cuda
```

## Current results

Five models share one architecture (ConvLSTM + U-Net, 708,753 parameters) and the same
split: 16 patterns for training, 2 for validation and 2 for testing. All numbers below
come from the lab GPU, where every model saw identical evaluation masks. MAE is in mm,
current / 200 ms.

| Model | Folder in `output/` | Paper masks, supported hidden | Large masks (70–90 %, stationary) | Real head, printer height* |
| --- | --- | ---: | ---: | ---: |
| **Mixed masks, 100 ep (primary)** | `servo_30fps_mixed_100ep_20261108` | 2.48 / 3.20 | 3.28 / 3.29 | 4.02 / 3.61 |
| Mixed masks, cloth-pretrained | `servo_30fps_mixed_finetune_20261108` | 2.49 / 3.50 | 3.61 / 3.65 | 2.97 / 2.27 |
| Gaussian masks, 100 ep | `servo_30fps_convlstm_100ep_20261107` | 2.47 / 3.17 | 36.31 / 36.57 | 15.85 / 16.84 |
| Gaussian masks, cloth-pretrained | `servo_30fps_finetune_20261107` | 2.48 / 3.51 | 28.08 / 19.68 | 6.59 / 5.22 |
| Gaussian masks, 30 ep (first paper model) | `servo_30fps_convlstm_20261105` | 2.60 / 3.49 | 20.61 / 20.41 | 2.64 / 2.35 |
| Last observation (supported cells only) | – | 3.58 / 4.40 | – | – |
| Repeat recordings, no model | – | – | – | 2.92 / 2.91 |

\*The real-head numbers are differences from an unregistered repeat recording made about
45 minutes earlier, not ground-truth errors. They were re-expressed as printer height
through a provisional four-dot registration (0.29 mm RMS). The two recordings already
differ by 2.92 mm on cells visible in both, so models near that floor cannot be ranked
by this test. Models trained only on 20 % masks fail when 70–90 % of the surface is
hidden; mixing head-scale masks into training fixes this.

The full tables, including camera-depth values and offsets, are in
[`docs/occlusion_forecasting/RESULTS_100EP_MIXED_MASKS_20261108.md`](docs/occlusion_forecasting/RESULTS_100EP_MIXED_MASKS_20261108.md).
The paper's Methodology and Results sections are in
[`docs/occlusion_forecasting/METHODOLOGY_REVISED.tex`](docs/occlusion_forecasting/METHODOLOGY_REVISED.tex)
and [`RESULTS.tex`](docs/occlusion_forecasting/RESULTS.tex). The training procedure is in
[`TRAINING_100EP_AND_MIXED_MASKS.md`](docs/occlusion_forecasting/TRAINING_100EP_AND_MIXED_MASKS.md).

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
