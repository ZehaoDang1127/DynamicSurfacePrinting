# Preliminary real-head repeat-run evaluation

`Dataset/Test` contains real-head recordings of patterns 21–25. The corresponding
`Dataset/Test_groundTruth` recordings are separate runs made about 45 minutes
earlier. They are repeat references, not simultaneous ground truth. Fixed scene
features show a viewpoint shift between runs; no rigid 3D registration has been
validated. Differences combine prediction error, changed viewpoint, imperfect
surface repeatability, temporal alignment, sensor noise, and mask error. Do not
report them as physical hidden-surface accuracy.

The preparation script checks every decoded native 16-bit depth frame against
the source checksum and preserves recorded camera timestamps and dropped-frame
gaps. It uses the same 23×45 camera-XY grid and depth gate as training. Values
are camera-axis depth, not printer-frame height. For each blocked frame it pairs
the nearest reference elapsed host-receipt time within 20 ms. The timing is
relative to each software-controlled motion start, and servo positions are
commanded, not measured. The preparation reports their phase and command
differences as well as disagreement on common visible cells.

The stationary head mask starts from a manually inspected RGB bounding box,
padded by 20 pixels. It projects each cell's training-only depth envelope through
the recorded factory depth-to-color calibration; envelope extrema come from at
most 120 evenly sampled frames per **training** session, padded by 20 mm. Grid
cells with no training observations are masked conservatively. No repeat-reference
measurements are used to define the mask or model inputs. The box, envelope,
camera pose, and registration are approximations and do not constitute a
validated segmentation. Margin sensitivity and support are in `report.json`.

For these five recordings, the 20-pixel margin masks 825/1,035 grid cells
(including cells without training support), leaving approximately 15–18% of
all grid cells observed. This is a substantial distribution shift from the
synthetic training masks, which nominally hide 20% of the grid. The model and
persistence baseline are compared on common supported cells; hidden cells may
have no persistence support when the mask is stationary. Model predictions and
differences in hidden regions are only approximate repeat-run comparisons.

From the project root in PowerShell, use **new output directories**:

```powershell
.\.venv\Scripts\python.exe -m tools.prepare_real_occlusion_repeats `
  --blocked Dataset\Test --repeat-reference Dataset\Test_groundTruth `
  --training-splits output\servo_30fps_camera_maps_20261105\splits.json `
  --output output\real_occlusion_prior_repeats_new

.\.venv-tds\Scripts\python.exe -m tools.evaluate_real_occlusion_repeats `
  --prepared output\real_occlusion_prior_repeats_new `
  --checkpoint output\servo_30fps_convlstm_20261105\best.pt `
  --output output\real_occlusion_repeat_evaluation_new --device cuda

.\.venv-tds\Scripts\python.exe -m tools.visualize_real_occlusion_repeat `
  --prepared output\real_occlusion_prior_repeats_new\21_amplitude_beats.npz `
  --predictions output\real_occlusion_repeat_evaluation_new\21_amplitude_beats.npz `
  --output output\real_occlusion_repeat_evaluation_new\example.png
```

The first step uses the OpenCV-enabled environment; the second uses the CUDA
environment. Evaluation checks checkpoint frame, stream, grid, transform, and
saved training/test depth calibration. It reports current reconstruction and a
six-frame (nominally 200 ms) forecast. Future scores require continuous timestamps
through the future frame and use the reference repeat paired to that frame; no
future observations enter the model. Output includes model estimates and
matched-support persistence metrics, with explicit reference and input counts.
The visualization is a qualitative current comparison in the unregistered
camera grid.
