# Recording experiments for surface prediction

The goal is to learn motion from repeated, timestamped height maps and assess
whether forecasts beat persistence and constant velocity on independent
trials. Start with the next frame, then select a longer horizon if measured
system latency requires it.

## 1. Fix the geometry and measurement procedure

Keep camera mounting, surface orientation, stream settings, grid bounds, cell
size, aggregation, and camera-to-output transform constant across a dataset.
Use `configs/d405_camera_frame.json` for a camera-coordinate prototype. Its
height is camera Z. Before interpreting outputs as printer Z, measure the
camera-to-printer rigid transform and verify it at multiple known positions.
Changing the transform or grid creates a new dataset geometry and requires
compatible training data.

Choose a print-region ROI visible throughout most of the motion. Record
occlusion and preserve invalid cells instead of treating them as zero height.
Avoid per-frame normalization that removes displacement. The training pipeline
supplies a mask and uses global training statistics.

If possible, measure imposed displacement with an independent reference and
synchronize that reference with the camera. Record reference accuracy and
synchronization uncertainty. A motion-stage command alone is not a measured
surface position, particularly for flexible material.

## 2. Record a practical pilot dataset

As an initial plan, record 30–60 seconds per trial and several independently
started repeats per condition. Choose displacement and speed ranges from the
actual intended experiment and measured field of view. This matrix describes
conditions to cover, not validated equipment limits:

| Condition | Vary across trials | What it checks |
| --- | --- | --- |
| Static surface | Height, tilt, and material | Sensor variation and persistence baseline |
| Periodic vertical motion | Small/medium/large amplitude and slow/medium/fast rate | Phase and changing velocity |
| Starts, stops, and reversals | Start time, dwell duration, and acceleration | Transients beyond steady oscillation |
| Local deformation | Location, footprint, and rate | Spatially varying motion |
| Representative printing scene | Visible fixtures and expected temporary occlusion | Coverage and real acquisition conditions |

Do not let static frames dominate every training trial. Include rest periods
so the model also learns stationary behavior, and review moving-region errors
separately. Capture repeats with new initial phases and independently restarted
motion. Keep unusual held-out speeds or deformations as an explicitly labelled
generalization test if they matter to the intended use.

For nominal 30 FPS, this requests 1,800 frames, approximately one minute of data
if the cadence is sustained:

```powershell
python -m surface_mapper --config configs/d405_camera_frame.json --frames 1800 --output output/train_trial_01 --save-depth
```

Use a new directory for each session. Record a short trial log alongside it:
condition, motion settings and measured displacement if available, material,
mounting/calibration identifier, reference clock details, operator observations,
and known interruptions. Preserve the manifest and timestamped NPZ frames;
the latest CSV alone is insufficient for training.

## 3. Reserve entire trials before training

Plan roughly 60% of independent sessions for training, 20% for validation, and
20% for testing as a pilot allocation. Include multiple repeats of relevant
conditions in each split when available. Counts depend on motion diversity;
many overlapping windows from one trial do not replace independent trials.

Populate `configs/splits.json` from the supplied example. Paths are relative to
that JSON file. Never put overlapping slices or renamed copies of the same
trial in different splits. Prefer holding out a complete acquisition run or
day when testing reproducibility. Use validation for tuning and early stopping;
look at reserved test results only after choosing the experiment.

## 4. Inspect timing and coverage

```powershell
python -m prediction inspect --splits configs/splits.json --config configs/convlstm.json
```

Requested 30 FPS is not a guarantee of recorded 30 FPS. Review timestamp
intervals, domains, frame ordering, coverage, and eligible window counts.
The default predictor expects 33.333 ms intervals within ±20% and at least 10%
valid cells per frame. These are initial data filters, not print-region
measurement-quality acceptance criteria. Assess ROI coverage separately and
tighten requirements for the intended task.

Investigate recordings with many rejected windows. Dropped frames, camera
restarts, or sparse coverage can leave little usable data even with many files.
Keep actual timestamps; do not relabel irregular samples as equally spaced.
Re-record problematic trials or deliberately design a resampling procedure
before using a different cadence.

## 5. Train and report results

Use the commands in [PREDICTION.md](../earlier_prediction/PREDICTION.md). Record the split file,
prediction config, calibration/config identifier, selected checkpoint, and
evaluation report together. Report per-trial and aggregate MAE, RMSE, and
95th-percentile errors in millimetres, coverage, and moving-region errors.
Compare ConvLSTM and both baselines on the same pixels and future horizon.
Inspect predicted, observed, and error maps during motion and at reversals,
not only stationary periods.

Define acceptable ROI error and deadline requirements from the printing
experiment before selecting a model. Require held-out improvements where they
matter and confirm results across independent trials. If ConvLSTM does not
beat the baselines, inspect timing, noise, data diversity, and overfitting
before adding model capacity.

## 6. Establish the usable prediction horizon

Measure acquisition-to-host delay and jitter, height-map projection, inference,
communication, and queued motion separately. Camera time and host monotonic
time require an established clock mapping; they cannot generally be directly
subtracted. Compare complete measured delay with `horizon * expected_dt_ms`.
Use a later trained horizon if the next-frame deadline cannot be met.

The recorded sensor target contains noise and may have missing regions. Check
physical displacement against the independent reference where available, and
keep that result separate from agreement with camera measurements. Live
prediction validates the software timing and recording path; printer motion
and its control validation are separate work.
