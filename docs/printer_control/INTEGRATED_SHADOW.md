# Live camera, prediction and planner integration

`controller shadow` runs raw D405 depth through the configured rigid transform,
height-map projection, causal kinematic predictor and arrival-time candidate
planner in one live process. The printer can remain disconnected.

```powershell
& ./.venv/Scripts/python.exe -m controller shadow `
  --config output/integrated_setup_20261025/controller.json `
  --camera-config output/integrated_setup_20261025/camera.json `
  --path output/integrated_setup_20261025/path.json `
  --frames 300 --output output/integration_check_01
```

The measured local setup also supplies a timestamped launcher:

```powershell
& ./output/integrated_setup_20261025/Run-Integration.ps1 -Frames 300
```

These measurement/configuration artifacts are local, ignored output files.
The generic `controller shadow` command is project code. On a Pi, copy the local
configuration JSON files and use the same Python command with its environment.
The PowerShell launcher selects the Windows project `.venv` and is for this laptop.

Stop other D405 streaming applications before running. The camera must retain its
calibrated pose. The configuration pins the physical camera serial and checks
the stream, transform, frame name and grid against the controller geometry hash.
Use `height_reference: "printer_z"` for maps already transformed into absolute
printer coordinates; no substrate datum is then added a second time.

Each proposal starts independently at the first path point, with a hypothetical
nozzle at the observed surface plus `desired_gap_mm`. It plans up to
`proposal_moves` waypoints, then evaluates a new independent proposal on the next
frame. This does not execute or simulate completion of the full path. No printer
transport is created, no G-code is sent, and there is no `--execute` option.

The current ConvLSTM adapter produces one fixed horizon. It cannot supply all
arrival times requested by this planner, so shadow planning explicitly requires
persistence, constant velocity or constant acceleration. A trained model also
needs the new geometry before any later adapter work.

The output contains observations, events, camera/controller configs and
`report.json`. The latest forecast is saved when prediction succeeds. An accepted
proposal additionally produces `candidate_preview.json`, containing XYZ, feed
and timing estimates with zero extrusion. Acquisition and logging losses are
reported. Missing support, excessive slope, staleness and horizon violations
are recorded as rejections, never silently filled or bypassed.

`READY` means at least one unexecuted candidate was produced. `BLOCKED` means no
candidate passed, with reasons in the report. `FAULT` identifies capture, clock,
configuration or logging failures. None of these states qualifies real motion.
The receipt-anchored camera clock has a provisional 0.1 s uncertainty; exposure,
USB and printer scheduling delays remain uncalibrated.

## 25 October 2026 local commissioning findings

The original four-dot color PnP fit had 0.107 px reprojection RMS, but actual
depth points transformed with its composed matrix were about 11 mm too high
and 12–14 mm offset in Y at the markers. Do not use that original color-derived
depth matrix for hardware positioning. The user confirmed both camera and
surface were unchanged; a fresh image showed dot centers shifted by at most
0.37 px. Factory and custom color intrinsics differ, but the exact cause of
the metric discrepancy has not been isolated.

A separate diagnostic fit uses the same measured printer XYZ and paired metric
depth observations from all four markers. It reconstructs each 3D point using
the SDK color ray and median depth samples within one image pixel, then fits
one rigid transform without scale. In-sample RMS is 0.816 mm, maximum 1.140 mm;
leave-one-out errors reach 2.385 mm. This has not been independently validated.
The original registration and first integration profiles are preserved.

The provisional depth-fit profile improves median observed grid coverage from
39.0% to 98.7%. On its 90-frame live run, it produced 86 forecasts but **zero
accepted proposals**: 74 slope rejections, 6 missing-support rejections and
6 horizon rejections. Prediction and planning took 0.80 ms median / 1.25 ms P95,
excluding exposure, USB and height-map acquisition/projection. No frames or logs
were dropped, and no printer commands were sent.

An estimator comparison on the exact same saved recording produced 63 accepted
persistence proposals, versus none for velocity or acceleration. The stationary
startup profile therefore uses persistence; `controller.velocity.json` preserves
the velocity profile for comparison. Persistence holds the latest height map
constant and does not establish moving-surface prediction capability.

The final 90-frame live stationary run produced 86 forecasts and **47 accepted
unexecuted candidates**. Median depth coverage was 98.5%; 39 candidates were
rejected for excessive slope or missing support. Prediction/planning took
1.33 ms median / 1.84 ms P95. Acquisition and logging drops were zero; printer
commands sent were zero. Result: `shadow_20261025_163533_868/report.json` in the
local setup folder.

The diagnostic profile uses a hypothetical 5 mm gap and a provisional 2 mm gap
error tolerance instead of the original example's 0.5 mm. This is a coarse
diagnostic setting, not a verified printing tolerance. Its calibration field is
1.2 mm, rounding up the maximum in-sample residual; the leave-one-out error is
larger and independent accuracy remains unknown. Numerical uncertainty fields
remain unqualified and must not be treated as measured safety bounds.

Next, measure an extra independently known point around X180/Y83 with its actual
Z, and compare against the depth transform. Characterize stationary depth noise
and forecast errors before choosing spatial aggregation, estimator and planning
limits. Then reconnect the printer for read-only discovery, measured elevated
cold positioning and schedule/gap characterization. Continuous moving-surface
hardware following and hot deposition still require implementation and validation.
