# First stationary extrusion line

`python -m controller.stationary` generates a fixed, explicitly positioned line
and can upload/start it through Moonraker. It uses an operator-selected absolute
nozzle Z. The camera registration and forecast are not used to correct this job.
The dynamic `python -m controller print` command retains its existing restrictions.

## Current local trial

The user selected nozzle **Z150 mm** and skipped the independent fifth-point check.
The local package is `output/stationary_print_20261025/pi_stationary_print_bundle.zip`.
It includes updated source, a printer configuration and a concrete G-code preview.

| Setting | Value |
|---|---|
| Required initial printer XYZ | X190 Y90 Z170 mm |
| Line | X170 to X190, Y83, Z150 mm |
| Travel Z | 170 mm |
| PLA nozzle setpoint | 220°C, retained from the earlier trial |
| XY speed / Z speed | 5 mm/s / 1 mm/s |
| Filament diameter / assumed bead area | 1.75 mm / 0.08 mm² |
| Calculated input filament | approximately 0.665203 mm |
| Final position | X190 Y90 Z170, nozzle heating off |

The start is a required position, not a claim about the printer's current pose.
Z150 is the selected nozzle coordinate, not a measured surface height or gap.
The flow/adhesion recipe is experimental; the earlier unsuccessful line is not
treated as a verified material profile. Bed/fan settings are left unchanged.

The generated G-code saves parsing state, sets millimetres/absolute XYZ/relative
E, lifts to travel Z, moves above the line start, waits for nozzle temperature,
descends vertically, prints one continuous line, turns nozzle heating off, lifts
and returns at travel Z. It restores parsing state without a restore move.
It contains no homing, probing, forced-coordinate commands or start/end macros.
Confirm the complete toolhead approach/line/return is clear and the surface is
stationary before using the file. Establish a trustworthy coordinate reference
with the fixture clear if that reference has been lost.

## Run on the Pi

Extract the package into a fresh directory and activate the existing Pi Python
environment. Install the controller dependencies if necessary:

```bash
python -m pip install -r requirements-controller.txt
```

The configuration uses `http://100.65.32.176:7125`. This address is reachable from
the user's Pi but timed out from the development laptop; no deployment or printer
execution was performed from the laptop. If the Pi uses a web proxy on port 80,
set the actual Moonraker URL in `controller.json` before continuing.

Read status without motion or heating:

```bash
python -m controller.stationary status \
  --config output/stationary_print_20261025/controller.json \
  --output output/printer_status_01
```

Review the reported XYZ, homing, offsets, heaters and mesh. The execution preflight
uses the existing controlled-setup convention: homed/idle, cold heaters off,
matching machine/G-code coordinates with zero offsets and inactive mesh/thermal
compensation, configured limits within installed limits, and the required initial
position. It never clears offsets/compensation or homes automatically.

The included JSON leaves factual readiness fields false. After actually checking
them, record `stationary_fixture_verified`, `stop_test_verified`, and
`exclusive_control_verified` in the controller config, and `loaded_and_primed`
and `path_clearance_verified` in the plan. These are setup facts, not calibration
accuracy claims. The stationary route accepts explicitly verified path clearance;
when a measured full-fixture top is also configured, its travel-clearance bound is
checked. Existing elevated cold-motion tools retain their full-fixture requirement.

Generate a new local preview after configuration changes:

```bash
python -m controller.stationary run \
  --config output/stationary_print_20261025/controller.json \
  --plan output/stationary_print_20261025/plan.json \
  --output output/stationary_line_preview_01
```

This does not contact the printer. To run the same plan once the setup conditions
hold, use a new output directory and explicitly add `--execute`:

```bash
python -m controller.stationary run \
  --config output/stationary_print_20261025/controller.json \
  --plan output/stationary_print_20261025/plan.json \
  --output output/stationary_line_execution_01 --execute
```

The command preflights, uploads a uniquely named job with autostart disabled and
a SHA256 checksum, reads it back and verifies the content, rechecks printer pose,
then sends one start request. Klipper executes the file, including its temperature
wait. Host-side long G-code-script requests are not used to wait through heating.
The command checks job identity, state, coordinate modes, offsets, path, temperature
and final position. Start requests are never retried after ambiguous replies.
Failures after a possible start request the dedicated emergency-stop endpoint,
which latches Klipper shutdown and requires manual reconciliation. Software polling
and controller-reported coordinates do not establish independent physical safety.

`job.json`, the exact G-code bytes, `execution_events.jsonl` and `execution.json`
record the trial. The code's software tests use simulated printers; real-machine
deposition has not been validated. A file can also be reviewed/uploaded through
the printer's normal interface, with the same required start and path conditions;
that route does not provide this command's host-side preflight/monitor.

References: [Moonraker file upload](https://moonraker.readthedocs.io/en/latest/external_api/file_manager/#file-upload),
[Moonraker print jobs and emergency stop](https://moonraker.readthedocs.io/en/latest/external_api/printer/).
