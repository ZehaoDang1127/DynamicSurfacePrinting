# Controller implementation and operation

Implemented 13 October 2026 against [the controller plan](PRINTER_CONTROLLER_PLAN.md).
Connection guidance updated 14 October 2026 for Pi 5 to H616 wired Ethernet.
Code lives in `controller/`; existing acquisition and forecasting code stays in place.

## What this release implements

This is the plan's executable first controller sprint, plus a guarded cold
characterization mode. It provides causal surface prediction, native XY patterns,
arrival-time iteration, constrained XYZ/E proposals, immutable submitted
endpoints, simulated execution, replay, run logs, fault injection, D405 observation,
and read-only printer discovery. An elevated stationary-fixture path can be
previewed and then executed through Moonraker or a local Klipper socket after its
hardware gates are filled in. The current example selects `moonraker`: the Pi 5
runs prediction and planning, while the existing H616 runs Moonraker and Klippy.
See [Moonraker Ethernet setup](MOONRAKER_ETHERNET_SETUP.md). The alternative local
socket requires controller and Klippy on the same Linux computer; its
[hardware comparison](DIRECT_KLIPPER_SETUP.md) remains available.

Dynamic prediction currently drives the **virtual printer**. Hardware `dry-run`
uses a **constant elevated Z over a stationary fixture**, with a deliberate `M400`
boundary after each short move to characterize timing. It does not apply live
camera corrections. `print` exits with a gate message before opening a connection.
Continuous moving-surface hardware tracking and hot deposition require the
schedule, fixture, and material validation described below. Setting a height or
changing readiness booleans does not enable them.

The simulator and software tests establish controller behavior, not printer
accuracy, actual execution timestamps, adhesion, or a physical collision guarantee.

## Installation and first run

Run from the repository root. The existing `.venv-tds` environment can be used:

```powershell
$python = ".\.venv-tds\Scripts\python.exe"
& $python -m pip install -r requirements-controller.txt
& $python -m controller --help

& $python -m controller simulate `
  --scenario moving-plane --virtual-extrusion `
  --output output/controller_demo
```

Always choose a new output directory. Commands refuse to overwrite a run.
`httpx==0.28.1` is the tested Moonraker network client. Local Klipper transport uses
the Python standard library on Linux and has no HTTP dependency. Synthetic baseline
control and controller-run replay use NumPy; the Moonraker backend needs HTTPX. ConvLSTM and
replay of existing `surface_mapper` manifests additionally need the existing
prediction environment (`requirements-prediction.txt`), because its strict
recording loader imports PyTorch. D405 observation uses `pyrealsense2`.

The default synthetic run uses a 0.5 mm amplitude, 0.1 Hz vertical sinusoid,
30 Hz observations, a 20 mm line, and the velocity baseline. `static-plane`
and `sloped-plane` are also supported. `--amplitude-mm`, `--frequency-hz`, and
`--fps` change the virtual fixture, without changing validated predictor settings.
A scenario exceeding those settings is expected to fault.

## Substrate height: leave it unset until measured

`configs/controller.example.json` deliberately contains:

```json
"height_reference": "relative",
"substrate_height_mm": null
```

The datum is the **substrate's reference top surface in homed printer Z**, in mm.
It is not camera depth, substrate thickness, distance from the camera, or nozzle
clearance. With relative height maps:

```text
nozzle Z = substrate_height_mm + predicted relative height + desired_gap_mm
```

For example, reference substrate Z = 42 mm, predicted displacement = +0.4 mm,
and desired gap = 5 mm produce nozzle Z = 47.4 mm. This is an example, not a
recommended machine height.

When you know the height, copy the example configuration to a local run file and
enter the measured number. You can also set a run-only override:

```powershell
& $python -m controller replay `
  --config configs/controller.json --substrate-height-mm 42 `
  --session output/calibrated_recording `
  --path configs/controller_path.example.json `
  --output output/controller_replay
```

The resolved configuration, including the override, is saved in the output.
`simulate` alone substitutes a clearly virtual 20 mm datum when the field is
unset; the example file stays unchanged. Relative-map replay and cold hardware
setup refuse an unset datum. Discovery and observation do not need it.

If the calibrated map already contains absolute printer Z, use
`height_reference: "printer_z"` and `measurement_frame: "printer"`:

```text
nozzle Z = predicted printer Z + desired_gap_mm
```

The predictor then ignores the datum when converting measured heights, so the
height is never added twice. Absolute-map offline planning does not require a
datum; cold stationary setup still requires it to define its elevated test plane.
Its `grid.z_min_mm`/`z_max_mm` must describe absolute Z. Relative-map grids instead
describe displacement about the reference. Changing conventions requires new
geometry identities and matching camera/checkpoint configuration.

## Modules and data flow

| Module | Responsibility |
|---|---|
| `types.py`, `config.py` | Explicit mm/seconds contracts, observation masks/counts, move timing, configuration hashes, unknown datum and hardware gates |
| `clock_sync.py` | Affine camera clock mapping, uncertainty, counter/domain/residual fault checks, network offset intervals |
| `predictor_adapter.py` | Bounded causal history, existing NumPy kinematic fits, supported horizons, strict ConvLSTM adapter |
| `path.py` | Subdivided line, rectangle and polyline patterns |
| `planner.py` | Forecast sampling, iterative timing, workspace/velocity/acceleration/flow checks, swept support and clearance gates |
| `transport.py` | Serialized submissions, identity/order/expiry/buffer checks, immutable joins, no ambiguous retries, fake printer |
| `connection.py`, `klipper_socket.py`, `moonraker.py`, `cold_run.py` | Explicit local/network backend selection, read-only discovery, status gates, dedicated emergency stop, bounded cold characterization batches |
| `supervisor.py` | Latched controller state transitions and fault response |
| `logging.py`, `runtime.py` | Bounded acquisition/log queues, offline execution and replay, diagnostic observation |
| `simulator.py`, `cli.py` | Analytical ground truth, scenario and CLI entry points |

Observations carry frame identity, height/validity/count arrays, geometry ID,
camera timestamp/domain, host receipt time, mapped observation time, and
uncertainty. Forecasts retain complete-history support and source frame IDs.
Trajectories carry segment/revision/predecessor IDs, a configuration hash, start
position/time, timed moves, extrusion, and expiry. Source arrays are copied into
the observation contract so later capture-buffer reuse does not change history.

The offline loop processes observation and virtual completion events in timestamp
order. At each completion, it plans from observations already available at that
time. A proposal can contain several moves; only its earliest move is submitted.
The remaining proposal is discarded and rebuilt using newer observations. The
submitted endpoint is immutable. This implements causal revision without editing
motion that was already sent.

The live observation worker holds only `history + 2` frames. On overflow it removes
the oldest frame and counts the drop. The predictor's timestamp rules still apply;
missing frames are not relabelled as a uniform sequence. Logging has a separate
bounded 2,048-item queue and counts discarded records. Disk work does not block
planning; finalization drains it after motion stops. A run with logging drops or
errors is marked incomplete and is refused as a controller replay input.

## Prediction and timing

Persistence, constant velocity, and constant acceleration reuse
`prediction.baselines.fit_kinematic_baselines`. Fits use actual time intervals.
The adapter caches fitted coefficients and samples them at each requested arrival
time. It requires measured support throughout the history, adequate point counts,
finite heights, configured Z range, and matching geometry. A global coverage
fraction cannot substitute for corridor support.

Frame/domain resets or excessive time gaps discard the history. A running
controller faults instead of silently warming up and resuming. Forecasts reject
stale input, past target times, and targets beyond `max_horizon_s`.

The optional ConvLSTM adapter calls the existing streaming predictor in memory.
Checkpoint history and full camera/grid/transform geometry must match. It accepts
only the checkpoint's exact trained horizon. There is no unvalidated interpolation,
extrapolation, or prediction feedback. A single fixed-horizon model cannot supply
the arbitrary times needed by the present planner, so selecting it usually causes
a supported-horizon gate before motion. A validated model bank or multi-output
model is the next integration step.

Camera clock mapping is an affine scale/offset with an explicit uncertainty and
residual limit. The `observe` command anchors the first camera timestamp to host
receipt for **diagnostics**, with 100 ms provisional uncertainty; it does not
calibrate exposure/USB delay. Ordinary recording replay uses a virtual camera
timeline. Network RTT intervals do not establish Klipper motion `print_time`.
Those limitations are recorded and cannot be used to arm dynamic hardware motion.

## Planning and extrusion

For each proposed point the planner estimates its arrival, samples future height,
sets Z, and recomputes the duration. It repeats until the arrival times agree within
`convergence_s`, or faults at `max_plan_iterations`. Spatial subdivisions are set
by `path_spacing_mm`; the example uses 1 mm.

Virtual motion is synchronized, straight-line, rest-to-rest travel. For each move,
the planner limits 3D speed, XY speed, Z speed, XY acceleration and Z acceleration.
It uses triangular or trapezoidal profiles and includes acceleration time. Stops
at each virtual waypoint give explicit joins and bounded accelerations. Stock
Klipper can blend ordinary `G1` moves; these profiles are **not a model of its
finalized continuous schedule**. Cold characterization drains the queue with
`M400`, intentionally introducing a synchronization boundary.

The acceptance checks cover machine workspace, the interpolation interior, the
swept tool footprint, configured gentle slopes, finite forecasts, the committed
join, and discretized gap error along the motion profile. Missing support is never
hole-filled. A provisional error allowance sums calibration, forecast, mechanical,
and surface-speed × time uncertainty. This is a conservative engineering gate
whose combined error still needs experiment; it is not a statistical confidence
guarantee. Sampled clearance checks also need physical envelope validation.

Extrusion is:

```text
relative surface length = sqrt(XY length² + spatial height change²)
filament advance = bead_area_mm2 × relative surface length
                   / (π × (filament_diameter_mm / 2)²)
```

The spatial height change is sampled at one future time. Uniform vertical motion
therefore adds no filament, even when the nozzle travels farther in machine Z.
Peak volumetric flow can reduce feed speed; excessive extrusion per machine-space
distance faults. This is a segment chord approximation for gentle surfaces and
vertical-only material motion. Subdivision and a calibrated bead area remain
necessary. Lateral material tracking and pressure dynamics are outside this release.

Generated commands use `G90` followed by `M83`, explicit XYZ, relative E when
virtual extrusion is selected, and feed in mm/min (`60 × internal mm/s`). Cold
hardware commands have no E and never set heater targets. Arbitrary slicer input,
startup macros, online offset correction, and automatic G-code retries are absent.

## Commands and outputs

| Command | Behavior |
|---|---|
| `simulate` | Dynamic virtual tracking; optional `--virtual-extrusion`; no printer connection |
| `replay` | Causal virtual planning from controller logs or matching finished mapper recordings |
| `discover` | Read-only Klipper inventory/status through the selected Moonraker or local backend |
| `observe` | D405 acquisition and mapped height logging; no printer connection |
| `shadow` | Live D405 acquisition, kinematic prediction and unexecuted candidate planning; [integration guide](INTEGRATED_SHADOW.md) |
| `camera-hash` | Canonical camera/geometry fingerprint for configuration matching |
| `dry-run` | Elevated cold path preview; `--execute` enables stationary-fixture hardware characterization |
| `stop` | Dedicated emergency-stop API request through the selected backend |
| `print` | Explains the remaining live-deposition gates and exits without a connection |

Example replay of a simulation, using its saved resolved configuration:

```powershell
& $python -m controller replay `
  --config output/controller_demo/config.json `
  --session output/controller_demo `
  --output output/controller_demo_replay

& $python -m controller simulate --dropout-frame 10 `
  --output output/controller_dropout
& $python -m controller simulate --lost-reply-segment 2 `
  --output output/controller_lost_reply
```

Fault runs exit unsuccessfully and leave a report. `report.json` includes state,
reason, completion, logging status, gap comparison, forecast horizon, frame age,
and planning-time percentiles. `events.jsonl` contains proposals, source identities,
submissions, acknowledgements, transitions, references, and faults. Observation
arrays are in `observation_*.npz`. `config.json` and the original-path event make
the run reproducible. Virtual runs also write `SUMMARY.md` and `preview.gcode`.
Cold runs write `cold_preview.json` and `cold_preview.gcode`. Virtual previews
are not calibrated hardware jobs and omit an approach from the machine's current
position. A replay's recorded-camera gap comparison is not independent physical
ground truth. Forecast maps can be recomputed from saved observations and the
resolved predictor configuration; checkpoint files must be preserved separately.

## State and fault behavior

The normal progression is `DISCONNECTED → READY → WARMUP → ARMED → RUNNING`.
Completion passes through `STOPPING` to `READY`. Faults latch `FAULT`; a new,
reconciled session is required. Reconnection does not resume a job.

The serialized writer checks segment order, duplicate attempts, predecessor,
configuration identity, join continuity, expiry and committed time horizon. An
attempt is recorded before network access. A reply loss, partial acceptance or
cancellation makes the session uncertain; motion is never blindly resent.
An acknowledgement does not mean physical arrival. The fake printer tests both
prefix execution and motion accepted before its reply is lost.

During cold execution, telemetry checks readiness, printer identity/eventtime,
modes, multipliers, offsets, compensation, competing prints, heater state and
active limits. A separate monitor checks between writer boundaries. Its 200 ms
poll interval plus network latency is not an independent hardware watchdog.
Ctrl+C attempts the same fault response. Faults after setup attempt the dedicated
emergency-stop API: `emergency_stop` on a separate local socket, or
`/printer/emergency_stop` for Moonraker. A failed host/socket or network can prevent
delivery, and neither endpoint holds the fixture or lifts the nozzle. Use the
physical arrangement in [the hardware guide](HARDWARE_SETUP.md).
These interface semantics follow the [Moonraker printer API](https://moonraker.readthedocs.io/en/latest/external_api/printer/)
and [Klipper status reference](https://www.klipper3d.org/Status_Reference.html).

## Validation and the next hardware gate

Run the complete regression suite after installing controller and prediction dependencies:

```powershell
& $python -m unittest discover -s tests -v
```

Controller acceptance tests cover datum conventions, malformed configuration,
offset/drift mapping, clock discontinuities, plane/slope/moving-plane planning,
surface-relative extrusion, corridor holes, stale input, horizon/workspace/slope/flow
limits, replay, duplicate/order/buffer gates, lost replies and prefix execution,
read-only HTTP discovery, cold readiness checks, previews without network effects,
and cold execution through an in-process HTTP fake.

Local transport tests additionally cover ETX framing, fragmented/coalesced messages,
out-of-order replies, notifications, lost replies without reconnect, malformed data,
disconnects, explicit backend selection and an independent stop connection. Both
backends retain the same hardware gates. Local deployment and the connection of an
external host to the actual MCU links remain untested on the physical printer.

The 13 October software smoke run completed 20 submitted moves and 185 observations
without logging loss. Against its ideal sinusoidal reference, absolute gap error
P95 was approximately 0.017 mm and maximum 0.020 mm. This result assumes noiseless
maps and the explicit virtual schedule; it does not predict real printer accuracy.

Before enabling dynamic hardware streaming, measure the scheduled/physical timeline
and arrival uncertainty on the installed SV08, establish a buffer duration that
fits the validated forecast horizon, and implement the tested autonomous fixture
response. Extend the transport with that measured schedule contract or a
version-pinned local bridge if needed. Then validate dynamic cold gap control and
finally a static deposition profile. The current `print` gate should be replaced
only by those explicit runtime checks, not simply removed.
