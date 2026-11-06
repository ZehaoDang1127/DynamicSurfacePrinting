# SV08 and D405 hardware setup guide

Companion to [the controller implementation](CONTROLLER_IMPLEMENTATION.md) and
[the controller plan](PRINTER_CONTROLLER_PLAN.md), updated 14 October 2026.

The substrate height can remain undecided during software development. The
example controller configuration leaves it null and the simulator supplies its
own virtual datum. Measure the actual reference substrate top Z only after the
fixture and printer coordinate frame are established.

## Arrangement

**Current arrangement:** run prediction and planning on the Pi 5, connect it with
one Ethernet cable directly to the existing SV08 H616 host, and retain Moonraker, Klippy and
the internal MCU connections on that host. The example selects
`hardware.backend: "moonraker"`. Follow the
[Chinese Moonraker Ethernet setup guide](MOONRAKER_ETHERNET_SETUP.md) for network,
printer configuration, authentication and read-only checks. Runtime control uses
the local network and does not require an Internet connection.

The [local-host comparison](DIRECT_KLIPPER_SETUP.md) and
[Pi-to-MCU migration notes](PI5_KLIPPER_CONNECTION.md) describe an alternative
deployment, rather than a prerequisite for this arrangement.

```text
D405 ─ USB ─ Pi 5 running prediction + controller
                  │
             wired Ethernet
                  │
          SV08 host / Moonraker
                  │
          Klipper / printer electronics
                  │
               XYZ motors

Independent physical stop ─ printer + vertical fixture
Independent displacement/gap reference ─ measurement log
```

Retain the existing SV08 host, electronics, hotend and extruder. Connect the Pi 5
and printer through a stable wired network, and connect the D405 directly to the
Pi through a reliable USB cable. Train on the workstation and benchmark capture,
inference and planning on the Pi with the intended checkpoint and camera cadence.
The architecture decision does not establish that a particular model meets its
deadline on the Pi.

Mount the camera on a rigid stationary support, with adjustment and strain relief.
Check the entire gantry/toolhead travel for cable and mount interference. The
camera must see the upcoming local path even when the nozzle is nearby. Begin
with a small region, around 40 × 40 mm if the fixture permits, and diffuse lighting.
Use the camera's usable close-range envelope and verify support on the actual
material; the plan references the manufacturer specifications. Account for toolhead
occlusion and heat when choosing position and angle.

Begin with a rigid coupon on a guided vertical fixture before a membrane. Its
stroke, maximum top height, loads, stopping distance, and hold behavior must be
known. For the cold characterization mode in this release, lock it stationary.
Its full mechanical envelope, including clamps and fasteners, must remain below
the elevated nozzle plane. Fixture actuators and controller firmware are not
supplied by this repository; selecting them requires your final geometry and loads.

## Inventory and backup

### What runs on the printer

The H616 keeps its existing Moonraker and Klippy services. The Pi runs this
repository's controller using `controller/moonraker.py`. Klippy retains ownership
of MCU communication and motor scheduling. A custom Klipper extension or macro
is not required by the present adapter. The alternative same-host local mode uses
`controller/klipper_socket.py`; both modes use `controller/cold_run.py` for guarded
stationary cold characterization.

```text
Pi 5 controller → wired Ethernet → H616 Moonraker → Klipper → printer MCUs
```

The optional printer-host bridge from section 8 of the plan has not been
implemented. It is a later service, deployed alongside Klipper if timing tests
show the stock interface is inadequate. Its responsibilities would include
unsent trajectory buffering, session ownership, expiry, a local heartbeat and
the validated fault response. A bridge or macro alone does not establish precise
execution timing or a physical fixture interlock.

### Connect the Pi or development workstation to Moonraker

1. Connect the SV08 and Pi with one Ethernet cable. Configure distinct fixed
   addresses on their wired interfaces, for example Pi `192.168.50.1/24` and
   H616 `192.168.50.2/24`, with no gateway or DNS on this isolated link. Choose
   a subnet that does not overlap existing Wi-Fi networks. Follow the
   [direct-cable procedure](MOONRAKER_ETHERNET_SETUP.md) for persistent settings.
2. Open the printer's existing web interface using its usual address and confirm
   Klipper is ready. The web interface and Moonraker may use different ports.
3. Set `hardware.moonraker_url` in the copied controller configuration. For example,
   `http://192.168.50.2:7125` uses the illustrative H616 address above. Port 7125 appears in the
   [published SV08 Moonraker configuration](https://raw.githubusercontent.com/Sovol3d/SV08/main/home/sovol/printer_data/config/moonraker.conf);
   confirm the installed configuration rather than assuming it is unchanged.
4. If authentication is required, set `MOONRAKER_API_KEY` locally to the
   existing API key before running the controller. Do not paste the key into a
   shared report or loosen authorization settings.
5. Run the discovery command below. It requires neither a substrate height nor
   homing and sends no motion or heating commands. Check that the saved printer
   state is `ready` and that the expected objects are available.

If connection fails, use `Test-NetConnection -ComputerName <printer-ip> -Port 7125`
with the actual address/port. Failure suggests address, routing, firewall or service
availability; an HTTP 401/403 instead indicates authorization. A reachable
Moonraker with Klipper in `startup`, `error` or `shutdown` requires printer-side
diagnosis. SSH is not required for ordinary controller communication; use it with
your configured credentials when inspecting services, backing up files or
deploying a future bridge.

### Inventory to record

1. Record the exact SV08 model, nozzle/toolhead, installed Klipper and Moonraker
   versions, and machine serial/identity. Back up the printer configuration and
   maintain a restore method before changes.
2. Record configured XYZ travel, Z velocity/acceleration, active XY motion limits,
   nozzle offsets, mesh/thermal compensation, pressure advance and extrusion settings.
   The example limits are provisional controller settings, not certified SV08 values.
3. Set a stable printer address. Use the existing authentication mechanism.
   If an API key is required, put it in the environment named by
   `hardware.api_key_env`, rather than committing it in JSON or a script.
   Moonraker's [authentication documentation](https://moonraker.readthedocs.io/en/latest/external_api/authorization/)
   defines the `X-Api-Key` HTTP header used by the client.
4. Copy `configs/controller.example.json` to
   `configs/controller.json`, set `hardware.backend` to `moonraker`, and set
   `hardware.moonraker_url` to the actual URL, including the confirmed port.
5. Run read-only discovery:

```powershell
$python = ".\.venv-tds\Scripts\python.exe"
& $python -m controller discover `
  --config configs/controller.json `
  --output output/sv08_inventory
```

Inspect `discovery.json`. Discovery queries only available objects and sends no
G-code. It also records an RTT sample and explicitly reports that the motion
schedule is unknown. Archive the report alongside the printer backup.

## Establish printer coordinates and measure the substrate datum

1. Remove or lower the fixture so ordinary homing, probing and gantry leveling
   cannot contact it. Perform the printer's documented procedure.
2. Establish the nozzle-tip reference in the homed machine frame. Clear ordinary
   bed-mesh compensation during controlled controller setup and verify the G-code
   offsets. Record all changes. Do not apply two height corrections to the same geometry.
3. Install the fixture/substrate with the printer stationary at verified clearance.
   Measure its reference receiving surface top Z using a suitable calibrated
   method. Include fixture base, coupon thickness, mounts and actual nozzle-tip
   reference in the coordinate conversion.
4. Enter that measured top Z in `substrate_height_mm`, in mm. Keep uncertainty
   and the measurement method in the calibration record. Do not use a guessed
   camera-depth value as the datum.
5. Determine the maximum Z of everything under the swept toolhead during all
   permitted fixture travel. Enter it in `hardware.fixture_top_z_mm`. It may be
   higher than the reference substrate datum because of stroke, clamps or geometry.

For relative-map operation:

```text
reference receiving surface: relative height = 0
actual surface Z = substrate datum + measured relative height
```

If instead the camera transform yields absolute printer Z, use `printer_z` mode
and `measurement_frame: "printer"`. Do not add the substrate datum to those
maps again. The datum still defines the cold setup plane. Recalibrate after
moving the camera, fixture mount, substrate datum or nozzle reference.

## Camera calibration

The existing `configs/d405_camera_frame.json` has an identity transform and
reports camera depth. It is not a printer calibration and is intentionally
rejected by the controller's frame/grid checks.

Create a new camera configuration from the existing `surface_mapper` format:

1. Measure a target with known printer XYZ coordinates and estimate the rigid
   camera-to-printer transform. Translation is in mm; incoming RealSense point
   clouds are in metres and the projector converts them once.
2. Validate at unused XY positions and multiple known heights. Raise a reference
   object by a known positive Z displacement and confirm positive reported Z.
3. Set `frame_name` to `printer_relative` or `printer`, matching the controller's
   `measurement_frame`. In relative mode the output frame has printer-aligned XY
   and Z measured relative to the reference surface datum. In absolute mode it
   has full printer XYZ.
4. Match the controller's grid exactly, including bounds, cell size, Z range,
   minimum point count and aggregation. Choose bounds inside the measured patch;
   keep the path and its footprint away from grid edges.
5. Save the measured transform, calibration record and a new `geometry_id`.
   Set `hardware.calibration_id` and a real ISO 8601 `expires_utc` with a timezone,
   based on your recalibration policy.
6. Compute the canonical geometry/stream hash:

```powershell
& $python -m controller camera-hash `
  --camera-config configs/d405_printer.json
```

Enter the returned `camera_config_sha256` in `hardware`. This hash includes
transform, frame, grid, image dimensions and FPS. Camera serial, warm-up count
and timeout are excluded. The hash checks consistency; it does not certify the
quality of a transform.

Record a stationary observation run:

```powershell
& $python -m controller observe `
  --config configs/controller.json `
  --camera-config configs/d405_printer.json --frames 300 `
  --output output/substrate_stationary
```

Observation does not move or heat the printer. Inspect coverage along the entire
intended path and footprint, point counts, noise, warm-up drift, known steps and
boundary artifacts. The current diagnostic clock anchor is not an exposure-time
calibration. Before dynamic control, measure camera delay, drift and frame age
against an independently timestamped reference.

The projector supplies depth-based cell validity; it does not identify substrate,
printed beads or the nozzle semantically. Choose a region/Z window and acquisition
geometry that exclude them for initial observation, or add and validate a substrate
mask before dynamic trials. Never treat interpolated holes as measured support.
Checkpoint cadence, geometry and preprocessing must match this calibrated setup.

## Elevated cold characterization

Use a locked stationary fixture, heaters off, and an independently tested physical
stop arrangement. Record the evidence before setting these configuration fields:

| Field | Evidence needed |
|---|---|
| `stationary_fixture_verified` | Fixture locked/held, no autonomous movement during the test |
| `stop_test_verified` | Physical stop behavior tested at large clearance, including gravity/stored energy |
| `exclusive_control_verified` | One application owns motion; no print file, jog controls or competing automation |
| `fixture_top_z_mm` | Maximum obstruction height over the entire swept toolhead region |
| `cold_clearance_mm` | Validated clearance above that full envelope; example is 5 mm |
| `calibration_id`, `camera_config_sha256`, `expires_utc` | Versioned valid geometry and datum record |

These booleans are recorded operator attestations, not electronically sensed
interlocks. They must correspond to completed setup work. The code also checks
homed axes, ready state, no active print/motion, cold temperatures, disabled heater
targets, unit multipliers, zero XYZ offsets, disabled compensation, actual workspace
and installed motion limits before it sends a setup script.

Cold nozzle Z is `substrate_height_mm + desired_gap_mm`. It must satisfy:

```text
nozzle Z >= fixture_top_z_mm + cold_clearance_mm
```

For example, a 42 mm substrate datum and a 43 mm full fixture envelope require
at least a 6 mm desired cold gap when clearance is 5 mm. Actual values come from
your measurements. A deposition gap is not suitable for this initial test.

Choose an elevated path within the measured workspace. Preview it first:

```powershell
& $python -m controller dry-run `
  --config configs/controller.json `
  --path configs/controller_path.example.json `
  --output output/cold_review
```

Inspect `cold_preview.json` and `cold_preview.gcode`: start position, every XY,
constant Z, feed and synchronization boundaries. This command makes no printer
connection. Use the printer UI to place the cold stationary nozzle at the reviewed
start via a separately checked approach. The controller does not generate that
approach, home the printer, or run an arbitrary startup macro.

Execute only after verifying the approach, clearance, fixture and stop behavior:

```powershell
& $python -m controller dry-run --execute `
  --config configs/controller.json `
  --path configs/controller_path.example.json `
  --output output/cold_timing_trial
```

Keep an operator at the physical stop. Each batch is time bounded and ends with
`M400`; expected start/end positions are checked through requested-motion telemetry.
The software never extrudes or sets a heater target in this mode. Record an
independent nozzle-position/time reference alongside the logs. Submission/reply
times and `motion_report.live_position` alone are not physical tracking measurements.

To request the dedicated network stop:

```powershell
& $python -m controller stop --config configs/controller.json
```

Ctrl+C during an executed cold run also attempts this response. Network delivery
can fail; neither software action replaces the physical stop. Do not automatically
retry failed moves or resume after reconnecting. Reconcile the machine and fixture,
then start a new session.

## Measurements before moving-surface control

Keep the following worksheet with each fixture/calibration version. Values are
deliberately not filled with assumed printer performance.

| Quantity | Measurement to save |
|---|---|
| Substrate datum and full obstruction envelope | mm in homed printer Z, method and uncertainty |
| Static height error/noise/drift | By XY and height, including full path footprint support |
| Camera timing | Exposure-to-host delay, cadence, gaps, offset/drift uncertainty |
| Printer timing | Submitted, scheduled and independently observed arrivals; P95/P99/max error |
| Replenishment/buffering | Worst processing/network delays, smooth execution window, underruns |
| Z tracking | Verified speed/acceleration and physical following error under load |
| Fixture response | Trigger latency, residual travel, hold behavior and independent watchdog |
| Gap success threshold | Allowed absolute error and separate minimum collision clearance |
| Material profile | Filament, substrate, nozzle, static successful gap, bead area, adhesion, fixed deposition settings |

For sinusoidal fixture displacement with amplitude A and frequency f, the required
peak vertical speed is `2πfA` and peak acceleration is `(2πf)²A`. Spatial slope adds
Z demand. Timing error contributes approximately surface speed × timing error to
gap error. Use the measurements to set forecast horizons and error budgets.

The moving fixture needs an autonomous, tested response to workstation freeze,
network loss and printer shutdown. Assess the complete residual trajectories:
immediate fixture hold can still collide with an already committed downward nozzle
move. A host polling monitor is not that watchdog. A shutdown also does not imply
a raised nozzle or a supported fixture.

Only after those checks should dynamic cold tracking be enabled through a measured
schedule integration. Reduce clearance gradually using an independent gap reference.
Then establish a static successful filament line on the actual substrate, calibrate
bead area/flow and the acceptable deposition gap, and compare moving-surface trials.
The present `print` command remains gated until this work is represented by
validated runtime contracts. Follow the staged acceptance criteria in the plan.
