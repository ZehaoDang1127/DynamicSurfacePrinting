# Hardware choices for local Klipper control

Alternative deployment notes, 13 October 2026. **Current decision, 14 October:**
use Pi 5 prediction and planning over wired Ethernet to Moonraker and Klippy on
the existing SV08 H616 host. Follow [Moonraker Ethernet setup](MOONRAKER_ETHERNET_SETUP.md).
The network-free, same-host alternatives below are retained for comparison and
are not needed for the selected arrangement. Substrate height remains unset.

For the step-by-step local API connection, use
[Pi 5 Klipper connection](PI5_KLIPPER_CONNECTION.md). That guide also records the
remaining physical migration gate: the official drawings do not establish a
plug-and-play external mainboard MCU cable route.

## What the linked Mainline project establishes

[Rappetor/Sovol-SV08-Mainline](https://github.com/Rappetor/Sovol-SV08-Mainline)
documents installing mainline Klipper on the SV08's existing onboard Linux system,
including host storage and both MCU firmware/configurations. It is a migration
guide, not a ready-made predictive surface controller. Its configuration has
separate mainboard and toolhead MCU identities. The USB0/USB-C procedure is for
host storage flashing; it does not establish an external runtime motor-control
connection. ST-Link is used for firmware work, not streaming trajectories.

The project is useful for a controlled, versioned Klipper installation. Migrating
to it is a separate hardware operation: preserve the working configuration,
firmware and storage image, verify board revision and toolhead identity, and use
the source's current instructions. This repository has not flashed or migrated
the printer, and does not replace those instructions with guessed pinouts.

## Two deployment choices

The original SV08's published host specifications are H616, 1 GB RAM and 8 GB eMMC.
[Sovol specifications](https://www.sovol3d.com/products/sovol-sv08-3d-printer).
That is the basis for benchmarking the onboard option before selecting it for
camera acquisition and learned prediction in addition to Klippy.

| Requirement | A: Existing SV08 Linux computer | B: External Linux computer |
|---|---|---|
| Where controller and Klippy run | Both on the onboard computer | Both on the external computer |
| Network in motion path | None | None |
| Connection to printer MCUs | Existing internal connections stay in place | Access/routing to both MCU links must be verified and transferred to the external host |
| Camera | D405 connected to an appropriate onboard USB host port | D405 connected directly to the external computer |
| Additional compute hardware | None initially | Existing suitable Linux workstation or another computer with measured sufficient performance |
| Initial prediction | Benchmark NumPy persistence/velocity first | NumPy baselines; benchmark the actual ConvLSTM checkpoint on CPU/GPU |
| Learned-model resources | Must measure RAM, inference time and interference with Klippy on the limited onboard system | Choose RAM/CPU/GPU from the measured capture + inference + planning workload |
| Hardware modification | No MCU rewiring for the basic local deployment | Potential MCU USB rerouting; exact connectors are not specified until board topology is verified |
| Operating system | Installed Linux with supported Python/camera runtime | Linux with supported Python/camera runtime and Klipper host installation |
| Firmware requirement | Existing compatible Klipper may already suffice for the local API | Host/MCU protocol compatibility and complete printer configuration must be established; do not assume a flash is always required |
| Principal uncertainty | Can capture, inference and planning share the host without disrupting motion scheduling? | Can both MCU links be exposed correctly, and can the host own the entire printer configuration? |

For the first stationary cold tests, option A requires less hardware work. Option B
is the stronger candidate if learned prediction cannot meet deadlines on the
onboard system. This is a conditional recommendation; no host purchase or wiring
choice is made from advertised compute specifications alone. Running only Klippy
locally while leaving prediction on a remote Windows workstation would still need
a live communication link, so it would not implement the requested single-host,
network-free architecture.

The current Windows checkout remains suitable for simulation, replay and code
development. Actual local deployment puts the controller beside Klippy on Linux.
Installing Linux, arranging camera drivers, and exposing printer hardware are
separate setup tasks, not operations performed by the Python CLI.

## Raspberry Pi 5 as the external host candidate

The Pi 5 is a stronger compute candidate than the original onboard host. Its
published CPU is a quad-core Cortex-A76 at 2.4 GHz, whereas the H616 uses
Cortex-A53 cores. A Pi 5 with 8 GB RAM would also provide substantially more memory
than the SV08 host's 1 GB. These specifications support evaluating it as option B;
they are not a benchmark of this camera/model workload.
[Pi 5 specifications](https://www.raspberrypi.com/products/raspberry-pi-5/),
[H616 product brief](https://www.allwinnertech.com/uploads/pdf/20200409170709ca.pdf).

Proposed evaluation hardware: Pi 5 with 8 GB RAM, active cooling, a suitable
5 V/5 A USB-C supply, storage, and D405 USB connection. Use a 64-bit Linux setup.
Train ConvLSTM on the workstation and deploy a fixed checkpoint. The current
inference code supports CPU or CUDA, so the Pi's integrated graphics are not an
accelerator for this implementation; benchmark CPU inference with camera capture,
planning and Klippy running together.

Pi 5 remains a candidate, not a confirmed hardware selection. Verify deadlines,
memory use, sustained temperature and both printer MCU connections before adopting
it. It improves the available compute budget but does not remove Klipper buffering
or establish a direct MCU connector pinout.

## Local communication architecture

Option A:

```text
D405 → SV08 onboard Linux [controller → local API socket → Klippy]
                                      ↓ existing MCU links
                               mainboard + toolhead MCUs
```

Option B, after physical connection verification:

```text
D405 → external Linux computer [controller → local API socket → Klippy]
                                          ↓ verified direct MCU links
                                   mainboard + toolhead MCUs
```

Klipper exposes a local Unix-domain socket API. It lets a process on the same host
query status, submit G-code and request emergency stop without HTTP. The controller
uses that API; Klippy owns low-level MCU communication. The local socket path is
distinct from the MCU serial paths. [Klipper API documentation](https://www.klipper3d.org/API_Server.html).

The MCU wire protocol is encoded Klipper commands. Sending ordinary G-code to a
bare MCU UART/USB device is not a supported substitute for the Klippy host.
[Klipper protocol documentation](https://www.klipper3d.org/Protocol.html).
Keep a single Klippy process as owner of both MCU links. A second controller board
driving motor step/direction signals would introduce different motion, heater and
interlock responsibilities; it is not necessary for either option above.

## Inventory before selecting or connecting hardware

Obtain these from the actual printer, using its existing interface or a local shell:

1. Mainboard revision, MCU firmware versions and current `printer.cfg` plus includes.
2. The exact `[mcu]` and toolhead MCU serial identities; confirm them individually.
3. The Klippy service command, Python environment and API socket path.
4. USB topology and physical connector routing for both MCUs and the D405.
5. Camera support/cadence on the proposed Linux host, with dropped-frame counts.
6. Host RAM usage, capture time, inference/planning P95/P99 and worst observed time
   while Klippy is active in a cold characterization trial.
7. Printer configuration backups, a restore procedure and the independent fixture
   stop arrangement from `HARDWARE_SETUP.md`.

Read-only Linux commands that help inventory the host:

```bash
systemctl cat klipper
ls -l /dev/serial/by-id/
lsusb -t
```

If the service has another name, find the installed Klippy service rather than
assuming it is called `klipper`. Inspect the command for its `-a` API socket
argument. An existing Moonraker configuration may also record the same socket as
`klippy_uds_address`; reading that setting does not put Moonraker in the control
path. Do not start a second Klippy instance while the installed one owns the MCUs.

For an external host, verify the board schematic/connector topology before
disconnecting MCU links. A connector's shape or the guide's storage-flashing
procedure does not identify a runtime MCU port. The motherboard and toolhead
must retain correct identity-to-pin configuration; swapping them is not an
acceptable shortcut. Exact external USB wiring remains pending this inventory.

## Configure the local adapter

`controller/klipper_socket.py` implements the local API transport and is selected
through `controller/connection.py`. To select this alternative, change the copied
configuration from its default `moonraker` backend to:

```json
"backend": "klipper_socket",
"klipper_socket_path": null
```

Both keys belong inside the existing `hardware` object. Copy the example to a
local configuration, then set the path obtained from the installed service, for
example `/home/sovol/printer_data/comms/klippy.sock` if that is the verified path.
It must be an API socket, not `/dev/serial/by-id/...`.

On the chosen Linux computer, in this checkout and a prepared environment:

```bash
python -m controller discover \
  --config configs/controller.json \
  --output output/klipper_local_inventory_01
```

Discovery sends only `info`, `objects/list` and `objects/query`. The saved report
records `backend: klipper_socket`, `network_required: false`, available objects,
versions, query timing and that the finalized motion schedule remains unknown.
The Linux user needs permission to access the existing API socket. Use the
installed service's user/group arrangement instead of making it world writable.
No API key or Moonraker URL is used in this mode.

The local client uses JSON messages terminated by ETX, matches replies by ID even
when they arrive out of order, keeps notifications in a bounded queue and rejects
malformed responses. Timeout or disconnect latches the connection failed; it does
not reconnect, repeat a command, or switch to the network backend. Emergency stop
uses a separate local API connection so it does not wait for an ordinary command's
reply. The same arming, expiry, clearance, no-extrusion and no-heating rules apply
to cold execution through either backend.

Once the measured datum and cold setup fields are filled, the existing preview
and execution commands use the selected backend:

```bash
python -m controller dry-run --config configs/controller.json \
  --path configs/controller_path.example.json --output output/local_cold_review

# After reviewing the path and verifying the stationary cold setup:
python -m controller dry-run --execute --config configs/controller.json \
  --path configs/controller_path.example.json --output output/local_cold_trial
```

The local backend has been tested with in-memory framed streams, including cold
execution, lost replies, disconnects, out-of-order replies and emergency stop while
another command waits. Native Linux connection to the actual printer remains a
hardware validation step. Native Windows deployment fails with an explicit Linux
host requirement; it does not fall back to Ethernet.

## What becomes faster, and what still needs measurement

Local control removes the Ethernet/HTTP layer from command submission. It does
not remove camera exposure/transfer delay, inference time, Klipper look-ahead,
MCU scheduling or physical Z acceleration limits. The present cold path still
uses deliberate `M400` boundaries and is not continuous adaptive printing.

Measure observation-to-scheduled and observation-to-physical arrival on the chosen
host. Select the buffering policy and supported prediction horizons from those
measurements. A future schedule-aware Klipper extension may still be needed;
the socket adapter alone does not allow edits to committed motion or prove a
bounded physical arrival error. The independent fixture response and the live
deposition gate remain necessary regardless of cable/interface choice.
