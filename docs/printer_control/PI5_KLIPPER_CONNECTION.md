# Alternative: moving the Klipper host to the Pi 5

**Current decision, 14 October 2026:** keep Klippy and Moonraker on the H616 and
connect Pi 5 prediction/planning over wired Ethernet. Start with
[Moonraker Ethernet setup](MOONRAKER_ETHERNET_SETUP.md). That arrangement retains
the printer's internal MCU wiring and does not require the migration below.

Alternative notes, 13 October 2026. This procedure prepares the software connection and identifies
the remaining physical migration gate for the original SV08. It does not assume
that the stock motherboard exposes a plug-and-play external MCU port.

## Connection status

The alternative arrangement considered here is:

```text
D405 → Pi 5 [prediction + planner → local Klippy API → Klipper host]
                                           ↓ verified physical MCU links
                                   SV08 mainboard + toolhead MCUs
```

The local controller-to-Klippy API is implemented. Moving MCU ownership from the
SV08's onboard H616 computer to the Pi is a separate hardware/firmware migration.
The linked Mainline guide installs Klippy on the onboard host; it is not an
external-Pi conversion procedure. No printer migration has been performed here.

The published board pin drawing shows USB0, UART0 and other host interfaces. The
published MCU sheet shows the main MCU's USB nets, but the supplied files do not
establish a complete, externally accessible Pi-to-main-MCU cable route. A USB-C
shape, UART label or storage-flashing procedure is insufficient to specify it.

The user has identified the mainboard ports under discussion as the external
USB-A ports. These are host ports of the onboard Linux computer, not an exposed
USB device port for the main MCU. Connecting a Pi USB host port to them does not
transfer MCU ownership. The mainboard connection gate therefore remains open.

## Toolhead USB header identified from the supplied drawing

The supplied toolhead pin drawing identifies a three-pin connector labelled USB
at the lower-left of the shown board. Its signals are:

| Toolhead signal | MCU pin | Connects to |
|---|---|---|
| `USB_N` | `PA11` | USB `D-` |
| `GND` | Ground | USB ground |
| `USB_P` | `PA12` | USB `D+` |

This is a USB data interface presented on a board connector, rather than a
standard USB socket. Use a correctly wired USB host cable/adapter with the
matching board connector, preferably preserving the existing toolhead harness.
If its host end is already a standard USB plug it can be rerouted; otherwise it
needs a matched connector-to-USB adapter. Connector family/pitch and physical
pin orientation must be verified on the real connector; the drawing gives signal
names, not a complete purchasable adapter specification. Match signals, not wire
colours or an assumed left-to-right order.

The drawing separately labels a `24V_POWER` connector. Retain the correct toolhead
power arrangement. The three-pin USB header has no labelled USB `+5V`/VBUS pin:
do not add USB 5 V to it, or connect toolhead 24 V to the Pi's USB wiring. This
header is not the toolhead's separate `uart` connector; a USB-to-TTL converter or
Pi UART GPIO is not an adapter for USB `D+`/`D-` signals.

After an independently verified, powered-off data-cable reroute, a read-only
enumeration check on the Pi can establish whether the toolhead is seen:

```bash
ls -l /dev/serial/by-id/
```

Identifying the toolhead alone is not sufficient to run the complete printer.
The stock onboard Klippy instance must not remain a competing owner of rerouted
devices. No direct external mainboard MCU path is specified by this toolhead
drawing. See [the published toolhead pin drawing](https://github.com/Sovol3d/SV08/blob/main/Motherboard/Extra_PIN_definition.pdf).

Sources inspected:

- [Official motherboard pin drawing](https://github.com/Sovol3d/SV08/blob/main/Motherboard/MCU_PIN_definition.pdf)
- [Official main MCU schematic](https://github.com/Sovol3d/SV08/blob/main/Motherboard/H616_JC_6Z_V1_2_MCU.pdf)
- [Mainline migration guide](https://github.com/Rappetor/Sovol-SV08-Mainline)

Do not connect two USB host ports together or wire a Pi to a labelled UART on the
assumption that it is the motor MCU. The physical step requires the actual board
revision, connector routing and an identified MCU interface. No speculative UART
pinout or firmware flash command is supplied.

## 1. Prepare a Linux Klipper host on the Pi

Use a 64-bit Pi-compatible Linux installation, proper power and cooling. Install
Klipper using the [official installation instructions](https://www.klipper3d.org/Installation.html)
or [KIAUH](https://github.com/dw-0/kiauh). KIAUH can install the host service; choose
Klipper, with Moonraker/Mainsail optional for inspection. Package downloads can
require a network during setup; the selected runtime motion path is local.

Copy the updated controller repository onto the Pi. The new controller files in
this Windows working checkout must be included; cloning an older repository
revision will not include uncommitted work. Do not copy a Windows `.venv` to Linux.

From that copied repository on the Pi:

```bash
python3 -m venv .venv-controller
source .venv-controller/bin/activate
python -m pip install -r requirements-controller.txt
python -m controller --help
```

This prepares configuration/discovery and baseline software. Camera runtime and
CPU PyTorch installation are additional tasks before physical acquisition and
learned inference. The base requirements' RealSense wheel is Windows-specific;
it does not install the Linux camera runtime automatically.

## 2. Transfer the printer configuration and establish MCU access

First preserve the current `printer.cfg`, all included configuration files,
required SV08 Klipper extras, firmware versions and recovery files. The Pi's
Klippy version, extras and MCU firmware must be compatible. Do not replace them
with an arbitrary generic CoreXY configuration.

After the physical MCU links have been verified and connected to the Pi, use:

```bash
ls -l /dev/serial/by-id/
lsusb -t
```

Identify the mainboard and toolhead individually. The example layout in the
Mainline configuration uses `[mcu]` and `[mcu extra_mcu]`; actual identities come
from your machine. A conceptual configuration is:

```ini
[mcu]
serial: /dev/serial/by-id/<verified-mainboard-device>

[mcu extra_mcu]
serial: /dev/serial/by-id/<verified-toolhead-device>
```

These are placeholders, not usable serial names. For a verified UART deployment,
serial paths and firmware interface settings would differ. Do not assume USB
enumeration is available before establishing the physical route.

Only one Klippy host may own the printer MCU links. The onboard instance must be
stopped/removed from those links as part of a reviewed migration, rather than
leaving two hosts competing. Test identity, configuration and readiness before
homing, heating or sending any motion.

## 3. Find Klippy's local API socket

Run on the Pi:

```bash
systemctl cat klipper
systemctl status klipper --no-pager
```

If the unit uses an `EnvironmentFile`, inspect the exact file listed there too.
Look for the Klippy `-a` option. It selects the local API socket, for example:

```text
-a /home/pi/printer_data/comms/klippy.sock
```

The username and path can differ. Verify the path rather than copying an example
from the old SV08 host. If the installed service has no API socket option, add
`-a <absolute-socket-path>` to its existing Klippy arguments while preserving its
configuration, log and other arguments, then restart that one service. Do not
launch a second Klippy instance manually. The socket's parent directory must
exist, and the controller user needs the appropriate service-user/group access.

Klipper's [API documentation](https://www.klipper3d.org/API_Server.html) defines
this local connection. It is distinct from the MCU serial interface.

## 4. Point the controller at that socket

Create a local controller config without overwriting an existing one:

```bash
cp -n configs/controller.example.json configs/controller.json
```

Edit these fields inside its existing `hardware` object, using the verified path:

```json
"backend": "klipper_socket",
"klipper_socket_path": "/home/pi/printer_data/comms/klippy.sock"
```

Leave `substrate_height_mm` null for discovery. No Moonraker URL or API key is
used by this backend.

## 5. Check the software connection without moving the printer

From the controller checkout and activated environment:

```bash
python -m controller discover \
  --config configs/controller.json \
  --output output/pi5_klipper_discovery_01
```

Choose a new output directory on each run. Inspect `discovery.json` for:

- `server.backend` equals `klipper_socket` and `network_required` equals `false`.
- `printer.state` and the queried `webhooks.state` are `ready`.
- Correct printer/config/version identity and the required toolhead, motion,
  gcode, heater and configuration objects.

If the API responds but reports `startup`, `error` or `shutdown`, the software
connection works but the printer is not ready for control. Diagnose the saved
state message and Klippy log. An API socket connection alone does not prove
correct MCU wiring, identity or firmware compatibility.

For an independent read-only API check, Klipper supplies `whconsole.py`:

```bash
python3 ~/klipper/scripts/whconsole.py /home/pi/printer_data/comms/klippy.sock
```

At its input, submit one JSON line:

```json
{"id":1,"method":"info","params":{}}
```

Use the actual installation and socket paths. This query sends no G-code.

## 6. Diagnose connection failures

| Result | Next check |
|---|---|
| Socket missing | Klippy service, its `-a` argument, parent directory and actual path |
| Permission denied | Controller user versus installed socket owner/group |
| Socket replies but printer not ready | Klippy log, MCU paths, firmware compatibility, required extras and configuration |
| Pi has no MCU devices | Physical route/interface first; the local API does not create USB devices |
| MCU protocol/version mismatch | Versioned host/firmware migration; preserve recovery files before flashing |

After discovery succeeds and the physical configuration has been validated,
follow `HARDWARE_SETUP.md` for homing/frame calibration, substrate datum,
fixture-envelope clearance and stationary cold trials. Do not use `dry-run
--execute` merely to test whether an unknown connector controls the printer.

To finalize exact Pi-to-SV08 wiring, obtain a clear image of the actual motherboard
revision and connector labels, plus the current mainboard/toolhead MCU config
sections. Those are the remaining machine-specific inputs. This procedure fully
specifies the local software connection; it deliberately leaves unverified
physical rewiring open rather than presenting it as a completed conversion.
