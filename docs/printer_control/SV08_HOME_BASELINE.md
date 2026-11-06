# SV08 default homing baseline

Recorded 2026-10-17 from [Sovol's published SV08 `printer.cfg`](https://github.com/Sovol3d/SV08/blob/main/home/sovol/printer_data/config/printer.cfg). **This is the published stock configuration, not a reading from the connected printer.** The printer was not reachable from this workstation at the setup guide's example address (`192.168.50.2:7125`), and no local copy of its installed configuration was found. Confirm these values against the machine before changing its homing configuration.

| Item | Published stock value |
| --- | --- |
| X homing direction and endstop coordinate | Positive X; `position_endstop: 355` mm |
| Y homing direction and endstop coordinate | Positive Y; `position_endstop: 364` mm |
| Z homing reference | `probe:z_virtual_endstop`; no fixed `position_endstop` |
| Z probing location in the `G28` override | `X191 Y165` mm |
| Position after a full `G28` finishes | `X191 Y165 Z10` mm, as commanded by the published override |

The full `G28` override temporarily moves to `X348` after X homing and `Y360` after Y homing, then moves to `X191 Y165`, homes Z with the probe, and raises to `Z10`. The final position is not the physical X/Y endstop location. A `G92` work origin changes G-code coordinates; it does not change these homing references.

**Installed-machine baseline:** not yet verified. Read the printer's actual `printer.cfg` and record its active offsets and `GET_POSITION` output when the connection is available, before making a homing change.
