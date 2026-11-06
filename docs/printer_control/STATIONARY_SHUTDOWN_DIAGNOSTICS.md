# Stationary-job shutdown diagnosis — 26 October 2026

The user-reported run uploaded and requested a start, then raised
`printer position left the submitted stationary path` and successfully requested
an emergency stop. The user observed no obvious movement before it stopped.
A later `/printer/info` response reports `ready` with
`v0.12.0-0-g02eeceb-dirty`; that later state does not identify the earlier faulting
position or prove that the previous homing reference remains valid.

The stop request was issued by this controller. The screenshot alone does not
establish a mechanical fault, nor does it establish a false alarm. The previous
release did not save the full rejected status sample before validation raised.
That diagnostic omission has been corrected locally: preflight, before-start
and every monitor sample are now saved before checking, with reported XYZ,
nearest path position and distance. The 0.1 mm path limit and stop behavior have
not been relaxed. The updated software is not yet installed on the user's Pi.

One hypothesis is an unavailable/zero interpolated `motion_report.live_position`
at startup. Upstream Klipper can return an all-zero initialized position when
there is no applicable trap-queue entry. This is not confirmed for this incident.
The installed Sovol fork and the actual status values need checking.
References: [Klipper status semantics](https://www.klipper3d.org/Status_Reference.html#motion_report),
[Klipper v0.12.0 motion-report implementation](https://github.com/Klipper3d/klipper/blob/v0.12.0/klippy/extras/motion_report.py).

## Read-only collection

Keep the machine stopped. Preserve the failed output folder. Run in the Pi bundle
directory; these commands send no motion, heating or restart requests:

```bash
curl -sS --connect-timeout 3 --max-time 6 http://192.168.50.2:7125/printer/info
```

```bash
curl -sS --connect-timeout 3 --max-time 6 'http://192.168.50.2:7125/printer/objects/query?gcode_move=position,gcode_position,homing_origin&motion_report=live_position,live_velocity&toolhead=position,homed_axes&print_stats=state,filename&virtual_sdcard=file_position,is_active'
```

The standalone `tools/collect_stationary_failure.py` can also be copied into the
Pi bundle directory and run as `python collect_stationary_failure.py`. It uses
the Python standard library. It writes a new diagnostic folder, reads the last
started failed job and its event log, queries current status, and requests only
a bounded tail of the Klipper log. It makes HTTP GET requests only and performs
no automatic recovery. Current readings are labeled as post-incident data.

Do not resume the failed job or rerun the old hot trial to obtain more data.
Confirm the firmware state, actual printer pose and retained homing state before
planning recovery. A restart/shutdown can invalidate the physical coordinate
reference; the raised fixture must be clear of any subsequent normal homing path.
No automatic homing, forced-position recovery or threshold bypass is supplied.
