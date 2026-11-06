# Servo + D405 acquisition

This is the code that recorded `Dataset/`. Run it on the Raspberry Pi connected to the
D405 and the goBILDA servo (GPIO18, 50 Hz PWM). These files are copied from
`Dataset/Test/source/` (5 Nov 2026), the newest version, which also recorded
`Test_groundTruth`. The training collection
`Dataset/recordings/servo_patterns_20_30fps_20261105/source/` holds the slightly earlier
snapshot that was actually used for those 20 recordings; it lacks `--resume` and RAM
buffering. Keep each dataset's own `source/` snapshot as its provenance.

| File | Role |
| --- | --- |
| `record_servo_d405.py` | Main recorder: camera warm-up, paired RGB/depth capture, servo playback, batch mode |
| `servo_patterns.py` | The 25 motion patterns (1–20 training set, 21–25 test set) and trajectory sampling |
| `servo_motion.py`, `gobilda_servo.py` | PWM servo control |
| `dataset_writer.py` | Lossless FFV1 (30 FPS) or PNG writer with per-frame SHA-256 |
| `requirements-camera.txt` | Pi-side dependencies |

```bash
python record_servo_d405.py --list-patterns
python record_servo_d405.py --patterns original --fps 30 --duration 30 --output <new folder>
python record_servo_d405.py --patterns new --fps 30 --duration 30 --output <new folder>   # 21–25
```

Each pattern folder contains `color.mkv`, `depth.mkv`, `frames.csv` (camera timestamps,
frame numbers, host times, commanded servo angle), `trajectory.csv`,
`servo_commands.csv` and `session.json` (intrinsics/extrinsics, depth scale, settings).
Servo labels are commanded positions at host receipt, not measured positions.

`--resume` imports `verify_servo_patterns`, which is not part of this snapshot. Resuming
needs that script from the Pi.
