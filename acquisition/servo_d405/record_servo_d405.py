#!/usr/bin/env python3
"""Record D405 RGB/depth sequences with the original sweep or 25 servo patterns."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import queue
import shutil
import signal
import subprocess
import threading
import time

import numpy as np
import pyrealsense2 as rs

from dataset_writer import ProcessDatasetWriter

from gobilda_servo import (
    GPIO18_CHANNEL, SPEED_DEG_PER_SEC, GoBildaServo, HardwarePWM, run_sweep,
)
from servo_patterns import PATTERNS, ORIGINAL_PATTERNS, NEW_PATTERNS, build_trajectory, resolve_pattern
from servo_motion import MotionCancelled, PatternMotion


class StopMotion(Exception):
    """Normal cancellation of the original sweep at its next PWM update."""


class LoggedServo(GoBildaServo):
    """Use the original motion implementation, with cancellable, logged commands."""

    def __init__(self, pwm, stop, command_file):
        super().__init__(pwm)
        self.stop_event = stop
        self.lock = threading.Lock()
        self.history = []
        self.origin_ns = None
        self.commands = csv.writer(command_file)
        self.commands.writerow([
            "host_monotonic_ns", "host_unix_ns", "event", "commanded_angle_deg", "pulse_us", "elapsed_s",
        ])
        if self.angle is not None:
            self._log("initial_pwm_state", self.pwm.get_duty_ns() / 1000)

    def _log(self, event, pulse):
        ns = time.monotonic_ns()
        elapsed = (ns - self.origin_ns) / 1e9 if self.origin_ns is not None else ""
        self.commands.writerow([ns, time.time_ns(), event, self.angle, pulse, elapsed])
        if self.angle is not None:
            self.history.append((ns, self.angle, pulse))

    def set_pulse_us(self, us):
        with self.lock:
            if self.stop_event.is_set():
                raise StopMotion()
            pulse = super().set_pulse_us(us)
            self._log("command", pulse)
            return pulse

    def snapshot_at(self, host_ns):
        with self.lock:
            for ns, angle, pulse in reversed(self.history):
                if ns <= host_ns:
                    return ns, angle, pulse
        return "", "", ""

    def release(self):
        # Serialize release with PWM writes, so cancellation cannot re-enable it.
        with self.lock:
            super().release()
            self._log("released", "")


FRAME_COLUMNS = [
    "index", "color_file", "depth_file", "elapsed_s", "host_monotonic_ns", "host_unix_ns",
    "color_frame_number", "depth_frame_number", "color_timestamp_ms", "depth_timestamp_ms",
    "color_timestamp_domain", "depth_timestamp_domain", "color_depth_skew_ms",
    "last_command_monotonic_ns", "commanded_angle_at_receive_deg", "commanded_pulse_at_receive_us",
    "color_sha256", "depth_sha256",
]


class DatasetWriter(ProcessDatasetWriter):
    """Isolate PNG compression and file I/O from the servo's Python thread."""

    def __init__(self, output, queue_size=32, **kwargs):
        super().__init__(output, FRAME_COLUMNS, queue_size, **kwargs)


def stream_calibration(profile):
    video = profile.as_video_stream_profile()
    intrinsics = video.get_intrinsics()
    return {
        "width": intrinsics.width, "height": intrinsics.height, "fps": video.fps(),
        "format": str(video.format()), "fx": intrinsics.fx, "fy": intrinsics.fy,
        "ppx": intrinsics.ppx, "ppy": intrinsics.ppy,
        "distortion_model": str(intrinsics.model), "coefficients": list(intrinsics.coeffs),
    }


def camera_calibration(profile):
    device = profile.get_device()
    depth = profile.get_stream(rs.stream.depth)
    color = profile.get_stream(rs.stream.color)
    extrinsics = depth.get_extrinsics_to(color)
    info = {}
    for key in ("name", "serial_number", "firmware_version", "usb_type_descriptor"):
        field = getattr(rs.camera_info, key)
        if device.supports(field):
            info[key] = device.get_info(field)
    return {
        **info, "depth_scale_m_per_unit": device.first_depth_sensor().get_depth_scale(),
        "color": stream_calibration(color), "depth": stream_calibration(depth),
        "depth_to_color_extrinsics": {
            "rotation_column_major": list(extrinsics.rotation),
            "translation_m": list(extrinsics.translation),
        },
    }


def write_manifest(output, data):
    temporary = output / "session.json.tmp"
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(output / "session.json")


def positive(text):
    value = float(text)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be finite and greater than zero")
    return value


def wait_for_frames(pipeline, timeout_ms, stop):
    """Poll without blocking Python's servo thread inside the camera SDK."""
    deadline = time.monotonic() + timeout_ms / 1000
    while not stop.is_set():
        frames = pipeline.poll_for_frames()
        if frames:
            return frames
        if time.monotonic() >= deadline:
            raise RuntimeError(f"D405 frame timeout after {timeout_ms} ms")
        if stop.wait(0.002):
            break
    raise StopMotion()


def parse_args(argv=None, default_pattern=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=positive, default=30.0, help="recording seconds (default: 30)")
    parser.add_argument("--output", type=Path, help="new session directory (default: recordings/<UTC time>)")
    parser.add_argument("--serial", help="D405 serial number; otherwise select the only attached D405")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, choices=(5, 15, 30), default=30, help="camera FPS (default: 30)")
    parser.add_argument("--storage-format", choices=("png", "ffv1"), help="lossless storage (default: ffv1 at 30 FPS, png otherwise)")
    parser.add_argument("--speed", type=positive, default=SPEED_DEG_PER_SEC, help="servo degrees/second")
    parser.add_argument("--start", type=float, help="sweep start / pattern lower bound (default: 0 / 30)")
    parser.add_argument("--end", type=float, help="sweep end / pattern upper bound (default: 300 / 270)")
    parser.add_argument("--pause", type=float, default=0.5, help="seconds at each end")
    parser.add_argument("--chip", help="override PWM controller path")
    parser.add_argument("--channel", type=int, default=GPIO18_CHANNEL)
    parser.add_argument("--camera-only", action="store_true", help="test recording without operating the servo")

    def pattern_name(value):
        try:
            return resolve_pattern(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(str(exc)) from exc

    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--pattern", type=pattern_name, help="pattern number (1-25) or name")
    selection.add_argument("--patterns", nargs="+", help="record a batch: original (1-20), new (21-25), all, or numbers/names")
    parser.add_argument("--seed", type=int, default=42, help="random waypoint seed (default: 42)")
    parser.add_argument("--list-patterns", action="store_true", help="list patterns without touching hardware")
    parser.add_argument("--dry-run", action="store_true", help="save planned trajectories without operating hardware")
    parser.add_argument("--resume", action="store_true", help="resume a batch in an existing --output; retain verified complete sessions")
    args = parser.parse_args(argv)
    if default_pattern and not args.pattern and not args.patterns:
        args.pattern = resolve_pattern(default_pattern)
    if args.patterns:
        try:
            groups = {"all": tuple(PATTERNS), "original": ORIGINAL_PATTERNS, "new": NEW_PATTERNS}
            args.patterns = list(groups[args.patterns[0]]) if len(args.patterns) == 1 and args.patterns[0] in groups else [resolve_pattern(v) for v in args.patterns]
        except ValueError as exc:
            parser.error(str(exc))
        if len(set(args.patterns)) != len(args.patterns):
            parser.error("each pattern may appear only once in a batch")
    use_patterns = args.pattern or args.patterns
    if args.storage_format is None:
        args.storage_format = "ffv1" if args.fps == 30 else "png"
    if args.start is None:
        args.start = 30.0 if use_patterns else 0.0
    if args.end is None:
        args.end = 270.0 if use_patterns else 300.0
    if any(not math.isfinite(a) or not 0 <= a <= 300 for a in (args.start, args.end)):
        parser.error("sweep angles must be between 0 and 300 degrees")
    if not math.isfinite(args.pause) or args.pause < 0:
        parser.error("pause must be finite and nonnegative")
    if args.width <= 0 or args.height <= 0 or args.channel < 0:
        parser.error("width/height must be positive and channel nonnegative")
    if use_patterns and args.start >= args.end:
        parser.error("patterns require --start < --end")
    if args.camera_only and use_patterns:
        parser.error("--camera-only cannot be combined with servo patterns")
    if args.dry_run and not use_patterns and not args.list_patterns:
        parser.error("--dry-run requires --pattern or --patterns")
    if args.resume and (not args.patterns or not args.output or args.dry_run):
        parser.error("--resume requires --patterns and --output, without --dry-run")
    return args


def record(args):
    output = (args.output or Path("recordings") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")).resolve()
    output.mkdir(parents=True, exist_ok=False)
    trajectory = build_trajectory(args.pattern, args.duration, args.start, args.end, args.speed, args.seed) if args.pattern else None
    if trajectory:
        trajectory.write_csv(output)
    stop = threading.Event()
    signal_reason = []

    def request_stop(signum, frame):
        signal_reason.append(signal.Signals(signum).name)
        stop.set()

    previous_signals = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    pipeline = rs.pipeline()
    streaming = False
    writer = servo = motor = command_file = None
    motor_errors = queue.SimpleQueue()
    motion_complete = threading.Event()
    start_ns = end_ns = None
    status, reason = "failed", "initializing"
    captured = 0
    gaps = {"color": 0, "depth": 0}
    duplicate_pairs = 0
    last_numbers = {}
    failure = None
    manifest = {
        "schema_version": 1, "status": "initializing", "requested_duration_s": args.duration,
        "created_utc": datetime.now(timezone.utc).isoformat(), "output": str(output),
        "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "storage": {
            "color": "lossless RGB uint8 PNG", "depth": "lossless native uint16 Z16 PNG",
            "depth_meters": "depth_pixel * camera.depth_scale_m_per_unit; zero means invalid",
            "alignment": "native frames, no alignment or depth filtering; calibration stored for offline alignment",
        },
        "timing": {
            "host_monotonic_ns": "host receipt time before conversion/storage, shared clock with servo_commands.csv",
            "host_unix_ns": "host wall clock at receipt; camera timestamps retain their recorded domains",
            "servo_labels": "commanded PWM position at host receipt, not measured angle or an exposure-synchronized label",
            "synchronization": "software concurrent capture and PWM; no hardware trigger",
        },
        "software": {name: importlib.metadata.version(name) for name in ("pyrealsense2", "numpy", "Pillow")},
        "servo_script_sha256": hashlib.sha256(Path(__file__).with_name("gobilda_servo.py").read_bytes()).hexdigest(),
        "recorder_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "writer_script_sha256": hashlib.sha256(Path(__file__).with_name("dataset_writer.py").read_bytes()).hexdigest(),
    }
    manifest["storage"]["format"] = args.storage_format
    if args.storage_format == "ffv1":
        manifest["schema_version"] = 2
        manifest["storage"].update(
            color="lossless RGB uint8 FFV1 in color.mkv (bgr0 coding)",
            depth="lossless native uint16 FFV1 in depth.mkv (gray16le coding)",
            video_frame_index="frames.csv index is the zero-based frame in both videos",
            video_timing="container PTS=index/fps for playback; true acquisition times and gaps are in frames.csv",
            pixel_checksums="SHA-256 of decoded rgb24 / gray16le pixel bytes for every frame",
            buffering="compressed videos and frame rows buffered in RAM during capture; 512 MiB compressed budget; persisted after PWM stops",
        )
    if trajectory:
        manifest["trajectory"] = trajectory.metadata()
        manifest["pattern_script_sha256"] = hashlib.sha256(Path(__file__).with_name("servo_patterns.py").read_bytes()).hexdigest()
        manifest["motion_script_sha256"] = hashlib.sha256(Path(__file__).with_name("servo_motion.py").read_bytes()).hexdigest()
        manifest["timing"]["trajectory"] = "trajectory.csv offsets share the session start_monotonic_ns origin; initial positioning precedes this origin"
        manifest["timing"]["motion_process"] = "PWM runs in a separate process; shared host monotonic command history labels camera frames; CSV journal is written after PWM release"
    write_manifest(output, manifest)
    print(f"Session: {output}", flush=True)
    try:
        if shutil.disk_usage(output).free < 1024**3:
            raise RuntimeError("Less than 1 GiB free; choose a different output drive.")
        if args.storage_format == "ffv1":
            if not shutil.which("ffmpeg"):
                raise RuntimeError("FFV1 recording requires ffmpeg")
            manifest["software"]["ffmpeg"] = subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0]
        context = rs.context()
        devices = [d for d in context.query_devices() if "D405" in d.get_info(rs.camera_info.name)]
        if args.serial:
            devices = [d for d in devices if d.get_info(rs.camera_info.serial_number) == args.serial]
        if len(devices) != 1:
            raise RuntimeError(f"Expected one selected D405, found {len(devices)}. Check USB or use --serial.")
        config = rs.config()
        config.enable_device(devices[0].get_info(rs.camera_info.serial_number))
        config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
        config.enable_stream(rs.stream.color, args.width, args.height, rs.format.rgb8, args.fps)
        profile = pipeline.start(config)
        streaming = True
        manifest["camera"] = camera_calibration(profile)
        print(f"D405: {args.width}x{args.height} at {args.fps} FPS; warming up...", flush=True)
        # Confirm actual paired frames before exporting/enabling any servo PWM.
        for _ in range(args.fps):
            if stop.is_set():
                raise StopMotion()
            frames = wait_for_frames(pipeline, 3000, stop)
            if not frames.get_color_frame() or not frames.get_depth_frame():
                raise RuntimeError("D405 did not return both color and depth.")
        # Buffer one standard sequence, capped at 256 MiB of raw image pairs.
        pair_bytes = args.width * args.height * 5
        memory_limit = max(1, (256 * 1024**2) // pair_bytes)
        queue_size = min(memory_limit, max(32, math.ceil(args.fps * args.duration) + 2))
        writer = DatasetWriter(output, queue_size, storage_format=args.storage_format,
                               width=args.width, height=args.height, fps=args.fps)
        manifest["storage"]["writer_queue_capacity_pairs"] = queue_size
        if trajectory:
            print(f"Pattern: {trajectory.pattern}; positioning to {trajectory.samples[0][1]:.1f} degrees...", flush=True)
            servo = PatternMotion(args, trajectory, output / "servo_commands.csv")
            servo.prepare(stop)
            motion_complete = servo.done
        elif not args.camera_only:
            command_file = (output / "servo_commands.csv").open("x", newline="", buffering=1)
            pwm = HardwarePWM(channel=args.channel, chip=args.chip)
            try:
                servo = LoggedServo(pwm, stop, command_file)
            except BaseException:
                pwm.release()
                raise

        def sweep():
            try:
                run_sweep(servo, args.speed, args.start, args.end, times=0, pause=args.pause)
            except StopMotion:
                pass
            except BaseException as exc:
                motor_errors.put(exc)
                stop.set()

        # Complete regular-file writes before assigning the trajectory clock origin.
        manifest.update(status="recording")
        write_manifest(output, manifest)
        start_ns = time.monotonic_ns()
        if servo and not trajectory:
            servo.origin_ns = start_ns
        manifest.update(status="recording", started_utc=datetime.now(timezone.utc).isoformat(), start_monotonic_ns=start_ns)
        # The camera is already producing verified frames when motion begins.
        if trajectory:
            servo.start(start_ns)
        elif servo:
            motor = threading.Thread(target=sweep, name="original-servo-sweep")
            motor.start()
        next_report = start_ns
        deadline_ns = start_ns + round(args.duration * 1e9)
        while not stop.is_set():
            now_ns = time.monotonic_ns()
            if trajectory and servo.error:
                raise RuntimeError(f"Servo worker failed: {servo.error}")
            if now_ns >= deadline_ns:
                break
            batch_stop = getattr(args, "batch_stop_file", None)
            if (output / "STOP").exists() or (batch_stop and batch_stop.exists()):
                signal_reason.append("STOP file")
                stop.set()
                break
            frames = wait_for_frames(pipeline, 1000, stop)
            host_ns, unix_ns = time.monotonic_ns(), time.time_ns()
            if stop.is_set() or host_ns >= deadline_ns:
                break
            color, depth = frames.get_color_frame(), frames.get_depth_frame()
            if not color or not depth:
                raise RuntimeError("D405 returned an incomplete color/depth pair.")
            numbers = {"color": color.get_frame_number(), "depth": depth.get_frame_number()}
            if any(key in last_numbers and value <= last_numbers[key] for key, value in numbers.items()):
                duplicate_pairs += 1
                continue
            for key, value in numbers.items():
                if key in last_numbers:
                    gaps[key] += max(0, value - last_numbers[key] - 1)
            last_numbers = numbers
            command_ns, angle, pulse = servo.snapshot_at(host_ns) if servo else ("", "", "")
            color_file, depth_file = (("color.mkv", "depth.mkv") if args.storage_format == "ffv1"
                                     else (f"color/{captured:06d}.png", f"depth/{captured:06d}.png"))
            row = dict(zip(FRAME_COLUMNS, [
                captured, color_file, depth_file,
                (host_ns - start_ns) / 1e9, host_ns, unix_ns, numbers["color"], numbers["depth"],
                color.get_timestamp(), depth.get_timestamp(), str(color.get_frame_timestamp_domain()),
                str(depth.get_frame_timestamp_domain()), color.get_timestamp() - depth.get_timestamp(),
                command_ns, angle, pulse,
            ]))
            writer.submit(row, np.asanyarray(color.get_data()), np.asanyarray(depth.get_data()))
            captured += 1
            if host_ns >= next_report:
                if shutil.disk_usage(output).free < 1024**3:
                    raise RuntimeError("Free disk space reached 1 GiB; recording stopped.")
                action = "encoded" if args.storage_format == "ffv1" else "saved"
                print(f"{row['elapsed_s']:5.1f}s / {args.duration:g}s: {captured} pairs captured, {writer.saved} {action}", flush=True)
                next_report = host_ns + 5_000_000_000
        if not motor_errors.empty():
            raise RuntimeError(f"Servo worker failed: {motor_errors.get()}")
        if trajectory and servo.error:
            raise RuntimeError(f"Servo worker failed: {servo.error}")
        if trajectory and not stop.is_set() and not motion_complete.is_set():
            raise RuntimeError("Pattern did not finish before the recording deadline")
        status = "interrupted" if stop.is_set() else "complete"
        reason = signal_reason[-1] if signal_reason else "duration reached"
    except (StopMotion, MotionCancelled):
        status, reason = "interrupted", signal_reason[-1] if signal_reason else "stopped during initialization"
    except BaseException as exc:
        failure = exc
        reason = f"{type(exc).__name__}: {exc}"
    finally:
        end_ns = time.monotonic_ns()
        stop.set()
        # Disable PWM promptly even if the sweep is currently in an endpoint pause.
        cleanup_errors = []
        if servo:
            try:
                servo.release()
            except BaseException as exc:
                cleanup_errors.append(f"PWM release: {exc}")
        if motor:
            motor.join()
        if not motor_errors.empty():
            status = "failed"
            reason += f"; Servo worker failed: {motor_errors.get()}"
        if command_file:
            command_file.close()
        if streaming:
            try:
                pipeline.stop()
            except BaseException as exc:
                cleanup_errors.append(f"Camera stop: {exc}")
        if writer:
            try:
                writer.close()
            except BaseException as exc:
                cleanup_errors.append(str(exc))
        if cleanup_errors:
            status = "failed"
            reason += "; " + "; ".join(cleanup_errors)
        manifest.update(
            status=status, stop_reason=reason, finished_utc=datetime.now(timezone.utc).isoformat(),
            end_monotonic_ns=end_ns, actual_capture_duration_s=(end_ns - start_ns) / 1e9 if start_ns else 0,
            captured_pairs=captured, saved_pairs=writer.saved if writer else 0,
            frame_number_gaps=gaps, skipped_duplicate_pairs=duplicate_pairs, cleanup_errors=cleanup_errors,
        )
        if trajectory:
            manifest["trajectory_completed"] = motion_complete.is_set()
        write_manifest(output, manifest)
        for sig, handler in previous_signals.items():
            signal.signal(sig, handler)
    print(f"{status}: saved {manifest['saved_pairs']} color/depth pairs to {output}", flush=True)
    if failure:
        raise failure
    if status == "failed":
        raise RuntimeError(reason)
    return output


def _write_batch_manifest(output, manifest):
    temporary = output / "batch.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(output / "batch.json")
    with (output / "sequences.csv").open("w", newline="") as handle:
        columns = ["pattern", "directory", "status", "saved_pairs", "actual_capture_duration_s"]
        rows = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        rows.writeheader()
        rows.writerows(manifest["sequences"])


def record_batch(args):
    """Run patterns sequentially and stop the batch on failure or cancellation."""
    prefix = "planned_patterns_" if args.dry_run else "patterns_"
    output = (args.output or Path("recordings") / (prefix + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))).resolve()
    if args.resume:
        from verify_servo_patterns import verify_sequence
        manifest = json.loads((output / "batch.json").read_text())
        for field, expected in (("requested_patterns", args.patterns), ("duration_per_pattern_s", args.duration),
                                ("seed", args.seed), ("fps", args.fps), ("storage_format", args.storage_format)):
            if manifest.get(field) != expected:
                raise ValueError(f"Cannot resume: {field} differs from the original batch")
        if manifest.get("error"):
            manifest.setdefault("previous_failures", []).append({"error": manifest.pop("error"), "finished_utc": manifest.get("finished_utc")})
        manifest.update(status="recording", resumed_utc=datetime.now(timezone.utc).isoformat())
    else:
        output.mkdir(parents=True, exist_ok=False)
        manifest = {
        "schema_version": 1, "status": "planning" if args.dry_run else "recording",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "requested_patterns": args.patterns, "duration_per_pattern_s": args.duration,
        "seed": args.seed, "sequences": [],
        "fps": args.fps, "storage_format": args.storage_format,
        }
    _write_batch_manifest(output, manifest)
    names = ("gobilda_servo.py", "servo_patterns.py", "servo_motion.py", "record_servo_d405.py", "dataset_writer.py", "requirements-camera.txt")
    revision = hashlib.sha256(b"".join(Path(__file__).with_name(name).read_bytes() for name in names)).hexdigest()[:12]
    source_name = f"source_{revision}" if args.resume else "source"
    source = output / source_name
    if not source.exists():
        source.mkdir()
        for name in names:
            shutil.copy2(Path(__file__).with_name(name), source / name)
    print(f"Batch: {output}", flush=True)
    try:
        for index, pattern in enumerate(args.patterns, 1):
            if (output / "STOP").exists():
                manifest["status"] = "interrupted"
                break
            child = argparse.Namespace(**vars(args))
            child.pattern, child.patterns = pattern, None
            child.output = output / pattern
            child.batch_stop_file = output / "STOP"
            entry = next((row for row in manifest["sequences"] if row["pattern"] == pattern), None)
            if args.resume and child.output.exists():
                session_path = child.output / "session.json"
                session = json.loads(session_path.read_text()) if session_path.exists() else {}
                if session.get("status") == "complete":
                    for field in ("duration", "fps", "storage_format", "width", "height", "speed", "start", "end", "seed", "channel", "chip", "serial"):
                        if session["settings"].get(field) != getattr(args, field):
                            raise ValueError(f"Cannot reuse {pattern}: {field} differs")
                    verify_sequence(child.output)
                    print(f"Keeping complete pattern {index}/{len(args.patterns)}: {pattern}", flush=True)
                    if entry:
                        entry.update(status="complete", saved_pairs=session["saved_pairs"], actual_capture_duration_s=session["actual_capture_duration_s"])
                    continue
                trials = output / "trials"
                trials.mkdir(exist_ok=True)
                archived = trials / (pattern + "_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
                child.output.rename(archived)
                if entry:
                    entry.setdefault("previous_attempts", []).append({"directory": str(archived.relative_to(output)), "status": session.get("status", "unknown")})
            if entry is None:
                entry = {"pattern": pattern, "directory": pattern}
                manifest["sequences"].append(entry)
            for field in ("saved_pairs", "actual_capture_duration_s", "stop_reason"):
                entry.pop(field, None)
            entry.update(status="initializing", source_snapshot=source_name)
            _write_batch_manifest(output, manifest)
            print(f"\nPattern {index}/{len(args.patterns)}: {pattern}", flush=True)
            try:
                if args.dry_run:
                    plan_only(child)
                else:
                    record(child)
            finally:
                session_file = child.output / "session.json"
                if session_file.exists():
                    session = json.loads(session_file.read_text())
                    for key in ("status", "saved_pairs", "actual_capture_duration_s", "stop_reason"):
                        if key in session:
                            entry[key] = session[key]
                _write_batch_manifest(output, manifest)
            if entry["status"] == "interrupted":
                manifest["status"] = "interrupted"
                break
        else:
            manifest["status"] = "planned" if args.dry_run else "complete"
    except KeyboardInterrupt:
        manifest["status"] = "interrupted"
    except BaseException as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        _write_batch_manifest(output, manifest)
    print(f"Batch {manifest['status']}: {len(manifest['sequences'])}/{len(args.patterns)} patterns; {output}", flush=True)
    return output


def plan_only(args):
    output = (args.output or Path("recordings") / ("planned_" + args.pattern + "_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))).resolve()
    trajectory = build_trajectory(args.pattern, args.duration, args.start, args.end, args.speed, args.seed)
    output.mkdir(parents=True, exist_ok=False)
    trajectory.write_csv(output)
    write_manifest(output, {"status": "planned", "requested_duration_s": args.duration,
                            "trajectory": trajectory.metadata(), "saved_pairs": 0})
    print(f"Planned: {output}", flush=True)
    return output


def main(argv=None, default_pattern=None):
    args = parse_args(argv, default_pattern)
    if args.list_patterns:
        for key, pattern in PATTERNS.items():
            print(f"{key:30} {pattern.description}")
        return
    if args.patterns:
        return record_batch(args)
    if args.dry_run:
        return plan_only(args)
    return record(args)


if __name__ == "__main__":
    main()
