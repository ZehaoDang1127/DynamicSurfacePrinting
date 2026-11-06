"""Pi commands; only live-print --execute can enable hardware actuation."""

import argparse
from datetime import datetime
import importlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = "profiles/stationary/controller.json"
DEFAULT_PLAN = "profiles/stationary/plan.json"


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor", help="check the environment; optionally enumerate cameras")
    doctor.add_argument("--camera", action="store_true")
    doctor.add_argument("--convlstm", action="store_true", help="also check CPU PyTorch")
    commands.add_parser("self-test", help="run synthetic acquisition, prediction and offline previews")
    commands.add_parser("live-self-test", help="test the closed-loop software using simulated hardware")
    for name in ("live-check", "live-shadow", "live-print"):
        sub = commands.add_parser(name, help="check or run experimental segmented ConvLSTM control")
        sub.add_argument("--config", type=Path, default=Path("profiles/live/controller.json"))
        sub.add_argument("--camera-config", type=Path, default=Path("profiles/live/camera.json"))
        sub.add_argument("--path", type=Path, default=Path("profiles/live/path.json"))
        sub.add_argument("--settings", type=Path, default=Path("profiles/live/settings.json"))
        sub.add_argument("--output", type=Path)
        if name == "live-print":
            sub.add_argument("--execute", action="store_true", help="enable motion/heating/extrusion after measured setup checks")
    for name in ("status", "diagnose", "preview"):
        sub = commands.add_parser(name)
        sub.add_argument("--config", type=Path, default=Path(DEFAULT_CONFIG))
        if name == "diagnose":
            sub.add_argument("--url", help="override the Moonraker URL in the profile")
            sub.add_argument("--session", type=Path, help="old failed execution folder; absolute path recommended")
            sub.add_argument("--offline", action="store_true")
        else:
            sub.add_argument("--output", type=Path)
        if name == "preview":
            sub.add_argument("--plan", type=Path, default=Path(DEFAULT_PLAN))
    for name, frames in (("capture", 90), ("baseline", 300)):
        sub = commands.add_parser(name, help="use the D405 without printer control")
        sub.add_argument("--camera-config", type=Path, default=Path("configs/d405_camera_frame.json"))
        sub.add_argument("--frames", type=int, default=frames, help="0 runs until Ctrl+C")
        sub.add_argument("--output", type=Path)
    return p


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def path_in_root(path, root):
    return path if path.is_absolute() else root / path


def fresh_output(root, name):
    return root / "output" / ("pi_" + name + "_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))


def command_for(args, root=ROOT):
    """Build argument vectors without a shell; actuation is an explicit option."""
    prefix = [sys.executable]
    if args.command == "live-self-test":
        return prefix + ["-m", "unittest", "discover", "-s", "tests", "-p", "test_live_print.py", "-v"]
    if args.command in {"live-check", "live-shadow", "live-print"}:
        action = {"live-check": "check", "live-shadow": "shadow", "live-print": "run"}[args.command]
        output = path_in_root(args.output, root) if args.output else fresh_output(root, args.command)
        command = prefix + ["-m", "controller.live_print", action, "--output", str(output)]
        for flag, value in (("--config", args.config), ("--camera-config", args.camera_config),
                            ("--path", args.path), ("--settings", args.settings)):
            command += [flag, str(path_in_root(value, root))]
        if getattr(args, "execute", False):
            command.append("--execute")
        return command
    if args.command == "diagnose":
        command = prefix + [str(root / "tools/collect_stationary_failure.py")]
        if args.offline:
            command.append("--offline")
        else:
            url = args.url
            if not url:
                url = read_json(path_in_root(args.config, root))["hardware"]["moonraker_url"]
            if not isinstance(url, str) or not url.startswith(("http://", "https://")):
                raise ValueError("Moonraker URL is missing; set hardware.moonraker_url or pass --url")
            command += ["--url", url]
        if args.session is not None:
            session = path_in_root(args.session, root).resolve()
            records = ("execution.json", "job.json", "execution_events.jsonl", "last_status_snapshot.json")
            if not session.is_dir():
                raise ValueError(f"--session must be an existing execution directory: {session}")
            if not any((session / name).is_file() for name in records):
                raise ValueError(f"--session contains no recognized execution records: {session}")
            command += ["--session", str(session)]
        return command
    output = path_in_root(args.output, root) if args.output else fresh_output(root, args.command)
    if args.command in {"status", "preview"}:
        command = prefix + ["-m", "controller.stationary", "status" if args.command == "status" else "run",
                            "--config", str(path_in_root(args.config, root)), "--output", str(output)]
        if args.command == "preview":
            command += ["--plan", str(path_in_root(args.plan, root))]
        return command
    if args.frames < 0:
        raise ValueError("--frames cannot be negative")
    camera = str(path_in_root(args.camera_config, root))
    if args.command == "capture":
        command = prefix + ["-m", "surface_mapper", "--config", camera, "--print-every", "30"]
    else:
        command = prefix + ["-m", "prediction", "baseline-live", "--camera-config", camera,
                            "--history", "5", "--record-input"]
    return command + ["--source", "realsense", "--frames", str(args.frames), "--output", str(output)]


def run_command(command, root=ROOT):
    return subprocess.run(command, cwd=root, check=False).returncode


def doctor(camera=False, convlstm=False):
    report = {"python": sys.version.split()[0], "executable": sys.executable,
              "system": platform.system(), "machine": platform.machine(),
              "bundle_root": str(ROOT), "dependencies": {}, "printer_commands_sent": 0}
    ok = True
    for name in ("numpy", "httpx") + (("pyrealsense2",) if camera else ()) + (("torch",) if convlstm else ()):
        try:
            module = importlib.import_module(name)
            report["dependencies"][name] = {"available": True, "version": getattr(module, "__version__", None)}
            if name == "torch":
                report["torch_cpu_result"] = float(module.ones(2, device="cpu").sum())
            if name == "pyrealsense2":
                context = module.context()
                devices = list(context.query_devices())
                report["cameras"] = [device.get_info(module.camera_info.name) for device in devices]
                if not devices:
                    ok = False
                    report["camera_note"] = "No camera found. Check USB connection, power, udev permissions and other camera processes."
        except (ImportError, OSError, RuntimeError) as exc:
            ok = False
            report["dependencies"][name] = {"available": False, "error": str(exc)}
    report["ok"] = ok
    report["target_platform"] = report["system"] == "Linux" and report["machine"] in {"aarch64", "arm64"}
    report["note"] = "Import/enumeration checks do not validate real acquisition or printer motion."
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if ok else 2


def self_test(root=ROOT):
    """Exercise actual offline workflows and verify their saved results."""
    out = fresh_output(root, "selftest")
    out.mkdir(parents=True, exist_ok=False)
    summary = {"ok": False, "output": str(out), "printer_commands_sent": 0,
               "scope": "offline synthetic software verification; no hardware validation", "checks": []}
    summary_file = out / "self_test.json"

    def save():
        summary_file.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    def invoke(name, argv):
        code = run_command([sys.executable, "-m", *map(str, argv)], root)
        summary["checks"].append({"name": name, "exit_code": code})
        save()
        if code != 0:
            raise RuntimeError(f"self-test {name} failed with exit code {code}; see {out}")

    save()
    try:
        camera = root / "configs/d405_camera_frame.json"
        invoke("capture", ["surface_mapper", "--source", "synthetic", "--config", camera,
                            "--frames", "12", "--print-every", "12", "--output", out / "capture"])
        capture = read_json(out / "capture/manifest.json")
        if not capture["complete"] or capture["frame_count"] != 12:
            raise RuntimeError("synthetic capture did not complete 12 frames")
        invoke("baseline", ["prediction", "baseline-live", "--source", "synthetic", "--camera-config", camera,
                             "--history", "5", "--frames", "12", "--record-input", "--output", out / "baseline"])
        forecasts = read_json(out / "baseline/manifest.json")
        if not forecasts["complete"] or forecasts["forecast_count"] < 1:
            raise RuntimeError("synthetic prediction did not produce forecasts")
        invoke("evaluation", ["prediction", "baseline-evaluate", "--session", out / "baseline/observations",
                               "--history", "5", "--output", out / "evaluation"])
        metrics = read_json(out / "evaluation/metrics.json")
        if metrics["windows"] < 1 or metrics["comparison_valid_pixels"] < 1:
            raise RuntimeError("baseline evaluation produced no scored support")
        virtual_path = out / "virtual_path.json"
        virtual_path.write_text(json.dumps({"kind": "line", "start_xy_mm": [110., 120.],
                                           "end_xy_mm": [112., 120.]}) + "\n", encoding="utf-8")
        invoke("simulation", ["controller", "simulate", "--scenario", "moving-plane", "--frames", "90",
                               "--config", root / "configs/controller.example.json",
                               "--path", virtual_path, "--output", out / "simulation"])
        simulation = read_json(out / "simulation/report.json")
        if simulation["state"] == "FAULT" or simulation["submitted_segments"] < 1 or not simulation["path_complete"]:
            raise RuntimeError("virtual controller did not complete its path")
        # A deliberate virtual fixture, independent of unmeasured physical readiness.
        from controller.stationary import template
        plan = template()
        plan.update(start_xyz_mm=[110., 120., 30.], travel_z_mm=35., nozzle_z_mm=25.,
                    line_start_xy_mm=[110., 120.], line_end_xy_mm=[130., 120.])
        plan_file = out / "virtual_plan.json"
        plan_file.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
        invoke("preview", ["controller.stationary", "run", "--config", root / "configs/controller.example.json",
                            "--plan", plan_file, "--output", out / "preview"])
        job = read_json(out / "preview/job.json")
        if job["line_length_mm"] != 20 or not (out / "preview" / job["filename"]).is_file():
            raise RuntimeError("stationary preview output is incomplete")
        summary.update(ok=True, captured_frames=capture["frame_count"], forecasts=forecasts["forecast_count"],
                       scored_windows=metrics["windows"], simulation_path_complete=simulation["path_complete"])
    except (ValueError, KeyError, OSError, RuntimeError) as exc:
        summary["error"] = str(exc)
        raise
    finally:
        save()
        print(json.dumps(summary, indent=2), flush=True)
    return 0


def main(argv=None):
    arguments = sys.argv[1:] if argv is None else argv
    args = parser().parse_args(arguments if arguments else ["--help"])
    os.chdir(ROOT)
    try:
        if args.command == "doctor":
            return doctor(args.camera, args.convlstm)
        if args.command == "self-test":
            return self_test()
        if args.command == "live-self-test" and doctor(convlstm=True) != 0:
            return 2
        return run_command(command_for(args))
    except (ValueError, KeyError, OSError, RuntimeError) as exc:
        print(f"Pi launcher: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
