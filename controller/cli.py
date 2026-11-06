"""Controller entry points. Hardware motion always starts with a reviewed cold path."""

import argparse
import asyncio
from dataclasses import replace
import json
from pathlib import Path
import sys

from .config import fingerprint, load_config
from .path import load_path
from .types import ControllerFault


def parser():
    p = argparse.ArgumentParser(description="Dynamic surface controller: live shadow planning, virtual tracking and guarded cold characterization")
    commands = p.add_subparsers(dest="command", required=True)
    for name in ("simulate", "replay", "discover", "observe", "shadow", "dry-run", "print", "stop", "camera-hash"):
        sub = commands.add_parser(name)
        sub.add_argument("--config", default="configs/controller.example.json")
        if name in {"simulate", "replay", "shadow", "dry-run", "print"}:
            sub.add_argument("--path", default="configs/controller_path.example.json")
            sub.add_argument("--substrate-height-mm", type=float,
                             help="Explicit run-only substrate top Z; recorded in the resolved config")
        if name in {"simulate", "replay", "discover", "observe", "shadow", "dry-run"}:
            sub.add_argument("--output", required=True, help="New output directory; existing directories are refused")
        if name == "simulate":
            sub.add_argument("--scenario", choices=["moving-plane", "static-plane", "sloped-plane"], default="moving-plane")
            sub.add_argument("--frames", type=int, default=600)
            sub.add_argument("--fps", type=float, default=30)
            sub.add_argument("--amplitude-mm", type=float, default=.5)
            sub.add_argument("--frequency-hz", type=float, default=.1)
            sub.add_argument("--dropout-frame", type=int)
            sub.add_argument("--lost-reply-segment", type=int)
            sub.add_argument("--virtual-extrusion", action="store_true", help="Generate XYZ/E in virtual preview only")
        if name == "replay":
            sub.add_argument("--session", required=True)
        if name in {"observe", "shadow", "camera-hash"}:
            sub.add_argument("--camera-config", required=True)
        if name in {"observe", "shadow"}:
            sub.add_argument("--frames", type=int, default=300)
        if name == "dry-run":
            sub.add_argument("--execute", action="store_true", help="Execute the reviewed elevated path on a stationary cold fixture")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        config = load_config(args.config)
        if getattr(args, "substrate_height_mm", None) is not None:
            config = replace(config, substrate_height_mm=args.substrate_height_mm)
        if args.command == "print":
            raise ControllerFault("live deposition is gated: first measure printer scheduling/physical gap and validate "
                                  "the fixture interlock and deposition profile; this release implements virtual tracking and cold characterization")
        if args.command == "camera-hash":
            from .runtime import geometry_signature
            from surface_mapper.config import load_config as camera_config
            result = {"camera_config_sha256": fingerprint(geometry_signature(camera_config(args.camera_config).to_dict()))}
        elif args.command == "simulate":
            if args.frames < 1 or not 0 < args.fps <= 1000:
                raise ValueError("frames and fps must be positive (fps <= 1000)")
            if config.substrate_height_mm is None:
                # This override exists only in the resolved virtual run, never in the hardware config.
                config = replace(config, substrate_height_mm=20.0)
            from .runtime import run_offline
            from .simulator import MovingPlane
            truth = MovingPlane(config, args.scenario, args.amplitude_mm, args.frequency_hz)
            observations = (truth.observation(i, i/args.fps, i == args.dropout_frame) for i in range(args.frames))
            result = asyncio.run(run_offline(config, load_path(args.path, config.path_spacing_mm), args.output,
                                observations, truth, fail_on_segment=args.lost_reply_segment,
                                allow_extrusion=args.virtual_extrusion))
        elif args.command == "replay":
            from .runtime import recording_observations, run_offline
            observations, geometry = recording_observations(config, args.session)
            result = asyncio.run(run_offline(config, load_path(args.path, config.path_spacing_mm), args.output,
                                            observations, model_geometry=geometry))
        elif args.command == "observe":
            if args.frames < 1:
                raise ValueError("frames must be positive")
            from .runtime import observe
            result = observe(config, args.camera_config, args.output, args.frames)
        elif args.command == "shadow":
            from .shadow import run_shadow
            result = run_shadow(config, args.camera_config,
                                load_path(args.path, config.path_spacing_mm), args.output, args.frames)
        elif args.command == "dry-run":
            from .cold_run import dry_run
            result = asyncio.run(dry_run(config, load_path(args.path, config.path_spacing_mm), args.output, args.execute))
        else:
            from .connection import create_printer_client
            async def printer_command():
                client = create_printer_client(config)
                try:
                    if args.command == "stop":
                        return {"emergency_stop": await client.emergency_stop(), "automatic_resume": False}
                    report = await client.discover()
                    output = Path(args.output)
                    output.mkdir(parents=True, exist_ok=False)
                    (output / "discovery.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
                    return report
                finally:
                    await client.close()
            result = asyncio.run(printer_command())
        print(json.dumps(result, indent=2, allow_nan=False))
        return 2 if result.get("state") == "FAULT" or result.get("reason") else 0
    except (ControllerFault, ValueError, OSError, RuntimeError) as exc:
        print(f"controller: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("controller: interrupted; use the tested physical stop if a printer is active", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
