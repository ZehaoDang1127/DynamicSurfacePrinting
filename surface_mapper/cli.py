"""Command-line entry point for D405-to-height-map capture."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys
import time

import numpy as np

from .camera import RealSenseD405, SyntheticSurfaceSource
from .config import load_config
from .heightmap import HeightMap, HeightMapProjector
from .recording import SessionWriter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture D405 point clouds and rasterize surface height maps."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/d405_camera_frame.json"),
        help="JSON camera, transform, and grid configuration",
    )
    parser.add_argument(
        "--source",
        choices=("realsense", "synthetic"),
        default="realsense",
        help="physical D405 or deterministic moving test surface",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=1,
        help="number of frames to capture; 0 runs until Ctrl+C",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="session output directory (default: output/session_TIMESTAMP)",
    )
    parser.add_argument(
        "--save-depth",
        action="store_true",
        help="also save each raw metric depth image (larger files)",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=1,
        help="print capture statistics every N saved frames",
    )
    return parser


def _height_range(heightmap: HeightMap) -> str:
    valid = heightmap.height_mm[heightmap.valid]
    if valid.size == 0:
        return "no valid heights"
    return f"z={float(np.min(valid)):.3f}..{float(np.max(valid)):.3f} mm"


def run(args: argparse.Namespace) -> int:
    if args.frames < 0:
        raise ValueError("--frames cannot be negative")
    if args.print_every <= 0:
        raise ValueError("--print-every must be positive")

    config = load_config(args.config)
    output = args.output or Path("output") / datetime.now().strftime(
        "session_%Y%m%d_%H%M%S"
    )
    if args.source == "realsense":
        source = RealSenseD405(config.camera)
    else:
        source = SyntheticSurfaceSource(config.grid, config.camera.fps)

    projector = HeightMapProjector(config.grid, config.camera_to_output)
    last_heightmap: HeightMap | None = None
    saved = 0
    started = time.perf_counter()
    clean_finish = False

    source.start()
    writer = SessionWriter(output, config, source.device_info, args.save_depth)
    try:
        writer.start()
        if config.frame_name.lower() == "camera" and args.source == "realsense":
            print(
                "WARNING: output frame is 'camera'; heights are camera Z, not "
                "calibrated printer Z.",
                file=sys.stderr,
            )
        print(f"source: {source.device_info}")
        print(
            f"grid: {config.grid.height}x{config.grid.width} cells in "
            f"'{config.frame_name}' frame; output: {output}"
        )
        while args.frames == 0 or saved < args.frames:
            frame = source.capture()
            last_heightmap = projector.project(frame.points_camera_m)
            writer.write(frame, last_heightmap)
            saved += 1
            if saved % args.print_every == 0:
                elapsed = max(time.perf_counter() - started, 1e-9)
                print(
                    f"saved={saved} camera_frame={frame.frame_number} "
                    f"camera_t={frame.camera_timestamp_ms:.3f} ms "
                    f"coverage={last_heightmap.valid_fraction:.1%} "
                    f"{_height_range(last_heightmap)} rate={saved / elapsed:.1f} fps"
                )
        clean_finish = True
    except KeyboardInterrupt:
        print("capture stopped by user")
        clean_finish = True
    finally:
        source.stop()
        if last_heightmap is not None:
            csv_path = writer.write_latest_csv(last_heightmap)
            print(f"latest height map: {csv_path}")
        if clean_finish:
            writer.finish()

    print(f"captured {saved} frame(s) in {time.perf_counter() - started:.2f} s")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run(args)
    except (FileExistsError, RuntimeError, ValueError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
