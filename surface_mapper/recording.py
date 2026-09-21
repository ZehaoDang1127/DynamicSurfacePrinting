"""Crash-resistant, frame-at-a-time recording for height-map sessions."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from .camera import PointCloudFrame
from .config import AppConfig
from .heightmap import HeightMap


class SessionWriter:
    def __init__(
        self,
        output_dir: str | Path,
        config: AppConfig,
        device_info: dict[str, Any],
        save_depth: bool = False,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.config = config
        self.device_info = dict(device_info)
        self.save_depth = save_depth
        self.frames: list[dict[str, Any]] = []
        self.started_utc = datetime.now(timezone.utc).isoformat()

    def start(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        existing_frames = list(self.output_dir.glob("frame_*.npz"))
        if existing_frames:
            raise FileExistsError(
                f"output directory already contains recorded frames: {self.output_dir}"
            )
        self._write_manifest(complete=False)

    def write(self, source: PointCloudFrame, heightmap: HeightMap) -> Path:
        filename = f"frame_{source.frame_number:06d}.npz"
        target = self.output_dir / filename
        temporary = self.output_dir / f".{filename}.tmp"
        arrays: dict[str, Any] = {
            "height_mm": heightmap.height_mm,
            "valid": heightmap.valid,
            "point_count": heightmap.point_count,
            "frame_number": np.int64(source.frame_number),
            "camera_timestamp_ms": np.float64(source.camera_timestamp_ms),
            "host_monotonic_ns": np.int64(source.host_monotonic_ns),
        }
        if self.save_depth and source.depth_m is not None:
            arrays["depth_m"] = source.depth_m.astype(np.float32, copy=False)
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(temporary, target)

        valid_heights = heightmap.height_mm[heightmap.valid]
        self.frames.append(
            {
                "file": filename,
                "frame_number": source.frame_number,
                "camera_timestamp_ms": source.camera_timestamp_ms,
                "host_monotonic_ns": source.host_monotonic_ns,
                "timestamp_domain": source.timestamp_domain,
                "valid_fraction": heightmap.valid_fraction,
                "accepted_point_count": heightmap.accepted_point_count,
                "height_min_mm": (
                    float(np.min(valid_heights)) if valid_heights.size else None
                ),
                "height_max_mm": (
                    float(np.max(valid_heights)) if valid_heights.size else None
                ),
            }
        )
        self._write_manifest(complete=False)
        return target

    def write_latest_csv(self, heightmap: HeightMap) -> Path:
        target = self.output_dir / "latest_height_mm.csv"
        temporary = self.output_dir / ".latest_height_mm.csv.tmp"
        np.savetxt(temporary, heightmap.height_mm, delimiter=",", fmt="%.6f")
        os.replace(temporary, target)
        return target

    def finish(self) -> None:
        self._write_manifest(complete=True)

    def _write_manifest(self, complete: bool) -> None:
        manifest = {
            "schema_version": 1,
            "complete": complete,
            "started_utc": self.started_utc,
            "finished_utc": datetime.now(timezone.utc).isoformat() if complete else None,
            "device": self.device_info,
            "save_depth": self.save_depth,
            "config": self.config.to_dict(),
            "grid": {
                "shape": [self.config.grid.height, self.config.grid.width],
                "x_centers_mm": self.config.grid.x_centers_mm().tolist(),
                "y_centers_mm": self.config.grid.y_centers_mm().tolist(),
            },
            "frame_count": len(self.frames),
            "frames": self.frames,
        }
        target = self.output_dir / "manifest.json"
        temporary = self.output_dir / ".manifest.json.tmp"
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
            handle.write("\n")
        os.replace(temporary, target)
