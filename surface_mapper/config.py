"""Configuration loading and validation for the height-map pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CameraConfig:
    serial: str | None = None
    width: int = 640
    height: int = 480
    fps: int = 30
    warmup_frames: int = 30
    timeout_ms: int = 5000

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0 or self.fps <= 0:
            raise ValueError("camera width, height, and fps must be positive")
        if self.warmup_frames < 0:
            raise ValueError("camera warmup_frames cannot be negative")
        if self.timeout_ms <= 0:
            raise ValueError("camera timeout_ms must be positive")


@dataclass(frozen=True)
class GridConfig:
    x_min_mm: float
    x_max_mm: float
    y_min_mm: float
    y_max_mm: float
    cell_size_mm: float
    z_min_mm: float
    z_max_mm: float
    min_points_per_cell: int = 1
    aggregation: str = "mean"

    def __post_init__(self) -> None:
        if not self.x_max_mm > self.x_min_mm:
            raise ValueError("grid x_max_mm must be greater than x_min_mm")
        if not self.y_max_mm > self.y_min_mm:
            raise ValueError("grid y_max_mm must be greater than y_min_mm")
        if not self.z_max_mm > self.z_min_mm:
            raise ValueError("grid z_max_mm must be greater than z_min_mm")
        if self.cell_size_mm <= 0:
            raise ValueError("grid cell_size_mm must be positive")
        if self.min_points_per_cell <= 0:
            raise ValueError("grid min_points_per_cell must be positive")
        if self.aggregation not in {"mean", "min", "max"}:
            raise ValueError("grid aggregation must be one of: mean, min, max")

    @property
    def width(self) -> int:
        return int(np.ceil((self.x_max_mm - self.x_min_mm) / self.cell_size_mm))

    @property
    def height(self) -> int:
        return int(np.ceil((self.y_max_mm - self.y_min_mm) / self.cell_size_mm))

    def x_centers_mm(self) -> np.ndarray:
        return self.x_min_mm + (np.arange(self.width) + 0.5) * self.cell_size_mm

    def y_centers_mm(self) -> np.ndarray:
        return self.y_min_mm + (np.arange(self.height) + 0.5) * self.cell_size_mm


@dataclass(frozen=True)
class AppConfig:
    frame_name: str
    camera: CameraConfig
    grid: GridConfig
    camera_to_output: np.ndarray

    def __post_init__(self) -> None:
        matrix = np.asarray(self.camera_to_output, dtype=np.float64)
        if matrix.shape != (4, 4):
            raise ValueError("camera_to_output must be a 4x4 matrix")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("camera_to_output must contain only finite values")
        if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
            raise ValueError("camera_to_output last row must be [0, 0, 0, 1]")
        rotation = matrix[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
            raise ValueError("camera_to_output upper-left block must be a rotation")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
            raise ValueError("camera_to_output rotation determinant must be +1")
        object.__setattr__(self, "camera_to_output", matrix)

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_name": self.frame_name,
            "camera": vars(self.camera),
            "grid": vars(self.grid),
            "camera_to_output": self.camera_to_output.tolist(),
        }


def load_config(path: str | Path) -> AppConfig:
    """Load a JSON configuration and reject missing/unknown structural errors."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a JSON object")
    try:
        frame_name = str(raw["frame_name"])
        camera = CameraConfig(**raw["camera"])
        grid = GridConfig(**raw["grid"])
        transform = np.asarray(raw["camera_to_output"], dtype=np.float64)
    except KeyError as exc:
        raise ValueError(f"missing required configuration field: {exc.args[0]}") from exc
    except TypeError as exc:
        raise ValueError(f"invalid configuration field: {exc}") from exc

    if not frame_name.strip():
        raise ValueError("frame_name cannot be empty")
    return AppConfig(frame_name, camera, grid, transform)
