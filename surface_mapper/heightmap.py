"""Point-cloud transformation and regular-grid height-map construction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import GridConfig


def apply_rigid_transform(points_mm: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply a 4x4 rigid transform to an N×3 point array in millimetres."""
    points = np.asarray(points_mm)
    transform = np.asarray(matrix, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_mm must have shape (N, 3)")
    if transform.shape != (4, 4):
        raise ValueError("matrix must have shape (4, 4)")
    return points @ transform[:3, :3].T + transform[:3, 3]


@dataclass(frozen=True)
class HeightMap:
    height_mm: np.ndarray
    valid: np.ndarray
    point_count: np.ndarray
    input_point_count: int
    accepted_point_count: int

    @property
    def valid_fraction(self) -> float:
        return float(np.count_nonzero(self.valid) / self.valid.size)


class HeightMapProjector:
    """Transform camera points and aggregate output Z into regular output XY cells."""

    def __init__(self, grid: GridConfig, camera_to_output: np.ndarray) -> None:
        self.grid = grid
        self.camera_to_output = np.asarray(camera_to_output, dtype=np.float64)

    def project(self, points_camera_m: np.ndarray) -> HeightMap:
        camera_points = np.asarray(points_camera_m)
        if camera_points.ndim != 2 or camera_points.shape[1] != 3:
            raise ValueError("points_camera_m must have shape (N, 3)")

        # RealSense point-cloud vertices are metres; all project geometry is mm.
        points_mm = apply_rigid_transform(
            camera_points.astype(np.float64, copy=False) * 1000.0,
            self.camera_to_output,
        )
        finite = np.all(np.isfinite(points_mm), axis=1)
        nonzero = np.any(camera_points != 0.0, axis=1)
        x = points_mm[:, 0]
        y = points_mm[:, 1]
        z = points_mm[:, 2]
        keep = (
            finite
            & nonzero
            & (x >= self.grid.x_min_mm)
            & (x < self.grid.x_max_mm)
            & (y >= self.grid.y_min_mm)
            & (y < self.grid.y_max_mm)
            & (z >= self.grid.z_min_mm)
            & (z <= self.grid.z_max_mm)
        )

        x = x[keep]
        y = y[keep]
        z = z[keep]
        ix = np.floor((x - self.grid.x_min_mm) / self.grid.cell_size_mm).astype(np.int64)
        iy = np.floor((y - self.grid.y_min_mm) / self.grid.cell_size_mm).astype(np.int64)
        flat_index = iy * self.grid.width + ix
        cell_count = self.grid.height * self.grid.width
        counts = np.bincount(flat_index, minlength=cell_count).astype(np.uint32)

        if self.grid.aggregation == "mean":
            sums = np.bincount(flat_index, weights=z, minlength=cell_count)
            values = np.full(cell_count, np.nan, dtype=np.float64)
            occupied = counts > 0
            values[occupied] = sums[occupied] / counts[occupied]
        elif self.grid.aggregation == "min":
            values = np.full(cell_count, np.inf, dtype=np.float64)
            np.minimum.at(values, flat_index, z)
        else:
            values = np.full(cell_count, -np.inf, dtype=np.float64)
            np.maximum.at(values, flat_index, z)

        valid = counts >= self.grid.min_points_per_cell
        values[~valid] = np.nan
        shape = (self.grid.height, self.grid.width)
        return HeightMap(
            height_mm=values.reshape(shape).astype(np.float32),
            valid=valid.reshape(shape),
            point_count=counts.reshape(shape),
            input_point_count=int(camera_points.shape[0]),
            accepted_point_count=int(np.count_nonzero(keep)),
        )
