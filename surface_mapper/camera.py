"""Depth sources for a physical RealSense D405 and deterministic test data."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Protocol

import numpy as np

from .config import CameraConfig, GridConfig


@dataclass(frozen=True)
class PointCloudFrame:
    frame_number: int
    camera_timestamp_ms: float
    host_monotonic_ns: int
    timestamp_domain: str
    points_camera_m: np.ndarray
    depth_m: np.ndarray | None = None


class PointCloudSource(Protocol):
    device_info: dict[str, Any]

    def start(self) -> None: ...

    def capture(self) -> PointCloudFrame: ...

    def stop(self) -> None: ...


class RealSenseD405:
    """Acquire metric point clouds directly from librealsense depth frames."""

    def __init__(self, config: CameraConfig) -> None:
        self.config = config
        self.device_info: dict[str, Any] = {}
        self._rs = None
        self._pipeline = None
        self._pointcloud = None
        self._started = False

    def start(self) -> None:
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise RuntimeError(
                "pyrealsense2 is not installed; install requirements.txt or use "
                "--source synthetic"
            ) from exc

        pipeline = rs.pipeline()
        rs_config = rs.config()
        if self.config.serial:
            rs_config.enable_device(self.config.serial)
        rs_config.enable_stream(
            rs.stream.depth,
            self.config.width,
            self.config.height,
            rs.format.z16,
            self.config.fps,
        )

        try:
            profile = pipeline.start(rs_config)
        except RuntimeError as exc:
            raise RuntimeError(
                "unable to start the RealSense depth stream; check the D405 "
                "connection, driver, stream mode, and whether another app owns it"
            ) from exc

        self._rs = rs
        self._pipeline = pipeline
        self._pointcloud = rs.pointcloud()
        self._started = True

        device = profile.get_device()
        for key, label in (
            (rs.camera_info.name, "name"),
            (rs.camera_info.serial_number, "serial"),
            (rs.camera_info.firmware_version, "firmware"),
        ):
            try:
                self.device_info[label] = device.get_info(key)
            except RuntimeError:
                pass
        try:
            depth_profile = profile.get_stream(rs.stream.depth).as_video_stream_profile()
            intrinsics = depth_profile.get_intrinsics()
            self.device_info["depth_stream"] = {
                "width": int(intrinsics.width),
                "height": int(intrinsics.height),
                "fps": int(depth_profile.fps()),
                "fx": float(intrinsics.fx),
                "fy": float(intrinsics.fy),
                "ppx": float(intrinsics.ppx),
                "ppy": float(intrinsics.ppy),
                "distortion_model": str(intrinsics.model),
                "distortion_coefficients": [float(value) for value in intrinsics.coeffs],
                "depth_scale_m_per_unit": float(
                    device.first_depth_sensor().get_depth_scale()
                ),
            }
        except RuntimeError:
            # Capture still works if a particular runtime cannot expose optional
            # metadata; the per-frame point cloud already has metric coordinates.
            pass

        try:
            for _ in range(self.config.warmup_frames):
                pipeline.wait_for_frames(self.config.timeout_ms)
        except Exception:
            self.stop()
            raise

    @staticmethod
    def _vertices_to_xyz(vertices: object, height: int, width: int) -> np.ndarray:
        raw = np.asanyarray(vertices)
        if raw.ndim == 2 and raw.shape == (height * width, 3):
            xyz = raw.astype(np.float32, copy=False)
        elif raw.ndim == 3 and raw.shape == (height, width, 3):
            xyz = raw.reshape(-1, 3).astype(np.float32, copy=False)
        else:
            try:
                xyz = raw.view(np.float32).reshape(-1, 3)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"unexpected RealSense vertex buffer shape/dtype: "
                    f"{raw.shape}/{raw.dtype}"
                ) from exc
        if xyz.shape[0] != height * width:
            raise RuntimeError(
                f"RealSense returned {xyz.shape[0]} points for a {width}x{height} frame"
            )
        return np.array(xyz, dtype=np.float32, copy=True)

    def capture(self) -> PointCloudFrame:
        if not self._started or self._pipeline is None or self._pointcloud is None:
            raise RuntimeError("RealSense source has not been started")

        frames = self._pipeline.wait_for_frames(self.config.timeout_ms)
        depth_frame = frames.get_depth_frame()
        if not depth_frame:
            raise RuntimeError("RealSense frameset did not contain a depth frame")

        host_ns = time.monotonic_ns()
        depth_raw = np.asanyarray(depth_frame.get_data())
        height, width = depth_raw.shape
        depth_scale = depth_frame.get_units()
        depth_m = depth_raw.astype(np.float32) * np.float32(depth_scale)
        points = self._pointcloud.calculate(depth_frame)
        xyz_m = self._vertices_to_xyz(points.get_vertices(), height, width)

        return PointCloudFrame(
            frame_number=int(depth_frame.get_frame_number()),
            camera_timestamp_ms=float(depth_frame.get_timestamp()),
            host_monotonic_ns=host_ns,
            timestamp_domain=str(depth_frame.get_frame_timestamp_domain()),
            points_camera_m=xyz_m,
            depth_m=depth_m,
        )

    def stop(self) -> None:
        if self._started and self._pipeline is not None:
            self._pipeline.stop()
        self._started = False

    def __enter__(self) -> "RealSenseD405":
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.stop()


class SyntheticSurfaceSource:
    """Generate a moving smooth height field for installation and pipeline checks."""

    def __init__(self, grid: GridConfig, fps: int = 30) -> None:
        self.grid = grid
        self.fps = fps
        self.device_info = {"name": "synthetic moving surface", "serial": "synthetic"}
        self._frame_number = 0
        self._started_ns = 0

    def start(self) -> None:
        self._frame_number = 0
        self._started_ns = time.monotonic_ns()

    def capture(self) -> PointCloudFrame:
        # Four samples per output cell exercise the aggregation and count masks.
        x_centers = self.grid.x_centers_mm()
        y_centers = self.grid.y_centers_mm()
        x, y = np.meshgrid(x_centers, y_centers)
        offsets = self.grid.cell_size_mm * np.array([-0.2, 0.2], dtype=np.float32)
        sample_x = np.stack([x + offsets[0], x + offsets[1], x + offsets[0], x + offsets[1]])
        sample_y = np.stack([y + offsets[0], y + offsets[0], y + offsets[1], y + offsets[1]])
        elapsed_s = self._frame_number / float(self.fps)
        z_mid = 0.5 * (self.grid.z_min_mm + self.grid.z_max_mm)
        amplitude = min(5.0, 0.1 * (self.grid.z_max_mm - self.grid.z_min_mm))
        wavelength = max(1.0, self.grid.x_max_mm - self.grid.x_min_mm)
        sample_z = z_mid + amplitude * np.sin(
            2.0 * np.pi * sample_x / wavelength + 2.0 * np.pi * 0.5 * elapsed_s
        )
        points_mm = np.column_stack(
            (sample_x.ravel(), sample_y.ravel(), sample_z.ravel())
        ).astype(np.float32)
        host_ns = time.monotonic_ns()
        frame = PointCloudFrame(
            frame_number=self._frame_number,
            camera_timestamp_ms=elapsed_s * 1000.0,
            host_monotonic_ns=host_ns,
            timestamp_domain="synthetic",
            points_camera_m=points_mm / 1000.0,
            depth_m=None,
        )
        self._frame_number += 1
        return frame

    def stop(self) -> None:
        pass

    def __enter__(self) -> "SyntheticSurfaceSource":
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.stop()
