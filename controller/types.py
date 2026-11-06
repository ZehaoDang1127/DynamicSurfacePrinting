"""Controller contracts. Distances are mm and elapsed times are monotonic seconds."""

from dataclasses import dataclass
from enum import Enum
import math

import numpy as np


class ControllerFault(RuntimeError):
    """A failed gate requires a new session, never an automatic motion retry."""


class State(str, Enum):
    DISCONNECTED = "DISCONNECTED"
    READY = "READY"
    WARMUP = "WARMUP"
    ARMED = "ARMED"
    RUNNING = "RUNNING"
    STOPPING = "STOPPING"
    FAULT = "FAULT"


@dataclass(frozen=True)
class Observation:
    frame_id: int
    height_mm: np.ndarray
    valid: np.ndarray
    point_count: np.ndarray
    geometry_id: str
    camera_time_ms: float
    timestamp_domain: str
    receipt_s: float
    observed_s: float
    uncertainty_s: float = 0.0

    def __post_init__(self):
        height = np.asarray(self.height_mm, dtype=np.float64)
        mask = np.asarray(self.valid, dtype=bool)
        count = np.asarray(self.point_count)
        if height.ndim != 2 or height.size == 0 or mask.shape != height.shape or count.shape != height.shape:
            raise ValueError("observation arrays must have matching nonempty 2D shapes")
        if not np.isfinite(count).all() or (count < 0).any():
            raise ValueError("invalid observation point counts")
        if not all(math.isfinite(v) for v in (self.camera_time_ms, self.receipt_s,
                                             self.observed_s, self.uncertainty_s)):
            raise ValueError("observation times must be finite")
        if self.uncertainty_s < 0 or self.observed_s > self.receipt_s + self.uncertainty_s:
            raise ValueError("invalid observation time/uncertainty")
        if type(self.frame_id) is not int or self.frame_id < 0 or not self.geometry_id or not self.timestamp_domain:
            raise ValueError("observation needs frame, geometry, and clock identities")
        object.__setattr__(self, "height_mm", height.copy())
        object.__setattr__(self, "valid", (mask & np.isfinite(height) & (count > 0)).copy())
        object.__setattr__(self, "point_count", count.copy())


@dataclass(frozen=True)
class ForecastMap:
    height_mm: np.ndarray
    support: np.ndarray
    source_frame_ids: tuple[int, ...]
    source_s: float
    target_s: float
    model_id: str
    error_bound_mm: float


@dataclass(frozen=True)
class Move:
    xyz: tuple[float, float, float]
    arrival_s: float
    duration_s: float
    feed_mm_s: float
    extrusion_mm: float
    surface_length_mm: float
    peak_z_speed_mm_s: float

    def __post_init__(self):
        if len(self.xyz) != 3 or not all(math.isfinite(v) for v in (*self.xyz, self.arrival_s,
                self.duration_s, self.feed_mm_s, self.extrusion_mm, self.surface_length_mm, self.peak_z_speed_mm_s)):
            raise ValueError("move values must be finite and XYZ must have three components")
        if min(self.duration_s, self.feed_mm_s, self.surface_length_mm) <= 0 or min(self.extrusion_mm, self.peak_z_speed_mm_s) < 0:
            raise ValueError("invalid move duration/feed/extrusion")


@dataclass(frozen=True)
class Trajectory:
    segment_id: int
    revision: int
    predecessor_id: int | None
    config_id: str
    start_xyz: tuple[float, float, float]
    start_s: float
    moves: tuple[Move, ...]
    source_frame_ids: tuple[int, ...]
    valid_until_s: float
    estimated_timing: bool = True

    def __post_init__(self):
        if type(self.segment_id) is not int or self.segment_id < 0 or type(self.revision) is not int or self.revision < 0:
            raise ValueError("invalid segment identity")
        if self.predecessor_id is not None and (type(self.predecessor_id) is not int or self.predecessor_id >= self.segment_id):
            raise ValueError("invalid committed predecessor")
        if not self.config_id or len(self.start_xyz) != 3 or not self.moves or not all(math.isfinite(t) for t in (
                *self.start_xyz, self.start_s, self.valid_until_s)):
            raise ValueError("invalid trajectory coordinates/times/configuration")
        if self.valid_until_s < self.start_s:
            raise ValueError("trajectory already expired at planned start")
        t = self.start_s
        for move in self.moves:
            t += move.duration_s
            if abs(t - move.arrival_s) > 1e-6:
                raise ValueError("trajectory arrival times disagree with move durations")

    @property
    def end_s(self):
        return self.moves[-1].arrival_s
