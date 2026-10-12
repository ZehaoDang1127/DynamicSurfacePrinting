"""Causal, per-cell kinematic depth forecasts using NumPy only.

Linear and quadratic least-squares fits use the complete supplied history and
actual timestamps. A quadratic fit is evaluated from its fitted current height,
velocity and acceleration; it is not anchored to the latest noisy observation.
With exactly three equally spaced frames and a one-frame horizon, this is
``3 * z[t] - 3 * z[t-1] + z[t-2]``. Longer histories reduce noise sensitivity.

Validity denotes measurement support for a fit, never prediction confidence.
Each fitted cell must be measured in every input frame. Unsupported predictions
remain NaN; there is no silent fallback to a different motion model.
"""

from collections import deque
from copy import deepcopy
from dataclasses import asdict, dataclass
import math
import time

import numpy as np


@dataclass(frozen=True)
class BaselineConfig:
    history: int = 5
    horizon_ms: float = 1000.0 / 30.0
    max_gap_ms: float = 100.0
    min_valid_fraction: float = 0.1

    def __post_init__(self) -> None:
        if type(self.history) is not int or self.history < 3:
            raise ValueError('history must be an integer of at least 3')
        for name in ('horizon_ms', 'max_gap_ms'):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if not math.isfinite(self.min_valid_fraction) or not 0 < self.min_valid_fraction <= 1:
            raise ValueError('min_valid_fraction must be in (0, 1]')

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class BaselineEstimate:
    prediction_mm: np.ndarray
    prediction_valid: np.ndarray
    velocity_mm_s: np.ndarray | None = None
    acceleration_mm_s2: np.ndarray | None = None


@dataclass(frozen=True)
class BaselineForecast:
    prediction_mm: np.ndarray
    observation_valid: np.ndarray
    history_observed: np.ndarray
    source_timestamp_ms: float
    target_timestamp_ms: float
    inference_ms: float
    timestamp_domain: str
    prediction_valid: np.ndarray
    velocity_mm_s: np.ndarray
    acceleration_mm_s2: np.ndarray


def fit_kinematic_baselines(heights, masks, timestamps_ms, horizon_ms: float) -> dict[str, BaselineEstimate]:
    """Fit recent depth maps and extrapolate ``horizon_ms`` beyond the last one.

    Input arrays have shape ``[T, H, W]``. At least two frames are required for
    velocity; acceleration is included only when at least three are available.
    Persistence uses the latest frame's valid cells. Linear and quadratic fits
    require finite, valid observations at every supplied timestamp per cell.

    Times are centered on the newest observation and scaled by the history span
    for numerical conditioning. Returned derivatives are at that newest time,
    in mm/s and mm/s^2, regardless of the input timestamp units (milliseconds).
    """
    height = np.asarray(heights, dtype=np.float64)
    valid = np.asarray(masks, dtype=bool)
    timestamps = np.asarray(timestamps_ms, dtype=np.float64)
    if height.ndim != 3 or valid.shape != height.shape or min(height.shape) < 1:
        raise ValueError('heights and masks must have matching nonempty [T, H, W] shapes')
    if height.shape[0] < 2:
        raise ValueError('at least two frames are required')
    if timestamps.shape != (height.shape[0],):
        raise ValueError('timestamps must have shape [T] matching the history')
    if not np.isfinite(timestamps).all() or not (np.diff(timestamps) > 0).all():
        raise ValueError('timestamps must be finite and strictly increasing')
    if not math.isfinite(horizon_ms) or horizon_ms <= 0:
        raise ValueError('horizon_ms must be finite and positive')

    valid = valid & np.isfinite(height)
    shape = height.shape[1:]
    latest_valid = valid[-1].copy()
    persistence = np.full(shape, np.nan, dtype=np.float32)
    persistence[latest_valid] = height[-1, latest_valid]
    estimates = {'persistence': BaselineEstimate(persistence, latest_valid)}
    support = np.logical_and.reduce(valid)

    # Subtract before converting to seconds, retaining precision for epoch times.
    relative_ms = timestamps - timestamps[-1]
    span_s = -relative_ms[0] / 1000.0
    normalized_time = relative_ms / -relative_ms[0]
    future = (horizon_ms / 1000.0) / span_s
    observations = height[:, support]

    for degree, name in ((1, 'constant_velocity'), (2, 'constant_acceleration')):
        if height.shape[0] < degree + 1:
            continue
        prediction = np.full(shape, np.nan, dtype=np.float32)
        velocity = np.full(shape, np.nan, dtype=np.float32)
        acceleration = np.full(shape, np.nan, dtype=np.float32) if degree == 2 else None
        design = np.column_stack([normalized_time ** power for power in range(degree + 1)])
        coefficients, _, rank, _ = np.linalg.lstsq(design, observations, rcond=None)
        if rank != degree + 1:
            raise ValueError('timestamps do not support a numerically stable motion fit')
        # This evaluates the fitted polynomial intercept, not the last raw height.
        prediction[support] = sum(coefficients[p] * future ** p for p in range(degree + 1))
        velocity[support] = coefficients[1] / span_s
        if acceleration is not None:
            acceleration[support] = 2.0 * coefficients[2] / span_s ** 2
        usable = support & np.isfinite(prediction) & np.isfinite(velocity)
        if acceleration is not None:
            usable &= np.isfinite(acceleration)
            acceleration[~usable] = np.nan
        prediction[~usable] = np.nan
        velocity[~usable] = np.nan
        estimates[name] = BaselineEstimate(prediction, usable, velocity, acceleration)
    return estimates


class StreamingBaselinePredictor:
    """Fit a causal sliding window; reset history after capture discontinuities.

    Skipped frame numbers are allowed when timestamps keep increasing and the
    gap is within ``max_gap_ms``; actual timestamps determine the fit. Duplicate
    or decreasing counters, changed domains, non-increasing times, excessive
    gaps and insufficient depth coverage discard the prior window.
    """

    def __init__(self, config: BaselineConfig, geometry: dict, shape) -> None:
        self.config = config
        self.geometry = deepcopy(geometry)
        self.shape = tuple(shape)
        if len(self.shape) != 2 or any(not isinstance(n, (int, np.integer)) or n < 1 for n in self.shape):
            raise ValueError('shape must contain two positive integers')
        self.buffer = deque(maxlen=config.history)
        self.reset_count = 0
        self.last_reset_reason = None

    def reset(self, reason: str = 'manual') -> None:
        self.buffer.clear()
        self.reset_count += 1
        self.last_reset_reason = reason

    def push(self, height, valid, timestamp_ms: float, timestamp_domain: str,
             frame_number: int | None = None) -> BaselineForecast | None:
        height = np.asarray(height, dtype=np.float32)
        mask = np.asarray(valid, dtype=bool)
        if height.ndim != 2 or mask.shape != height.shape or height.shape != self.shape:
            self.reset('grid shape changed')
            raise ValueError('height and validity must match the configured 2D grid shape')
        if not np.isfinite(timestamp_ms):
            self.reset('invalid timestamp')
            raise ValueError('timestamp must be finite')
        timestamp_ms = float(timestamp_ms)
        timestamp_domain = str(timestamp_domain)
        mask = mask & np.isfinite(height)
        if mask.mean() < self.config.min_valid_fraction:
            self.reset('insufficient valid depth')
            return None
        if self.buffer:
            previous = self.buffer[-1]
            dt = timestamp_ms - previous[2]
            if timestamp_domain != previous[3]:
                self.reset('timestamp domain changed')
            elif dt <= 0 or dt > self.config.max_gap_ms:
                self.reset('timestamp discontinuity')
            elif frame_number is not None and previous[4] is not None and frame_number <= previous[4]:
                self.reset('frame counter reset or duplicate')
        self.buffer.append((height.copy(), mask.copy(), timestamp_ms, timestamp_domain, frame_number))
        if len(self.buffer) < self.config.history:
            return None
        started = time.perf_counter()
        estimates = fit_kinematic_baselines(
            np.stack([item[0] for item in self.buffer]),
            np.stack([item[1] for item in self.buffer]),
            [item[2] for item in self.buffer], self.config.horizon_ms,
        )
        estimate = estimates['constant_acceleration']
        elapsed = (time.perf_counter() - started) * 1000.0
        return BaselineForecast(
            prediction_mm=estimate.prediction_mm,
            observation_valid=mask.copy(),
            history_observed=np.logical_or.reduce([item[1] for item in self.buffer]),
            source_timestamp_ms=timestamp_ms,
            target_timestamp_ms=timestamp_ms + self.config.horizon_ms,
            inference_ms=elapsed,
            timestamp_domain=timestamp_domain,
            prediction_valid=estimate.prediction_valid,
            velocity_mm_s=estimate.velocity_mm_s,
            acceleration_mm_s2=estimate.acceleration_mm_s2,
        )
