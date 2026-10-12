"""Causal sliding-window inference shared by replay and live capture."""

from collections import deque
from dataclasses import dataclass
import time

import numpy as np
import torch

from .training import load_checkpoint


@dataclass(frozen=True)
class Forecast:
    prediction_mm: np.ndarray
    observation_valid: np.ndarray
    history_observed: np.ndarray
    source_timestamp_ms: float
    target_timestamp_ms: float
    inference_ms: float
    timestamp_domain: str


class StreamingPredictor:
    """Reset on gaps; each full window starts the recurrent state from zero.

    Support masks describe past observations, never confidence in future values.
    The target timestamp uses the checkpoint's fixed training cadence/horizon.
    """

    def __init__(self, checkpoint_path, device: str = 'auto') -> None:
        self.model, self.config, self.normalization, self.checkpoint = load_checkpoint(
            checkpoint_path, device
        )
        self.device = next(self.model.parameters()).device
        self.geometry = self.checkpoint['geometry']
        self.buffer = deque(maxlen=self.config.history)
        self.reset_count = 0
        self.last_reset_reason = None

    def reset(self, reason: str = 'manual') -> None:
        self.buffer.clear()
        self.reset_count += 1
        self.last_reset_reason = reason

    def push(self, height_mm, valid, timestamp_ms: float,
             timestamp_domain: str, frame_number: int | None = None) -> Forecast | None:
        height = np.asarray(height_mm, dtype=np.float32)
        mask = np.asarray(valid, dtype=bool)
        if height.ndim != 2 or mask.shape != height.shape:
            raise ValueError('height and validity must have matching 2D shapes')
        expected_shape = self.checkpoint.get('shape')
        if expected_shape is not None and tuple(expected_shape) != height.shape:
            raise ValueError('input shape differs from checkpoint grid')
        if not np.isfinite(timestamp_ms):
            self.reset('invalid timestamp')
            raise ValueError('timestamp must be finite')
        mask = mask & np.isfinite(height)
        if mask.mean() < self.config.min_valid_fraction:
            self.reset('insufficient valid depth')
            return None
        if self.buffer:
            previous = self.buffer[-1]
            dt = timestamp_ms - previous[2]
            limit = self.config.expected_dt_ms * self.config.timing_tolerance
            if timestamp_domain != previous[3]:
                self.reset('timestamp domain changed')
            elif height.shape != previous[0].shape:
                self.reset('grid shape changed')
                raise ValueError('grid shape changed during prediction')
            elif abs(dt - self.config.expected_dt_ms) > limit + 1e-6:
                self.reset('timestamp discontinuity')
            elif frame_number is not None and previous[4] is not None and frame_number <= previous[4]:
                self.reset('frame counter reset or duplicate')
        self.buffer.append((height.copy(), mask.copy(), float(timestamp_ms),
                            str(timestamp_domain), frame_number))
        if len(self.buffer) < self.config.history:
            return None
        started = time.perf_counter()
        inputs = np.stack([
            np.stack((self.normalization.normalize(h, m), m.astype(np.float32)))
            for h, m, _, _, _ in self.buffer
        ])
        with torch.inference_mode():
            tensor = torch.from_numpy(inputs[None]).to(self.device)
            normalized = self.model(tensor)[0, 0].cpu().numpy()
        prediction = np.asarray(self.normalization.denormalize(normalized), dtype=np.float32)
        if not np.isfinite(prediction).all():
            raise RuntimeError('model produced non-finite depth predictions')
        elapsed = (time.perf_counter() - started) * 1000.0
        return Forecast(
            prediction, mask.copy(), np.logical_or.reduce([item[1] for item in self.buffer]),
            float(timestamp_ms),
            float(timestamp_ms + self.config.horizon * self.config.expected_dt_ms),
            elapsed, str(timestamp_domain),
        )
