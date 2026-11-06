"""Causal paired clean/masked windows for supervised occlusion experiments.

Recordings must be clean reference sessions; a sensor-valid nozzle measurement
is not a surface label. Split sessions before constructing these datasets. The
synthetic grid-space masks approximate occlusion, not a calibrated tool.
"""

from __future__ import annotations

import hashlib
import math

import numpy as np

from .data import Normalization, Session, _load_arrays


def _settings(config: dict, seed: int) -> dict:
    names = ('history', 'horizon', 'expected_dt_ms', 'timing_tolerance',
             'min_valid_fraction', 'age_scale_ms', 'occlusion_fraction')
    try:
        values = {name: config[name] for name in names}
    except KeyError as exc:
        raise ValueError(f'missing occlusion dataset setting: {exc.args[0]}') from exc
    for name in ('history', 'horizon'):
        if type(values[name]) is not int or values[name] < 1:
            raise ValueError(f'{name} must be a positive integer')
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('seed must be an integer in [0, 2**32)')
    for name in names[2:]:
        value = values[name]
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
            raise ValueError(f'{name} must be finite and numeric')
    if values['expected_dt_ms'] <= 0 or values['age_scale_ms'] <= 0:
        raise ValueError('expected_dt_ms and age_scale_ms must be positive')
    if not 0 <= values['timing_tolerance'] < 1:
        raise ValueError('timing_tolerance must be in [0, 1)')
    if not 0 < values['min_valid_fraction'] <= 1:
        raise ValueError('min_valid_fraction must be in (0, 1]')
    if not 0 <= values['occlusion_fraction'] <= 1:
        raise ValueError('occlusion_fraction must be in [0, 1]')
    values['occlusion_shape'] = config.get('occlusion_shape', 'rectangle')
    if values['occlusion_shape'] not in ('rectangle', 'gaussian'):
        raise ValueError('occlusion_shape must be rectangle or gaussian')
    values['gaussian_step_interval'] = config.get('gaussian_step_interval', 1)
    if type(values['gaussian_step_interval']) is not int or values['gaussian_step_interval'] < 1:
        raise ValueError('gaussian_step_interval must be a positive integer')
    return values


def _masks(shape: tuple[int, int], count: int, fraction: float, rng: np.random.Generator) -> np.ndarray:
    """Move one contiguous rectangle at at most one grid cell per frame.

    The requested fraction is approximate because the grid is discrete. Motion
    reflects at borders; a full-grid mask remains stationary.
    """
    height, width = shape
    result = np.zeros((count, height, width), dtype=bool)
    if fraction == 0:
        return result
    # Choose integer dimensions whose area approximates the requested area and
    # whose aspect ratio remains roughly similar to the grid.
    target = fraction * height * width
    rect_h, rect_w = min(
        ((h, w) for h in range(1, height + 1)
         for w in {max(1, min(width, int(target / h))),
                   max(1, min(width, int(target / h) + 1))}),
        key=lambda pair: (abs(pair[0] * pair[1] - target),
                          abs(pair[0] / height - pair[1] / width)),
    )
    limits = (height - rect_h, width - rect_w)
    pos = [int(rng.integers(limit + 1)) for limit in limits]
    velocity = [int(rng.choice((-1, 0, 1))) if limit else 0 for limit in limits]
    for i in range(count):
        result[i, pos[0]:pos[0] + rect_h, pos[1]:pos[1] + rect_w] = True
        for axis, limit in enumerate(limits):
            if pos[axis] + velocity[axis] < 0 or pos[axis] + velocity[axis] > limit:
                velocity[axis] *= -1
            pos[axis] += velocity[axis]
    return result


def _gaussian_masks(shape: tuple[int, int], count: int, fraction: float,
                    rng: np.random.Generator, step_interval: int = 1) -> np.ndarray:
    """Threshold a moving, anisotropic Gaussian to form a binary occluder.

    The largest Gaussian values are selected so each frame masks approximately
    the requested fraction, including when the contour touches a grid edge.
    Its centre moves by at most one cell per axis per frame and reflects at
    boundaries every `step_interval` frames; its orientation and aspect ratio stay fixed within the window.
    This is a synthetic silhouette, not a measured nozzle profile.
    """
    height, width = shape
    result = np.zeros((count, height, width), dtype=bool)
    if fraction == 0:
        return result
    pixels = min(height * width, max(1, round(fraction * height * width)))
    if pixels == height * width:
        result[:] = True
        return result
    center = [int(rng.integers(height)), int(rng.integers(width))]
    velocity = [int(rng.choice((-1, 0, 1))) if limit > 1 else 0
                for limit in shape]
    theta = rng.uniform(0, np.pi)
    aspect = rng.uniform(.6, 1.7)
    ys, xs = np.indices(shape, dtype=np.float64)
    for i in range(count):
        dy, dx = ys - center[0], xs - center[1]
        along = dx * np.cos(theta) + dy * np.sin(theta)
        across = -dx * np.sin(theta) + dy * np.cos(theta)
        # Squared Mahalanobis distance is monotonic in Gaussian intensity.
        distance = (along / aspect) ** 2 + (across * aspect) ** 2
        selected = np.argpartition(distance.ravel(), pixels - 1)[:pixels]
        result[i].flat[selected] = True
        if (i + 1) % step_interval == 0:
            for axis, size in enumerate(shape):
                if center[axis] + velocity[axis] < 0 or center[axis] + velocity[axis] >= size:
                    velocity[axis] *= -1
                center[axis] += velocity[axis]
    return result


def encode_history(heights, valid_masks, timestamps_ms, normalization: Normalization,
                   age_scale_ms: float) -> np.ndarray:
    """Encode real observations, with age relative only to this history.

    Inputs are [T,H,W]. Never-observed and stale cells both use capped age 1;
    the final channel distinguishes them. Non-finite heights are not observed.
    """
    heights = np.asarray(heights)
    valid = np.asarray(valid_masks, dtype=bool)
    timestamps = np.asarray(timestamps_ms, dtype=np.float64)
    if heights.ndim != 3 or heights.shape != valid.shape or not all(heights.shape):
        raise ValueError('heights and valid_masks must have equal nonempty [T,H,W] shape')
    if timestamps.shape != (len(heights),) or not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0):
        raise ValueError('timestamps must be finite and strictly increasing')
    if isinstance(age_scale_ms, bool) or not math.isfinite(age_scale_ms) or age_scale_ms <= 0:
        raise ValueError('age_scale_ms must be finite and positive')
    last_seen = np.zeros(heights.shape[1:], dtype=np.float64)
    seen = np.zeros(heights.shape[1:], dtype=bool)
    inputs = []
    for height, mask, timestamp in zip(heights, valid, timestamps):
        observed = mask & np.isfinite(height)
        last_seen[observed] = timestamp
        seen |= observed
        age = np.where(seen, np.clip((timestamp - last_seen) / age_scale_ms, 0, 1), 1)
        inputs.append(np.stack((normalization.normalize(height, observed),
                                observed.astype(np.float32), age.astype(np.float32),
                                (~seen).astype(np.float32))))
    return np.stack(inputs)


class OcclusionDataset:
    """NumPy DataLoader-compatible samples with window-local observation age.

    Input channels are normalized observed height, surface observation mask,
    clipped age / age_scale_ms, and never-observed flag. Never-observed cells
    have age 1; the fourth channel distinguishes them from old observations.
    Targets are current and next `horizon` clean heights with independent
    reference validity. The mask applies only to history, never to labels.
    """

    def __init__(self, sessions: list[Session], config: dict, normalization: Normalization,
                 *, seed: int = 42, cache_frames: bool = False, window_stride: int = 1):
        self.config = _settings(config, seed)
        if type(cache_frames) is not bool:
            raise ValueError('cache_frames must be a boolean')
        if type(window_stride) is not int or window_stride < 1:
            raise ValueError('window_stride must be a positive integer')
        self.sessions = list(sessions)
        self.normalization = normalization
        self.seed = seed
        self.window_stride = window_stride
        self._frame_cache = {} if cache_frames else None
        self.windows: list[tuple[int, int]] = []
        self._candidates = self._timing_rejected = self._coverage_rejected = 0
        seen = set()
        geometry = None
        span = self.config['history'] + self.config['horizon']
        for session_index, session in enumerate(self.sessions):
            if session.path in seen:
                raise ValueError(f'duplicate session in dataset: {session.path}')
            seen.add(session.path)
            if geometry is not None and session.geometry != geometry:
                raise ValueError('dataset contains incompatible recording geometry')
            geometry = session.geometry
            times = np.asarray([frame.camera_timestamp_ms for frame in session.frames])
            interval_ok = np.abs(np.diff(times) - self.config['expected_dt_ms']) <= (
                self.config['expected_dt_ms'] * self.config['timing_tolerance'] + 1e-6)
            coverage = np.asarray([frame.valid_fraction for frame in session.frames])
            for start in range(0, max(0, len(session.frames) - span + 1), window_stride):
                self._candidates += 1
                if not np.all(interval_ok[start:start + span - 1]):
                    self._timing_rejected += 1
                    continue
                if np.any(coverage[start:start + span] < self.config['min_valid_fraction']):
                    self._coverage_rejected += 1
                    continue
                self.windows.append((session_index, start))

    def __len__(self) -> int:
        return len(self.windows)

    def _read_frame(self, session: Session, index: int):
        path = session.frames[index].path
        if self._frame_cache is None:
            return _load_arrays(path, session.shape)
        if path not in self._frame_cache:
            self._frame_cache[path] = _load_arrays(path, session.shape)
        return self._frame_cache[path]

    def __getitem__(self, index: int) -> dict:
        session_index, start = self.windows[index]
        session = self.sessions[session_index]
        history = self.config['history']
        origin = start + history - 1
        # Stable across processes and dataset iteration order. Future content is
        # deliberately absent from the mask seed and observation calculation.
        identity = f'{self.seed}\0{session.path}\0{start}'.encode('utf-8')
        mask_seed = int.from_bytes(hashlib.sha256(identity).digest()[:8], 'little')
        rng = np.random.default_rng(mask_seed)
        if self.config['occlusion_shape'] == 'gaussian':
            masks = _gaussian_masks(session.shape, history, self.config['occlusion_fraction'], rng,
                                    self.config['gaussian_step_interval'])
        else:
            masks = _masks(session.shape, history, self.config['occlusion_fraction'], rng)
        heights, observed_masks, timestamps = [], [], []
        current_valid = current_observed = None
        for offset in range(history):
            frame_index = start + offset
            height, valid = self._read_frame(session, frame_index)
            observed = valid & ~masks[offset]
            heights.append(height)
            observed_masks.append(observed)
            timestamps.append(session.frames[frame_index].camera_timestamp_ms)
            current_valid, current_observed = valid, observed
        targets, target_valid = [], []
        for frame_index in range(origin, origin + self.config['horizon'] + 1):
            height, valid = self._read_frame(session, frame_index)
            targets.append(self.normalization.normalize(height, valid)[None])
            target_valid.append(valid[None])
        return {
            'inputs': encode_history(heights, observed_masks, timestamps, self.normalization,
                                     self.config['age_scale_ms']),
            'targets': np.stack(targets),
            'target_valid': np.stack(target_valid),
            'hidden_origin': (current_valid & ~current_observed)[None],
        }

    def summary(self) -> dict:
        return {'sessions': len(self.sessions),
                'frames': sum(len(session.frames) for session in self.sessions),
                'candidate_windows': self._candidates, 'windows': len(self.windows),
                'rejected_timing': self._timing_rejected,
                'rejected_coverage': self._coverage_rejected,
                'history': self.config['history'], 'horizon': self.config['horizon'],
                'seed': self.seed, 'occlusion_fraction': self.config['occlusion_fraction'],
                'occlusion_shape': self.config['occlusion_shape'],
                'gaussian_step_interval': self.config['gaussian_step_interval'],
                'window_stride': self.window_stride, 'cache_frames': self._frame_cache is not None}
