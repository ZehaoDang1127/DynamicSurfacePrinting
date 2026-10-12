"""Validated recording sessions and finite, masked NumPy training examples.

Frames stay on disk rather than loading entire recordings into RAM. Splits are
made at session level before windows are constructed to prevent temporal leakage.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from surface_mapper.config import AppConfig, CameraConfig, GridConfig

from .config import PredictionConfig


@dataclass(frozen=True)
class Normalization:
    mean_mm: float
    scale_mm: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.mean_mm):
            raise ValueError('normalization mean_mm must be finite')
        if not math.isfinite(self.scale_mm) or self.scale_mm < 1e-3:
            raise ValueError('normalization scale_mm must be finite and at least 1e-3')

    def to_dict(self) -> dict:
        return {'mean_mm': self.mean_mm, 'scale_mm': self.scale_mm}

    @classmethod
    def from_dict(cls, raw: dict) -> 'Normalization':
        return cls(**raw)

    def normalize(self, height: np.ndarray, valid: np.ndarray) -> np.ndarray:
        height = np.asarray(height)
        valid = np.asarray(valid, dtype=bool)
        if height.shape != valid.shape:
            raise ValueError('height and validity mask shapes differ')
        mask = valid & np.isfinite(height)
        result = np.zeros(height.shape, dtype=np.float32)
        normalized = (height[mask].astype(np.float64) - self.mean_mm) / self.scale_mm
        if np.any(np.abs(normalized) > np.finfo(np.float32).max):
            raise ValueError('normalized depth exceeds float32 range')
        result[mask] = normalized
        return result

    def denormalize(self, array: np.ndarray) -> np.ndarray:
        return np.asarray(array) * self.scale_mm + self.mean_mm


@dataclass(frozen=True)
class FrameRecord:
    path: Path
    frame_number: int
    camera_timestamp_ms: float
    host_monotonic_ns: int
    timestamp_domain: str
    valid_fraction: float
    file_identity: tuple[int, int] | None = None


@dataclass(frozen=True)
class Session:
    path: Path
    manifest: dict
    shape: tuple[int, int]
    frames: tuple[FrameRecord, ...]
    geometry: dict


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding='utf-8-sig'))
    except (OSError, ValueError) as exc:
        raise ValueError(f'cannot read JSON {path}: {exc}') from exc
    if not isinstance(value, dict):
        raise ValueError(f'{path}: expected a JSON object')
    return value


def geometry_signature(config_dict: dict) -> dict:
    """Canonical geometry/stream settings, excluding device and startup options."""
    try:
        if not isinstance(config_dict, dict):
            raise ValueError('config must be an object')
        frame_name = config_dict['frame_name']
        if not isinstance(frame_name, str) or not frame_name.strip():
            raise ValueError('frame_name must be a nonempty string')
        # Reject infinities in any numeric configuration before computing grids.
        json.dumps(config_dict, allow_nan=False)
        camera = CameraConfig(**config_dict['camera'])
        for name in ('width', 'height', 'fps'):
            if type(getattr(camera, name)) is not int:
                raise ValueError(f'camera {name} must be an integer')
        grid = GridConfig(**config_dict['grid'])
        if type(grid.min_points_per_cell) is not int:
            raise ValueError('min_points_per_cell must be an integer')
        app = AppConfig(frame_name, camera, grid, config_dict['camera_to_output'])
        signature = app.to_dict()
        signature['camera'] = {
            key: value for key, value in signature['camera'].items()
            if key not in {'serial', 'warmup_frames', 'timeout_ms'}
        }
        return json.loads(json.dumps(signature, allow_nan=False))
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'invalid recording geometry: {exc}') from exc


def _integer(value: Any, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f'{label} must be a nonnegative integer')
    return value


def _timestamp(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError(f'{label} must be finite and numeric')
    return float(value)


def _load_arrays(path: Path, shape: tuple[int, int], metadata: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            height = archive['height_mm']
            valid = archive['valid']
            count = archive['point_count']
            if height.shape != shape or valid.shape != shape or count.shape != shape:
                raise ValueError(f'frame shape disagrees with grid shape {shape}')
            if height.dtype.kind not in 'fi' or valid.dtype.kind != 'b':
                raise ValueError('height_mm must be real numeric and valid must be boolean')
            if count.dtype.kind not in 'iu' or np.any(count < 0):
                raise ValueError('point_count must contain nonnegative integers')
            if np.any(np.abs(height[np.isfinite(height)]) > np.finfo(np.float32).max):
                raise ValueError('height_mm exceeds float32 range')
            if metadata is not None:
                for name in ('frame_number', 'camera_timestamp_ms', 'host_monotonic_ns'):
                    value = archive[name]
                    if value.shape != () or value.dtype.kind not in 'iuf':
                        raise ValueError(f'{name} must be a numeric scalar')
                    scalar = value.item()
                    if not math.isfinite(scalar) or scalar != metadata[name]:
                        raise ValueError(f'{name} disagrees with manifest')
                    if name != 'camera_timestamp_ms' and value.dtype.kind not in 'iu':
                        raise ValueError(f'{name} must be an integer scalar')
                if 'timestamp_domain' in archive:
                    domain = archive['timestamp_domain']
                    if domain.shape != () or domain.item() != metadata['timestamp_domain']:
                        raise ValueError('timestamp_domain disagrees with manifest')
                if 'valid_fraction' in metadata:
                    fraction = _timestamp(metadata['valid_fraction'], 'valid_fraction')
                    if not 0 <= fraction <= 1 or not np.isclose(fraction, valid.mean(), atol=1e-7, rtol=0):
                        raise ValueError('valid_fraction disagrees with stored validity mask')
            valid = valid & np.isfinite(height)
            # Zero invalid values so baselines and torch batches never carry NaNs.
            height = np.where(valid, height, 0).astype(np.float32)
            return height, valid
    except (OSError, KeyError, ValueError, TypeError, EOFError) as exc:
        raise ValueError(f'invalid frame {path}: {exc}') from exc


def load_session(path: str | Path) -> Session:
    """Load a finished recorder session, validating every referenced frame."""
    directory = Path(path).expanduser().resolve()
    manifest = _read_json(directory / 'manifest.json')
    try:
        if manifest.get('schema_version') != 1:
            raise ValueError('unsupported schema_version (expected 1)')
        if manifest.get('complete') is not True:
            raise ValueError('recording is incomplete; finish recording before using it')
        geometry = geometry_signature(manifest['config'])
        grid = GridConfig(**geometry['grid'])
        shape_raw = manifest['grid']['shape']
        if not isinstance(shape_raw, list) or len(shape_raw) != 2 or any(type(v) is not int or v < 1 for v in shape_raw):
            raise ValueError('grid shape must contain two positive integers')
        shape = tuple(shape_raw)
        if shape != (grid.height, grid.width):
            raise ValueError('grid shape disagrees with config geometry')
        for axis, expected in [('x', grid.x_centers_mm()), ('y', grid.y_centers_mm())]:
            centers = np.asarray(manifest['grid'][f'{axis}_centers_mm'], dtype=float)
            if centers.shape != expected.shape or not np.all(np.isfinite(centers)) or not np.allclose(centers, expected, atol=1e-6, rtol=0):
                raise ValueError(f'grid {axis} centers disagree with config geometry')
        raw_frames = manifest['frames']
        if not isinstance(raw_frames, list) or not raw_frames:
            raise ValueError('frames must be a nonempty list')
        if _integer(manifest['frame_count'], 'frame_count') != len(raw_frames):
            raise ValueError('frame_count disagrees with frames list')
        frames: list[FrameRecord] = []
        paths: set[Path] = set()
        identities: set[tuple[int, int]] = set()
        domain = None
        for raw in raw_frames:
            if not isinstance(raw, dict):
                raise ValueError('every frame entry must be an object')
            filename = raw['file']
            if not isinstance(filename, str) or '/' in filename or '\\' in filename or not filename.startswith('frame_') or not filename.endswith('.npz'):
                raise ValueError('frame file must be a local frame_*.npz basename without traversal')
            frame_path = (directory / filename).resolve()
            if frame_path.parent != directory:
                raise ValueError('frame file resolves outside session directory')
            if frame_path in paths:
                raise ValueError(f'duplicate frame file: {filename}')
            stat = frame_path.stat()
            identity = (stat.st_dev, stat.st_ino)
            if stat.st_ino and identity in identities:
                raise ValueError(f'duplicate frame file identity: {filename}')
            paths.add(frame_path)
            identities.add(identity)
            number = _integer(raw['frame_number'], 'frame_number')
            timestamp = _timestamp(raw['camera_timestamp_ms'], 'camera_timestamp_ms')
            host = _integer(raw['host_monotonic_ns'], 'host_monotonic_ns')
            current_domain = raw['timestamp_domain']
            if not isinstance(current_domain, str) or not current_domain.strip():
                raise ValueError('timestamp_domain must be a nonempty string')
            if domain is not None and domain != current_domain:
                raise ValueError('timestamp_domain changes within recording')
            domain = current_domain
            if frames and (timestamp <= frames[-1].camera_timestamp_ms or host <= frames[-1].host_monotonic_ns or number <= frames[-1].frame_number):
                raise ValueError('frame numbers and camera/host timestamps must be strictly increasing')
            _, valid = _load_arrays(frame_path, shape, raw)
            frames.append(FrameRecord(frame_path, number, timestamp, host, domain, float(valid.mean()), identity))
        return Session(directory, manifest, shape, tuple(frames), geometry)
    except (KeyError, TypeError, ValueError, OSError, OverflowError) as exc:
        raise ValueError(f'invalid session {directory}: {exc}') from exc


def load_splits(path: str | Path) -> dict[str, list[Session]]:
    """Resolve session paths relative to a split JSON and reject data leakage."""
    split_path = Path(path).expanduser().resolve()
    raw = _read_json(split_path)
    if set(raw) != {'train', 'validation', 'test'}:
        raise ValueError('split JSON must contain exactly train, validation, and test')
    result: dict[str, list[Session]] = {}
    seen_sessions: set[Path] = set()
    seen_frames: set[Path] = set()
    seen_identities: set[tuple[int, int]] = set()
    geometry = None
    for name in ('train', 'validation', 'test'):
        entries = raw[name]
        if not isinstance(entries, list) or not entries:
            raise ValueError(f'{name} split must be a nonempty list of session paths')
        result[name] = []
        for entry in entries:
            if not isinstance(entry, str) or not entry.strip():
                raise ValueError(f'{name} split contains an invalid session path')
            session_path = (split_path.parent / Path(entry).expanduser()).resolve()
            if session_path in seen_sessions:
                raise ValueError(f'session occurs more than once across splits: {session_path}')
            seen_sessions.add(session_path)
            session = load_session(session_path)
            if geometry is not None and session.geometry != geometry:
                raise ValueError(f'incompatible recording geometry: {session.path}')
            geometry = session.geometry
            for frame in session.frames:
                identity = frame.file_identity
                if identity is None:
                    stat = frame.path.stat()
                    identity = (stat.st_dev, stat.st_ino)
                if frame.path in seen_frames or (identity[1] and identity in seen_identities):
                    raise ValueError(f'frame occurs more than once across splits: {frame.path}')
                seen_frames.add(frame.path)
                seen_identities.add(identity)
            result[name].append(session)
    return result


def fit_normalization(sessions: list[Session]) -> Normalization:
    """Population statistics over each valid training frame once (not windows)."""
    total = 0
    mean = 0.0
    m2 = 0.0
    seen: set[Path] = set()
    for session in sessions:
        if session.path in seen:
            continue
        seen.add(session.path)
        for frame in session.frames:
            height, valid = _load_arrays(frame.path, session.shape)
            values = height[valid].astype(np.float64)
            count = values.size
            if not count:
                continue
            batch_mean = float(values.mean())
            batch_m2 = float(np.sum((values - batch_mean) ** 2))
            delta = batch_mean - mean
            combined = total + count
            m2 += batch_m2 + delta * delta * total * count / combined
            mean += delta * count / combined
            total = combined
    if total == 0:
        raise ValueError('training sessions contain no finite valid depth measurements')
    return Normalization(mean, max(math.sqrt(m2 / total), 1e-3))


class HeightSequenceDataset:
    """NumPy dataset compatible with torch DataLoader, without importing torch."""

    def __init__(self, sessions: list[Session], config: PredictionConfig, normalization: Normalization,
                 *, cache_frames: bool = False, window_stride: int = 1):
        if type(window_stride) is not int or window_stride < 1:
            raise ValueError('window_stride must be a positive integer')
        self.sessions = list(sessions)
        self.config = config
        self.normalization = normalization
        self.window_stride = window_stride
        self._frame_cache = {} if cache_frames else None
        self.windows: list[tuple[int, int, int]] = []
        self._candidates = 0
        self._timing_rejected = 0
        self._coverage_rejected = 0
        geometry = None
        seen: set[Path] = set()
        for session_index, session in enumerate(self.sessions):
            if session.path in seen:
                raise ValueError(f'duplicate session in dataset: {session.path}')
            seen.add(session.path)
            if geometry is not None and session.geometry != geometry:
                raise ValueError('dataset contains incompatible recording geometry')
            geometry = session.geometry
            timestamps = np.asarray([frame.camera_timestamp_ms for frame in session.frames])
            intervals = np.diff(timestamps)
            tolerance = config.expected_dt_ms * config.timing_tolerance
            interval_ok = np.abs(intervals - config.expected_dt_ms) <= tolerance + 1e-6
            coverage = np.asarray([frame.valid_fraction for frame in session.frames])
            span = config.history + config.horizon
            for start in range(0, max(0, len(session.frames) - span + 1), window_stride):
                self._candidates += 1
                last = start + config.history - 1
                target = last + config.horizon
                if not np.all(interval_ok[start:target]):
                    self._timing_rejected += 1
                    continue
                if np.any(coverage[start:last + 1] < config.min_valid_fraction) or coverage[target] < config.min_valid_fraction:
                    self._coverage_rejected += 1
                    continue
                self.windows.append((session_index, start, target))

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
        session_index, start, target_index = self.windows[index]
        session = self.sessions[session_index]
        last_index = start + self.config.history - 1
        history = [self._read_frame(session, i) for i in range(start, last_index + 1)]
        target, target_valid = self._read_frame(session, target_index)
        inputs = np.stack([
            np.stack((self.normalization.normalize(height, valid), valid.astype(np.float32)))
            for height, valid in history
        ])
        last = session.frames[last_index]
        previous = session.frames[last_index - 1]
        target_frame = session.frames[target_index]
        return {
            'inputs': inputs,
            'target': self.normalization.normalize(target, target_valid)[None],
            'target_valid': target_valid[None],
            # Physical-unit history is exclusively past observations; kinematic
            # baselines fit it using actual capture times, without target leakage.
            'history_mm': np.stack([height for height, _ in history]),
            'history_valid': np.stack([valid for _, valid in history]),
            'history_timestamps_ms': np.asarray([
                session.frames[i].camera_timestamp_ms
                for i in range(start, last_index + 1)
            ], dtype=np.float64),
            'last_mm': history[-1][0],
            'previous_mm': history[-2][0],
            'last_valid': history[-1][1],
            'previous_valid': history[-2][1],
            'dt_ms': last.camera_timestamp_ms - previous.camera_timestamp_ms,
            'target_dt_ms': target_frame.camera_timestamp_ms - last.camera_timestamp_ms,
            'session': str(session.path),
            'target_timestamp_ms': target_frame.camera_timestamp_ms,
            'last_timestamp_ms': last.camera_timestamp_ms,
        }

    def summary(self) -> dict:
        return {
            'sessions': len(self.sessions),
            'frames': sum(len(session.frames) for session in self.sessions),
            'candidate_windows': self._candidates,
            'windows': len(self.windows),
            'rejected_timing': self._timing_rejected,
            'rejected_coverage': self._coverage_rejected,
            'history': self.config.history,
            'horizon': self.config.horizon,
            'window_stride': self.window_stride,
        }
