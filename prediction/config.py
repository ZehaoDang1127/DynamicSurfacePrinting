"""Shared, checkpointed settings for training and inference."""

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class PredictionConfig:
    history: int = 10
    horizon: int = 1
    expected_dt_ms: float = 1000.0 / 30.0
    timing_tolerance: float = 0.2
    min_valid_fraction: float = 0.1
    hidden_channels: tuple[int, ...] = (32, 64)
    kernel_size: int = 3
    batch_size: int = 8
    learning_rate: float = 0.001
    epochs: int = 50
    patience: int = 8
    grad_clip: float = 1.0
    seed: int = 42

    def __post_init__(self) -> None:
        for name in ('history', 'horizon', 'kernel_size', 'batch_size', 'epochs', 'patience'):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if self.history < 2:
            raise ValueError('history must be at least 2 for velocity comparison')
        if self.kernel_size % 2 != 1:
            raise ValueError('kernel_size must be odd')
        if not self.hidden_channels or any(type(c) is not int or c < 1 for c in self.hidden_channels):
            raise ValueError('hidden_channels must contain positive integers')
        object.__setattr__(self, 'hidden_channels', tuple(self.hidden_channels))
        for name in ('expected_dt_ms', 'learning_rate', 'grad_clip'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if not 0 <= self.timing_tolerance < 1:
            raise ValueError('timing_tolerance must be in [0, 1)')
        if not 0 < self.min_valid_fraction <= 1:
            raise ValueError('min_valid_fraction must be in (0, 1]')
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError('seed must be an integer in [0, 2**32)')

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict) -> 'PredictionConfig':
        return cls(**raw)


def load_prediction_config(path: str | Path) -> PredictionConfig:
    return PredictionConfig.from_dict(json.loads(Path(path).read_text(encoding='utf-8-sig')))
