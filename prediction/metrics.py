"""Physical-unit forecast error statistics, independent of PyTorch."""

import numpy as np


class ErrorStats:
    """Exact sums and a bounded uniform sample for the 95th percentile."""

    def __init__(self, max_samples: int = 100_000) -> None:
        self.count = 0
        self.absolute_sum = 0.0
        self.square_sum = 0.0
        self.max_samples = max_samples
        self.samples = np.empty(0, dtype=np.float64)
        self.priorities = np.empty(0, dtype=np.float64)
        self.rng = np.random.default_rng(42)

    def update(self, errors) -> None:
        errors = np.asarray(errors, dtype=np.float64).ravel()
        if not np.isfinite(errors).all():
            raise ValueError('non-finite errors cannot be scored')
        self.count += errors.size
        self.absolute_sum += float(np.abs(errors).sum())
        self.square_sum += float(np.square(errors).sum())
        values = np.concatenate((self.samples, np.abs(errors)))
        priorities = np.concatenate((self.priorities, self.rng.random(errors.size)))
        if values.size > self.max_samples:
            keep = np.argpartition(priorities, self.max_samples - 1)[:self.max_samples]
            values, priorities = values[keep], priorities[keep]
        self.samples, self.priorities = values, priorities

    def result(self) -> dict:
        return {
            'valid_pixels': int(self.count),
            'mae_mm': self.absolute_sum / self.count if self.count else None,
            'rmse_mm': float(np.sqrt(self.square_sum / self.count)) if self.count else None,
            'p95_absolute_error_mm': float(np.percentile(self.samples, 95)) if self.count else None,
            'p95_sample_count': int(self.samples.size),
            'p95_is_sampled': self.count > self.max_samples,
        }
