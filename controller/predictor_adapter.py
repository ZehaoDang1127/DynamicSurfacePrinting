"""Causal forecasts with an opt-in, bounded ConvLSTM temporal interpolation.

The network predicts one fixed training horizon. The linear_to_horizon policy
explicitly approximates intermediate maps between the newest measurement and
that endpoint; it neither changes the learned horizon nor extrapolates past it.
Its configured error allowance is provisional, not an accuracy measurement.
"""

from collections import deque
from dataclasses import replace

import numpy as np

from prediction.baselines import fit_kinematic_baselines

from .types import ControllerFault, ForecastMap


class PredictorAdapter:
    def __init__(self, config, model_geometry=None):
        self.config = config
        self.buffer = deque(maxlen=config.predictor.history)
        self.reset_reason = None
        self.model = None
        self.model_horizon_s = None
        self.fixed = None
        self.coefficients = None
        if config.predictor.kind == "convlstm":
            from prediction.inference import StreamingPredictor
            self.model = StreamingPredictor(config.predictor.checkpoint, config.predictor.device)
            if model_geometry != self.model.geometry:
                raise ValueError("ConvLSTM checkpoint geometry differs from calibrated input geometry")
            if self.model.config.history != config.predictor.history:
                raise ValueError("controller history differs from ConvLSTM checkpoint")
            try:
                fps = float(model_geometry["camera"]["fps"])
                configured_dt = 1000.0 / fps
            except (KeyError, TypeError, ValueError, ZeroDivisionError):
                raise ValueError("ConvLSTM input geometry must specify a valid camera cadence") from None
            if not np.isfinite(configured_dt) or configured_dt <= 0 or abs(
                    configured_dt - self.model.config.expected_dt_ms) > 1e-6:
                raise ValueError("camera cadence differs from ConvLSTM checkpoint training cadence")
            self.model_horizon_s = self.model.config.horizon * self.model.config.expected_dt_ms / 1000.0

    @property
    def ready(self):
        return len(self.buffer) == self.buffer.maxlen and (self.model is None or self.fixed is not None)

    def reset(self, reason):
        self.buffer.clear()
        self.coefficients = None
        self.fixed = None
        self.reset_reason = reason
        if self.model:
            self.model.reset(reason)

    def push(self, observation):
        expected = (self.config.grid.height, self.config.grid.width)
        if observation.height_mm.shape != expected or observation.geometry_id != self.config.geometry_id:
            self.reset("geometry changed")
            raise ControllerFault("observation grid/transform identity differs from run configuration")
        observation = replace(observation, valid=observation.valid &
                              (observation.point_count >= self.config.grid.min_points_per_cell) &
                              (observation.height_mm >= self.config.grid.z_min_mm) &
                              (observation.height_mm <= self.config.grid.z_max_mm))
        self.reset_reason = None
        if self.buffer:
            prev = self.buffer[-1]
            dt = observation.observed_s - prev.observed_s
            if observation.timestamp_domain != prev.timestamp_domain:
                self.reset("timestamp domain changed")
            elif observation.frame_id <= prev.frame_id or not 0 < dt <= self.config.predictor.max_gap_s:
                self.reset("frame/timestamp discontinuity")
        self.buffer.append(observation)
        if self.model:
            self.fixed = self.model.push(observation.height_mm, observation.valid,
                                         observation.camera_time_ms, observation.timestamp_domain,
                                         observation.frame_id)
        if not self.ready:
            return False
        if self.model is None:
            estimates = fit_kinematic_baselines(
                np.stack([f.height_mm for f in self.buffer]),
                np.stack([f.valid & (f.point_count >= self.config.grid.min_points_per_cell) for f in self.buffer]),
                np.asarray([f.observed_s for f in self.buffer]) * 1000, 1.0)
            estimate = estimates[self.config.predictor.kind]
            velocity = np.zeros(expected) if estimate.velocity_mm_s is None else estimate.velocity_mm_s
            acceleration = np.zeros(expected) if estimate.acceleration_mm_s2 is None else estimate.acceleration_mm_s2
            # Existing fits return derivatives at the newest observation, and prediction at +1 ms.
            intercept = estimate.prediction_mm.astype(float) - velocity * .001 - acceleration * .0000005
            # Complete-history support is conservative even for persistence.
            support = np.logical_and.reduce([f.valid & (f.point_count >= self.config.grid.min_points_per_cell)
                                             for f in self.buffer]) & estimate.prediction_valid
            self.coefficients = intercept, velocity, acceleration, support
        return True

    def forecast(self, target_s, now_s):
        if not self.ready:
            raise ControllerFault("predictor history is incomplete")
        latest = self.buffer[-1]
        age = now_s - latest.observed_s
        if age < -latest.uncertainty_s or age + latest.uncertainty_s > self.config.predictor.max_frame_age_s:
            raise ControllerFault("camera observation is stale or in the future")
        horizon = target_s - latest.observed_s
        if not np.isfinite(horizon) or horizon < -1e-9 or horizon > self.config.predictor.max_horizon_s:
            raise ControllerFault("arrival time is outside the validated forecast horizon")
        model_id = self.config.predictor.kind
        if self.model:
            supported = (self.fixed.target_timestamp_ms - self.fixed.source_timestamp_ms) / 1000
            if not np.isfinite(supported) or supported <= 0 or abs(supported - self.model_horizon_s) > 1e-6:
                raise ControllerFault("ConvLSTM forecast timestamp differs from its checkpoint horizon")
            endpoint = self.fixed.prediction_mm
            support = np.logical_and.reduce([
                f.valid & (f.point_count >= self.config.grid.min_points_per_cell) & np.isfinite(f.height_mm)
                for f in self.buffer]) & np.isfinite(endpoint)
            if self.config.predictor.temporal_policy == "linear_to_horizon":
                if horizon < -1e-9 or horizon > supported + 1e-9:
                    raise ControllerFault("ConvLSTM temporal interpolation cannot extrapolate beyond [0, checkpoint horizon]")
                # Only absorb floating-point endpoint rounding; never extrapolate.
                fraction = min(1.0, max(0.0, horizon / supported))
                values = np.full(endpoint.shape, np.nan, dtype=float)
                values[support] = latest.height_mm[support] + fraction * (
                    endpoint[support] - latest.height_mm[support])
                model_id = "convlstm+linear_temporal_interpolation"
            else:
                if abs(horizon - supported) > 1e-6:
                    raise ControllerFault("ConvLSTM provides one fixed horizon; requested arrival is unsupported")
                values = endpoint
        else:
            intercept, velocity, acceleration, support = self.coefficients
            values = intercept + velocity * horizon + .5 * acceleration * horizon ** 2
        return ForecastMap(values, support & np.isfinite(values),
                           tuple(f.frame_id for f in self.buffer), latest.observed_s,
                           target_s, model_id, self.config.predictor.forecast_error_mm
                           + self.config.predictor.interpolation_error_mm)
