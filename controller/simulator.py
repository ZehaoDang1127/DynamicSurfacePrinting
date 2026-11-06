"""Printer-frame analytical surfaces for causal controller and gap-error tests."""

import math

import numpy as np

from .types import Observation


class MovingPlane:
    def __init__(self, config, scenario="moving-plane", amplitude_mm=.5, frequency_hz=.1):
        if scenario not in {"moving-plane", "static-plane", "sloped-plane"}:
            raise ValueError("unknown synthetic scenario")
        if not math.isfinite(amplitude_mm) or amplitude_mm < 0 or not math.isfinite(frequency_hz) or frequency_hz < 0:
            raise ValueError("invalid synthetic motion")
        self.config = config
        self.amplitude = amplitude_mm if scenario == "moving-plane" else 0.0
        self.frequency = frequency_hz
        self.slope = .05 if scenario == "sloped-plane" else 0.0
        self.x, self.y = np.meshgrid(config.grid.x_centers_mm(), config.grid.y_centers_mm())

    def height(self, xy, at_s):
        relative = self.slope * (np.asarray(xy)[..., 0]-self.config.grid.x_min_mm) + self.amplitude * math.sin(
            2*math.pi*self.frequency*at_s)
        return relative if self.config.height_reference == "relative" else self.config.substrate_height_mm + relative

    def observation(self, frame_id, at_s, dropout=False):
        height = self.height(np.stack((self.x, self.y), axis=-1), at_s)
        valid = np.ones_like(height, dtype=bool)
        if dropout:
            valid[:] = False
        return Observation(frame_id, height, valid, np.full(height.shape, 4, dtype=np.uint32),
                           self.config.geometry_id, at_s*1000, "virtual_camera", at_s, at_s)
