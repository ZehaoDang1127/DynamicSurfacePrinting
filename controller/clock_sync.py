"""Affine clock mapping with explicit uncertainty; receipt time is not exposure time."""

from dataclasses import dataclass
import math

from .types import ControllerFault


@dataclass
class ClockMapper:
    domain: str
    source_origin_ms: float
    host_origin_s: float
    scale: float = 1.0
    uncertainty_s: float = 0.0
    max_residual_s: float = 0.02
    last_source_ms: float | None = None
    last_frame: int | None = None

    def __post_init__(self):
        if not self.domain or not all(math.isfinite(x) for x in (
                self.source_origin_ms, self.host_origin_s, self.scale, self.uncertainty_s, self.max_residual_s)):
            raise ValueError("invalid clock mapping")
        if self.scale <= 0 or self.uncertainty_s < 0 or self.max_residual_s <= 0:
            raise ValueError("invalid clock scale/uncertainty")

    def map(self, camera_ms, domain, receipt_s, frame_id):
        if not math.isfinite(camera_ms) or not math.isfinite(receipt_s):
            raise ControllerFault("non-finite camera/host timestamp")
        if domain != self.domain:
            raise ControllerFault("camera timestamp domain changed; rebuild clock mapping and history")
        if self.last_source_ms is not None and (camera_ms <= self.last_source_ms or frame_id <= self.last_frame):
            raise ControllerFault("camera clock or frame counter restarted")
        mapped = self.host_origin_s + (camera_ms - self.source_origin_ms) * self.scale / 1000.0
        if abs(receipt_s - mapped) > self.max_residual_s + self.uncertainty_s:
            raise ControllerFault("camera clock residual exceeds mapping envelope")
        self.last_source_ms, self.last_frame = camera_ms, frame_id
        return mapped


def network_offset_bounds(send_s, receive_s, server_s):
    """Offset interval for a server event inside an RTT; never a motion-clock mapping."""
    if not all(math.isfinite(t) for t in (send_s, receive_s, server_s)) or receive_s < send_s:
        raise ValueError("invalid round trip")
    return server_s - receive_s, server_s - send_s
