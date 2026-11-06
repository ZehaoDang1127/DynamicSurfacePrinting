"""Strict run configuration; an unknown substrate datum is represented by null."""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

from surface_mapper.config import GridConfig

from .types import ControllerFault


def fingerprint(raw: dict) -> str:
    return hashlib.sha256(json.dumps(raw, sort_keys=True, allow_nan=False).encode()).hexdigest()


def positive(value, name, allow_zero=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite and numeric")
    if value < 0 or (not allow_zero and value == 0):
        raise ValueError(f"{name} must be {'nonnegative' if allow_zero else 'positive'}")


@dataclass(frozen=True)
class MotionLimits:
    xyz_min_mm: tuple[float, ...] = (0, 0, 0)
    xyz_max_mm: tuple[float, ...] = (350, 350, 300)
    xy_speed_mm_s: float = 5.0
    xyz_speed_mm_s: float = 10.0
    z_speed_mm_s: float = 2.0
    xy_accel_mm_s2: float = 50.0
    z_accel_mm_s2: float = 20.0
    max_slope: float = 0.2
    max_flow_mm3_s: float = 2.0
    max_extrude_ratio: float = 0.3

    def __post_init__(self):
        lo, hi = tuple(self.xyz_min_mm), tuple(self.xyz_max_mm)
        if len(lo) != 3 or len(hi) != 3 or not all(math.isfinite(v) for v in lo + hi):
            raise ValueError("machine bounds need three finite coordinates")
        if any(a >= b for a, b in zip(lo, hi)):
            raise ValueError("machine bounds must increase")
        object.__setattr__(self, "xyz_min_mm", lo)
        object.__setattr__(self, "xyz_max_mm", hi)
        for name, value in vars(self).items():
            if name not in {"xyz_min_mm", "xyz_max_mm"}:
                positive(value, name)


@dataclass(frozen=True)
class PredictorConfig:
    kind: str = "constant_velocity"
    history: int = 5
    max_gap_s: float = 0.08
    max_frame_age_s: float = 0.10
    max_horizon_s: float = 1.0
    forecast_error_mm: float = 0.1
    checkpoint: str | None = None
    device: str = "auto"
    temporal_policy: str = "fixed"
    interpolation_error_mm: float = 0.0

    def __post_init__(self):
        if self.kind not in {"persistence", "constant_velocity", "constant_acceleration", "convlstm"}:
            raise ValueError("unsupported predictor")
        if type(self.history) is not int or self.history < 3:
            raise ValueError("predictor history must be an integer >= 3")
        for key in ("max_gap_s", "max_frame_age_s", "max_horizon_s"):
            positive(getattr(self, key), key)
        positive(self.forecast_error_mm, "forecast_error_mm", True)
        if self.temporal_policy not in {"fixed", "linear_to_horizon"}:
            raise ValueError("temporal_policy must be fixed or linear_to_horizon")
        positive(self.interpolation_error_mm, "interpolation_error_mm", True)
        if self.temporal_policy == "linear_to_horizon":
            if self.kind != "convlstm":
                raise ValueError("linear_to_horizon requires the ConvLSTM predictor")
            if self.interpolation_error_mm <= 0:
                raise ValueError("linear_to_horizon requires a positive explicit interpolation_error_mm")
        if self.kind == "convlstm" and not self.checkpoint:
            raise ValueError("convlstm requires a checkpoint")


@dataclass(frozen=True)
class HardwareConfig:
    backend: str = "moonraker"  # Preserve existing saved configurations that omit backend.
    klipper_socket_path: str | None = None
    moonraker_url: str | None = None
    api_key_env: str = "MOONRAKER_API_KEY"
    request_timeout_s: float = 3.0
    calibration_id: str | None = None
    camera_config_sha256: str | None = None
    expires_utc: str | None = None
    stationary_fixture_verified: bool = False
    stop_test_verified: bool = False
    exclusive_control_verified: bool = False
    cold_clearance_mm: float = 5.0
    fixture_top_z_mm: float | None = None
    max_cold_temperature_c: float = 45.0
    max_batch_s: float = 2.0

    def __post_init__(self):
        if self.backend not in {"klipper_socket", "moonraker"}:
            raise ValueError("hardware.backend must be klipper_socket or moonraker")
        if self.klipper_socket_path is not None and (not isinstance(self.klipper_socket_path, str)
                or not self.klipper_socket_path.startswith("/") or "\x00" in self.klipper_socket_path):
            raise ValueError("klipper_socket_path must be an absolute Unix socket path")
        for key in ("stationary_fixture_verified", "stop_test_verified", "exclusive_control_verified"):
            if type(getattr(self, key)) is not bool:
                raise ValueError(f"{key} must be boolean")
        for key in ("request_timeout_s", "cold_clearance_mm", "max_cold_temperature_c", "max_batch_s"):
            positive(getattr(self, key), key)
        if self.fixture_top_z_mm is not None:
            positive(self.fixture_top_z_mm, "fixture_top_z_mm", True)
        if not self.api_key_env or not isinstance(self.api_key_env, str):
            raise ValueError("api_key_env must be an environment variable name")


@dataclass(frozen=True)
class ControllerConfig:
    grid: GridConfig
    schema_version: int = 1
    units: str = "mm_s"
    measurement_frame: str = "printer_relative"
    height_reference: str = "relative"
    substrate_height_mm: float | None = None
    geometry_id: str = "uncalibrated-example"
    desired_gap_mm: float = 5.0
    allowed_gap_error_mm: float = 0.5
    calibration_error_mm: float = 0.1
    mechanical_error_mm: float = 0.1
    timing_uncertainty_s: float = 0.01
    footprint_radius_mm: float = 1.0
    path_spacing_mm: float = 1.0
    proposal_moves: int = 2
    max_plan_iterations: int = 30
    convergence_s: float = 0.0005
    bead_area_mm2: float = 0.08
    filament_diameter_mm: float = 1.75
    limits: MotionLimits = field(default_factory=MotionLimits)
    predictor: PredictorConfig = field(default_factory=PredictorConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1 or self.units != "mm_s":
            raise ValueError("controller requires schema 1 and mm_s units")
        if self.height_reference not in {"relative", "printer_z"}:
            raise ValueError("height_reference must be relative or printer_z")
        if not self.geometry_id or not self.measurement_frame:
            raise ValueError("geometry_id and measurement_frame are required")
        if self.substrate_height_mm is not None:
            positive(self.substrate_height_mm, "substrate_height_mm", True)
        for key in ("desired_gap_mm", "allowed_gap_error_mm", "path_spacing_mm", "convergence_s",
                    "bead_area_mm2", "filament_diameter_mm"):
            positive(getattr(self, key), key)
        for key in ("calibration_error_mm", "mechanical_error_mm", "timing_uncertainty_s", "footprint_radius_mm"):
            positive(getattr(self, key), key, True)
        for key in ("proposal_moves", "max_plan_iterations"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f"{key} must be a positive integer")
        if self.provisional_error_mm >= self.allowed_gap_error_mm:
            raise ValueError("provisional calibration/forecast/interpolation/mechanical error exceeds gap budget")

    @property
    def provisional_error_mm(self):
        return (self.calibration_error_mm + self.predictor.forecast_error_mm
                + self.predictor.interpolation_error_mm + self.mechanical_error_mm)

    @property
    def config_id(self):
        return fingerprint(self.to_dict())

    def to_dict(self):
        return asdict(self)

    def surface_z(self, measured_mm: float):
        if self.height_reference == "printer_z":
            return measured_mm
        if self.substrate_height_mm is None:
            raise ControllerFault("substrate_height_mm is null; enter the measured substrate top Z in printer coordinates")
        return self.substrate_height_mm + measured_mm

    def require_cold_hardware(self):
        # No inference about hardware readiness is made from example numerical limits.
        if self.substrate_height_mm is None:
            raise ControllerFault("substrate_height_mm is null; cold setup requires the measured substrate top Z")
        hw = self.hardware
        connection = hw.klipper_socket_path if hw.backend == "klipper_socket" else hw.moonraker_url
        if not connection or not hw.calibration_id or not hw.camera_config_sha256:
            raise ControllerFault("cold motion requires a configured printer connection and versioned camera calibration/hash")
        if not all((hw.stationary_fixture_verified, hw.stop_test_verified, hw.exclusive_control_verified)):
            raise ControllerFault("cold motion requires stationary fixture, tested physical stop, and exclusive control")
        if hw.fixture_top_z_mm is None or not hw.expires_utc:
            raise ControllerFault("cold motion requires fixture_top_z_mm and expires_utc")
        try:
            expiry = datetime.fromisoformat(hw.expires_utc.replace("Z", "+00:00"))
            if expiry.tzinfo is None or expiry <= datetime.now(timezone.utc):
                raise ValueError()
        except (TypeError, ValueError):
            raise ControllerFault("hardware calibration expiry is invalid or expired") from None


def from_dict(raw: dict) -> ControllerConfig:
    raw = dict(raw)
    try:
        raw["grid"] = GridConfig(**raw["grid"])
        for name, cls in (("limits", MotionLimits), ("predictor", PredictorConfig), ("hardware", HardwareConfig)):
            raw[name] = cls(**raw.get(name, {}))
        json.dumps(asdict(raw["grid"]), allow_nan=False)
        return ControllerConfig(**raw)
    except (TypeError, KeyError) as exc:
        raise ValueError(f"invalid controller configuration: {exc}") from exc


def load_config(path: str | Path) -> ControllerConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(raw, dict):
        raise ValueError("controller config must be an object")
    json.dumps(raw, allow_nan=False)
    return from_dict(raw)
