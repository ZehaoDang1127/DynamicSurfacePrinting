"""Twenty-five reproducible, speed-bounded servo trajectories (no hardware imports)."""

from bisect import bisect_right
import csv
from dataclasses import dataclass
import math
from pathlib import Path
import random
import time

from gobilda_servo import UPDATE_INTERVAL


@dataclass(frozen=True)
class Pattern:
    key: str
    description: str


PATTERNS = {
    key: Pattern(key, description) for key, description in [
        ("01_triangle_sweep", "Constant-speed sweep out and back"),
        ("02_sine_sweep", "Sinusoidal sweep with smooth reversals"),
        ("03_cubic_sweep", "Cubic ease-in/ease-out sweep"),
        ("04_quintic_sweep", "Quintic sweep with gentle acceleration changes"),
        ("05_asymmetric_sweep", "Slow outward sweep and faster return"),
        ("06_sweep_with_holds", "Sweep with extended holds at both ends"),
        ("07_staircase", "Five-level ascending and descending staircase"),
        ("08_random_waypoints", "Seeded random waypoints with smooth transitions"),
        ("09_small_oscillation", "Four small sinusoidal oscillations"),
        ("10_medium_oscillation", "Two medium sinusoidal oscillations"),
        ("11_expanding_oscillation", "Oscillations with increasing amplitude"),
        ("12_contracting_oscillation", "Oscillations with decreasing amplitude"),
        ("13_chirp_up", "Oscillation frequency increases over time"),
        ("14_chirp_down", "Oscillation frequency decreases over time"),
        ("15_two_frequency", "Superimposed slow and fast oscillations"),
        ("16_center_out", "Alternating center-to-edge excursions"),
        ("17_double_pulse", "Two ramped pulses with different heights"),
        ("18_stop_start", "Progressive sweep interrupted by holds"),
        ("19_damped_oscillation", "Exponentially damped oscillation"),
        ("20_biased_oscillation", "Oscillation around a gradually shifting center"),
        ("21_amplitude_beats", "Oscillation with a repeating beat envelope"),
        ("22_slow_fast_slow", "Slow oscillation, faster middle section, then slow again"),
        ("23_three_plateau", "Three unequal position holds with smooth transitions"),
        ("24_seeded_random_walk", "Seeded incremental random walk with smooth transitions"),
        ("25_interrupted_chirp", "Oscillation, a center hold, then an accelerating oscillation"),
    ]
}

ORIGINAL_PATTERNS = tuple(key for key in PATTERNS if int(key[:2]) <= 20)
NEW_PATTERNS = tuple(key for key in PATTERNS if int(key[:2]) > 20)


def resolve_pattern(value):
    """Accept a full name, an unnumbered name, or a number from 1 to 25."""
    for key in PATTERNS:
        if value in (key, key[3:], str(int(key[:2])), key[:2]):
            return key
    raise ValueError(f"Unknown pattern {value!r}; use --list-patterns")


def _piecewise(u, points, curve="linear"):
    index = min(len(points) - 2, bisect_right([p[0] for p in points], u) - 1)
    t0, y0 = points[index]
    t1, y1 = points[index + 1]
    v = max(0.0, min(1.0, (u - t0) / (t1 - t0)))
    if curve == "cubic":
        v = v * v * (3 - 2 * v)
    elif curve == "quintic":
        v = v**3 * (10 - 15 * v + 6 * v * v)
    elif curve == "cosine":
        v = (1 - math.cos(math.pi * v)) / 2
    return y0 + (y1 - y0) * v


def _shape(key, u, random_points, walk_points):
    number = int(key[:2])
    if number in (1, 3, 4):
        curve = {1: "linear", 3: "cubic", 4: "quintic"}[number]
        return _piecewise(u, [(0, -1), (0.5, 1), (1, -1)], curve)
    if number == 2:
        return -math.cos(2 * math.pi * u)
    if number == 5:
        return _piecewise(u, [(0, -1), (0.7, 1), (1, -1)])
    if number == 6:
        return _piecewise(u, [(0, -1), (0.1, -1), (0.4, 1), (0.6, 1), (0.9, -1), (1, -1)])
    if number == 7:
        levels = [-1, -0.5, 0, 0.5, 1, 0.5, 0, -0.5, -1]
        points = [(0, levels[0])]
        for i, level in enumerate(levels[1:]):
            points.extend([((i + 0.35) / 8, levels[i]), ((i + 1) / 8, level)])
        return _piecewise(u, points)
    if number == 8:
        return _piecewise(u, random_points, "cubic")
    if number == 9:
        return 0.35 * math.sin(8 * math.pi * u)
    if number == 10:
        return 0.7 * math.sin(4 * math.pi * u)
    if number == 11:
        return (0.15 + 0.85 * u) * math.sin(6 * math.pi * u)
    if number == 12:
        return (1 - 0.85 * u) * math.sin(6 * math.pi * u)
    if number == 13:
        return 0.65 * math.sin(2 * math.pi * (u + 2 * u * u))
    if number == 14:
        return 0.65 * math.sin(2 * math.pi * (5 * u - 2 * u * u))
    if number == 15:
        return 0.65 * math.sin(2 * math.pi * u) + 0.25 * math.sin(10 * math.pi * u)
    if number == 16:
        levels = [0, 0.5, 0, -0.5, 0, 1, 0, -1, 0]
        return _piecewise(u, [(i / 8, y) for i, y in enumerate(levels)], "cosine")
    if number == 17:
        return _piecewise(u, [(0, 0), (0.1, 0), (0.2, 1), (0.3, 1), (0.4, 0),
                              (0.55, 0), (0.65, -0.6), (0.75, -0.6), (0.85, 0), (1, 0)])
    if number == 18:
        return _piecewise(u, [(0, -1), (0.15, -0.4), (0.25, -0.4), (0.4, 0.3),
                              (0.5, 0.3), (0.65, 1), (0.75, 1), (1, -1)])
    if number == 19:
        return math.exp(-2.5 * u) * math.sin(8 * math.pi * u)
    if number == 20:
        return -0.5 + u + 0.4 * math.sin(6 * math.pi * u)
    if number == 21:
        return 0.9 * math.cos(2 * math.pi * u) * math.sin(8 * math.pi * u)
    if number == 22:
        phase = 2 * u if u < 0.25 else (0.5 + 6 * (u - 0.25) if u < 0.75 else 3.5 + 2 * (u - 0.75))
        return 0.65 * math.sin(2 * math.pi * phase)
    if number == 23:
        return _piecewise(u, [(0, 0), (0.1, -0.8), (0.25, -0.8), (0.4, 0.35),
                              (0.55, 0.35), (0.7, 0.9), (0.85, 0.9), (1, 0)], "quintic")
    if number == 24:
        return _piecewise(u, walk_points, "cubic")
    if number == 25:
        if u < 0.4:
            return 0.6 * math.sin(2 * math.pi * 2.5 * u)
        if u < 0.6:
            return 0.0
        v = u - 0.6
        return 0.6 * math.sin(2 * math.pi * (2.5 * v + 6.25 * v * v))
    raise ValueError(key)


@dataclass(frozen=True)
class Trajectory:
    pattern: str
    description: str
    duration_s: float
    seed: int
    samples: tuple  # (elapsed seconds, commanded degrees)
    amplitude_scale_deg: float
    max_speed_deg_s: float

    def metadata(self):
        angles = [angle for _, angle in self.samples]
        return {
            "pattern": self.pattern, "description": self.description, "seed": self.seed,
            "duration_s": self.duration_s, "sample_count": len(self.samples),
            "nominal_update_interval_s": UPDATE_INTERVAL,
            "min_angle_deg": min(angles), "max_angle_deg": max(angles),
            "amplitude_scale_deg": self.amplitude_scale_deg,
            "max_planned_speed_deg_s": self.max_speed_deg_s,
            "amplitude_policy": "shrink about range center as needed to honor speed and duration",
        }

    def write_csv(self, output):
        with (Path(output) / "trajectory.csv").open("x", newline="") as handle:
            rows = csv.writer(handle)
            rows.writerow(["elapsed_s", "commanded_angle_deg", "pulse_us"])
            for elapsed, angle in self.samples:
                rows.writerow([f"{elapsed:.9f}", f"{angle:.9f}", f"{500 + angle * 2000 / 300:.9f}"])


def build_trajectory(pattern, duration=30.0, start=30.0, end=270.0, speed=30.0, seed=42):
    """Sample a complete pattern and scale its amplitude to respect the speed cap."""
    key = resolve_pattern(pattern)
    if not all(math.isfinite(v) for v in (duration, start, end, speed)):
        raise ValueError("Trajectory settings must be finite")
    if duration <= 0 or speed <= 0 or not 0 <= start < end <= 300:
        raise ValueError("Require duration/speed > 0 and 0 <= start < end <= 300")
    # Leave a final hold so the last trajectory command occurs before capture ends.
    motion_duration = duration - min(0.25, duration / 10)
    steps = max(1, math.ceil(motion_duration / UPDATE_INTERVAL))
    times = [min(i * UPDATE_INTERVAL, motion_duration) for i in range(steps + 1)]
    rng = random.Random(seed)
    random_points = [(0, 0)] + [(i / 8, rng.uniform(-1, 1)) for i in range(1, 8)] + [(1, 0)]
    walk_points = [(0, 0)]
    for i in range(1, 13):
        walk_points.append((i / 12, max(-1.0, min(1.0, walk_points[-1][1] + rng.uniform(-0.45, 0.45)))))
    values = [_shape(key, t / motion_duration, random_points, walk_points) for t in times]
    max_value = max(1.0, max(abs(value) for value in values))
    max_slope = max(abs(b - a) / (t1 - t0)
                    for (t0, a), (t1, b) in zip(zip(times, values), zip(times[1:], values[1:])))
    amplitude = min((end - start) / (2 * max_value), speed / max_slope) if max_slope else (end - start) / 2
    center = (start + end) / 2
    samples = tuple((t, center + amplitude * value) for t, value in zip(times, values))
    return Trajectory(key, PATTERNS[key].description, motion_duration, seed, samples,
                      amplitude, amplitude * max_slope)


def play_trajectory(servo, trajectory, origin_ns, stop):
    """Follow absolute deadlines; cancellation interrupts holds as well as moves."""
    for elapsed, angle in trajectory.samples:
        remaining = (origin_ns + round(elapsed * 1e9) - time.monotonic_ns()) / 1e9
        if stop.wait(max(0.0, remaining)):
            return False
        servo.set_angle(angle)
    return True
