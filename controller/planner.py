"""Receding-horizon prototype with explicit rest-to-rest timing and rejection gates.

These timings describe the simulator. Stock Klipper may blend G1 moves; they are
not claimed to be its finalized timeline. Live dynamic streaming needs measured
schedule integration before this planner can drive it.
"""

import math

import numpy as np

from .types import ControllerFault, Move, Trajectory


def sample_map(values, support, grid, xy):
    x, y = xy
    fx = (x - grid.x_min_mm) / grid.cell_size_mm - .5
    fy = (y - grid.y_min_mm) / grid.cell_size_mm - .5
    ix, iy = math.floor(fx), math.floor(fy)
    if not 0 <= ix < grid.width - 1 or not 0 <= iy < grid.height - 1:
        raise ControllerFault("path leaves the measured grid interpolation interior")
    patch = values[iy:iy+2, ix:ix+2]
    if not support[iy:iy+2, ix:ix+2].all() or not np.isfinite(patch).all():
        raise ControllerFault("missing interpolation support on path")
    dx, dy = fx - ix, fy - iy
    return float((1-dy) * ((1-dx)*patch[0, 0] + dx*patch[0, 1])
                 + dy * ((1-dx)*patch[1, 0] + dx*patch[1, 1]))


def corridor_cells(grid, a_xy, b_xy, radius):
    # Every grid cell intersecting a conservative swept bounding rectangle.
    lo = np.minimum(a_xy, b_xy) - radius
    hi = np.maximum(a_xy, b_xy) + radius
    if lo[0] < grid.x_min_mm or lo[1] < grid.y_min_mm or hi[0] >= grid.x_max_mm or hi[1] >= grid.y_max_mm:
        raise ControllerFault("tool footprint leaves calibrated grid")
    ix0, iy0 = np.floor((lo - [grid.x_min_mm, grid.y_min_mm]) / grid.cell_size_mm).astype(int)
    ix1, iy1 = np.floor((hi - [grid.x_min_mm, grid.y_min_mm]) / grid.cell_size_mm).astype(int)
    return slice(iy0, iy1+1), slice(ix0, ix1+1)


def motion_profile(a, b, limits, speed_cap=None):
    delta = np.asarray(b) - np.asarray(a)
    distance = float(np.linalg.norm(delta))
    xy = float(np.linalg.norm(delta[:2]))
    if distance < 1e-9 or xy < 1e-9:
        raise ControllerFault("controller path requires positive XY travel")
    velocity = min(limits.xyz_speed_mm_s, limits.xy_speed_mm_s * distance / xy)
    accel = limits.xy_accel_mm_s2 * distance / xy
    if abs(delta[2]) > 1e-9:
        velocity = min(velocity, limits.z_speed_mm_s * distance / abs(delta[2]))
        accel = min(accel, limits.z_accel_mm_s2 * distance / abs(delta[2]))
    if speed_cap is not None:
        velocity = min(velocity, speed_cap)
    peak = min(velocity, math.sqrt(distance * accel))
    duration = distance / peak + peak / accel
    return duration, peak, accel


def profile_fraction(elapsed, duration, distance, peak):
    # Solve distance = peak*(duration-ramp), valid for triangle and trapezoid.
    ramp = duration - distance / peak
    accel = peak / ramp
    elapsed = min(max(elapsed, 0), duration)
    if elapsed <= ramp:
        travelled = .5 * accel * elapsed ** 2
    elif elapsed < duration-ramp:
        travelled = .5 * peak * ramp + peak * (elapsed-ramp)
    else:
        travelled = distance - .5 * accel * (duration-elapsed) ** 2
    return min(max(travelled / distance, 0), 1)


class Planner:
    def __init__(self, config):
        self.config = config

    def _bounds(self, xyz):
        if not np.isfinite(xyz).all() or np.any(np.asarray(xyz) < self.config.limits.xyz_min_mm) or np.any(
                np.asarray(xyz) > self.config.limits.xyz_max_mm):
            raise ControllerFault("trajectory exceeds configured machine workspace")

    def plan(self, predictor, start_xyz, next_xy, start_s, now_s, segment_id, predecessor_id=None, extrude=False):
        cfg = self.config
        self._bounds(start_xyz)
        if not np.isfinite(start_s) or start_s < now_s:
            raise ControllerFault("planning deadline already missed")
        points = np.asarray(next_xy, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or not len(points) or not np.isfinite(points).all():
            raise ControllerFault("invalid proposed XY section")
        committed = tuple(float(x) for x in start_xyz)
        arrivals = []
        xyz = committed
        t = start_s
        for xy in points:
            guess = (*xy, xyz[2])
            duration, _, _ = motion_profile(xyz, guess, cfg.limits)
            t += duration
            arrivals.append(t)
            xyz = guess
        for _ in range(cfg.max_plan_iterations):
            xyz, t = committed, start_s
            moves, new_arrivals = [], []
            for xy, guessed_arrival in zip(points, arrivals):
                forecast = predictor.forecast(guessed_arrival, now_s)
                surface = sample_map(forecast.height_mm, forecast.support, cfg.grid, xy)
                target = (float(xy[0]), float(xy[1]), cfg.surface_z(surface) + cfg.desired_gap_mm)
                self._bounds(target)
                corridor = corridor_cells(cfg.grid, xyz[:2], xy, cfg.footprint_radius_mm)
                if not forecast.support[corridor].all():
                    raise ControllerFault("missing support in swept path/tool footprint")
                # Spatial change evaluated at one time: uniform temporal translation adds no filament.
                old_surface = sample_map(forecast.height_mm, forecast.support, cfg.grid, xyz[:2])
                xy_length = float(np.linalg.norm(np.asarray(xy)-xyz[:2]))
                slope = abs(surface - old_surface) / xy_length
                if slope > cfg.limits.max_slope:
                    raise ControllerFault("surface slope exceeds vertical-nozzle envelope")
                surface_length = math.hypot(xy_length, surface - old_surface)
                extrusion = cfg.bead_area_mm2 * surface_length / (math.pi * (cfg.filament_diameter_mm/2)**2) if extrude else 0
                distance = float(np.linalg.norm(np.asarray(target)-xyz))
                ratio = extrusion / distance
                if ratio > cfg.limits.max_extrude_ratio:
                    raise ControllerFault("extrusion per machine-space mm exceeds configured limit")
                speed_cap = None
                if extrusion:
                    # Peak volumetric flow, rather than the mean over the move.
                    volume = cfg.bead_area_mm2 * surface_length
                    speed_cap = cfg.limits.max_flow_mm3_s * distance / volume
                duration, feed, _ = motion_profile(xyz, target, cfg.limits, speed_cap)
                t += duration
                new_arrivals.append(t)
                peak_z = feed * abs(target[2]-xyz[2]) / distance
                moves.append(Move(target, t, duration, feed, extrusion, surface_length, peak_z))
                xyz = target
            residual = max(abs(a-b) for a, b in zip(new_arrivals, arrivals))
            arrivals = new_arrivals
            if residual <= cfg.convergence_s:
                break
        else:
            raise ControllerFault("arrival-time/path iteration did not converge")
        # Verify the whole discretized swept corridor, including the immutable join.
        xyz, t = committed, start_s
        for move in moves:
            distance = float(np.linalg.norm(np.asarray(move.xyz)-xyz))
            count = max(4, int(math.ceil(distance / (cfg.grid.cell_size_mm/2))))
            corridor = corridor_cells(cfg.grid, xyz[:2], move.xyz[:2], cfg.footprint_radius_mm)
            for elapsed in np.linspace(0, move.duration_s, count+1):
                fraction = profile_fraction(elapsed, move.duration_s, distance, move.feed_mm_s)
                position = np.asarray(xyz) + fraction * (np.asarray(move.xyz)-xyz)
                forecast = predictor.forecast(t + elapsed, now_s)
                if not forecast.support[corridor].all():
                    raise ControllerFault("incomplete support along swept corridor")
                height = sample_map(forecast.height_mm, forecast.support, cfg.grid, position[:2])
                gap_error = abs(position[2] - cfg.surface_z(height) - cfg.desired_gap_mm)
                latest = predictor.buffer[-1]
                # Finite-difference temporal surface speed includes the selected predictor's motion.
                dt = 1e-4
                if forecast.target_s + dt - latest.observed_s <= cfg.predictor.max_horizon_s:
                    future = predictor.forecast(forecast.target_s + dt, now_s)
                    speed = abs(sample_map(future.height_mm, future.support, cfg.grid, position[:2])-height) / dt
                else:
                    raise ControllerFault("no timing margin at maximum forecast horizon")
                budget = cfg.provisional_error_mm + speed * (cfg.timing_uncertainty_s + latest.uncertainty_s)
                if gap_error + budget > cfg.allowed_gap_error_mm:
                    raise ControllerFault("trajectory/join and provisional error budget exceed allowed gap error")
                # Reject a high neighbour under the clearance footprint; no hole filling.
                local = corridor_cells(cfg.grid, position[:2], position[:2], cfg.footprint_radius_mm)
                highest = float(np.max(forecast.height_mm[local]))
                if position[2] - cfg.surface_z(highest) <= budget:
                    raise ControllerFault("tool footprint clearance is insufficient")
            xyz, t = move.xyz, move.arrival_s
        return Trajectory(segment_id, 0, predecessor_id, cfg.config_id, committed, start_s,
                          tuple(moves), tuple(f.frame_id for f in predictor.buffer),
                          predictor.buffer[-1].observed_s + cfg.predictor.max_frame_age_s)
