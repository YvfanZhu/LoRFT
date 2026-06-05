from __future__ import annotations

from typing import Iterable

import numpy as np


LOCAL_HEADING_WINDOW = 5
LOCAL_HEADING_MIN_DISP = 1.0


def _normalize_mode(mode: str | None, default: str) -> str:
    value = str(mode or default).strip().lower().replace('-', '_')
    return value or default


def wrap_to_pi(angle: float | np.ndarray) -> float | np.ndarray:
    """Wrap angles to [-pi, pi)."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def heading_angle_difference(theta_a: float, theta_b: float) -> float:
    """Return the direction-aware angular distance in [0, pi]."""
    return float(abs(wrap_to_pi(theta_a - theta_b)))


def align_heading_to_motion(theta_car: float, heading_map: float) -> float:
    """Flip the map heading by pi when it points against the motion direction."""
    aligned_heading = float(wrap_to_pi(heading_map))
    if heading_angle_difference(theta_car, aligned_heading) > (np.pi / 2.0):
        aligned_heading = float(wrap_to_pi(aligned_heading + np.pi))
    return aligned_heading


def compute_lane_flow_dtheta(theta_car: float, heading_map: float, mode: str | None = 'direct') -> float:
    """Compute d_theta against the requested map-heading semantics."""
    mode_norm = _normalize_mode(mode, 'direct')
    aligned_heading = float(heading_map)
    if mode_norm in {'motion_aligned', 'mixed'}:
        aligned_heading = align_heading_to_motion(theta_car, aligned_heading)
    elif mode_norm not in {'direct', 'raw', 'none'}:
        raise ValueError(f'Unsupported d_theta mode: {mode}')
    return float(wrap_to_pi(theta_car - aligned_heading))


def _resolve_local_heading(
    positions: np.ndarray,
    idx: int,
    last_heading: float,
    window: int = LOCAL_HEADING_WINDOW,
    min_disp: float = LOCAL_HEADING_MIN_DISP,
) -> float:
    num_points = len(positions)
    if num_points == 0:
        return float(last_heading)

    half_window = max(int(window) // 2, 1)
    lo = max(0, idx - half_window)
    hi = min(num_points - 1, idx + half_window)
    if hi <= lo:
        return float(last_heading)

    delta = positions[hi] - positions[lo]
    disp = float(np.linalg.norm(delta))
    if disp < float(min_disp):
        return float(last_heading)
    return float(np.arctan2(delta[1], delta[0]))


def estimate_mixed_headings(
    positions_xy: Iterable[Iterable[float]],
    velocities_xy: Iterable[Iterable[float]],
    low_speed_threshold: float,
    window: int = LOCAL_HEADING_WINDOW,
    min_disp: float = LOCAL_HEADING_MIN_DISP,
) -> np.ndarray:
    """Estimate theta_car using velocity first, local motion fallback second."""
    positions = np.asarray(list(positions_xy), dtype=np.float64)
    velocities = np.asarray(list(velocities_xy), dtype=np.float64)
    num_points = len(positions)
    if num_points == 0:
        return np.zeros((0,), dtype=np.float64)

    headings = np.zeros((num_points,), dtype=np.float64)
    last_heading = 0.0
    low_speed_threshold = float(low_speed_threshold)

    for idx in range(num_points):
        vx = float(velocities[idx, 0]) if idx < len(velocities) else 0.0
        vy = float(velocities[idx, 1]) if idx < len(velocities) else 0.0
        speed = float(np.hypot(vx, vy))
        if speed >= low_speed_threshold:
            heading = float(np.arctan2(vy, vx))
        else:
            heading = _resolve_local_heading(positions, idx, last_heading, window=window, min_disp=min_disp)
        headings[idx] = heading
        last_heading = heading

    return headings


def estimate_headings(
    positions_xy: Iterable[Iterable[float]],
    velocities_xy: Iterable[Iterable[float]],
    low_speed_threshold: float,
    mode: str | None = 'motion_aligned',
    window: int = LOCAL_HEADING_WINDOW,
    min_disp: float = LOCAL_HEADING_MIN_DISP,
) -> np.ndarray:
    """Estimate theta_car with a configurable heading strategy."""
    mode_norm = _normalize_mode(mode, 'motion_aligned')
    positions = np.asarray(list(positions_xy), dtype=np.float64)
    velocities = np.asarray(list(velocities_xy), dtype=np.float64)
    num_points = len(positions)
    if num_points == 0:
        return np.zeros((0,), dtype=np.float64)

    if mode_norm in {'motion_aligned', 'mixed'}:
        return estimate_mixed_headings(
            positions,
            velocities,
            low_speed_threshold=low_speed_threshold,
            window=window,
            min_disp=min_disp,
        )
    if mode_norm not in {'velocity', 'velocity_only'}:
        raise ValueError(f'Unsupported heading estimation mode: {mode}')

    headings = np.zeros((num_points,), dtype=np.float64)
    last_heading = 0.0
    for idx in range(num_points):
        vx = float(velocities[idx, 0]) if idx < len(velocities) else 0.0
        vy = float(velocities[idx, 1]) if idx < len(velocities) else 0.0
        speed = float(np.hypot(vx, vy))
        if speed >= float(low_speed_threshold):
            heading = float(np.arctan2(vy, vx))
        else:
            heading = float(last_heading)
        headings[idx] = heading
        last_heading = heading
    return headings


def estimate_mixed_heading_from_recent(
    recent_positions_xy: Iterable[Iterable[float]],
    velocity_xy: Iterable[float],
    low_speed_threshold: float,
    fallback_heading: float,
    window: int = LOCAL_HEADING_WINDOW,
    min_disp: float = LOCAL_HEADING_MIN_DISP,
) -> float:
    """Estimate one-step theta_car for dynamic refresh using the same rule as training/inference."""
    velocity = np.asarray(list(velocity_xy), dtype=np.float64)
    if velocity.size >= 2:
        vx = float(velocity[0])
        vy = float(velocity[1])
        speed = float(np.hypot(vx, vy))
        if speed >= float(low_speed_threshold):
            return float(np.arctan2(vy, vx))

    positions = np.asarray(list(recent_positions_xy), dtype=np.float64)
    if len(positions) >= 2:
        return _resolve_local_heading(
            positions,
            len(positions) - 1,
            float(fallback_heading),
            window=window,
            min_disp=min_disp,
        )
    return float(fallback_heading)


def estimate_heading_from_recent(
    recent_positions_xy: Iterable[Iterable[float]],
    velocity_xy: Iterable[float],
    low_speed_threshold: float,
    fallback_heading: float,
    mode: str | None = 'motion_aligned',
    window: int = LOCAL_HEADING_WINDOW,
    min_disp: float = LOCAL_HEADING_MIN_DISP,
) -> float:
    """Estimate one-step theta_car with a configurable heading strategy."""
    mode_norm = _normalize_mode(mode, 'motion_aligned')
    velocity = np.asarray(list(velocity_xy), dtype=np.float64)
    if velocity.size >= 2:
        vx = float(velocity[0])
        vy = float(velocity[1])
        speed = float(np.hypot(vx, vy))
        if speed >= float(low_speed_threshold):
            return float(np.arctan2(vy, vx))

    if mode_norm in {'motion_aligned', 'mixed'}:
        return estimate_mixed_heading_from_recent(
            recent_positions_xy,
            velocity,
            low_speed_threshold=low_speed_threshold,
            fallback_heading=fallback_heading,
            window=window,
            min_disp=min_disp,
        )
    if mode_norm in {'velocity', 'velocity_only'}:
        return float(fallback_heading)
    raise ValueError(f'Unsupported heading estimation mode: {mode}')
