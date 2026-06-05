import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

from utils.geometry import heading_angle_difference

try:
    from pykalman import KalmanFilter
    from shapely.geometry import Point, LineString as SLineString
except ImportError:
    raise ImportError("Missing required packages. Please run: pip install pykalman shapely")


class RTSSmoother:

    def __init__(self, dt, process_noise_pos, process_noise_vel, measurement_noise):
        self.dt = dt
        self.F = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1,  0],
                           [0, 0, 0,  1]], dtype=np.float64)
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]], dtype=np.float64)
        q_pos = process_noise_pos ** 2
        q_vel = process_noise_vel ** 2
        self.Q = np.diag([q_pos, q_pos, q_vel, q_vel])
        self.R = np.eye(2) * (measurement_noise ** 2)

    def smooth(self, measurements: np.ndarray) -> np.ndarray:

        N = len(measurements)
        if N < 2:
            res = np.zeros((N, 4), dtype=np.float64)
            if N == 1:
                res[0, :2] = measurements[0]
            return res
        init_vx = (measurements[1, 0] - measurements[0, 0]) / self.dt
        init_vy = (measurements[1, 1] - measurements[0, 1]) / self.dt
        init_state = [measurements[0, 0], measurements[0, 1], init_vx, init_vy]
        init_cov = np.eye(4) * 10.0

        kf = KalmanFilter(
            transition_matrices=self.F, observation_matrices=self.H,
            transition_covariance=self.Q, observation_covariance=self.R,
            initial_state_mean=init_state, initial_state_covariance=init_cov
        )
        try:
            smoothed, _ = kf.smooth(measurements)
            return smoothed
        except Exception:
            print("Warning: pykalman smooth failed. Using finite differences for velocity.")
            res = np.zeros((N, 4), dtype=np.float64)
            res[:, :2] = measurements
            if N > 1:
                res[1:, 2] = (measurements[1:, 0] - measurements[:-1, 0]) / self.dt
                res[1:, 3] = (measurements[1:, 1] - measurements[:-1, 1]) / self.dt
                res[0, 2:] = res[1, 2:]
            return res


class MapHelper:

    def __init__(self, map_data_dir: str, logger: Optional[logging.Logger] = None):
        self.map_dir = Path(map_data_dir)
        self.cache = {}
        self.logger = logger if logger is not None else logging.getLogger(__name__)

    def load_scene(self, scene_name: str) -> bool:
        if scene_name in self.cache:
            return True
        json_path = self.map_dir / f"{scene_name}.json"
        if not json_path.exists():
            return False
        try:
            with open(json_path, 'r') as f:
                raw = json.load(f)
            parsed = {'centerlines': {}, 'boundaries': {}}
            for k, v in raw.get('centerlines', {}).items():
                if len(v['points']) > 1:
                    parsed['centerlines'][k] = SLineString(v['points'])
            for k, v in raw.get('boundaries', {}).items():
                if len(v['points']) > 1:
                    parsed['boundaries'][k] = SLineString(v['points'])

            parsed['zone_meta'] = {k: v for k, v in raw.get('zone_meta', {}).items()}
            self.cache[scene_name] = parsed
            return True
        except Exception:
            return False

    def query(self, scene: str, x: float, y: float, vx: float = None, vy: float = None,
              prev_map_feat: dict = None, next_map_feat: dict = None,
              direction_filter: int = None) -> dict:

        if not self.load_scene(scene):
            raise RuntimeError(f"Failed to load scene map: {scene}")

        map_data = self.cache[scene]
        point = Point(x, y)

        vehicle_heading = None
        if vx is not None and vy is not None:
            if abs(vx) > 0.1 or abs(vy) > 0.1:
                vehicle_heading = np.arctan2(vy, vx)
            else:
                vehicle_heading = None

        candidate_zones = []
        all_zones = []

        for zone, line in map_data['centerlines'].items():
            dist = line.distance(point)
            all_zones.append((zone, line, dist))
            if dist < 1000:
                candidate_zones.append((zone, line, dist))

        if len(candidate_zones) == 0:
            all_zones.sort(key=lambda item: item[2])
            nearest_zone, nearest_cl, nearest_dist = all_zones[0]

            start_point = nearest_cl.interpolate(0)
            end_point = nearest_cl.interpolate(nearest_cl.length)
            dist_to_start = np.linalg.norm([point.x - start_point.x, point.y - start_point.y])
            dist_to_end = np.linalg.norm([point.x - end_point.x, point.y - end_point.y])

            candidate_zones = [(nearest_zone, nearest_cl, nearest_dist)]
            self.logger.warning(f"Fallback: point ({x:.1f},{y:.1f}) is {nearest_dist:.1f}px from the nearest centerline; using endpoint extension.")

        candidate_zones.sort(key=lambda item: item[2])

        if direction_filter is not None:
            import re as _re
            _zone_meta = self.cache[scene].get('zone_meta', {})
            if _zone_meta:
                _filtered = [
                    (z, cl, d) for z, cl, d in candidate_zones
                    if _zone_meta.get(_re.sub(r'_\d+$', '', z), {}).get('direction') == direction_filter
                ]
                if _filtered:
                    candidate_zones = _filtered

        top_candidates = candidate_zones[:min(2, len(candidate_zones))]

        best_zone = None
        best_cl = None
        min_angle_diff = float('inf')

        for zone, cl, dist in top_candidates:
            d_proj = cl.project(point)

            if d_proj < 0:

                p_proj_point = cl.interpolate(0)
                p_next_point = cl.interpolate(min(cl.length, 1.0))
                vec_t = np.array([p_next_point.x - p_proj_point.x, p_next_point.y - p_proj_point.y])
                norm = np.linalg.norm(vec_t)
                if norm > 1e-6:
                    vec_t /= norm

                p0 = np.array([p_proj_point.x, p_proj_point.y])
                p_vehicle = np.array([point.x, point.y])
                proj_dist = np.dot(p_vehicle - p0, vec_t)
                p_proj = p0 + proj_dist * vec_t
            elif d_proj > cl.length:

                p_proj_point = cl.interpolate(cl.length)
                p_prev_point = cl.interpolate(max(0, cl.length - 1.0))
                vec_t = np.array([p_proj_point.x - p_prev_point.x, p_proj_point.y - p_prev_point.y])
                norm = np.linalg.norm(vec_t)
                if norm > 1e-6:
                    vec_t /= norm

                p0 = np.array([p_proj_point.x, p_proj_point.y])
                p_vehicle = np.array([point.x, point.y])
                proj_dist = np.dot(p_vehicle - p0, vec_t)
                p_proj = p0 + proj_dist * vec_t
            else:

                p_proj_point = cl.interpolate(d_proj)
                p_proj = np.array([p_proj_point.x, p_proj_point.y])

                p_next_point = cl.interpolate(min(cl.length, d_proj + 1.0))
                vec_t = np.array([p_next_point.x - p_proj_point.x, p_next_point.y - p_proj_point.y])
                norm = np.linalg.norm(vec_t)
                if norm > 1e-6:
                    vec_t /= norm

            cl_heading = np.arctan2(vec_t[1], vec_t[0])

            if vehicle_heading is not None:

                angle_diff = heading_angle_difference(vehicle_heading, cl_heading)
            else:

                angle_diff = dist

            if angle_diff < min_angle_diff:
                min_angle_diff = angle_diff
                best_zone = zone
                best_cl = cl

        if best_zone is None:
            raise RuntimeError(
                f"Map matching failed: point ({x:.1f}, {y:.1f}) in scene '{scene}' cannot be assigned to a centerline."
            )

        cl = best_cl
        d_proj = cl.project(point)

        left_key = next(
            (k for k in map_data['boundaries']
             if k.startswith(f"{best_zone}:") and "left" in k), None)
        right_key = next(
            (k for k in map_data['boundaries']
             if k.startswith(f"{best_zone}:") and "right" in k), None)

        use_virtual = False

        if not left_key or not right_key:
            use_virtual = True
            self.logger.warning(f"Missing boundary for zone '{best_zone}' in scene '{scene}'; using virtual extension.")

        if d_proj < -50 or d_proj > cl.length + 50:
            use_virtual = True

        reference_width = None
        reference_dist_left = None
        reference_dist_right = None

        if not use_virtual:

            if d_proj < 0:
                p_proj_point = cl.interpolate(0)
                p_next_point = cl.interpolate(min(cl.length, 1.0))
                vec_t = np.array([p_next_point.x - p_proj_point.x, p_next_point.y - p_proj_point.y])
                norm = np.linalg.norm(vec_t)
                if norm > 1e-6:
                    vec_t /= norm
                heading = np.arctan2(vec_t[1], vec_t[0])
                p0 = np.array([p_proj_point.x, p_proj_point.y])
                p_vehicle = np.array([point.x, point.y])
                proj_dist = np.dot(p_vehicle - p0, vec_t)
                p_proj = Point(p0[0] + proj_dist * vec_t[0], p0[1] + proj_dist * vec_t[1])
            elif d_proj > cl.length:
                p_proj_point = cl.interpolate(cl.length)
                p_prev_point = cl.interpolate(max(0, cl.length - 1.0))
                vec_t = np.array([p_proj_point.x - p_prev_point.x, p_proj_point.y - p_prev_point.y])
                norm = np.linalg.norm(vec_t)
                if norm > 1e-6:
                    vec_t /= norm
                heading = np.arctan2(vec_t[1], vec_t[0])
                p0 = np.array([p_proj_point.x, p_proj_point.y])
                p_vehicle = np.array([point.x, point.y])
                proj_dist = np.dot(p_vehicle - p0, vec_t)
                p_proj = Point(p0[0] + proj_dist * vec_t[0], p0[1] + proj_dist * vec_t[1])
            else:
                p_proj = cl.interpolate(d_proj)
                p_next = cl.interpolate(min(cl.length, d_proj + 1.0))
                vec_t = np.array([p_next.x - p_proj.x, p_next.y - p_proj.y])
                norm = np.linalg.norm(vec_t)
                if norm > 1e-6:
                    vec_t /= norm
                heading = np.arctan2(vec_t[1], vec_t[0])

            dist_l = map_data['boundaries'][left_key].distance(point)
            dist_r = map_data['boundaries'][right_key].distance(point)

            W = dist_l + dist_r
            if W < 1.0:
                self.logger.warning(f"Abnormal road width (W={W:.2f}); using virtual extension.")
                use_virtual = True
            else:

                x_left = x - dist_l
                return {
                    'dist_left': dist_l,
                    'dist_right': dist_r,
                    'road_width': W,
                    'heading': heading,
                    'x_left': x_left,
                    'valid': True,
                    'fallback_type': 'none',
                    'query_x': x,
                    'query_y': y
                }

        if use_virtual:
            self.logger.debug(f"Boundary missing or out of range: zone '{best_zone}', scene '{scene}', point ({x:.1f},{y:.1f}); returning invalid for inheritance.")
            return {
                'valid': False,
                'fallback_type': 'no_boundary',
                'query_x': x,
                'query_y': y
            }
