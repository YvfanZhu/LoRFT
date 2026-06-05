import logging
from typing import List, Tuple

import numpy as np

from utils.geometry import compute_lane_flow_dtheta, estimate_headings, wrap_to_pi
from utils.map_matching import MapHelper, RTSSmoother

logger = logging.getLogger(__name__)


class FeatureExtractor:


    def __init__(self, config, map_helper: MapHelper):
        self.cfg = config
        self.map = map_helper
        try:
            self.img_h = config['image']['height']
            self.vel_scale = config['features']['velocity_scale']
            rts_cfg = config['rts_smoother']
            dt = rts_cfg['dt']
            process_noise_pos = rts_cfg['process_noise_pos']
            process_noise_vel = rts_cfg['process_noise_vel']
            measurement_noise = rts_cfg['measurement_noise']
        except KeyError as e:
            raise KeyError(f"Configuration is missing required parameters: {e}")
        self.dt = dt
        self.low_speed_threshold = float(config.get('map_matching', {}).get('low_speed_threshold', 2.0))
        heading_cfg = config.get('heading_estimation', {})
        self.heading_estimation_mode = str(heading_cfg.get('mode', 'motion_aligned'))
        self.heading_window = int(heading_cfg.get('window', 5))
        self.heading_min_step_px = float(heading_cfg.get('min_step_px', 1.0))
        self.dtheta_mode = str(config.get('heading_estimation', {}).get('dtheta_mode', 'motion_aligned'))
        self.kf = RTSSmoother(dt, process_noise_pos, process_noise_vel, measurement_noise)

    def process_trajectory(self, scene: str, track_data: np.ndarray,
                           do_flip: bool, is_downstream: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        features, abs_info, validity_mask, _ = self._process_trajectory_core(
            scene,
            track_data,
            do_flip,
            is_downstream=is_downstream,
            return_metadata=False,
        )
        return features, abs_info, validity_mask

    def process_trajectory_with_metadata(self, scene: str, track_data: np.ndarray,
                                         do_flip: bool, is_downstream: bool = False):
        return self._process_trajectory_core(
            scene,
            track_data,
            do_flip,
            is_downstream=is_downstream,
            return_metadata=True,
        )

    def _process_trajectory_core(self, scene: str, track_data: np.ndarray,
                                 do_flip: bool, is_downstream: bool = False,
                                 return_metadata: bool = False):

        cx = track_data[:, 1] + track_data[:, 3] / 2.0
        cy = track_data[:, 2] + track_data[:, 4]
        w_raw = track_data[:, 3]
        h_raw = track_data[:, 4]

        states = self.kf.smooth(np.column_stack((cx, cy)))

        features = []
        abs_info = []
        frame_metadata: List[dict] = []
        N = len(states)
        motion_headings = estimate_headings(
            states[:, :2],
            states[:, 2:4],
            low_speed_threshold=self.low_speed_threshold,
            mode=self.heading_estimation_mode,
            window=self.heading_window,
            min_disp=self.heading_min_step_px,
        ) if N > 0 else np.zeros((0,), dtype=np.float64)

        _dir_filter = -1 if is_downstream else 1

        map_features_raw = []
        for i in range(N):
            real_x, real_y = states[i, 0], states[i, 1]
            real_vx, real_vy = states[i, 2], states[i, 3]

            map_feat = self.map.query(
                scene, real_x, real_y, real_vx, real_vy,
                prev_map_feat=None, next_map_feat=None,
                direction_filter=_dir_filter,
            )
            map_features_raw.append(map_feat)

        _cold_start_feat = None
        for _i in range(N):
            _rf = map_features_raw[_i]
            if _rf.get('valid', False) and _rf.get('fallback_type') == 'none':
                _cold_start_feat = _rf
                break

        if _cold_start_feat is None:
            empty_features = np.array([], dtype=np.float32)
            empty_abs_info = np.array([], dtype=np.float32)
            empty_validity = np.array([], dtype=bool)
            return empty_features, empty_abs_info, empty_validity, []

        map_features = []
        _last_valid_feat = _cold_start_feat
        for i in range(N):
            raw_feat = map_features_raw[i]
            real_x = states[i, 0]
            real_y = states[i, 1]

            if raw_feat.get('valid', False) and raw_feat.get('fallback_type') == 'none':
                map_features.append(raw_feat)
                _last_valid_feat = raw_feat
            else:
                _ref = _last_valid_feat
                if _ref is not None:
                    W_ref = _ref['road_width']
                    heading_ref = _ref['heading']

                    _dy = real_y - _ref.get('query_y', real_y)
                    _tan_h = np.tan(heading_ref)

                    if abs(_tan_h) > 0.05:
                        _dx_lane = np.clip(_dy / _tan_h, -50.0, 50.0)
                    else:
                        _dx_lane = 0.0
                    x_left_curr = _ref['x_left'] + _dx_lane

                    dist_l_new = real_x - x_left_curr
                    dist_r_new = W_ref - dist_l_new
                    map_features.append({
                        'road_width': W_ref,
                        'dist_left': dist_l_new,
                        'dist_right': dist_r_new,
                        'heading': heading_ref,
                        'x_left': x_left_curr,
                        'valid': False,
                        'fallback_type': 'inherited',
                        'query_x': real_x,
                        'query_y': real_y,
                    })
                else:
                    map_features.append({
                        'road_width': 100.0,
                        'dist_left': 50.0,
                        'dist_right': 50.0,
                        'heading': 0.0,
                        'x_left': real_x - 50.0,
                        'valid': False,
                        'fallback_type': 'default',
                        'query_x': real_x,
                        'query_y': real_y,
                    })

        _u_arr = np.array([
            (map_features[i]['dist_left']) / map_features[i]['road_width']
            if map_features[i]['road_width'] > 1 else 0.5
            for i in range(N)
        ], dtype=np.float64)
        _y_norm_arr = states[:, 1] / self.img_h
        _v_u_arr = np.zeros(N, dtype=np.float64)
        _v_y_arr = np.zeros(N, dtype=np.float64)
        if N > 1:
            _v_u_arr[1:] = (_u_arr[1:] - _u_arr[:-1]) / self.dt * self.vel_scale
            _v_y_arr[1:] = (_y_norm_arr[1:] - _y_norm_arr[:-1]) / self.dt * self.vel_scale
            _v_u_arr[0] = _v_u_arr[1]
            _v_y_arr[0] = _v_y_arr[1]

        for i in range(N):
            real_x, real_y = states[i, 0], states[i, 1]
            w, h = w_raw[i], h_raw[i]

            map_feat = map_features[i]
            W = map_feat['road_width']
            dist_l = map_feat['dist_left']
            dist_r = map_feat['dist_right']
            x_left = map_feat['x_left']
            heading = map_feat['heading']

            u = dist_l / W
            y_norm = real_y / self.img_h
            w_norm = w / W
            h_norm = h / W

            v_u = _v_u_arr[i]
            v_y = _v_y_arr[i]

            theta_car = float(motion_headings[i]) if len(motion_headings) > i else 0.0

            _has_real_geo = (
                map_feat.get('valid', False) or
                map_feat.get('fallback_type') in ('inherited', 'interpolate', 'prev_frame')
            )
            if _has_real_geo:
                d_theta = compute_lane_flow_dtheta(theta_car, heading, mode=self.dtheta_mode)
                d_left = dist_l / W
                d_right = dist_r / W
                f_lat = (dist_l - dist_r) / W
            else:
                d_theta = 0.0
                d_left, d_right = 0.5, 0.5
                f_lat = 0.0

            frame_metadata.append({
                'W': float(W),
                'x_left': float(x_left),
                'heading': float(heading),
                'phy_x': float(real_x),
                'phy_y': float(real_y),
                'dist_left': float(dist_l),
                'dist_right': float(dist_r),
                'valid': bool(map_feat.get('valid', False)),
                'fallback_type': str(map_feat.get('fallback_type', 'none')),
                'query_x': float(map_feat.get('query_x', real_x)),
                'query_y': float(map_feat.get('query_y', real_y)),
            })

            if do_flip:
                u = 1.0 - u
                v_y = -v_y
                d_left, d_right = d_right, d_left
                f_lat = -f_lat
                d_theta = float(wrap_to_pi(-d_theta))

            features.append([u, y_norm, w_norm, h_norm,
                             v_u, v_y,
                             f_lat, d_theta, d_left, d_right])
            abs_info.append([u, y_norm, W, x_left])

        n_none = sum(1 for f in map_features if f.get('fallback_type') == 'none')
        n_inherited = sum(1 for f in map_features if f.get('fallback_type') == 'inherited')
        n_default = sum(1 for f in map_features if f.get('fallback_type') == 'default')
        total_frames = len(map_features)
        inherited_ratio = (n_inherited + n_default) / total_frames if total_frames > 0 else 0
        if n_inherited + n_default > 0:
            logger.info(
                f"Map coverage [{scene}|id?]: "
                f"direct={n_none}/{total_frames} ({(n_none/total_frames)*100:.1f}%), "
                f"inherited={n_inherited}, default={n_default}, "
                f"inherit_ratio={inherited_ratio*100:.1f}%  "
                f"-> window-level filtering will check each frame"
            )

        validity_mask = np.array(
            [f.get('fallback_type') == 'none' for f in map_features], dtype=bool
        )

        features_arr = np.array(features, dtype=np.float32)
        abs_info_arr = np.array(abs_info, dtype=np.float32)
        return features_arr, abs_info_arr, validity_mask, frame_metadata if return_metadata else []
