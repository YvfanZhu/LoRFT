from __future__ import annotations


import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List
from copy import deepcopy

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.seq2seq_model import build_model
from utils.config import load_config_bundle, validate_time_config_contract, resolve_canonical_window_config
from utils.geometry import compute_lane_flow_dtheta, estimate_headings, heading_angle_difference, wrap_to_pi
from utils.map_matching import MapHelper as SharedMapHelper
from utils.trajectory_features import FeatureExtractor as SharedFeatureExtractor

try:
    from pykalman import KalmanFilter
    HAS_PYKALMAN = True
except ImportError:
    HAS_PYKALMAN = False
    print("Warning: pykalman not installed. Finite differences will be used.")


class RTSSmoother:

    def __init__(self, dt=0.04, process_noise_pos=1.0, process_noise_vel=5.0, measurement_noise=10.0):
        self.dt = dt
        self.F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64)
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)
        self.Q = np.diag([process_noise_pos**2, process_noise_pos**2,
                          process_noise_vel**2, process_noise_vel**2])
        self.R = np.eye(2) * (measurement_noise ** 2)

    def smooth(self, observations: np.ndarray) -> np.ndarray:
        N = len(observations)
        if N < 2:
            res = np.zeros((N, 4))
            res[:, :2] = observations
            return res
        if HAS_PYKALMAN:
            init_vx = (observations[1, 0] - observations[0, 0]) / self.dt
            init_vy = (observations[1, 1] - observations[0, 1]) / self.dt
            kf = KalmanFilter(
                transition_matrices=self.F, observation_matrices=self.H,
                transition_covariance=self.Q, observation_covariance=self.R,
                initial_state_mean=[observations[0, 0], observations[0, 1], init_vx, init_vy],
                initial_state_covariance=np.eye(4) * 10.0
            )
            smoothed_states, _ = kf.smooth(observations)
            return smoothed_states
        else:
            res = np.zeros((N, 4))
            res[:, :2] = observations
            if N > 1:
                res[1:, 2] = (observations[1:, 0] - observations[:-1, 0]) / self.dt
                res[1:, 3] = (observations[1:, 1] - observations[:-1, 1]) / self.dt
                res[0, 2:] = res[1, 2:]
            return res


class MapInterface:

    def __init__(self, map_data_dir=None):
        self.maps = {}
        self.map_data_dir = map_data_dir
        try:
            from shapely.geometry import Point, LineString
            self.has_shapely = True
            self.Point = Point
            self.LineString = LineString
        except ImportError:
            self.has_shapely = False

    def load_scene_map(self, scene_name):
        if not self.has_shapely:
            return False
        if scene_name in self.maps:
            return True
        if not self.map_data_dir:
            return False
        try:
            json_path = os.path.join(self.map_data_dir, f"{scene_name}.json")
            if not os.path.exists(json_path):
                return False
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.maps[scene_name] = {
                'centerlines': {k: self.LineString(v['points'])
                                for k, v in data.get('centerlines', {}).items() if len(v['points']) > 1},
                'boundaries': {k: self.LineString(v['points'])
                               for k, v in data.get('boundaries', {}).items() if len(v['points']) > 1},
                'zone_meta': {k: v for k, v in data.get('zone_meta', {}).items()},
            }
            return True
        except Exception as e:
            print(f"Map load error: {e}")
            return False

    def query(self, scene: str, x: float, y: float,
              vx: float = None, vy: float = None,
              prev_map_feat: dict = None,
              direction_filter: int = None) -> dict:


        if not self.has_shapely or scene not in self.maps:
            return {'dist_left': 50, 'dist_right': 50, 'road_width': 100,
                    'heading': 0, 'x_left': x - 50, 'valid': False}

        map_data = self.maps[scene]
        point = self.Point(x, y)

        vehicle_heading = None
        if vx is not None and vy is not None and (abs(vx) > 0.1 or abs(vy) > 0.1):
            vehicle_heading = np.arctan2(vy, vx)


        candidate_zones = []
        for zone, cl in map_data['centerlines'].items():
            dist = cl.distance(point)
            if dist < 2000:
                candidate_zones.append((zone, cl, dist))
        if not candidate_zones:
            min_dist = float('inf')
            best_z, best_cl = None, None
            for zone, cl in map_data['centerlines'].items():
                d = cl.distance(point)
                if d < min_dist:
                    min_dist = d
                    best_z, best_cl = zone, cl
            if best_z:
                candidate_zones.append((best_z, best_cl, min_dist))
            else:
                return {'dist_left': 50, 'dist_right': 50, 'road_width': 100,
                        'heading': 0, 'x_left': x - 50, 'valid': False}

        candidate_zones.sort(key=lambda item: item[2])


        if direction_filter is not None:
            import re as _re
            _zone_meta = map_data.get('zone_meta', {})
            if _zone_meta:
                _filtered = [
                    (z, cl, d) for z, cl, d in candidate_zones
                    if _zone_meta.get(_re.sub(r'_\d+$', '', z), {}).get('direction') == direction_filter
                ]
                if _filtered:
                    candidate_zones = _filtered


        top_candidates = candidate_zones[:min(2, len(candidate_zones))]
        best_zone, best_cl = None, None
        min_score = float('inf')
        for zone, cl, dist in top_candidates:
            d_proj = cl.project(point)
            if d_proj >= cl.length - 1e-6:
                p_curr = cl.interpolate(max(0, cl.length - 1.0))
                p_next = cl.interpolate(cl.length)
            else:
                p_curr = cl.interpolate(d_proj)
                p_next = cl.interpolate(min(cl.length, d_proj + 1.0))
            vec_t = np.array([p_next.x - p_curr.x, p_next.y - p_curr.y])
            norm = np.linalg.norm(vec_t)
            vec_t = vec_t / norm if norm > 1e-6 else np.array([0, 1])
            cl_heading = np.arctan2(vec_t[1], vec_t[0])
            score = dist
            if vehicle_heading is not None:
                angle_diff = heading_angle_difference(vehicle_heading, cl_heading)
                score += angle_diff * 200.0
            if score < min_score:
                min_score = score
                best_zone = zone
                best_cl = cl

        if not best_zone:
            best_zone, best_cl = top_candidates[0][0], top_candidates[0][1]

        import re as _re
        boundary_zone = _re.sub(r'_\d+$', '', best_zone)
        l_key = next((k for k in map_data['boundaries']
                      if k.startswith(f"{best_zone}:") and "left" in k), None)
        r_key = next((k for k in map_data['boundaries']
                      if k.startswith(f"{best_zone}:") and "right" in k), None)

        if l_key and r_key:
            dist_l = map_data['boundaries'][l_key].distance(point)
            dist_r = map_data['boundaries'][r_key].distance(point)
            W = dist_l + dist_r
            if W < 1.0:
                W = 100.0
                dist_l = 50.0
                dist_r = 50.0
            d_proj = best_cl.project(point)
            p_c = best_cl.interpolate(min(best_cl.length, max(0, d_proj)))
            p_n = best_cl.interpolate(min(best_cl.length, d_proj + 1))
            vt = np.array([p_n.x - p_c.x, p_n.y - p_c.y])
            norm_vt = np.linalg.norm(vt)
            h = np.arctan2(vt[1], vt[0]) if norm_vt > 1e-6 else 0.0
            x_l = x - dist_l
            result = {
                'dist_left': dist_l, 'dist_right': dist_r,
                'road_width': W, 'heading': h, 'x_left': x_l,
                'valid': True, 'zone': best_zone,
                'query_x': x, 'query_y': y
            }
        else:
            result = {'valid': False, 'query_x': x, 'query_y': y}


        if not result.get('valid', False):
            if prev_map_feat is not None and prev_map_feat.get('valid', False):
                W = prev_map_feat['road_width']
                heading = prev_map_feat['heading']
                _dy = y - prev_map_feat.get('query_y', y)
                _tan_h = np.tan(heading)
                _dx_lane = np.clip(_dy / _tan_h, -50.0, 50.0) if abs(_tan_h) > 0.05 else 0.0
                x_left = prev_map_feat['x_left'] + _dx_lane
                dist_l = x - x_left
                dist_r = W - dist_l
                result = {
                    'dist_left': dist_l, 'dist_right': dist_r,
                    'road_width': W, 'heading': heading, 'x_left': x_left,
                    'valid': True, 'zone': prev_map_feat.get('zone', best_zone),
                    'query_x': x, 'query_y': y
                }
            else:
                result = {
                    'dist_left': 50.0, 'dist_right': 50.0,
                    'road_width': 100.0, 'heading': 0.0, 'x_left': x - 50.0,
                    'valid': False, 'zone': best_zone,
                    'query_x': x, 'query_y': y
                }

        if 'zone' not in result:
            result['zone'] = best_zone
        return result


class TrajectoryPredictor:
    def __init__(self, config):
        self.config = config
        requested_device = config.get('inference', {}).get('device', 'cpu')
        if str(requested_device).lower() == 'auto':
            requested_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(requested_device if torch.cuda.is_available() or str(requested_device) == 'cpu' else 'cpu')
        self.dynamic_geo_enabled = bool(config.get('model', {}).get('dynamic_geo', {}).get('enabled', False))
        self.model = self._load_model()

        map_dir = config.get('map', {}).get('data_dir')
        self.map_interface = MapInterface(map_data_dir=map_dir)
        self.shared_map_helper = SharedMapHelper(map_data_dir=map_dir)
        self.scene_name = config.get('scene', {}).get('name')

        self.img_width = config['image']['width']
        self.img_height = config['image']['height']
        self.obs_len = config['sliding_window']['obs_len']
        self.pred_len = config['sliding_window']['pred_len']
        self.dt = config['rts_smoother']['dt']
        self.decode_max_delta = float(config.get('model', {}).get('decode', {}).get('max_delta', 0.1))

        vs_loss = config.get('loss', {}).get('velocity_scale')
        if vs_loss is None:
            raise ValueError('Configuration is missing loss.velocity_scale')
        self.velocity_scale = vs_loss
        self.low_speed_threshold = float(config.get('map_matching', {}).get('low_speed_threshold', 2.0))
        heading_cfg = config.get('heading_estimation', {})
        self.heading_estimation_mode = str(heading_cfg.get('mode', 'motion_aligned'))
        self.heading_window = int(heading_cfg.get('window', 5))
        self.heading_min_step_px = float(heading_cfg.get('min_step_px', 1.0))
        self.dtheta_mode = str(config.get('heading_estimation', {}).get('dtheta_mode', 'motion_aligned'))
        motion_clamp_cfg = config.get('inference', {}).get('motion_clamp', {})
        self.motion_clamp_enabled = bool(motion_clamp_cfg.get('enabled', False))
        self.motion_clamp_scale_u = float(motion_clamp_cfg.get('max_step_scale_u', 1.0))
        self.motion_clamp_scale_y = float(motion_clamp_cfg.get('max_step_scale_y', 1.0))
        self.motion_clamp_scale_u_upstream = float(motion_clamp_cfg.get('max_step_scale_u_upstream', self.motion_clamp_scale_u))
        self.motion_clamp_scale_u_downstream = float(motion_clamp_cfg.get('max_step_scale_u_downstream', self.motion_clamp_scale_u))
        self.motion_clamp_scale_y_upstream = float(motion_clamp_cfg.get('max_step_scale_y_upstream', self.motion_clamp_scale_y))
        self.motion_clamp_scale_y_downstream = float(motion_clamp_cfg.get('max_step_scale_y_downstream', self.motion_clamp_scale_y))
        self.motion_clamp_floor_u = float(motion_clamp_cfg.get('step_floor_u', 0.002))
        self.motion_clamp_floor_y = float(motion_clamp_cfg.get('step_floor_y', 0.002))

        rts_cfg = config['inference']['rts_smoother']
        self.rts = RTSSmoother(
            dt=self.dt,
            process_noise_pos=rts_cfg['process_noise_pos'],
            process_noise_vel=rts_cfg['process_noise_vel'],
            measurement_noise=rts_cfg['measurement_noise'],
        )
        self.shared_feature_extractor = SharedFeatureExtractor(config, self.shared_map_helper)

        if not self.scene_name or not self.map_interface.load_scene_map(self.scene_name):
            raise RuntimeError(f'Map load failed: {self.scene_name}')

    def _load_model(self):
        ckpt_path = self.config.get('model', {}).get('checkpoint_path')
        if not ckpt_path:
            ckpt_path = str(Path(self.config['checkpoint']['save_dir']) / 'best_model.pth')
        ckpt_file = Path(ckpt_path)
        if not ckpt_file.exists():
            raise FileNotFoundError(
                f'Checkpoint not found: {ckpt_file}. Train a model first or set model.checkpoint_path in configs/local.yaml.'
            )
        print(f'Loading model from {ckpt_file}')
        ckpt = torch.load(ckpt_file, map_location=self.device, weights_only=False)
        model = build_model(self.config)
        state_dict = ckpt['model_state'] if 'model_state' in ckpt else ckpt
        if any(k.startswith('module.') for k in state_dict):
            state_dict = {k.removeprefix('module.'): v for k, v in state_dict.items()}
        if any(k.startswith('decoder.output_head') for k in state_dict):
            state_dict = {k.replace('decoder.output_head', 'decoder.fc_out'): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        model.to(self.device).eval()
        return model

    def _predict_autoregressive(self, model, obs_tensor: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            hidden, cell = model.encoder(obs_tensor)
            prev_features = obs_tensor[:, -1:, :]
            prev_pos = obs_tensor[:, -1:, 0:2]
            fixed_wh = obs_tensor[:, -1:, 2:4]
            context_geo = obs_tensor[:, -1:, 6:10]

            predictions = []
            for _ in range(self.pred_len):
                velocity_output, hidden, cell = model.decoder(prev_features, hidden, cell)
                velocity_output = torch.clamp(velocity_output, -self.decode_max_delta, self.decode_max_delta)
                curr_pos = prev_pos + velocity_output
                predictions.append(curr_pos.squeeze(0).cpu().numpy())

                vel_feat = velocity_output / self.dt * self.velocity_scale
                new_features = torch.cat([curr_pos, fixed_wh, vel_feat, context_geo], dim=2)
                prev_pos = curr_pos
                prev_features = new_features

        return np.vstack(predictions)

    def _xywh_to_bottom_center(self, boxes):
        bc = np.zeros_like(boxes, dtype=np.float64)
        bc[:, 0] = boxes[:, 0] + boxes[:, 2] / 2.0
        bc[:, 1] = boxes[:, 1] + boxes[:, 3]
        bc[:, 2] = boxes[:, 2]
        bc[:, 3] = boxes[:, 3]
        return bc

    def _bottom_center_to_xywh(self, bc):
        tl = np.zeros_like(bc)
        tl[:, 0] = bc[:, 0] - bc[:, 2] / 2.0
        tl[:, 1] = bc[:, 1] - bc[:, 3]
        tl[:, 2] = bc[:, 2]
        tl[:, 3] = bc[:, 3]
        return tl

    def generate_features(self, bc_boxes: np.ndarray,
                          precomputed_velocities: np.ndarray = None,
                          is_downstream: bool = False) -> Tuple[np.ndarray, list]:


        N = len(bc_boxes)
        if precomputed_velocities is not None:
            vx, vy = precomputed_velocities[:, 0], precomputed_velocities[:, 1]
        else:
            smoothed = self.rts.smooth(bc_boxes[:, :2])
            vx, vy = smoothed[:, 2], smoothed[:, 3]

        motion_headings = estimate_headings(
            bc_boxes[:, :2],
            np.column_stack((vx, vy)),
            low_speed_threshold=self.low_speed_threshold,
            mode=self.heading_estimation_mode,
            window=self.heading_window,
            min_disp=self.heading_min_step_px,
        ) if N > 0 else np.zeros((0,), dtype=np.float64)

        features = []
        abs_info_list = []
        last_valid_map_feat = None
        _dir_filter = -1 if is_downstream else 1

        _cold_start_feat = None
        for _i in range(N):
            _mf = self.map_interface.query(
                self.scene_name, float(bc_boxes[_i, 0]), float(bc_boxes[_i, 1]),
                float(vx[_i] if precomputed_velocities is not None else 0),
                float(vy[_i] if precomputed_velocities is not None else 0),
                direction_filter=_dir_filter)
            if _mf.get('valid', False):
                _cold_start_feat = _mf
                break
        if _cold_start_feat is None:
            return None, None

        for i in range(N):
            real_x, real_y = bc_boxes[i, 0], bc_boxes[i, 1]
            w, h = bc_boxes[i, 2], bc_boxes[i, 3]
            phys_vx, phys_vy = vx[i], vy[i]

            map_feat = self.map_interface.query(
                self.scene_name, real_x, real_y, phys_vx, phys_vy,
                direction_filter=_dir_filter)

            if not map_feat['valid']:
                _ref = last_valid_map_feat if last_valid_map_feat is not None else _cold_start_feat
                if _ref is not None:
                    W = _ref['road_width']
                    heading = _ref['heading']
                    _dy = real_y - _ref.get('query_y', real_y)
                    _tan_h = np.tan(heading)
                    _dx_lane = np.clip(_dy / _tan_h, -50.0, 50.0) if abs(_tan_h) > 0.05 else 0.0
                    x_left = _ref['x_left'] + _dx_lane
                    dist_l = real_x - x_left
                    dist_r = W - dist_l
                else:
                    W, dist_l, dist_r = 100.0, 50.0, 50.0
                    x_left = real_x - dist_l
                    heading = np.arctan2(phys_vy, phys_vx) if (abs(phys_vx) + abs(phys_vy) > 0.1) else 0.0
            else:
                W = map_feat['road_width']
                dist_l = map_feat['dist_left']
                dist_r = map_feat['dist_right']
                x_left = map_feat['x_left']
                heading = map_feat['heading']
                last_valid_map_feat = map_feat.copy()

            u = dist_l / W
            y_norm = real_y / self.img_height
            w_norm = w / W
            h_norm = h / W
            v_u = (phys_vx / W) * self.velocity_scale
            v_y = (phys_vy / self.img_height) * self.velocity_scale
            d_left = dist_l / W
            d_right = dist_r / W
            f_lat = (dist_l - dist_r) / W
            theta_car = float(motion_headings[i]) if len(motion_headings) > i else float(heading)
            d_theta = compute_lane_flow_dtheta(theta_car, heading, mode=self.dtheta_mode)

            features.append([u, y_norm, w_norm, h_norm, v_u, v_y, f_lat, d_theta, d_left, d_right])
            abs_info_list.append({
                'W': W, 'x_left': x_left, 'phy_x': real_x, 'phy_y': real_y,
                'heading': heading, 'dist_left': dist_l, 'dist_right': dist_r,
                'valid': map_feat.get('valid', False), 'zone': map_feat.get('zone'),
                'query_x': real_x, 'query_y': real_y,
            })

        features = np.array(features, dtype=np.float32)

        if N > 1:
            features[1:, 4] = (features[1:, 0] - features[:-1, 0]) / self.dt * self.velocity_scale
            features[1:, 5] = (features[1:, 1] - features[:-1, 1]) / self.dt * self.velocity_scale
            features[0, 4] = features[1, 4]
            features[0, 5] = features[1, 5]

        return features, abs_info_list

    def _apply_downstream_transform(self, features_raw: np.ndarray) -> np.ndarray:

        f = features_raw[::-1].copy()
        f[:, 0] = 1.0 - f[:, 0]
        f[:, 6] = -f[:, 6]
        f[:, 7] = wrap_to_pi(-f[:, 7])
        f[:, 8], f[:, 9] = f[:, 9].copy(), f[:, 8].copy()
        N = len(f)
        if N > 1:
            f[1:, 4] = (f[1:, 0] - f[:-1, 0]) / self.dt * self.velocity_scale
            f[1:, 5] = (f[1:, 1] - f[:-1, 1]) / self.dt * self.velocity_scale
            f[0, 4] = f[1, 4]
            f[0, 5] = f[1, 5]
        return f



    def apply_relative_coords(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        origin = features[-1, 0:2].copy()
        start_pos = origin.astype(np.float32)
        rel = features.copy()
        rel[:, 0] -= origin[0]
        rel[:, 1] -= origin[1]
        return rel, start_pos


    def _apply_motion_clamp(self,
                            preds_abs: np.ndarray,
                            obs_features: np.ndarray,
                            start_pos: np.ndarray,
                            is_downstream: bool) -> np.ndarray:
        if not self.motion_clamp_enabled or len(preds_abs) < 2 or len(obs_features) < 2:
            return preds_abs

        obs_steps = np.diff(obs_features[:, 0:2], axis=0)
        obs_step_u = float(np.median(np.abs(obs_steps[:, 0])))
        obs_step_y = float(np.median(np.abs(obs_steps[:, 1])))
        scale_u = self.motion_clamp_scale_u_downstream if is_downstream else self.motion_clamp_scale_u_upstream
        scale_y = self.motion_clamp_scale_y_downstream if is_downstream else self.motion_clamp_scale_y_upstream
        limit_u = max(obs_step_u * scale_u, self.motion_clamp_floor_u)
        limit_y = max(obs_step_y * scale_y, self.motion_clamp_floor_y)

        clamped = preds_abs.copy()
        prev = start_pos[0:2].copy()
        for i in range(len(clamped)):
            step = clamped[i, 0:2] - prev
            step[0] = float(np.clip(step[0], -limit_u, limit_u))
            step[1] = float(np.clip(step[1], -limit_y, limit_y))
            clamped[i, 0:2] = prev + step
            prev = clamped[i, 0:2].copy()
        return clamped

    def predict_step(self, obs_features: np.ndarray, anchor_abs: np.ndarray | None = None, is_downstream: bool = False) -> np.ndarray:
        obs_tensor = torch.from_numpy(obs_features).float().unsqueeze(0).to(self.device)

        if self.dynamic_geo_enabled:
            if anchor_abs is None:
                raise ValueError('dynamic_geo inference requires anchor_abs')
            anchor_tensor = torch.from_numpy(anchor_abs).float().unsqueeze(0).to(self.device)
            flipped_tensor = torch.tensor([bool(is_downstream)], dtype=torch.bool, device=self.device)
            with torch.no_grad():
                pred_pos, _ = self.model(
                    obs_tensor,
                    target_seq=None,
                    teacher_forcing_ratio=0.0,
                    pred_len=self.pred_len,
                    scene_names=[self.scene_name],
                    anchor_abs=anchor_tensor,
                    is_flipped=flipped_tensor,
                )
            return pred_pos.squeeze(0).cpu().numpy()

        return self._predict_autoregressive(self.model, obs_tensor)

    def _inherit_recovery_geometry(self, prev_map_feat: dict, provisional_x: float, pixel_y: float) -> dict:
        road_width = max(float(prev_map_feat.get('road_width', prev_map_feat.get('W', 100.0))), 1.0)
        heading = float(prev_map_feat.get('heading', 0.0))
        prev_x_left = float(prev_map_feat.get('x_left', provisional_x - road_width * 0.5))
        prev_query_y = float(prev_map_feat.get('query_y', pixel_y))
        tan_h = np.tan(heading)
        dx_lane = np.clip((pixel_y - prev_query_y) / tan_h, -50.0, 50.0) if abs(tan_h) > 0.05 else 0.0
        x_left = prev_x_left + float(dx_lane)
        dist_left = float(provisional_x) - x_left
        dist_right = road_width - dist_left
        return {
            'road_width': road_width,
            'x_left': float(x_left),
            'heading': heading,
            'dist_left': float(dist_left),
            'dist_right': float(dist_right),
            'query_x': float(provisional_x),
            'query_y': float(pixel_y),
            'valid': False,
            'fallback_type': 'inherited',
            'zone': prev_map_feat.get('zone'),
        }

    def build_recovery_geo_trace(self,
                                 pred_phys: np.ndarray,
                                 anchor_abs_info: dict,
                                 anchor_bc: np.ndarray,
                                 is_downstream: bool) -> list:
        n_preds = len(pred_phys)
        trace = [None] * n_preds
        prev_map_feat = {
            'road_width': max(float(anchor_abs_info['W']), 1.0),
            'x_left': float(anchor_abs_info['x_left']),
            'heading': float(anchor_abs_info.get('heading', 0.0)),
            'dist_left': float(anchor_abs_info.get('dist_left', 0.0)),
            'dist_right': float(anchor_abs_info.get('dist_right', max(float(anchor_abs_info['W']), 1.0))),
            'query_x': float(anchor_bc[0]),
            'query_y': float(anchor_bc[1]),
            'valid': True,
            'zone': anchor_abs_info.get('zone'),
        }
        iter_range = range(n_preds - 1, -1, -1) if is_downstream else range(n_preds)
        direction_filter = -1 if is_downstream else 1

        for i in iter_range:
            u_phys = float(pred_phys[i, 0])
            pixel_y = float(pred_phys[i, 1]) * self.img_height
            prev_width = max(float(prev_map_feat.get('road_width', 100.0)), 1.0)
            prev_x_left = float(prev_map_feat.get('x_left', float(anchor_abs_info['x_left'])))
            provisional_x = prev_x_left + u_phys * prev_width
            vx = (provisional_x - float(prev_map_feat.get('query_x', provisional_x))) / self.dt
            vy = (pixel_y - float(prev_map_feat.get('query_y', pixel_y))) / self.dt

            map_res = self.map_interface.query(
                self.scene_name,
                provisional_x,
                pixel_y,
                vx=vx,
                vy=vy,
                prev_map_feat=prev_map_feat,
                direction_filter=direction_filter,
            )
            if map_res.get('valid', False) and 'road_width' in map_res and 'x_left' in map_res:
                curr = dict(map_res)
                curr['road_width'] = max(float(curr['road_width']), 1.0)
                curr['x_left'] = float(curr['x_left'])
                curr['heading'] = float(curr.get('heading', prev_map_feat.get('heading', 0.0)))
                curr['query_x'] = float(curr.get('query_x', provisional_x))
                curr['query_y'] = float(curr.get('query_y', pixel_y))
            else:
                curr = self._inherit_recovery_geometry(prev_map_feat, provisional_x, pixel_y)

            trace[i] = curr
            prev_map_feat = curr

        return trace

    def denormalize_predictions(self, pred_positions: np.ndarray,
                                anchor_abs_info: dict,
                                anchor_bc: np.ndarray,
                                is_downstream: bool,
                                geo_trace: list | None = None) -> np.ndarray:


        n_preds = len(pred_positions)
        preds_bc = np.zeros((n_preds, 4), dtype=np.float64)

        W_anchor = max(float(anchor_abs_info['W']), 1.0)
        W_curr = W_anchor
        x_left_curr = anchor_abs_info['x_left']
        heading = anchor_abs_info['heading']
        anchor_w, anchor_h = anchor_bc[2], anchor_bc[3]

        prev_px = float(anchor_bc[0])
        prev_py = anchor_bc[1]
        last_map_feat = {
            'road_width': float(W_curr),
            'x_left': float(x_left_curr),
            'heading': float(heading),
            'query_x': float(anchor_bc[0]),
            'query_y': float(anchor_bc[1]),
            'valid': True,
        }

        iter_range = range(n_preds - 1, -1, -1) if is_downstream else range(n_preds)

        for step_idx, i in enumerate(iter_range):
            u_phys = pred_positions[i, 0]
            y_norm = pred_positions[i, 1]
            pixel_y = y_norm * self.img_height

            if geo_trace is not None and geo_trace[i] is not None:
                step_geo = geo_trace[i]
                W_curr = max(float(step_geo.get('road_width', step_geo.get('W', W_curr))), 1.0)
                x_left_curr = float(step_geo.get('x_left', x_left_curr))
            else:
                dy = pixel_y - prev_py
                _tan_h = np.tan(heading)
                if abs(_tan_h) > 0.05:
                    dx_lane = np.clip(dy / _tan_h, -50.0, 50.0)
                    x_left_curr += dx_lane
            pixel_x = x_left_curr + u_phys * W_curr

            accumulated_ratio = pixel_y / anchor_bc[1] if anchor_bc[1] > 1 else 1.0
            scale = np.clip(accumulated_ratio, 0.3, 3.0)
            pixel_w = np.clip(anchor_w * scale, 10, 800)
            pixel_h = np.clip(anchor_h * scale, 10, 800)

            pixel_x = np.clip(pixel_x, -100, self.img_width + 100)
            pixel_y = np.clip(pixel_y, -100, self.img_height + 100)

            preds_bc[i] = [pixel_x, pixel_y, pixel_w, pixel_h]
            prev_px = pixel_x
            prev_py = pixel_y

        return preds_bc

    def process_track(self, track_df: pd.DataFrame, is_downstream: bool) -> Tuple[np.ndarray, int]:


        track_df = track_df.sort_values('frame')
        min_frame, max_frame = track_df['frame'].min(), track_df['frame'].max()
        if len(track_df) != (max_frame - min_frame + 1):
            track_df = track_df.set_index('frame')
            full_idx = np.arange(min_frame, max_frame + 1)
            track_df = track_df.reindex(full_idx).interpolate(method='linear').reset_index()

        boxes_tl = track_df[['x', 'y', 'w', 'h']].values.astype(np.float64)
        boxes_bc = self._xywh_to_bottom_center(boxes_tl)
        output_obs_bc = boxes_bc.copy()

        if len(track_df) < self.obs_len:
            return self._bottom_center_to_xywh(output_obs_bc), 0

        raw_track = track_df[['frame', 'x', 'y', 'w', 'h']].values.astype(np.float64)
        shared_features_phys, shared_abs_info_phys, _, frame_metadata = self.shared_feature_extractor.process_trajectory_with_metadata(
            self.scene_name,
            raw_track,
            do_flip=is_downstream,
            is_downstream=is_downstream,
        )
        if len(shared_features_phys) == 0:
            return None, 0

        if is_downstream:
            features_model = shared_features_phys[::-1].copy()
            shared_abs_info_model = shared_abs_info_phys[::-1].copy()
            n_frames = len(features_model)
            if n_frames > 1:
                features_model[1:, 4] = (
                    (features_model[1:, 0] - features_model[:-1, 0])
                    / self.shared_feature_extractor.dt
                    * self.shared_feature_extractor.vel_scale
                )
                features_model[1:, 5] = (
                    (features_model[1:, 1] - features_model[:-1, 1])
                    / self.shared_feature_extractor.dt
                    * self.shared_feature_extractor.vel_scale
                )
                features_model[0, 4] = features_model[1, 4]
                features_model[0, 5] = features_model[1, 5]
        else:
            features_model = shared_features_phys
            shared_abs_info_model = shared_abs_info_phys

        obs_features = features_model[-self.obs_len:]
        if len(obs_features) < 2:
            return None, 0

        anchor_idx_phys = 0 if is_downstream else -1
        anchor_metadata = frame_metadata[anchor_idx_phys]
        anchor_abs_info = {
            'W': anchor_metadata['W'],
            'x_left': anchor_metadata['x_left'],
            'heading': anchor_metadata['heading'],
            'phy_x': anchor_metadata['phy_x'],
            'phy_y': anchor_metadata['phy_y'],
            'dist_left': anchor_metadata['dist_left'],
            'dist_right': anchor_metadata['dist_right'],
            'valid': anchor_metadata['valid'],
            'fallback_type': anchor_metadata['fallback_type'],
            'query_x': anchor_metadata['query_x'],
            'query_y': anchor_metadata['query_y'],
        }
        anchor_bc = output_obs_bc[anchor_idx_phys]


        rel_features, start_pos = self.apply_relative_coords(obs_features)
        anchor_abs = np.array(
            [start_pos[0], start_pos[1], anchor_abs_info['W'], anchor_abs_info['x_left']],
            dtype=np.float32,
        )
        preds_rel = self.predict_step(rel_features, anchor_abs=anchor_abs, is_downstream=is_downstream)

        preds_abs = preds_rel.copy()
        preds_abs[:, 0] += start_pos[0]
        preds_abs[:, 1] += start_pos[1]
        preds_abs[:, 0] = np.clip(preds_abs[:, 0], -0.5, 1.5)
        preds_abs[:, 1] = np.clip(preds_abs[:, 1], -0.2, 1.2)
        preds_abs = self._apply_motion_clamp(
            preds_abs, obs_features, start_pos, is_downstream
        )
        preds_abs[:, 0] = np.clip(preds_abs[:, 0], -0.5, 1.5)
        preds_abs[:, 1] = np.clip(preds_abs[:, 1], -0.2, 1.2)
        if is_downstream:
            pred_phys = preds_abs[::-1].copy()
            pred_phys[:, 0] = 1.0 - pred_phys[:, 0]
        else:
            pred_phys = preds_abs.copy()

        recovery_geo_trace = (
            self.build_recovery_geo_trace(pred_phys, anchor_abs_info, anchor_bc, is_downstream)
            if self.dynamic_geo_enabled else None
        )
        preds_bc = self.denormalize_predictions(
            pred_phys, anchor_abs_info, anchor_bc, is_downstream, geo_trace=recovery_geo_trace)

        if is_downstream:
            final_track_bc = np.vstack([preds_bc, output_obs_bc])
        else:
            final_track_bc = np.vstack([output_obs_bc, preds_bc])

        return self._bottom_center_to_xywh(final_track_bc), len(preds_bc)


def _load_track_dataframe(input_path: str) -> pd.DataFrame:
    df = pd.read_csv(input_path, header=None)
    if df.shape[1] >= 10:
        df = df.iloc[:, :10].copy()
        df.columns = ['frame', 'id', 'x', 'y', 'w', 'h', 'c', 'd', 'e', 'label']
    elif df.shape[1] == 9:
        df.columns = ['frame', 'id', 'x', 'y', 'w', 'h', 'c', 'd', 'e']
    elif df.shape[1] == 8:
        df.columns = ['frame', 'id', 'x', 'y', 'w', 'h', 'c', 'd']
    elif df.shape[1] == 7:
        df.columns = ['frame', 'id', 'x', 'y', 'w', 'h', 'label']
    elif df.shape[1] == 6:
        df.columns = ['frame', 'id', 'x', 'y', 'w', 'h']
    else:
        raise ValueError(f'Unsupported track format: {input_path}, columns={df.shape[1]}')
    for col in ['frame', 'id']:
        df[col] = df[col].astype(int)
    return df.sort_values(['id', 'frame']).reset_index(drop=True)


def _prepare_inference_dataframe(config: dict, scene_name: str, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    input_cfg = config['inference']['input']
    use_label_filter = bool(input_cfg.get('use_label_filter', False))
    label_column = str(input_cfg.get('label_column', 'label'))
    observed_label = int(input_cfg.get('observed_label', 0))

    source_track_count = int(df['id'].nunique()) if not df.empty else 0
    if use_label_filter:
        if label_column not in df.columns:
            raise ValueError(
                f'use_label_filter=True but label_column "{label_column}" is missing; '
                f'available columns: {list(df.columns)}'
            )
        filtered_df = df[df[label_column].astype(int) == observed_label].copy()
    else:
        filtered_df = df.copy()

    filtered_df = filtered_df.sort_values(['id', 'frame']).reset_index(drop=True)
    filtered_track_count = int(filtered_df['id'].nunique()) if not filtered_df.empty else 0
    return filtered_df, {
        'source_rows': int(len(df)),
        'filtered_rows': int(len(filtered_df)),
        'dropped_rows': int(len(df) - len(filtered_df)),
        'source_track_count': source_track_count,
        'filtered_track_count': filtered_track_count,
        'dropped_tracks': int(source_track_count - filtered_track_count),
    }


def _resolve_scene_names(config: dict) -> List[str]:
    inf_cfg = config.get('inference', {})
    if inf_cfg.get('use_scene_split', False):
        split_name = inf_cfg.get('split_name', 'test_scenes')
        split_file = Path(config['paths']['scene_split_file'])
        if not split_file.exists():
            raise FileNotFoundError(
                f'Scene split file not found: {split_file}. Run python run_preprocess.py first, or set inference.use_scene_split=false and provide inference.scenes.'
            )
        split_info = json.loads(split_file.read_text(encoding='utf-8'))
        if split_name in split_info:
            return list(split_info[split_name])
        raise KeyError(f'Split "{split_name}" not found in {split_file}')

    scenes = inf_cfg.get('scenes') or []
    if scenes:
        return list(scenes)

    scene_name = config.get('scene', {}).get('name')
    if scene_name:
        return [scene_name]

    raise ValueError('No inference scenes configured.')




def _resolve_scene_input_items(config: dict, scene_name: str, scenes: List[str]) -> List[tuple[str, str, str]]:
    input_cfg = config['inference']['input']
    base_path = input_cfg.get('data_path')
    if not base_path:
        raise ValueError('Input path missing')

    base = Path(base_path)
    if base.is_file():
        scene_count = len(scenes)
        if scene_count != 1:
            raise ValueError(
                'exact-file mode requires exactly one scene; '
                f'scene_count={scene_count}; scenes={list(scenes)}'
            )
        return [(scene_name, scene_name, str(base))]

    file_pattern = input_cfg.get('file_pattern', '{scene}/gt/gt.txt')
    try:
        relative_path = file_pattern.format(scene=scene_name)
    except Exception as exc:
        raise ValueError(
            'Failed to resolve inference input path: '
            f'scene_name={scene_name}; data_path={base_path}; file_pattern={file_pattern}'
        ) from exc

    expected_path = base / relative_path
    if expected_path.is_file():
        return [(scene_name, scene_name, str(expected_path))]

    nested_paths = sorted((base / scene_name).glob('*/gt/gt.txt'))
    if nested_paths:
        return [
            (scene_name, f'{scene_name}/{path.parent.parent.name}', str(path))
            for path in nested_paths
        ]

    if not expected_path.is_file():
        raise FileNotFoundError(
            'Inference input file not found: '
            f'scene_name={scene_name}; data_path={base_path}; '
            f'file_pattern={file_pattern}; expected_path={expected_path}'
        )
    return [(scene_name, scene_name, str(expected_path))]


def _sanitize_output_dataframe(rows: list, image_width: float, image_height: float) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=['frame', 'id', 'x', 'y', 'w', 'h', 'c', 'd', 'e', 'f'])
    df = df.sort_values(['id', 'frame'])
    coord_cols = ['x', 'y', 'w', 'h']
    for col in coord_cols:
        df.loc[~np.isfinite(df[col]), col] = 10.0 if col in ['w', 'h'] else 0.0
    df['x'] = np.clip(df['x'], 0, image_width)
    df['y'] = np.clip(df['y'], 0, image_height)
    df['w'] = np.clip(df['w'], 10, 800)
    df['h'] = np.clip(df['h'], 10, 800)
    df[coord_cols] = np.rint(df[coord_cols]).astype(int)
    return df


def _scene_output_paths(config: dict, scene_name: str) -> tuple:
    out_cfg = config['inference']['output']
    prediction_only_dir = Path(out_cfg.get('prediction_only_dir', Path(config['paths']['prediction_dir']) / 'prediction_only'))
    obs_and_prediction_dir = Path(out_cfg.get('obs_and_prediction_dir', Path(config['paths']['prediction_dir']) / 'obs_and_prediction'))
    prediction_only_pattern = out_cfg.get('prediction_only_file_pattern', '{scene}_prediction_only.txt')
    obs_and_prediction_pattern = out_cfg.get('obs_and_prediction_file_pattern', '{scene}_obs_and_prediction.txt')
    prediction_only_dir.mkdir(parents=True, exist_ok=True)
    obs_and_prediction_dir.mkdir(parents=True, exist_ok=True)
    pred_only_path = prediction_only_dir / prediction_only_pattern.format(scene=scene_name)
    obs_pred_path = obs_and_prediction_dir / obs_and_prediction_pattern.format(scene=scene_name)
    pred_only_path.parent.mkdir(parents=True, exist_ok=True)
    obs_pred_path.parent.mkdir(parents=True, exist_ok=True)
    return pred_only_path, obs_pred_path


def _run_scene(config: dict, map_scene_name: str, output_scene_name: str, input_path: str) -> dict:
    scene_config = deepcopy(config)
    scene_config.setdefault('scene', {})['name'] = map_scene_name
    predictor = TrajectoryPredictor(scene_config)

    print(f'Reading [{output_scene_name}] with map [{map_scene_name}]: {input_path}')
    source_df = _load_track_dataframe(input_path)
    df, input_stats = _prepare_inference_dataframe(scene_config, output_scene_name, source_df)
    obs_pred_rows = []
    pred_only_rows = []
    skipped = 0
    short_tracks = 0
    predicted_tracks = 0
    if df.empty:
        print(f'No label-0 observation rows found for [{scene_name}], skipping prediction.')

    frame_cfg = scene_config.get('inference', {}).get('frame_range', {})
    if frame_cfg.get('enabled', False):
        min_frame = frame_cfg.get('start_frame')
        max_frame = frame_cfg.get('end_frame')
        if min_frame is None:
            min_frame = -float('inf')
        if max_frame is None:
            max_frame = float('inf')
    else:
        min_frame = -float('inf')
        max_frame = float('inf')

    for tid, group in df.groupby('id'):
        group = group.sort_values('frame')
        raw_frames = group['frame'].values
        boxes_bc_check = predictor._xywh_to_bottom_center(group[['x', 'y', 'w', 'h']].values)
        dy_bc = boxes_bc_check[-1, 1] - boxes_bc_check[0, 1]
        is_downstream = dy_bc > 0

        try:
            full_track, pred_count = predictor.process_track(group, is_downstream)
            if full_track is None:
                skipped += 1
                continue
            if pred_count == 0:
                short_tracks += 1
            else:
                predicted_tracks += 1

            first_frame = raw_frames[0]
            start_frame = first_frame - pred_count if is_downstream else first_frame

            pred_start_idx = 0 if is_downstream else max(0, len(full_track) - pred_count)
            pred_end_idx = pred_count if is_downstream else len(full_track)

            for i in range(len(full_track)):
                curr_f = int(start_frame + i)
                if curr_f < min_frame or curr_f > max_frame:
                    continue
                row = [curr_f, tid, *full_track[i], 1, 1, 1, 1]
                obs_pred_rows.append(row)
                if pred_count > 0 and pred_start_idx <= i < pred_end_idx:
                    pred_only_rows.append(row)

        except Exception as e:
            print(f'Error [{output_scene_name}] tid {tid}: {e}')
            import traceback
            traceback.print_exc()

    pred_only_path, obs_pred_path = _scene_output_paths(scene_config, output_scene_name)
    if pred_only_rows:
        _sanitize_output_dataframe(pred_only_rows, predictor.img_width, predictor.img_height).to_csv(pred_only_path, header=False, index=False)
        print(f'Saved prediction-only [{output_scene_name}]: {pred_only_path}')
    if obs_pred_rows:
        _sanitize_output_dataframe(obs_pred_rows, predictor.img_width, predictor.img_height).to_csv(obs_pred_path, header=False, index=False)
        print(f'Saved observation+prediction [{output_scene_name}]: {obs_pred_path}')

    filtered_track_count = int(df['id'].nunique()) if not df.empty else 0
    source_track_count = int(source_df['id'].nunique()) if not source_df.empty else 0
    report = {
        'scene': output_scene_name,
        'map_scene': map_scene_name,
        'input_path': input_path,
        'prediction_only_path': str(pred_only_path),
        'obs_and_prediction_path': str(obs_pred_path),
        'source_rows': int(input_stats.get('source_rows', len(source_df))),
        'observation_rows': int(input_stats.get('filtered_rows', len(df))),
        'dropped_input_rows': int(input_stats.get('dropped_rows', 0)),
        'source_track_count': int(input_stats.get('source_track_count', source_track_count)),
        'track_count': int(input_stats.get('filtered_track_count', filtered_track_count)),
        'dropped_input_tracks': int(input_stats.get('dropped_tracks', 0)),
        'processed_tracks': int(filtered_track_count - skipped),
        'predicted_tracks': int(predicted_tracks),
        'short_tracks': int(short_tracks),
        'skipped_tracks': int(skipped),
        'prediction_only_rows': int(len(pred_only_rows)),
        'obs_and_prediction_rows': int(len(obs_pred_rows)),
    }
    return report


def main(config_dir=None):
    cfg_root = Path(config_dir) if config_dir is not None else (PROJECT_ROOT / 'configs')
    config = load_config_bundle(cfg_root, ['data', 'model', 'train', 'predict'])
    validate_time_config_contract(config)
    resolve_canonical_window_config(config)
    scenes = _resolve_scene_names(config)
    reports = []
    for scene_name in scenes:
        input_items = _resolve_scene_input_items(config, scene_name, scenes)
        for map_scene_name, output_scene_name, input_path in input_items:
            reports.append(_run_scene(config, map_scene_name, output_scene_name, input_path))

    out_cfg = config['inference']['output']
    report_path = Path(out_cfg.get('report_path', Path(config['paths']['prediction_dir']) / 'infer_report.json'))
    report_path.parent.mkdir(parents=True, exist_ok=True)

    inf_cfg = config.get('inference', {})
    summary = {
        'scene_source': inf_cfg.get('split_name') if inf_cfg.get('use_scene_split', False) else 'explicit',
        'split_file': config['paths'].get('scene_split_file'),
        'scenes': scenes,
        'expanded_scenes': [item['scene'] for item in reports],
        'reports': reports,
    }
    report_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'Saved infer report: {report_path}')


if __name__ == '__main__':
    main()
