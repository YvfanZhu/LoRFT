from __future__ import annotations

import json
import os
import random
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from .seq2seq_model import TrajectoryDecoder, TrajectoryEncoder
from utils.geometry import compute_lane_flow_dtheta, estimate_heading_from_recent, heading_angle_difference, wrap_to_pi


class DynamicMapInterface:
    def __init__(self, map_data_dir: str | None = None):
        self.maps: Dict[str, dict] = {}
        self.map_data_dir = map_data_dir
        try:
            from shapely.geometry import LineString, Point

            self.has_shapely = True
            self.LineString = LineString
            self.Point = Point
        except ImportError:
            self.has_shapely = False

    def load_scene_map(self, scene_name: str) -> bool:
        if not self.has_shapely or not scene_name:
            return False
        if scene_name in self.maps:
            return True
        if not self.map_data_dir:
            return False
        json_path = os.path.join(self.map_data_dir, f'{scene_name}.json')
        if not os.path.exists(json_path):
            return False
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception:
            return False

        self.maps[scene_name] = {
            'centerlines': {
                key: self.LineString(value['points'])
                for key, value in data.get('centerlines', {}).items()
                if len(value.get('points', [])) > 1
            },
            'boundaries': {
                key: self.LineString(value['points'])
                for key, value in data.get('boundaries', {}).items()
                if len(value.get('points', [])) > 1
            },
            'zone_meta': {key: value for key, value in data.get('zone_meta', {}).items()},
        }
        return True

    def query(
        self,
        scene: str,
        x: float,
        y: float,
        vx: float | None = None,
        vy: float | None = None,
        prev_map_feat: dict | None = None,
        direction_filter: int | None = None,
    ) -> dict:
        if not self.has_shapely or scene not in self.maps:
            return {
                'dist_left': 50.0,
                'dist_right': 50.0,
                'road_width': 100.0,
                'heading': 0.0,
                'x_left': x - 50.0,
                'valid': False,
            }

        map_data = self.maps[scene]
        point = self.Point(x, y)

        vehicle_heading = None
        if vx is not None and vy is not None and (abs(vx) > 0.1 or abs(vy) > 0.1):
            vehicle_heading = np.arctan2(vy, vx)

        candidate_zones = []
        for zone, centerline in map_data['centerlines'].items():
            dist = centerline.distance(point)
            if dist < 2000:
                candidate_zones.append((zone, centerline, dist))
        if not candidate_zones:
            min_dist = float('inf')
            best_zone = None
            best_centerline = None
            for zone, centerline in map_data['centerlines'].items():
                dist = centerline.distance(point)
                if dist < min_dist:
                    min_dist = dist
                    best_zone = zone
                    best_centerline = centerline
            if best_zone is None or best_centerline is None:
                return {
                    'dist_left': 50.0,
                    'dist_right': 50.0,
                    'road_width': 100.0,
                    'heading': 0.0,
                    'x_left': x - 50.0,
                    'valid': False,
                }
            candidate_zones.append((best_zone, best_centerline, min_dist))

        candidate_zones.sort(key=lambda item: item[2])

        if direction_filter is not None:
            import re as _re

            zone_meta = map_data.get('zone_meta', {})
            if zone_meta:
                filtered = [
                    (zone, centerline, dist)
                    for zone, centerline, dist in candidate_zones
                    if zone_meta.get(_re.sub(r'_\d+$', '', zone), {}).get('direction') == direction_filter
                ]
                if filtered:
                    candidate_zones = filtered

        best_zone = None
        best_centerline = None
        min_score = float('inf')
        for zone, centerline, dist in candidate_zones[: min(2, len(candidate_zones))]:
            projected_dist = centerline.project(point)
            if projected_dist >= centerline.length - 1e-6:
                point_curr = centerline.interpolate(max(0.0, centerline.length - 1.0))
                point_next = centerline.interpolate(centerline.length)
            else:
                point_curr = centerline.interpolate(projected_dist)
                point_next = centerline.interpolate(min(centerline.length, projected_dist + 1.0))
            tangent = np.array([point_next.x - point_curr.x, point_next.y - point_curr.y])
            norm = np.linalg.norm(tangent)
            tangent = tangent / norm if norm > 1e-6 else np.array([0.0, 1.0])
            lane_heading = np.arctan2(tangent[1], tangent[0])
            score = dist
            if vehicle_heading is not None:
                angle_diff = heading_angle_difference(vehicle_heading, lane_heading)
                score += angle_diff * 200.0
            if score < min_score:
                min_score = score
                best_zone = zone
                best_centerline = centerline

        if best_zone is None or best_centerline is None:
            best_zone, best_centerline = candidate_zones[0][0], candidate_zones[0][1]

        left_key = next((key for key in map_data['boundaries'] if key.startswith(f'{best_zone}:') and 'left' in key), None)
        right_key = next((key for key in map_data['boundaries'] if key.startswith(f'{best_zone}:') and 'right' in key), None)

        if left_key and right_key:
            dist_left = map_data['boundaries'][left_key].distance(point)
            dist_right = map_data['boundaries'][right_key].distance(point)
            road_width = dist_left + dist_right
            if road_width < 1.0:
                road_width = 100.0
                dist_left = 50.0
                dist_right = 50.0
            projected_dist = best_centerline.project(point)
            point_curr = best_centerline.interpolate(min(best_centerline.length, max(0.0, projected_dist)))
            point_next = best_centerline.interpolate(min(best_centerline.length, projected_dist + 1.0))
            tangent = np.array([point_next.x - point_curr.x, point_next.y - point_curr.y])
            norm = np.linalg.norm(tangent)
            heading = np.arctan2(tangent[1], tangent[0]) if norm > 1e-6 else 0.0
            x_left = x - dist_left
            result = {
                'dist_left': float(dist_left),
                'dist_right': float(dist_right),
                'road_width': float(road_width),
                'heading': float(heading),
                'x_left': float(x_left),
                'valid': True,
                'zone': best_zone,
                'query_x': float(x),
                'query_y': float(y),
            }
        else:
            result = {'valid': False, 'zone': best_zone, 'query_x': float(x), 'query_y': float(y)}

        if not result.get('valid', False) and prev_map_feat is not None and prev_map_feat.get('valid', False):
            road_width = float(prev_map_feat['road_width'])
            heading = float(prev_map_feat['heading'])
            dy = float(y) - float(prev_map_feat.get('query_y', y))
            tan_h = np.tan(heading)
            dx_lane = np.clip(dy / tan_h, -50.0, 50.0) if abs(tan_h) > 0.05 else 0.0
            x_left = float(prev_map_feat['x_left']) + float(dx_lane)
            dist_left = float(x) - x_left
            dist_right = road_width - dist_left
            result = {
                'dist_left': float(dist_left),
                'dist_right': float(dist_right),
                'road_width': float(road_width),
                'heading': float(heading),
                'x_left': float(x_left),
                'valid': True,
                'zone': prev_map_feat.get('zone', best_zone),
                'query_x': float(x),
                'query_y': float(y),
            }

        if not result.get('valid', False):
            result = {
                'dist_left': 50.0,
                'dist_right': 50.0,
                'road_width': 100.0,
                'heading': 0.0,
                'x_left': float(x) - 50.0,
                'valid': False,
                'zone': result.get('zone', best_zone),
                'query_x': float(x),
                'query_y': float(y),
            }
        return result


class TrajectorySeq2SeqDynamicGeo(nn.Module):
    supports_dynamic_geo = True

    def __init__(self, config: dict):
        super().__init__()
        enc_cfg = config['model']['encoder']
        dec_cfg = config['model']['decoder']
        loss_cfg = config['loss']
        dyn_cfg = config.get('model', {}).get('dynamic_geo', {})
        heading_cfg = config.get('heading_estimation', {})

        self.velocity_scale = float(loss_cfg['velocity_scale'])
        self.dt = float(config['model']['dt'])
        self.max_delta = float(config.get('model', {}).get('decode', {}).get('max_delta', 0.1))
        self.img_height = float(config['image']['height'])
        self.refresh_every = int(dyn_cfg.get('refresh_every', 5))
        self.low_speed_threshold = float(config.get('map_matching', {}).get('low_speed_threshold', 2.0))
        self.heading_estimation_mode = str(heading_cfg.get('mode', 'motion_aligned'))
        self.local_heading_window = int(heading_cfg.get('window', 5))
        self.local_heading_min_disp = float(heading_cfg.get('min_step_px', 1.0))
        self.dtheta_mode = str(dyn_cfg.get('dtheta_mode', config.get('heading_estimation', {}).get('dtheta_mode', 'motion_aligned')))
        map_data_dir = dyn_cfg.get('map_data_dir') or config.get('map', {}).get('data_dir') or config.get('paths', {}).get('map_root')

        self.encoder = TrajectoryEncoder(
            input_dim=enc_cfg['input_dim'],
            hidden_dim=enc_cfg['hidden_dim'],
            num_layers=enc_cfg['num_layers'],
            dropout=enc_cfg['dropout'],
            bidirectional=enc_cfg['bidirectional'],
        )
        self.decoder = TrajectoryDecoder(
            input_dim=dec_cfg['input_dim'],
            hidden_dim=dec_cfg['hidden_dim'],
            output_dim=dec_cfg['output_dim'],
            num_layers=dec_cfg['num_layers'],
            dropout=dec_cfg['dropout'],
        )
        self.map_interface = DynamicMapInterface(map_data_dir=map_data_dir)
        self.scene_id_to_name = self._build_scene_id_to_name(config)

    def _build_scene_id_to_name(self, config: dict) -> Dict[int, str]:
        ordered_scenes: List[str] = []

        def _append(scene_name: str | None) -> None:
            scene_name = str(scene_name or '').strip()
            if scene_name and scene_name not in ordered_scenes:
                ordered_scenes.append(scene_name)

        for scene_name in config.get('scenes', []) or []:
            _append(scene_name)
        split_cfg = config.get('split_by_scene', {})
        for key in ('train_scenes', 'val_scenes', 'test_scenes'):
            for scene_name in split_cfg.get(key, []) or []:
                _append(scene_name)
        return {idx: scene_name for idx, scene_name in enumerate(ordered_scenes)}

    def _normalize_scene_names(
        self,
        scene_names: Sequence[str] | str | None,
        batch_size: int,
        scene_ids: Optional[torch.Tensor] = None,
    ) -> List[str]:
        if scene_ids is not None:
            scene_id_list = [int(item) for item in scene_ids.detach().cpu().tolist()]
            resolved = []
            for scene_id in scene_id_list:
                if scene_id not in self.scene_id_to_name:
                    raise KeyError(f'Unknown scene id for dynamic_geo: {scene_id}')
                resolved.append(self.scene_id_to_name[scene_id])
            if len(resolved) == 1 and batch_size > 1:
                return resolved * batch_size
            if len(resolved) == batch_size:
                return resolved
            if len(resolved) >= batch_size:
                return resolved[:batch_size]
            raise ValueError(f'scene_ids size mismatch: got {len(resolved)}, expected at least {batch_size}')

        if scene_names is None:
            raise ValueError('dynamic_geo requires scene_names or scene_ids')
        if isinstance(scene_names, str):
            return [scene_names] * batch_size
        scene_list = list(scene_names)
        if len(scene_list) == 1 and batch_size > 1:
            return scene_list * batch_size
        if len(scene_list) >= batch_size:
            return scene_list[:batch_size]
        raise ValueError(f'scene_names size mismatch: got {len(scene_list)}, expected at least {batch_size}')

    def _initialize_states(
        self,
        input_seq: torch.Tensor,
        scene_names: Sequence[str],
        anchor_abs: torch.Tensor,
        is_flipped: torch.Tensor,
    ) -> List[dict]:
        anchor_np = anchor_abs.detach().cpu().numpy()
        last_vel_np = input_seq[:, -1, 4:6].detach().cpu().numpy()
        flipped_np = is_flipped.detach().cpu().numpy().astype(bool)
        states: List[dict] = []

        for idx, scene_name in enumerate(scene_names):
            flipped = bool(flipped_np[idx])
            u_model = float(anchor_np[idx, 0])
            y_norm = float(anchor_np[idx, 1])
            road_width = max(float(anchor_np[idx, 2]), 1.0)
            x_left = float(anchor_np[idx, 3])
            u_phys = 1.0 - u_model if flipped else u_model
            pixel_x = x_left + u_phys * road_width
            pixel_y = y_norm * self.img_height
            vx = float(last_vel_np[idx, 0]) / self.velocity_scale * road_width
            vy = float(last_vel_np[idx, 1]) / self.velocity_scale * self.img_height * (-1.0 if flipped else 1.0)

            state = {
                'scene': scene_name,
                'is_flipped': flipped,
                'W': road_width,
                'x_left': x_left,
                'prev_map_feat': None,
                'recent_abs_positions': [],
            }

            if self.map_interface.load_scene_map(scene_name):
                map_res = self.map_interface.query(
                    scene_name,
                    pixel_x,
                    pixel_y,
                    vx=vx,
                    vy=vy,
                    direction_filter=-1 if flipped else 1,
                )
                if map_res.get('valid', False):
                    state['W'] = max(float(map_res['road_width']), 1.0)
                    state['x_left'] = float(map_res['x_left'])
                state['prev_map_feat'] = map_res

            state['recent_abs_positions'].append((pixel_x, pixel_y))
            states.append(state)
        return states

    def _resolve_lane_geometry(self, state: dict) -> tuple[float, float]:
        road_width = max(float(state.get('W', 1.0)), 1.0)
        x_left = float(state.get('x_left', 0.0))
        return road_width, x_left

    def _append_recent_abs_positions(self, abs_pos_model: torch.Tensor, states: List[dict]) -> None:
        abs_pos_np = abs_pos_model.squeeze(1).detach().cpu().numpy()
        max_history = max(self.local_heading_window, 2)
        for idx, state in enumerate(states):
            road_width, x_left = self._resolve_lane_geometry(state)
            abs_u_model = float(abs_pos_np[idx, 0])
            abs_y_model = float(abs_pos_np[idx, 1])
            flipped = bool(state.get('is_flipped', False))
            u_phys = 1.0 - abs_u_model if flipped else abs_u_model
            pixel_x = x_left + u_phys * road_width
            pixel_y = abs_y_model * self.img_height
            recent = state.setdefault('recent_abs_positions', [])
            recent.append((pixel_x, pixel_y))
            if len(recent) > max_history:
                del recent[:-max_history]

    def _estimate_refresh_theta(self, state: dict, vx: float, vy: float, fallback_theta: float) -> float:
        recent = state.get('recent_abs_positions', [])
        return estimate_heading_from_recent(
            recent,
            (vx, vy),
            low_speed_threshold=self.low_speed_threshold,
            fallback_heading=fallback_theta,
            mode=self.heading_estimation_mode,
            window=self.local_heading_window,
            min_disp=self.local_heading_min_disp,
        )

    def _refresh_geo_features(
        self,
        abs_pos_model: torch.Tensor,
        next_vel_feat: torch.Tensor,
        states: List[dict],
        current_geo_feat: torch.Tensor,
    ) -> torch.Tensor:
        abs_pos_np = abs_pos_model.squeeze(1).detach().cpu().numpy()
        vel_np = next_vel_feat.squeeze(1).detach().cpu().numpy()
        geo_np = current_geo_feat.squeeze(1).detach().cpu().numpy().copy()

        for idx, state in enumerate(states):
            scene_name = state['scene']
            if not scene_name or not self.map_interface.load_scene_map(scene_name):
                continue

            flipped = bool(state['is_flipped'])
            abs_u_model = float(abs_pos_np[idx, 0])
            abs_y_model = float(abs_pos_np[idx, 1])
            u_phys = 1.0 - abs_u_model if flipped else abs_u_model
            pixel_x = float(state['x_left']) + u_phys * float(state['W'])
            pixel_y = abs_y_model * self.img_height
            vx = float(vel_np[idx, 0]) / self.velocity_scale * float(state['W'])
            vy = float(vel_np[idx, 1]) / self.velocity_scale * self.img_height * (-1.0 if flipped else 1.0)

            map_res = self.map_interface.query(
                scene_name,
                pixel_x,
                pixel_y,
                vx=vx,
                vy=vy,
                prev_map_feat=state.get('prev_map_feat'),
                direction_filter=-1 if flipped else 1,
            )

            if not map_res.get('valid', False):
                continue

            state['W'] = max(float(map_res['road_width']), 1.0)
            state['x_left'] = float(map_res['x_left'])
            state['prev_map_feat'] = map_res

            dist_left = float(map_res['dist_left'])
            dist_right = float(map_res['dist_right'])
            road_width = max(dist_left + dist_right, 1e-6)
            heading = float(map_res['heading'])
            fallback_theta = np.arctan2(vy, vx) if (abs(vx) + abs(vy) > 0.1) else heading
            theta_car = self._estimate_refresh_theta(state, vx, vy, fallback_theta)
            d_theta = compute_lane_flow_dtheta(theta_car, heading, mode=self.dtheta_mode)
            f_lat = (dist_left - dist_right) / road_width
            d_left = dist_left / road_width
            d_right = dist_right / road_width

            if flipped:
                f_lat = -f_lat
                d_theta = float(wrap_to_pi(-d_theta))
                d_left, d_right = d_right, d_left

            geo_np[idx] = np.array([f_lat, d_theta, d_left, d_right], dtype=np.float32)

        return torch.as_tensor(geo_np, dtype=current_geo_feat.dtype, device=current_geo_feat.device).unsqueeze(1)

    def forward(
        self,
        input_seq: torch.Tensor,
        target_seq: Optional[torch.Tensor],
        teacher_forcing_ratio: float = 0.5,
        pred_len: int = 75,
        scene_names: Sequence[str] | str | None = None,
        scene_ids: Optional[torch.Tensor] = None,
        anchor_abs: Optional[torch.Tensor] = None,
        is_flipped: Optional[torch.Tensor] = None,
    ):
        if anchor_abs is None:
            raise ValueError('dynamic_geo requires anchor_abs')

        batch_size = input_seq.shape[0]
        device = input_seq.device
        if is_flipped is None:
            is_flipped = torch.zeros(batch_size, dtype=torch.bool, device=device)
        else:
            is_flipped = is_flipped.to(device=device, dtype=torch.bool)
        anchor_abs = anchor_abs.to(device)
        scene_names = self._normalize_scene_names(scene_names, batch_size, scene_ids=scene_ids)
        states = self._initialize_states(input_seq, scene_names, anchor_abs, is_flipped)

        hidden, cell = self.encoder(input_seq)
        decoder_input = input_seq[:, -1, :].unsqueeze(1)
        current_pos = input_seq[:, -1, 0:2].unsqueeze(1)
        fixed_wh = input_seq[:, -1, 2:4].unsqueeze(1)
        current_geo_feat = input_seq[:, -1, 6:10].unsqueeze(1)
        anchor_pos = anchor_abs[:, 0:2].unsqueeze(1)

        pred_pos_list = []
        pred_vel_list = []
        target_len = int(target_seq.shape[1]) if target_seq is not None else int(pred_len)

        for step_idx in range(target_len):
            output, hidden, cell = self.decoder(decoder_input, hidden, cell)
            output = torch.clamp(output, -self.max_delta, self.max_delta)
            pred_vel_list.append(output)

            predicted_next_pos = current_pos + output
            pred_pos_list.append(predicted_next_pos)

            use_teacher_forcing = bool(target_seq is not None and random.random() < teacher_forcing_ratio)
            if use_teacher_forcing:
                next_pos_ref = target_seq[:, step_idx:step_idx + 1, 0:2]
                next_vel_feat = target_seq[:, step_idx:step_idx + 1, 4:6]
            else:
                next_pos_ref = predicted_next_pos
                next_vel_feat = (output / self.dt) * self.velocity_scale

            abs_pos_model = next_pos_ref + anchor_pos
            self._append_recent_abs_positions(abs_pos_model, states)

            if self.refresh_every > 0 and ((step_idx + 1) % self.refresh_every == 0):
                current_geo_feat = self._refresh_geo_features(abs_pos_model, next_vel_feat, states, current_geo_feat)

            decoder_input = torch.cat([next_pos_ref, fixed_wh, next_vel_feat, current_geo_feat], dim=2)
            current_pos = next_pos_ref

        pred_pos = torch.cat(pred_pos_list, dim=1)
        pred_vel = torch.cat(pred_vel_list, dim=1)
        return pred_pos, pred_vel
