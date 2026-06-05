import json
import logging
import os
import pickle
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import load_config_bundle
from utils.geometry import compute_lane_flow_dtheta, estimate_headings, heading_angle_difference, wrap_to_pi
from utils.map_matching import MapHelper, RTSSmoother
from utils.trajectory_features import FeatureExtractor

try:
    from pykalman import KalmanFilter
    from shapely.geometry import Point, LineString as SLineString
except ImportError:
    raise ImportError("Missing required packages. Please run: pip install pykalman shapely")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataPreprocessor:
    def __init__(self, config_source):
        if isinstance(config_source, dict):
            self.cfg = config_source
        else:
            with open(config_source, encoding='utf-8') as f:
                self.cfg = yaml.safe_load(f)

        try:
            map_dir = self.cfg['paths']['map_root']
        except KeyError as e:
            raise KeyError(f"Configuration is missing map path: {e}")
        if not os.path.exists(map_dir):
            raise FileNotFoundError(f"Map path does not exist: {map_dir}")

        self.map_helper = MapHelper(map_dir, logger=logger)
        self.extractor = FeatureExtractor(self.cfg, self.map_helper)
        direction_cfg = self.cfg.get('direction_consistency', {})
        self.max_mean_vy_after_flip = float(direction_cfg.get('max_mean_vy_after_flip', 0.05))

    def _recompute_model_time_velocity(self, feats: np.ndarray, previous_pos: Optional[np.ndarray] = None) -> None:
        if len(feats) == 0:
            return

        if previous_pos is not None:
            feats[0, 4:6] = (
                (feats[0, 0:2] - previous_pos.astype(np.float32))
                / self.extractor.dt
                * self.extractor.vel_scale
            )
        elif len(feats) > 1:
            feats[0, 4:6] = (
                (feats[1, 0:2] - feats[0, 0:2])
                / self.extractor.dt
                * self.extractor.vel_scale
            )
        else:
            feats[0, 4:6] = 0.0

        if len(feats) > 1:
            feats[1:, 4:6] = (
                (feats[1:, 0:2] - feats[:-1, 0:2])
                / self.extractor.dt
                * self.extractor.vel_scale
            )

    def _extract_model_time_chunk(self,
                                  scene: str,
                                  raw_model_chunk: np.ndarray,
                                  do_flip: bool,
                                  is_downstream: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        raw_for_extractor = raw_model_chunk[::-1].copy() if do_flip else raw_model_chunk
        feats, abs_info, validity_mask = self.extractor.process_trajectory(
            scene,
            raw_for_extractor,
            do_flip,
            is_downstream=is_downstream,
        )
        if len(feats) == 0:
            return feats, abs_info, validity_mask

        if do_flip:
            feats = feats[::-1].copy()
            abs_info = abs_info[::-1].copy()
            validity_mask = validity_mask[::-1].copy()
            self._recompute_model_time_velocity(feats)

        return feats, abs_info, validity_mask

    def _resolve_scene_split(self, scenes: List[str]) -> Tuple[Dict[str, List[str]], List[str], str]:
        split_cfg = self.cfg['data_split']
        ordered_scenes = sorted(scenes) if split_cfg.get('sort_scenes_before_shuffle', False) else list(scenes)
        seeded_scene_order = ordered_scenes.copy()
        rng = random.Random(split_cfg['random_seed'])
        rng.shuffle(seeded_scene_order)

        explicit_split = {
            'train': list(split_cfg.get('train_scenes') or []),
            'val': list(split_cfg.get('val_scenes') or []),
            'test': list(split_cfg.get('test_scenes') or []),
        }

        if any(explicit_split.values()):
            if not all(explicit_split.values()):
                raise ValueError('When explicit split lists are configured, train/val/test scene lists must all be provided.')
            split_map = explicit_split
            split_source = 'config_explicit'
            shuffled_scenes = seeded_scene_order
        else:
            shuffled_scenes = seeded_scene_order.copy()
            train_count = split_cfg['train_count']
            val_count = split_cfg['val_count']
            test_count = split_cfg['test_count']
            split_map = {
                'train': shuffled_scenes[:train_count],
                'val': shuffled_scenes[train_count:train_count + val_count],
                'test': shuffled_scenes[train_count + val_count:train_count + val_count + test_count],
            }
            split_source = 'generated_from_seed'

        self._validate_scene_split(split_map, scenes)
        return split_map, shuffled_scenes, split_source

    def _validate_scene_split(self, split_map: Dict[str, List[str]], scenes: List[str]) -> None:
        split_cfg = self.cfg['data_split']
        expected_counts = {
            'train': split_cfg['train_count'],
            'val': split_cfg['val_count'],
            'test': split_cfg['test_count'],
        }
        actual_counts = {name: len(items) for name, items in split_map.items()}
        if actual_counts != expected_counts:
            raise ValueError(f'Scene split counts do not match: expected={expected_counts}, actual={actual_counts}')

        flat_scenes = split_map['train'] + split_map['val'] + split_map['test']
        if len(flat_scenes) != len(set(flat_scenes)):
            raise ValueError('Duplicate scenes found in scene split configuration.')

        missing = sorted(set(scenes) - set(flat_scenes))
        extra = sorted(set(flat_scenes) - set(scenes))
        if missing or extra:
            raise ValueError(f'Scene split does not match the configured scene set: missing={missing}, extra={extra}')

    def _save_scene_split(self, split_map: Dict[str, List[str]], shuffled_scenes: List[str], split_source: str) -> None:
        split_cfg = self.cfg['data_split']
        split_path = Path(self.cfg['paths']['scene_split_file'])
        split_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            'source': split_source,
            'random_seed': split_cfg['random_seed'],
            'sort_scenes_before_shuffle': split_cfg.get('sort_scenes_before_shuffle', False),
            'train_count': split_cfg['train_count'],
            'val_count': split_cfg['val_count'],
            'test_count': split_cfg['test_count'],
            'shuffled_scene_order': shuffled_scenes,
            'train_scenes': split_map['train'],
            'val_scenes': split_map['val'],
            'test_scenes': split_map['test'],
        }

        with split_path.open('w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        print(f'Saved scene split: {split_path}')

    def _discover_gt_files(self, gt_root: str, scene: str) -> List[Tuple[str, Path]]:
        scene_root = Path(gt_root) / scene
        direct_gt = scene_root / 'gt' / 'gt.txt'
        if direct_gt.exists():
            return [(scene, direct_gt)]

        nested = sorted(scene_root.glob('*/gt/gt.txt'))
        return [(f'{scene}/{path.parent.parent.name}', path) for path in nested]

    def run(self):
        print('=' * 60)
        print('Training data preprocessing (dynamic road-width normalization)')
        print('=' * 60)

        scenes = list(self.cfg['scenes'])
        gt_root = self.cfg['paths']['gt_root']

        split_map, shuffled_scenes, split_source = self._resolve_scene_split(scenes)
        scene_to_split = {scene: split_name for split_name, scene_list in split_map.items() for scene in scene_list}
        all_samples = {'train': [], 'val': [], 'test': []}

        self._save_scene_split(split_map, shuffled_scenes, split_source)

        print(f'Scene split source: {split_source}')
        for split_name in ('train', 'val', 'test'):
            print(f"{split_name.title()} scenes ({len(split_map[split_name])}): {sorted(split_map[split_name])}")

        stats = defaultdict(int)
        stats['max_mean_vy_after_flip'] = self.max_mean_vy_after_flip
        split_stats = {
            'train': defaultdict(int),
            'val': defaultdict(int),
            'test': defaultdict(int),
        }
        dtheta_values = {
            'overall': [],
            'upstream': [],
            'downstream': [],
        }

        for scene in scenes:
            dtype = scene_to_split[scene]
            scene_gt_files = self._discover_gt_files(gt_root, scene)
            if not scene_gt_files:
                print(f"  [SKIP] {scene}: no gt.txt found")
                continue

            scene_total_count = 0
            print(f"Processing scene: {scene} ({len(scene_gt_files)} clip(s))")

            for clip_scene, gt_path in scene_gt_files:
                try:
                    df = pd.read_csv(gt_path, header=None)
                    if df.shape[1] >= 6:
                        df = df.iloc[:, :6]
                        df.columns = ['frame', 'id', 'x', 'y', 'w', 'h']
                    else:
                        print(f"  [SKIP] {clip_scene}: columns < 6")
                        continue
                except Exception as e:
                    print(f"  [SKIP] {clip_scene}: {e}")
                    continue

                clip_count = 0
                for tid, track in df.groupby('id'):
                    if len(track) < self.cfg['sliding_window']['min_track_len']:
                        continue

                    track = track.sort_values('frame')
                    raw = track[['frame', 'x', 'y', 'w', 'h']].values

                    y0_bc = raw[0, 2] + raw[0, 4]
                    y1_bc = raw[-1, 2] + raw[-1, 4]
                    is_downstream = (y1_bc - y0_bc) > 0

                    do_flip = False
                    if is_downstream and self.cfg['time_reversal']['enabled']:
                        do_flip = True
                        stats['flipped'] += 1

                    obs_len = self.cfg['sliding_window']['obs_len']
                    pred_len = self.cfg['sliding_window']['pred_len']
                    seq_len = obs_len + pred_len
                    step = self.cfg['sliding_window'].get(f'{dtype}_step', self.cfg['sliding_window']['step'])

                    max_invalid_ratio = self.cfg['sliding_window']['max_invalid_ratio']
                    win_total = 0
                    win_skipped = 0
                    track_has_sample = False
                    model_raw = raw[::-1].copy() if do_flip else raw

                    for i in range(0, len(model_raw) - seq_len + 1, step):
                        obs_raw_model = model_raw[i: i + obs_len]
                        pred_raw_model = model_raw[i + obs_len: i + seq_len]
                        obs_feats, obs_abs, obs_mask = self._extract_model_time_chunk(
                            scene, obs_raw_model, do_flip, is_downstream=is_downstream)
                        pred_feats, pred_abs, pred_mask = self._extract_model_time_chunk(
                            scene, pred_raw_model, do_flip, is_downstream=is_downstream)
                        win_total += 1
                        if len(obs_feats) == 0 or len(pred_feats) == 0:
                            win_skipped += 1
                            logger.debug(
                                f"Window skipped [{clip_scene}|tid={tid}|i={i}]: "
                                "feature extraction failed for obs or pred chunk"
                            )
                            continue

                        self._recompute_model_time_velocity(pred_feats, previous_pos=obs_feats[-1, 0:2])

                        win_mask = np.concatenate([obs_mask, pred_mask])
                        split_stats[dtype]['frame_count'] += int(len(win_mask))
                        split_stats[dtype]['invalid_frame_count'] += int((~win_mask).sum())
                        invalid_ratio = 1.0 - win_mask.mean()
                        if invalid_ratio > max_invalid_ratio:
                            win_skipped += 1
                            logger.debug(
                                f"Window skipped [{clip_scene}|tid={tid}|i={i}]: "
                                f"invalid_ratio={invalid_ratio*100:.1f}% > {max_invalid_ratio*100:.0f}%"
                            )
                            continue

                        win_feats = np.vstack([obs_feats, pred_feats])
                        if do_flip:
                            win_vy = win_feats[:, 5]
                            if win_vy.mean() > self.max_mean_vy_after_flip:
                                win_skipped += 1
                                stats['direction_consistency_skipped'] += 1
                                split_stats[dtype]['direction_consistency_skipped'] += 1
                                logger.debug(
                                    f"Window skipped [{clip_scene}|tid={tid}|i={i}]: "
                                    f"mean(v_y) after flipping={win_vy.mean():.3f} > {self.max_mean_vy_after_flip:.3f}; direction consistency failed"
                                )
                                continue

                        finite_dtheta = win_feats[np.isfinite(win_feats[:, 7]), 7]
                        if finite_dtheta.size > 0:
                            dtheta_values['overall'].extend(finite_dtheta.astype(np.float64).tolist())
                            dtheta_values['downstream' if do_flip else 'upstream'].extend(
                                finite_dtheta.astype(np.float64).tolist()
                            )

                        sample = {
                            'obs_seq': obs_feats,
                            'pred_seq': pred_feats,
                            'start_pos': obs_abs[0],
                            'anchor_abs': obs_abs[-1],
                            'scene': scene,
                            'clip_scene': clip_scene,
                            'track_id': int(tid),
                            'is_flipped': do_flip,
                            'direction': -1 if do_flip else 1,
                        }
                        all_samples[dtype].append(sample)
                        clip_count += 1
                        scene_total_count += 1
                        stats['total'] += 1
                        split_stats[dtype]['samples'] += 1
                        track_has_sample = True

                    if track_has_sample:
                        split_stats[dtype]['track_count'] += 1

                    if win_skipped > 0:
                        logger.info(
                            f"  Window filtering [{clip_scene}|tid={tid}]: "
                            f"skipped {win_skipped}/{win_total} windows "
                            f"(map invalid ratio>{max_invalid_ratio*100:.0f}%), "
                            f"kept {win_total - win_skipped}"
                        )
                    stats['win_skipped'] += win_skipped
                    stats['win_total'] += win_total
                    split_stats[dtype]['win_total'] += win_total
                    split_stats[dtype]['win_skipped'] += win_skipped

                print(f"  [{dtype.upper()}] {clip_scene}: {clip_count} samples")

            print(f"  [{dtype.upper()}] {scene}: {scene_total_count} samples total")

        for dtype, samples in all_samples.items():
            path = self.cfg['paths'][f'{dtype}_data']
            Path(path).parent.mkdir(parents=True, exist_ok=True)

            data_wrapper = {
                'samples': samples,
                'feature_dim': 10,
                'normalization': 'dynamic_road_width',
                'feature_names': ['u', 'y_norm', 'w_norm', 'h_norm',
                                  'v_u', 'v_y', 'f_lat', 'd_theta',
                                  'd_left', 'd_right'],
                'abs_info_names': ['u', 'y_norm', 'W', 'x_left'],
                'config': self.cfg,
                'split': dtype
            }

            with open(path, 'wb') as f:
                pickle.dump(data_wrapper, f)
            print(f"Saved {dtype}: {len(samples)} samples -> {path}")

        win_total = stats.get('win_total', 0)
        win_skipped = stats.get('win_skipped', 0)
        win_kept = win_total - win_skipped
        keep_rate = (win_kept / win_total * 100) if win_total > 0 else 0.0
        print(f"\nDone. Total samples: {stats['total']}, flipped: {stats.get('flipped', 0)}")
        print(f"Window filtering summary: total={win_total}, "
              f"skipped={win_skipped} ({100-keep_rate:.1f}%), "
              f"kept={win_kept} ({keep_rate:.1f}%)")
        self._save_stats(split_map, split_source, stats, split_stats)
        self._save_dtheta_stats(dtheta_values)

    def _save_stats(self,
                    split_map: Dict[str, List[str]],
                    split_source: str,
                    stats: Dict[str, int],
                    split_stats: Dict[str, defaultdict]) -> None:
        if not bool(self.cfg.get('logging', {}).get('save_stats', False)):
            return

        stats_path_value = self.cfg.get('paths', {}).get('stats_report')
        if not stats_path_value:
            return

        stats_path = Path(stats_path_value)
        stats_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            'split_source': split_source,
            'scene_split_file': self.cfg['paths'].get('scene_split_file'),
            'max_mean_vy_after_flip': self.max_mean_vy_after_flip,
            'overall': {
                'samples': int(stats.get('total', 0)),
                'flipped_tracks': int(stats.get('flipped', 0)),
                'window_total': int(stats.get('win_total', 0)),
                'window_skipped': int(stats.get('win_skipped', 0)),
                'direction_consistency_skipped': int(stats.get('direction_consistency_skipped', 0)),
            },
            'splits': {},
        }

        for split_name, split_scene_names in split_map.items():
            curr = split_stats[split_name]
            frame_count = int(curr.get('frame_count', 0))
            invalid_frame_count = int(curr.get('invalid_frame_count', 0))
            payload['splits'][split_name] = {
                'scenes': list(split_scene_names),
                'track_count': int(curr.get('track_count', 0)),
                'samples': int(curr.get('samples', 0)),
                'window_total': int(curr.get('win_total', 0)),
                'window_skipped': int(curr.get('win_skipped', 0)),
                'direction_consistency_skipped': int(curr.get('direction_consistency_skipped', 0)),
                'frame_count': frame_count,
                'invalid_frame_count': invalid_frame_count,
                'map_fallback_ratio': (invalid_frame_count / frame_count) if frame_count > 0 else 0.0,
            }

        stats_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
        print(f'Saved data quality statistics: {stats_path}')

    def _save_dtheta_stats(self, dtheta_values: Dict[str, List[float]]) -> None:
        stats_path_value = self.cfg.get('paths', {}).get('dtheta_stats_report')
        if not stats_path_value:
            return

        stats_path = Path(stats_path_value)
        stats_path.parent.mkdir(parents=True, exist_ok=True)

        def _summary(values: List[float]) -> Dict[str, Union[float, int, None]]:
            arr = np.asarray(values, dtype=np.float64)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                return {
                    'count': 0,
                    'mean': None,
                    'median': None,
                    'p90': None,
                    'abs_gt_2p6_ratio': None,
                }
            return {
                'count': int(arr.size),
                'mean': float(arr.mean()),
                'median': float(np.median(arr)),
                'p90': float(np.percentile(arr, 90)),
                'abs_gt_2p6_ratio': float(np.mean(np.abs(arr) > 2.6)),
            }

        payload = {
            'overall': _summary(dtheta_values.get('overall', [])),
            'upstream': _summary(dtheta_values.get('upstream', [])),
            'downstream': _summary(dtheta_values.get('downstream', [])),
        }
        stats_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
        print(f'Saved d_theta statistics: {stats_path}')

def main(config_dir=None):
    cfg_root = Path(config_dir) if config_dir is not None else (PROJECT_ROOT / 'configs')
    config = load_config_bundle(cfg_root, ['data', 'model'])
    DataPreprocessor(config).run()


if __name__ == '__main__':
    main()
