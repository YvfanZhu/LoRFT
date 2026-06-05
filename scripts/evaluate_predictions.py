from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.evaluation import compute_scene_metrics
from utils.track_segments import load_tracking_rows
from utils.config import load_config_bundle, validate_time_config_contract, resolve_canonical_window_config


def parse_gt_label_filter(eval_cfg: dict) -> int | None:
    if 'gt_label_filter' not in eval_cfg or eval_cfg.get('gt_label_filter') is None:
        return None

    value = eval_cfg.get('gt_label_filter')
    if type(value) is int and value in (0, 1):
        return value

    raise ValueError(
        'evaluation.gt_label_filter must be integer 0, integer 1, or null; '
        f'got {value!r} (type={type(value).__name__})'
    )


def parse_rmse_seconds(eval_cfg: dict) -> list[int]:
    if 'rmse_seconds' not in eval_cfg or eval_cfg.get('rmse_seconds') is None:
        raise ValueError('evaluation.rmse_seconds is required and must be a non-empty list of positive integers.')

    raw_value = eval_cfg.get('rmse_seconds')
    if not isinstance(raw_value, (list, tuple)) or len(raw_value) == 0:
        raise ValueError(
            'evaluation.rmse_seconds must be a non-empty list of positive integers; '
            f'got {raw_value!r} (type={type(raw_value).__name__})'
        )

    parsed: list[int] = []
    for sec in raw_value:
        if type(sec) is not int or sec <= 0:
            raise ValueError(
                'evaluation.rmse_seconds must contain only positive integers; '
                f'got {sec!r} (type={type(sec).__name__})'
            )
        parsed.append(sec)

    return sorted(set(parsed))


def resolve_eval_scenes(config: dict) -> tuple[list[str], str]:
    eval_cfg = config['evaluation']
    if eval_cfg.get('scenes'):
        return list(eval_cfg['scenes']), 'config_list'
    if eval_cfg.get('scene'):
        return [str(eval_cfg['scene'])], 'single_scene'
    if bool(eval_cfg.get('use_scene_split', True)):
        split_name = eval_cfg.get('split_name', 'test_scenes')
        split_file = Path(config['paths']['scene_split_file'])
        if not split_file.exists():
            raise FileNotFoundError(
                f'Scene split file not found: {split_file}. Run python run_preprocess.py first, or set evaluation.use_scene_split=false and provide evaluation.scenes.'
            )
        split_info = json.loads(split_file.read_text(encoding='utf-8'))
        return list(split_info[split_name]), split_name
    raise ValueError('No evaluation scenes specified and scene split usage is disabled.')


def resolve_prediction_path(config: dict, scene: str) -> Path:
    eval_cfg = config['evaluation']
    prediction_path = eval_cfg.get('prediction_path')
    if prediction_path:
        return Path(str(prediction_path).format(scene=scene))
    prediction_dir = Path(eval_cfg['prediction_dir'])
    return prediction_dir / eval_cfg['prediction_file_pattern'].format(scene=scene)


def resolve_scene_gt_items(gt_root: Path, scene: str) -> List[Tuple[str, Path]]:
    direct_gt = gt_root / scene / 'gt' / 'gt.txt'
    if direct_gt.exists():
        return [(scene, direct_gt)]

    nested = sorted((gt_root / scene).glob('*/gt/gt.txt'))
    return [(f'{scene}/{path.parent.parent.name}', path) for path in nested]


def build_directional_motion_summary(track_metrics: List[dict], matched_parts: List[pd.DataFrame]) -> dict:
    motion_by_track: dict[tuple[str, int, str], dict] = {}
    if matched_parts:
        matched = pd.concat(matched_parts, ignore_index=True)
        if not matched.empty:
            for (scene, track_id, target_order), group in matched.groupby(['scene', 'id', 'target_order'], sort=False):
                group = group.sort_values('step')
                pred_disp = float(np.hypot(
                    group['px_pred'].iloc[-1] - group['px_pred'].iloc[0],
                    group['py_pred'].iloc[-1] - group['py_pred'].iloc[0],
                ))
                gt_disp = float(np.hypot(
                    group['px_gt'].iloc[-1] - group['px_gt'].iloc[0],
                    group['py_gt'].iloc[-1] - group['py_gt'].iloc[0],
                ))
                motion_by_track[(str(scene), int(track_id), str(target_order))] = {
                    'pred_disp': pred_disp,
                    'gt_disp': gt_disp,
                    'disp_ratio': float(pred_disp / max(gt_disp, 1e-6)),
                }

    payload = {
        'thresholds': {
            'low_motion_ratio': 0.5,
        },
        'directions': {},
    }
    for direction_name, target_order in (('upstream', '0_to_1'), ('downstream', '1_to_0')):
        direction_tracks = [item for item in track_metrics if item.get('target_order') == target_order]
        ade_values = [float(item['ade']) for item in direction_tracks]
        fde_values = [float(item['fde']) for item in direction_tracks]
        ratios = []
        pred_disps = []
        gt_disps = []
        low_motion_count = 0
        for item in direction_tracks:
            key = (str(item['scene']), int(item['id']), str(item.get('target_order')))
            motion = motion_by_track.get(key)
            if motion is None:
                continue
            ratio = float(motion['disp_ratio'])
            ratios.append(ratio)
            pred_disps.append(float(motion['pred_disp']))
            gt_disps.append(float(motion['gt_disp']))
            if ratio < 0.5:
                low_motion_count += 1

        payload['directions'][direction_name] = {
            'track_count': int(len(direction_tracks)),
            'ade': float(np.mean(ade_values)) if ade_values else None,
            'fde': float(np.mean(fde_values)) if fde_values else None,
            'pred_disp_mean': float(np.mean(pred_disps)) if pred_disps else None,
            'gt_disp_mean': float(np.mean(gt_disps)) if gt_disps else None,
            'pred_disp_ratio_mean': float(np.mean(ratios)) if ratios else None,
            'pred_disp_ratio_median': float(np.median(ratios)) if ratios else None,
            'low_motion_track_count': int(low_motion_count),
        }
    return payload


def main(config_dir=None) -> None:
    cfg_root = Path(config_dir) if config_dir is not None else (PROJECT_ROOT / 'configs')
    config = load_config_bundle(cfg_root, ['data', 'predict', 'eval'])
    validate_time_config_contract(config)
    eval_cfg = config['evaluation']
    window_cfg = resolve_canonical_window_config(config)
    scenes, scene_source = resolve_eval_scenes(config)
    gt_label_filter = parse_gt_label_filter(eval_cfg)
    fps = float(config['rts_smoother']['fps'])
    rmse_seconds = parse_rmse_seconds(eval_cfg)
    pred_len = window_cfg['pred_len']
    gt_root = Path(eval_cfg.get('gt_root') or config['inference']['input']['data_path'])

    output_dir = Path(eval_cfg['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    scene_rows = []
    track_rows = []
    matched_parts = []
    total_matched = 0
    total_ade_sum = 0.0
    total_fde_values = []
    total_rmse_accumulators = {f'rmse_{sec}s': {'sum_sq': 0.0, 'count': 0} for sec in rmse_seconds}
    total_skipped = defaultdict(int)
    missing_predictions = []

    expanded_scenes = []
    for scene in scenes:
        scene_gt_items = resolve_scene_gt_items(gt_root, scene)
        if not scene_gt_items:
            scene_gt_items = [(scene, gt_root / scene / 'gt' / 'gt.txt')]

        for output_scene, gt_path in scene_gt_items:
            expanded_scenes.append(output_scene)
            pred_path = resolve_prediction_path(config, output_scene)
            if not pred_path.exists():
                missing_predictions.append(str(pred_path))
                continue

            pred_df = load_tracking_rows(pred_path, keep_label=False)
            gt_df = load_tracking_rows(gt_path, keep_label=True)
            if gt_label_filter is not None and 'label' not in gt_df.columns:
                raise ValueError(
                    f'evaluation.gt_label_filter={gt_label_filter} but GT file has no explicit label column: {gt_path}'
                )
            metrics = compute_scene_metrics(
                output_scene,
                pred_df,
                gt_df,
                pred_coord_type=eval_cfg.get('prediction_coordinate_type', 'tlwh'),
                gt_coord_type=eval_cfg.get('gt_coordinate_type', 'tlwh'),
                fps=fps,
                rmse_seconds=rmse_seconds,
                pred_len=pred_len,
                support_backward=bool(eval_cfg.get('support_backward', False)),
                gt_label_filter=gt_label_filter,
            )

            matched_points = metrics.pop('matched_points')
            rmse_metrics = metrics.pop('rmse_metrics')
            rmse_accumulators = metrics.pop('rmse_accumulators')
            skipped = metrics.pop('skipped')

            scene_row = {
                'scene': output_scene,
                'map_scene': scene,
                'pred_path': str(pred_path),
                'gt_path': str(gt_path),
                'pred_rows': metrics['pred_rows'],
                'gt_rows': metrics['gt_rows'],
                'matched_rows': metrics['matched_rows'],
                'pred_ids': metrics['pred_ids'],
                'matched_ids': metrics['matched_ids'],
                'ade': metrics['ade'],
                'fde': metrics['fde'],
                'missing_target': skipped['missing_target'],
                'short_target': skipped['short_target'],
                'missing_prediction': skipped['missing_prediction'],
                'short_prediction': skipped['short_prediction'],
            }
            scene_row.update(rmse_metrics)
            scene_rows.append(scene_row)
            track_rows.extend(metrics['track_metrics'])
            if not matched_points.empty:
                matched_parts.append(matched_points.copy())

            if metrics['matched_rows'] > 0:
                total_matched += metrics['matched_rows']
                total_ade_sum += metrics['ade'] * metrics['matched_rows']
                total_fde_values.extend([row['fde'] for row in metrics['track_metrics']])
            for key, acc in rmse_accumulators.items():
                total_rmse_accumulators[key]['sum_sq'] += acc['sum_sq']
                total_rmse_accumulators[key]['count'] += acc['count']
            for key, value in skipped.items():
                total_skipped[key] += int(value)

    overall_rmse_metrics = {}
    for sec in rmse_seconds:
        key = f'rmse_{sec}s'
        count = total_rmse_accumulators[key]['count']
        overall_rmse_metrics[key] = float(np.sqrt(total_rmse_accumulators[key]['sum_sq'] / count)) if count > 0 else None

    directional_motion_summary = build_directional_motion_summary(track_rows, matched_parts)

    summary = {
        'scene_source': scene_source,
        'split_file': config['paths'].get('scene_split_file'),
        'prediction_coordinate_type': eval_cfg.get('prediction_coordinate_type', 'tlwh'),
        'gt_coordinate_type': eval_cfg.get('gt_coordinate_type', 'tlwh'),
        'gt_root': str(gt_root),
        'prediction_dir': eval_cfg.get('prediction_dir'),
        'prediction_file_pattern': eval_cfg.get('prediction_file_pattern'),
        'fps': fps,
        'rmse_seconds': rmse_seconds,
        'required_track_points': pred_len,
        'scenes': scenes,
        'expanded_scenes': expanded_scenes,
        'missing_prediction_files': missing_predictions,
        'overall_skipped': total_skipped,
        'overall': {
            'scene_count': len(scene_rows),
            'matched_rows': total_matched,
            'ade': (total_ade_sum / total_matched) if total_matched > 0 else None,
            'fde': (float(np.mean(total_fde_values)) if total_fde_values else None),
            **overall_rmse_metrics,
        },
        'directional_motion_summary': directional_motion_summary,
    }

    summary_path = output_dir / 'summary.json'
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')

    motion_summary_path = output_dir / 'downstream_motion_summary.json'
    motion_summary_path.write_text(
        json.dumps(directional_motion_summary, indent=2, ensure_ascii=False),
        encoding='utf-8',
    )

    scene_csv = output_dir / 'scene_metrics.csv'
    with scene_csv.open('w', newline='', encoding='utf-8') as f:
        fieldnames = [
            'scene', 'map_scene', 'pred_path', 'gt_path', 'pred_rows', 'gt_rows', 'matched_rows',
            'pred_ids', 'matched_ids', 'ade', 'fde',
            'missing_target', 'short_target', 'missing_prediction', 'short_prediction',
        ] + [f'rmse_{sec}s' for sec in rmse_seconds]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(scene_rows)

    track_csv = output_dir / 'track_metrics.csv'
    with track_csv.open('w', newline='', encoding='utf-8') as f:
        fieldnames = ['scene', 'id', 'target_order', 'start_frame', 'end_frame', 'points', 'ade', 'fde'] + [f'rmse_{sec}s' for sec in rmse_seconds]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(track_rows)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
