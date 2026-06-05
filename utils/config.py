from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
from typing import Any, Iterable, Optional

import yaml


def load_yaml(path: Path) -> dict:
    with path.open('r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise TypeError(f'Config file must contain a mapping: {path}')
    return data


def deep_merge(base: dict, override: dict) -> dict:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _substitute_tokens(value: Any, *, project_root: Path, config_dir: Path) -> Any:
    if isinstance(value, dict):
        return {
            key: _substitute_tokens(item, project_root=project_root, config_dir=config_dir)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _substitute_tokens(item, project_root=project_root, config_dir=config_dir)
            for item in value
        ]
    if isinstance(value, str):
        return (
            value
            .replace('{project_root}', str(project_root))
            .replace('{config_dir}', str(config_dir))
        )
    return value


def validate_time_config_contract(config: dict, *, rel_tol: float = 1e-9, abs_tol: float = 1e-9) -> dict[str, float]:
    root_rts_cfg = config.get('rts_smoother', {})
    if 'fps' not in root_rts_cfg or root_rts_cfg.get('fps') is None:
        raise ValueError('rts_smoother.fps is required as the canonical experiment time configuration.')
    if 'dt' not in root_rts_cfg or root_rts_cfg.get('dt') is None:
        raise ValueError('rts_smoother.dt is required as the canonical experiment time configuration.')

    try:
        root_fps = float(root_rts_cfg['fps'])
        root_dt = float(root_rts_cfg['dt'])
    except (TypeError, ValueError) as exc:
        raise ValueError('rts_smoother.fps and rts_smoother.dt must be numeric values.') from exc

    if root_fps <= 0:
        raise ValueError(f'rts_smoother.fps must be positive, got {root_fps}.')
    if root_dt <= 0:
        raise ValueError(f'rts_smoother.dt must be positive, got {root_dt}.')

    expected_dt = 1.0 / root_fps
    if not math.isclose(root_dt, expected_dt, rel_tol=rel_tol, abs_tol=abs_tol):
        raise ValueError(
            'Canonical time config mismatch: rts_smoother.dt must equal 1 / rts_smoother.fps; '
            f'rts_smoother.fps={root_fps}, rts_smoother.dt={root_dt}, expected_dt={expected_dt}'
        )

    inference_rts_cfg = config.get('inference', {}).get('rts_smoother', {})
    if 'fps' in inference_rts_cfg and inference_rts_cfg.get('fps') is not None:
        try:
            inference_fps = float(inference_rts_cfg['fps'])
        except (TypeError, ValueError) as exc:
            raise ValueError('inference.rts_smoother.fps must be numeric when provided.') from exc
        if not math.isclose(inference_fps, root_fps, rel_tol=rel_tol, abs_tol=abs_tol):
            raise ValueError(
                'inference.rts_smoother.fps must match rts_smoother.fps; '
                f'inference.rts_smoother.fps={inference_fps}, rts_smoother.fps={root_fps}'
            )

    evaluation_cfg = config.get('evaluation', {})
    if 'fps' in evaluation_cfg and evaluation_cfg.get('fps') is not None:
        try:
            evaluation_fps = float(evaluation_cfg['fps'])
        except (TypeError, ValueError) as exc:
            raise ValueError('evaluation.fps must be numeric when provided.') from exc
        if not math.isclose(evaluation_fps, root_fps, rel_tol=rel_tol, abs_tol=abs_tol):
            raise ValueError(
                'evaluation.fps must match rts_smoother.fps; '
                f'evaluation.fps={evaluation_fps}, rts_smoother.fps={root_fps}'
            )

    model_cfg = config.get('model', {})
    if 'dt' in model_cfg and model_cfg.get('dt') is not None:
        try:
            model_dt = float(model_cfg['dt'])
        except (TypeError, ValueError) as exc:
            raise ValueError('model.dt must be numeric when provided.') from exc
        if not math.isclose(model_dt, root_dt, rel_tol=rel_tol, abs_tol=abs_tol):
            raise ValueError(
                'model.dt must match rts_smoother.dt; '
                f'model.dt={model_dt}, rts_smoother.dt={root_dt}'
            )

    config.setdefault('model', {})['dt'] = root_dt
    return {'fps': root_fps, 'dt': root_dt}


def resolve_canonical_window_config(config: dict) -> dict[str, int]:
    sliding_cfg = config.get('sliding_window', {})
    if 'obs_len' not in sliding_cfg or sliding_cfg.get('obs_len') is None:
        raise ValueError('sliding_window.obs_len is required as the canonical observation window source.')
    if 'pred_len' not in sliding_cfg or sliding_cfg.get('pred_len') is None:
        raise ValueError('sliding_window.pred_len is required as the canonical prediction window source.')

    try:
        obs_len = int(sliding_cfg['obs_len'])
        pred_len = int(sliding_cfg['pred_len'])
    except (TypeError, ValueError) as exc:
        raise ValueError('sliding_window.obs_len and sliding_window.pred_len must both be integers.') from exc

    sequence_cfg = config.get('sequence', {})
    if 'obs_len' in sequence_cfg and sequence_cfg.get('obs_len') is not None:
        try:
            sequence_obs_len = int(sequence_cfg['obs_len'])
        except (TypeError, ValueError) as exc:
            raise ValueError('sequence.obs_len must be an integer when provided.') from exc
        if sequence_obs_len != obs_len:
            raise ValueError(
                'sequence.obs_len must match sliding_window.obs_len when provided; '
                f'sequence.obs_len={sequence_obs_len}, sliding_window.obs_len={obs_len}'
            )

    if 'pred_len' in sequence_cfg and sequence_cfg.get('pred_len') is not None:
        try:
            sequence_pred_len = int(sequence_cfg['pred_len'])
        except (TypeError, ValueError) as exc:
            raise ValueError('sequence.pred_len must be an integer when provided.') from exc
        if sequence_pred_len != pred_len:
            raise ValueError(
                'sequence.pred_len must match sliding_window.pred_len when provided; '
                f'sequence.pred_len={sequence_pred_len}, sliding_window.pred_len={pred_len}'
            )

    evaluation_cfg = config.get('evaluation', {})
    if 'required_track_points' in evaluation_cfg and evaluation_cfg.get('required_track_points') is not None:
        try:
            required_track_points = int(evaluation_cfg['required_track_points'])
        except (TypeError, ValueError) as exc:
            raise ValueError('evaluation.required_track_points must be an integer when provided.') from exc
        if required_track_points != pred_len:
            raise ValueError(
                'evaluation.required_track_points must match sliding_window.pred_len when provided; '
                f'evaluation.required_track_points={required_track_points}, sliding_window.pred_len={pred_len}'
            )

    return {'obs_len': obs_len, 'pred_len': pred_len}


def load_config_bundle(
    config_dir: Path,
    stage_files: Optional[Iterable[str]] = None,
    *,
    local_filename: str = 'local.yaml',
) -> dict:
    config_dir = Path(config_dir)
    merged = load_yaml(config_dir / 'base.yaml')

    for name in list(stage_files or []):
        merged = deep_merge(merged, load_yaml(config_dir / f'{name}.yaml'))

    local_path = config_dir / local_filename
    if local_path.exists():
        merged = deep_merge(merged, load_yaml(local_path))

    return _substitute_tokens(
        merged,
        project_root=config_dir.parent,
        config_dir=config_dir,
    )
