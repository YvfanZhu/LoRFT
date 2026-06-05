from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from utils.track_segments import validate_scene_segments


def require_box_size(df: pd.DataFrame, coord_type: str, target_type: str) -> None:
    if df[["w", "h"]].isna().any().any():
        raise ValueError(
            f"Cannot convert from {coord_type} to {target_type} without width/height columns"
        )


def to_top_left(df: pd.DataFrame, coord_type: str) -> pd.DataFrame:
    out = df.copy()
    if coord_type in ("tlwh", "top_left"):
        out["tx"] = out["x"]
        out["ty"] = out["y"]
    elif coord_type == "bottom_center":
        require_box_size(out, coord_type, "top_left")
        out["tx"] = out["x"] - out["w"] / 2.0
        out["ty"] = out["y"] - out["h"]
    elif coord_type == "bbox_center":
        require_box_size(out, coord_type, "top_left")
        out["tx"] = out["x"] - out["w"] / 2.0
        out["ty"] = out["y"] - out["h"] / 2.0
    else:
        raise ValueError(f"Unsupported coordinate type: {coord_type}")
    return out


def add_eval_points(df: pd.DataFrame, src_coord_type: str, target_coord_type: str) -> pd.DataFrame:
    out = df.copy()
    if src_coord_type == target_coord_type:
        out["px"] = out["x"]
        out["py"] = out["y"]
        return out

    out = to_top_left(out, src_coord_type)
    if target_coord_type in ("tlwh", "top_left"):
        out["px"] = out["tx"]
        out["py"] = out["ty"]
    elif target_coord_type == "bottom_center":
        require_box_size(out, src_coord_type, target_coord_type)
        out["px"] = out["tx"] + out["w"] / 2.0
        out["py"] = out["ty"] + out["h"]
    elif target_coord_type == "bbox_center":
        require_box_size(out, src_coord_type, target_coord_type)
        out["px"] = out["tx"] + out["w"] / 2.0
        out["py"] = out["ty"] + out["h"] / 2.0
    else:
        raise ValueError(f"Unsupported target coordinate type: {target_coord_type}")
    return out


def _sort_for_eval(df: pd.DataFrame, order: str) -> pd.DataFrame:
    ascending = order != "1_to_0"
    return df.sort_values("frame", ascending=ascending).reset_index(drop=True)


def _select_target_rows(segment, gt_label_filter: int | None) -> pd.DataFrame:
    if gt_label_filter in (None, 1):
        return segment.target_rows.copy()
    if gt_label_filter == 0:
        return segment.obs_rows.copy()
    raise ValueError(f"Unsupported gt_label_filter: {gt_label_filter}")


def compute_scene_metrics(
    scene: str,
    pred_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    pred_coord_type: str,
    gt_coord_type: str,
    fps: float | None,
    rmse_seconds: Iterable[int],
    pred_len: int,
    support_backward: bool = False,
    gt_label_filter: int | None = None,
) -> dict:
    rmse_seconds = sorted({int(sec) for sec in rmse_seconds if int(sec) > 0})
    pred_len = int(pred_len)
    segments = validate_scene_segments(gt_df, scene=scene)
    rmse_accumulators = {
        f"rmse_{sec}s": {"sum_sq": 0.0, "count": 0}
        for sec in rmse_seconds
    }
    skipped = {
        "missing_target": 0,
        "short_target": 0,
        "missing_prediction": 0,
        "short_prediction": 0,
        "unsupported_backward_case": 0,
    }
    matched_parts = []
    track_metrics = []
    fde_values = []

    pred_groups = {
        int(track_id): group.copy()
        for track_id, group in pred_df.groupby("id", sort=True)
    }

    for track_id, segment in segments.items():
        if segment.order == "1_to_0" and not support_backward:
            skipped["unsupported_backward_case"] += 1
            continue
        target_rows = _select_target_rows(segment, gt_label_filter)
        if target_rows.empty:
            skipped["missing_target"] += 1
            continue
        if len(target_rows) < pred_len:
            skipped["short_target"] += 1
            continue

        pred_rows = pred_groups.get(int(track_id))
        if pred_rows is None or pred_rows.empty:
            skipped["missing_prediction"] += 1
            continue
        if len(pred_rows) < pred_len:
            skipped["short_prediction"] += 1
            continue

        target_rows = _sort_for_eval(target_rows, segment.order).iloc[:pred_len].copy()
        pred_rows = _sort_for_eval(pred_rows, segment.order).iloc[:pred_len].copy()

        pred_eval = add_eval_points(pred_rows, pred_coord_type, gt_coord_type)
        gt_eval = add_eval_points(target_rows, gt_coord_type, gt_coord_type)
        aligned = pd.DataFrame(
            {
                "scene": scene,
                "id": int(track_id),
                "step": np.arange(1, pred_len + 1, dtype=np.int32),
                "frame_pred": pred_eval["frame"].to_numpy(dtype=np.int32),
                "frame_gt": gt_eval["frame"].to_numpy(dtype=np.int32),
                "px_pred": pred_eval["px"].to_numpy(dtype=np.float64),
                "py_pred": pred_eval["py"].to_numpy(dtype=np.float64),
                "px_gt": gt_eval["px"].to_numpy(dtype=np.float64),
                "py_gt": gt_eval["py"].to_numpy(dtype=np.float64),
                "target_order": segment.order,
            }
        )
        aligned["dist"] = np.sqrt(
            (aligned["px_pred"] - aligned["px_gt"]) ** 2
            + (aligned["py_pred"] - aligned["py_gt"]) ** 2
        )
        aligned["dist_sq"] = aligned["dist"] ** 2
        matched_parts.append(aligned)

        ade = float(aligned["dist"].mean())
        fde = float(aligned.iloc[-1]["dist"])
        fde_values.append(fde)
        track_metric = {
            "scene": scene,
            "id": int(track_id),
            "target_order": segment.order,
            "start_frame": int(target_rows["frame"].min()),
            "end_frame": int(target_rows["frame"].max()),
            "points": int(len(aligned)),
            "ade": ade,
            "fde": fde,
        }
        for sec in rmse_seconds:
            key = f"rmse_{sec}s"
            if fps is None:
                track_metric[key] = None
                continue
            step = int(round(sec * fps))
            if step <= 0 or len(aligned) < step:
                track_metric[key] = None
                continue
            dist_sq = float(aligned.iloc[step - 1]["dist_sq"])
            track_metric[key] = float(np.sqrt(dist_sq))
            rmse_accumulators[key]["sum_sq"] += dist_sq
            rmse_accumulators[key]["count"] += 1
        track_metrics.append(track_metric)

    if matched_parts:
        matched = pd.concat(matched_parts, ignore_index=True)
    else:
        matched = pd.DataFrame(
            columns=[
                "scene",
                "id",
                "step",
                "frame_pred",
                "frame_gt",
                "px_pred",
                "py_pred",
                "px_gt",
                "py_gt",
                "target_order",
                "dist",
                "dist_sq",
            ]
        )

    rmse_metrics = {}
    for sec in rmse_seconds:
        key = f"rmse_{sec}s"
        count = rmse_accumulators[key]["count"]
        rmse_metrics[key] = float(np.sqrt(rmse_accumulators[key]["sum_sq"] / count)) if count > 0 else None

    return {
        "scene": scene,
        "pred_rows": int(len(pred_df)),
        "gt_rows": int(len(gt_df)),
        "matched_rows": int(len(matched)),
        "pred_ids": int(pred_df["id"].nunique()) if not pred_df.empty else 0,
        "matched_ids": int(len(track_metrics)),
        "ade": float(matched["dist"].mean()) if not matched.empty else None,
        "fde": float(np.mean(fde_values)) if fde_values else None,
        "track_metrics": track_metrics,
        "matched_points": matched,
        "rmse_metrics": rmse_metrics,
        "rmse_accumulators": rmse_accumulators,
        "skipped": skipped,
    }
