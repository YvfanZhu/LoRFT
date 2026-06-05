"""Generate a clean single-scene README teaser from LoRFT trajectory labels.

The output is intentionally simple: one fixed-camera frame with a small number
of trajectory snippets and vehicle boxes. It visualizes only trajectory labels,
not maps or model predictions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


COLS = ["frame", "id", "x", "y", "w", "h", "c", "d", "e", "label"]
OBS_COLOR = (185, 115, 28)  # BGR blue
REF_COLOR = (0, 116, 214)  # BGR orange


def load_gt(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, names=COLS)
    df["bc_x"] = df["x"] + df["w"] / 2.0
    df["bc_y"] = df["y"] + df["h"]
    return df


def read_frame(video_path: Path, frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(frame_idx - 1, 0))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_idx} from {video_path}")
    return frame


def select_unit(summary: pd.DataFrame, dataset_root: Path, preferred_scene: str) -> tuple[str, str, Path, Path]:
    rows = summary[summary["scene"] == preferred_scene].copy()
    if rows.empty:
        rows = summary.copy()
    rows = rows.sort_values(["vehicle_trajectories", "vehicle_bounding_boxes"], ascending=False)
    for row in rows.itertuples(index=False):
        gt_path = dataset_root / "gt" / row.scene / row.unit / "gt" / "gt.txt"
        video_path = dataset_root / "video" / f"{row.unit}.mp4"
        if gt_path.exists() and video_path.exists():
            return row.scene, row.unit, gt_path, video_path
    raise RuntimeError("No usable GT/video pair found.")


def select_frame(gt: pd.DataFrame) -> int:
    frame_stats = (
        gt.groupby("frame")
        .agg(n=("id", "count"), n_obs=("label", lambda x: int((x == 0).sum())), n_ref=("label", lambda x: int((x == 1).sum())))
        .reset_index()
    )
    frame_stats["score"] = frame_stats["n_obs"] * 3 + frame_stats["n_ref"] * 2 + frame_stats["n"]
    frame_stats = frame_stats[(frame_stats["n_obs"] >= 5) & (frame_stats["n_ref"] >= 2)]
    if frame_stats.empty:
        frame_stats = gt.groupby("frame").size().reset_index(name="score")
    return int(frame_stats.sort_values("score", ascending=False).iloc[0]["frame"])


def select_visible_tracks(gt: pd.DataFrame, frame_idx: int) -> list[int]:
    current = gt[gt["frame"] == frame_idx].copy()
    current["area"] = current["w"] * current["h"]
    obs = current[current["label"] == 0].sort_values("area", ascending=False)["id"].head(5).astype(int).tolist()
    ref = current[current["label"] == 1].sort_values("area", ascending=False)["id"].head(3).astype(int).tolist()
    track_ids = []
    for track_id in obs + ref:
        if track_id not in track_ids:
            track_ids.append(track_id)
    return track_ids


def draw_polyline(canvas: np.ndarray, pts: np.ndarray, color: tuple[int, int, int], thickness: int = 2) -> None:
    if len(pts) < 2:
        return
    pts_i = np.round(pts).astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(canvas, [pts_i], isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA)
    for x, y in pts_i.reshape((-1, 2))[:: max(1, len(pts_i) // 5)]:
        cv2.circle(canvas, (int(x), int(y)), 2, color, -1, lineType=cv2.LINE_AA)


def draw_box(canvas: np.ndarray, row, color: tuple[int, int, int]) -> None:
    x1, y1 = int(round(row.x)), int(round(row.y))
    x2, y2 = int(round(row.x + row.w)), int(round(row.y + row.h))
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2, lineType=cv2.LINE_AA)


def draw_teaser(gt: pd.DataFrame, frame: np.ndarray, frame_idx: int, output: Path) -> None:
    canvas = frame.copy()
    overlay = canvas.copy()
    track_ids = select_visible_tracks(gt, frame_idx)

    for track_id in track_ids:
        track = gt[gt["id"] == track_id].sort_values("frame")
        track = track[(track["frame"] >= frame_idx - 110) & (track["frame"] <= frame_idx + 80)]
        for label, color in [(0, OBS_COLOR), (1, REF_COLOR)]:
            part = track[track["label"] == label]
            pts = part[["bc_x", "bc_y"]].to_numpy(dtype=float)
            draw_polyline(overlay, pts, color, thickness=2)

    cv2.addWeighted(overlay, 0.82, canvas, 0.18, 0, canvas)

    current = gt[gt["frame"] == frame_idx].copy()
    current = current[current["id"].isin(track_ids)]
    current["area"] = current["w"] * current["h"]
    for row in current.sort_values("area", ascending=False).itertuples(index=False):
        color = OBS_COLOR if int(row.label) == 0 else REF_COLOR
        draw_box(canvas, row, color)

    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), canvas)


def make_figure(dataset_root: Path, output: Path, preferred_scene: str) -> None:
    summary = pd.read_csv(dataset_root / "dataset_counts_summary.csv")
    _, _, gt_path, video_path = select_unit(summary, dataset_root, preferred_scene)
    gt = load_gt(gt_path)
    frame_idx = select_frame(gt)
    frame = read_frame(video_path, frame_idx)
    draw_teaser(gt, frame, frame_idx, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scene", default="G5013-K207")
    args = parser.parse_args()
    make_figure(args.dataset_root, args.output, args.scene)


if __name__ == "__main__":
    main()
