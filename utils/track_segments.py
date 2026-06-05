from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

TRACK_COLUMNS = ["frame", "id", "x", "y", "w", "h"]


@dataclass
class TrackSegments:
    track_id: int
    order: str
    obs_rows: pd.DataFrame
    target_rows: pd.DataFrame
    full_rows: pd.DataFrame


def load_tracking_rows(path, keep_label: bool = False) -> pd.DataFrame:
    path = Path(path) if not hasattr(path, "exists") else path
    if not path.exists():
        columns = list(TRACK_COLUMNS)
        if keep_label:
            columns.append("label")
        return pd.DataFrame(columns=columns)

    df = pd.read_csv(path, header=None)
    if df.shape[1] < 4:
        raise ValueError(f"Unsupported tracking file format: {path}, columns={df.shape[1]}")

    if keep_label and df.shape[1] >= 7:
        out = df.iloc[:, :6].copy() if df.shape[1] >= 6 else df.iloc[:, :4].copy()
        if out.shape[1] == 4:
            out.columns = ["frame", "id", "x", "y"]
            out["w"] = float("nan")
            out["h"] = float("nan")
            out = out[TRACK_COLUMNS]
        else:
            out.columns = TRACK_COLUMNS
        out["label"] = df.iloc[:, -1].astype(int)
    elif df.shape[1] >= 6:
        out = df.iloc[:, :6].copy()
        out.columns = TRACK_COLUMNS
    else:
        out = df.iloc[:, :4].copy()
        out.columns = ["frame", "id", "x", "y"]
        out["w"] = float("nan")
        out["h"] = float("nan")
        out = out[TRACK_COLUMNS]

    out["frame"] = out["frame"].astype(int)
    out["id"] = out["id"].astype(int)
    for col in ["x", "y", "w", "h"]:
        out[col] = out[col].astype(float)
    if keep_label and "label" in out.columns:
        out["label"] = out["label"].astype(int)
    return out.sort_values(["id", "frame"]).reset_index(drop=True)


def _label_runs(labels) -> List[int]:
    runs: List[int] = []
    prev: Optional[int] = None
    for raw in labels:
        value = int(raw)
        if value not in (0, 1):
            raise ValueError(f"Only label 0/1 is supported, got {value}")
        if value != prev:
            runs.append(value)
            prev = value
    return runs


def extract_track_segments(track_df: pd.DataFrame, scene: str = "") -> TrackSegments:
    if "label" not in track_df.columns:
        raise ValueError("Track label extraction requires a label column.")
    track_df = track_df.sort_values("frame").reset_index(drop=True)
    track_id = int(track_df.iloc[0]["id"]) if not track_df.empty else -1
    runs = _label_runs(track_df["label"].tolist())
    if len(runs) > 2:
        raise ValueError(f"Non-contiguous label segments detected for scene={scene} id={track_id}: {runs}")

    obs_rows = track_df[track_df["label"].astype(int) == 0].copy().reset_index(drop=True)
    target_rows = track_df[track_df["label"].astype(int) == 1].copy().reset_index(drop=True)

    if runs == [0]:
        order = "0_only"
    elif runs == [1]:
        order = "1_only"
    elif runs == [0, 1]:
        order = "0_to_1"
    elif runs == [1, 0]:
        order = "1_to_0"
    else:
        raise ValueError(f"Unsupported label order for scene={scene} id={track_id}: {runs}")

    return TrackSegments(
        track_id=track_id,
        order=order,
        obs_rows=obs_rows,
        target_rows=target_rows,
        full_rows=track_df.copy(),
    )


def validate_scene_segments(df: pd.DataFrame, scene: str = "") -> Dict[int, TrackSegments]:
    if "label" not in df.columns:
        raise ValueError(f"Scene {scene} is missing label column.")
    segments: Dict[int, TrackSegments] = {}
    for track_id, group in df.groupby("id", sort=True):
        segments[int(track_id)] = extract_track_segments(group.copy(), scene=scene)
    return segments
