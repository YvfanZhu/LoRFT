"""Generate a README image plate from organized LoRFT trajectory data.

The figure visualizes trajectory labels only. It does not draw road maps,
predictions, or experimental results.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import pandas as pd


COLS = ["frame", "id", "x", "y", "w", "h", "c", "d", "e", "label"]
OBS_COLOR = "#1F77B4"
REF_COLOR = "#D55E00"
BOX_ALPHA = 0.78


def load_gt(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, names=COLS)
    df["bc_x"] = df["x"] + df["w"] / 2.0
    df["bc_y"] = df["y"] + df["h"]
    return df


def read_frame(video_path: Path, frame_idx: int):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(frame_idx - 1, 0))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_idx} from {video_path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def choose_units(summary: pd.DataFrame, dataset_root: Path, n: int = 4) -> list[tuple[str, str, Path, Path]]:
    preferred = ["G5013-K207", "G5013-K212", "G0512-K82", "G5-K2310", "G5-K1846", "SA2-K188"]
    pref_rank = {scene: i for i, scene in enumerate(preferred)}
    rows = summary.copy()
    rows["rank"] = rows["scene"].map(pref_rank).fillna(999).astype(int)
    rows = rows.sort_values(
        ["rank", "vehicle_trajectories", "vehicle_bounding_boxes"],
        ascending=[True, False, False],
    )
    chosen: list[tuple[str, str, Path, Path]] = []
    used_scenes: set[str] = set()
    for row in rows.itertuples(index=False):
        if row.scene in used_scenes:
            continue
        gt_path = dataset_root / "gt" / row.scene / row.unit / "gt" / "gt.txt"
        video_path = dataset_root / "video" / f"{row.unit}.mp4"
        if not gt_path.exists() or not video_path.exists():
            continue
        gt = load_gt(gt_path)
        if gt["label"].nunique() < 2:
            continue
        chosen.append((row.scene, row.unit, gt_path, video_path))
        used_scenes.add(row.scene)
        if len(chosen) >= n:
            break
    if len(chosen) < n:
        raise RuntimeError(f"Only found {len(chosen)} usable scene examples.")
    return chosen


def choose_frame(gt: pd.DataFrame) -> int:
    counts = gt.groupby("frame").size().sort_values(ascending=False)
    for frame in counts.index[:200]:
        labels = set(gt.loc[gt["frame"] == frame, "label"].astype(int))
        if 0 in labels or 1 in labels:
            return int(frame)
    return int(counts.index[0])


def choose_tracks(gt: pd.DataFrame, frame: int, max_tracks: int = 8) -> list[int]:
    present = set(gt.loc[gt["frame"] == frame, "id"].astype(int))
    scored = []
    for track_id, track in gt.groupby("id"):
        n_obs = int((track["label"] == 0).sum())
        n_ref = int((track["label"] == 1).sum())
        if n_obs == 0 and n_ref == 0:
            continue
        if int(track_id) not in present:
            continue
        score = len(track) + min(n_ref, 80)
        scored.append((score, int(track_id)))
    scored.sort(reverse=True)
    return [track_id for _, track_id in scored[:max_tracks]]


def draw_box(ax, row) -> None:
    color = OBS_COLOR if int(row.label) == 0 else REF_COLOR
    rect = plt.Rectangle(
        (row.x, row.y),
        row.w,
        row.h,
        fill=False,
        edgecolor=color,
        linewidth=1.0,
        alpha=BOX_ALPHA,
    )
    ax.add_patch(rect)


def draw_panel(ax, scene: str, unit: str, gt_path: Path, video_path: Path) -> None:
    gt = load_gt(gt_path)
    frame_idx = choose_frame(gt)
    frame = read_frame(video_path, frame_idx)
    ax.imshow(frame)

    tracks = choose_tracks(gt, frame_idx)
    for track_id in tracks:
        track = gt[gt["id"] == track_id].sort_values("frame")
        lo = frame_idx - 180
        hi = frame_idx + 180
        track = track[(track["frame"] >= lo) & (track["frame"] <= hi)]
        obs = track[track["label"] == 0]
        ref = track[track["label"] == 1]
        if len(obs) >= 2:
            ax.plot(obs["bc_x"], obs["bc_y"], color=OBS_COLOR, lw=1.0, alpha=0.65)
        if len(ref) >= 2:
            ax.plot(ref["bc_x"], ref["bc_y"], color=REF_COLOR, lw=1.0, alpha=0.78)

    current = gt[gt["frame"] == frame_idx].copy()
    current["area"] = current["w"] * current["h"]
    for row in current.sort_values("area", ascending=False).head(18).itertuples(index=False):
        draw_box(ax, row)

    ax.set_title(scene, loc="left", fontsize=10, fontweight="bold", pad=4)
    ax.set_xlim(0, frame.shape[1])
    ax.set_ylim(frame.shape[0], 0)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#D0D7DE")
        spine.set_linewidth(0.8)


def make_figure(dataset_root: Path, output: Path) -> None:
    summary = pd.read_csv(dataset_root / "dataset_counts_summary.csv")
    examples = choose_units(summary, dataset_root, n=4)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "sans-serif"],
            "font.size": 8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(9.6, 6.1),
        dpi=220,
        gridspec_kw={"wspace": 0.08, "hspace": 0.18},
    )
    for ax, example in zip(axes.flat, examples):
        draw_panel(ax, *example)

    legend_handles = [
        plt.Line2D([0], [0], color=OBS_COLOR, lw=2, label="Observed segment"),
        plt.Line2D([0], [0], color=REF_COLOR, lw=2, label="Distant reference"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, 0.0),
    )
    fig.tight_layout(rect=[0, 0.055, 1, 1], h_pad=0.8, w_pad=0.25)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    make_figure(args.dataset_root, args.output)


if __name__ == "__main__":
    main()
