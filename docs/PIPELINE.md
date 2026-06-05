# Map-RSTNet Pipeline

This document describes the public reproduction pipeline for Map-RSTNet, a map-aware residual Seq2Seq LSTM method for long-range vehicle trajectory reconstruction.

## 1. Repository Inputs

The public pipeline expects two data sources under the project root:

```text
data/
  gt/<scene>/gt/gt.txt
  gt/<scene>/<clip>/gt/gt.txt
  map_files/<scene>.json
```

The first GT layout is for a single clip per scene. The second layout is for multiple clips under the same road scene; all clips use the parent scene map file.

`gt.txt` files are comma-separated tracking files with 10 columns:

```text
frame,id,x,y,w,h,c,d,e,label
```

The label protocol is:

- `label=0`: observed trajectory rows exposed to public inference.
- `label=1`: target trajectory rows used by public evaluation.

Training preprocessing does not filter GT rows by label. It builds supervised sliding-window samples from the full labeled trajectory sequence. The label protocol is enforced in the public inference and evaluation stages.

Each map JSON must contain at least:

- `centerlines`: road centerline polylines.
- `boundaries`: left/right road boundary polylines.
- `zone_meta`: zone metadata, including direction information when available.

If `data/map_files` contains symbolic links, package the linked JSON targets when creating a release archive. Broken symlinks will prevent map loading in a fresh clone.

## 2. Configuration Files

Official configuration files live in `configs/`:

```text
base.yaml      Shared project paths and scene list
data.yaml      Sliding-window, split, feature, and data path settings
model.yaml     Network, loss, and dynamic geometry settings
train.yaml     Training, checkpoint, scheduler, and seed settings
predict.yaml   Inference input/output settings
eval.yaml      Evaluation metric and target-label settings
```

The loader merges configs in stage order, then merges `configs/local.yaml` last if it exists. `local.yaml` is intended only for machine-specific overrides such as local data paths, checkpoint paths, or reduced settings for quick validation. Do not include local overrides in a release unless they are intentional and documented.

The default reconstruction setting is:

- observation length: `60` frames.
- prediction length: `125` frames.
- frame rate: `25 FPS`.
- split: `14` training scenes, `4` validation scenes, and `4` test scenes.

## 3. Preprocessing

Command:

```bash
python run_preprocess.py
```

Config files used:

```text
base.yaml + data.yaml + model.yaml [+ local.yaml if present]
```

Main inputs:

- `data/gt/<scene>/gt/gt.txt` or `data/gt/<scene>/<clip>/gt/gt.txt`
- `data/map_files/<scene>.json`

Main outputs:

```text
outputs/experiments/map_rstnet/data/preprocessed/scene_split.json
outputs/experiments/map_rstnet/data/train_ready/train_data.pkl
outputs/experiments/map_rstnet/data/train_ready/val_data.pkl
outputs/experiments/map_rstnet/data/train_ready/test_data.pkl
```

The preprocessor:

- resolves the configured scene split;
- extracts map-aware normalized trajectory features;
- applies upstream/downstream direction handling;
- filters windows with too many invalid map matches;
- saves train, validation, and test pickle files for model training.

## 4. Training

Command:

```bash
python run_train.py
```

Config files used:

```text
base.yaml + data.yaml + model.yaml + train.yaml [+ local.yaml if present]
```

Main inputs:

```text
outputs/experiments/map_rstnet/data/train_ready/train_data.pkl
outputs/experiments/map_rstnet/data/train_ready/val_data.pkl
```

Main outputs:

```text
outputs/experiments/map_rstnet/checkpoints/best_model.pth
outputs/experiments/map_rstnet/checkpoints/latest.pth
outputs/experiments/map_rstnet/checkpoints/epoch_<n>.pth
outputs/experiments/map_rstnet/logs/
```

`best_model.pth` is selected by the configured validation monitor, currently `rmse_5s`. `latest.pth` is used for resume training when enabled.

## 5. Inference

Command:

```bash
python run_predict.py
```

Config files used:

```text
base.yaml + data.yaml + model.yaml + train.yaml + predict.yaml [+ local.yaml if present]
```

Main inputs:

- `outputs/experiments/map_rstnet/checkpoints/best_model.pth`
- `data/gt/<scene>/gt/gt.txt` or `data/gt/<scene>/<clip>/gt/gt.txt`
- `data/map_files/<scene>.json`
- `outputs/experiments/map_rstnet/data/preprocessed/scene_split.json`

By default, inference uses the configured test split and filters the input GT to `label=0` rows only.

Main outputs:

```text
outputs/experiments/map_rstnet/predictions/prediction_only/<scene>_prediction_only.txt
outputs/experiments/map_rstnet/predictions/prediction_only/<scene>/<clip>_prediction_only.txt
outputs/experiments/map_rstnet/predictions/obs_and_prediction/<scene>_obs_and_prediction.txt
outputs/experiments/map_rstnet/predictions/obs_and_prediction/<scene>/<clip>_obs_and_prediction.txt
outputs/experiments/map_rstnet/predictions/infer_report.json
```

Prediction files use the standard tracking-row layout:

```text
frame,id,x,y,w,h,c,d,e,f
```

## 6. Evaluation

Command:

```bash
python run_evaluate.py
```

Config files used:

```text
base.yaml + data.yaml + predict.yaml + eval.yaml [+ local.yaml if present]
```

Main inputs:

- `outputs/experiments/map_rstnet/predictions/prediction_only/<scene>_prediction_only.txt`
- `outputs/experiments/map_rstnet/predictions/prediction_only/<scene>/<clip>_prediction_only.txt`
- `data/gt/<scene>/gt/gt.txt` or `data/gt/<scene>/<clip>/gt/gt.txt`
- `outputs/experiments/map_rstnet/data/preprocessed/scene_split.json`

By default, evaluation uses `label=1` GT rows as targets and supports both `0_to_1` and `1_to_0` target order.

Main outputs:

```text
outputs/experiments/map_rstnet/eval/summary.json
outputs/experiments/map_rstnet/eval/scene_metrics.csv
outputs/experiments/map_rstnet/eval/track_metrics.csv
outputs/experiments/map_rstnet/eval/downstream_motion_summary.json
```

Reported metrics include ADE, FDE, and RMSE at configured horizons from 1 to 5 seconds.

## 7. End-to-End Reproduction

From a clean repository with data and map files available:

```bash
pip install -r requirements.txt
python run_preprocess.py
python run_train.py
python run_predict.py
python run_evaluate.py
```

For a public artifact, verify this sequence in a fresh clone or fresh environment before release.

## 8. Release Checklist

Before publishing code and data:

- Ensure `configs/local.yaml` is absent unless intentionally documented.
- Ensure `data/map_files/*.json` are real readable JSON files in the release, not broken symlinks.
- Exclude generated `outputs/`, `__pycache__/`, checkpoints, pickle files, and logs unless a trained checkpoint is intentionally released.
- Keep README, this pipeline document, and default configs consistent.
- Record the exact train/validation/test scene split.
- State clearly that training uses full labeled trajectories while public inference uses `label=0` and public evaluation uses `label=1`.
- Run the four public commands from a clean environment and save the resulting `summary.json` for comparison.
