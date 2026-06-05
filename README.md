<div align="center">

# LoRFT / Map-RSTNet

**Long-Range Vehicle Trajectory Reconstruction from Fixed Highway Cameras**

<p>
  <a href="#quick-start">Quick Start</a> |
  <a href="#dataset">Dataset</a> |
  <a href="docs/PIPELINE.md">Pipeline</a> |
  <a href="#citation">Citation</a>
</p>

<p>
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c" alt="PyTorch">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
  <a href="https://huggingface.co/datasets/YvfanZhu/LoRFT">
    <img src="https://img.shields.io/badge/Dataset-LoRFT-yellow" alt="Dataset">
  </a>
</p>

</div>

LoRFT is an open benchmark for long-range vehicle trajectory reconstruction from fixed highway cameras. The task starts from a reliable near-field vehicle tracklet and reconstructs the distant continuation of the same vehicle in the original fixed-camera image plane.

This repository provides the LoRFT benchmark resources, evaluation pipeline, and Map-RSTNet reference implementation. Map-RSTNet is a map-aware residual Seq2Seq model that reconstructs distant vehicle trajectories in a road-aligned state space and projects the results back to the image plane.

The benchmark targets a common limitation of highway surveillance videos: fixed cameras provide continuous monitoring, but automatic tracking often becomes fragmented or terminates early in distant regions because of perspective compression, scale decay, occlusion, and unstable association.

## News

- Initial release of the LoRFT benchmark codebase and Map-RSTNet reference implementation.
- The pipeline supports scene-level splits and multi-clip ground-truth folders.
- Original video data are hosted separately on Hugging Face: [YvfanZhu/LoRFT](https://huggingface.co/datasets/YvfanZhu/LoRFT).

## Highlights

- **Long-range reconstruction task.** LoRFT evaluates whether the distant continuation of a known vehicle can be reconstructed from a reliable near-field tracklet.
- **Fixed-camera highway setting.** The benchmark preserves image-space degradation from routine surveillance cameras, including perspective compression, scale decay, occlusion, and track fragmentation.
- **Manual trajectory verification.** The released annotations include observed and distant reference segments for the same vehicle, with vehicle bounding boxes and scene-level road geometry.
- **Map-aware baseline.** Map-RSTNet uses road centerlines, road boundaries, and zone metadata to reconstruct trajectories in a road-aligned state space.
- **Scene-level protocol.** Predefined train/validation/test splits separate road scenes to reduce leakage from camera viewpoint, road layout, and background appearance.

## Benchmark Overview

LoRFT contains:

- 22 expressway surveillance scenes.
- 366,109 video frames.
- 6,601 manually verified vehicle trajectories.
- 2,694,889 vehicle bounding boxes.
- Scene-level road-geometry annotations.
- Predefined scene-level splits and evaluation scripts.

Under the default benchmark configuration, Map-RSTNet achieves `ADE=12.32`, `FDE=21.71`, and `RMSE-5s=27.47` pixels.

## Repository Structure

```text
Map-RSTNet/
|-- configs/              # YAML configuration files
|-- data/
|   |-- gt/               # Labeled trajectory GT files
|   |-- map_files/        # Preprocessed map JSON files
|   `-- README.md         # Data format notes
|-- docs/
|   `-- PIPELINE.md       # Detailed pipeline description
|-- examples/             # Minimal format examples
|-- models/               # Dataset and model definitions
|-- scripts/              # Preprocessing, training, inference, evaluation modules
|-- utils/                # Configuration, geometry, map matching, and metrics
|-- run_preprocess.py     # Build training/evaluation samples
|-- run_train.py          # Train Map-RSTNet
|-- run_predict.py        # Run public inference
|-- run_evaluate.py       # Evaluate predictions
`-- requirements.txt
```

Generated artifacts such as processed samples, checkpoints, predictions, logs, and evaluation tables are written to `outputs/`, which is ignored by Git.

## Dataset

The repository expects two processed data components for training and evaluation:

- `data/gt`: labeled vehicle trajectory files.
- `data/map_files`: preprocessed map JSON files containing road centerlines, boundaries, and zone metadata.

The original video data are available from the LoRFT dataset page:

- [https://huggingface.co/datasets/YvfanZhu/LoRFT](https://huggingface.co/datasets/YvfanZhu/LoRFT)

The released training and evaluation pipeline uses the processed trajectory and map files. The videos are mainly needed if you want to inspect the original scenes, regenerate tracking labels, or create qualitative visualizations.

### GT Layout

A scene may contain one GT file directly:

```text
data/gt/<scene>/gt/gt.txt
```

or multiple clips under the same road scene:

```text
data/gt/<scene>/<clip>/gt/gt.txt
```

All clips under the same `<scene>` share:

```text
data/map_files/<scene>.json
```

GT files are comma-separated text files with 10 columns:

```text
frame,id,x,y,w,h,c,d,e,label
```

The label column defines the reconstruction protocol:

- `label=0`: observed rows used as model input.
- `label=1`: distant reference rows used for evaluation.

Training preprocessing builds supervised sliding windows from the labeled trajectories. The default benchmark configuration uses `obs_len=60`, `pred_len=125`, and 25 FPS data.

## Installation

Create a Python environment and install the dependencies:

```bash
pip install -r requirements.txt
```

The main dependencies are PyTorch, NumPy, pandas, PyYAML, pykalman, Shapely, and TensorBoard.

## Quick Start

Run the full pipeline from the repository root:

```bash
python run_preprocess.py
python run_train.py
python run_predict.py
python run_evaluate.py
```

By default, the pipeline reads:

```text
data/gt
data/map_files
```

and writes results under:

```text
outputs/experiments/map_rstnet
```

## Configuration

The default configuration is assembled from the YAML files in `configs/`:

```text
configs/base.yaml      # project paths and scene list
configs/data.yaml      # sliding-window settings and scene split
configs/model.yaml     # model architecture
configs/train.yaml     # optimizer, epochs, checkpointing, device settings
configs/predict.yaml   # inference input/output settings
configs/eval.yaml      # evaluation paths and metrics
```

The default scene split is defined in `configs/data.yaml`. Training uses 14 scenes, validation uses 4 scenes, and testing uses 4 scenes.

## Training

Preprocess the GT and map files:

```bash
python run_preprocess.py
```

Train Map-RSTNet:

```bash
python run_train.py
```

Training writes stable checkpoint names to the configured checkpoint directory:

- `best_model.pth`: best validation checkpoint, used by inference by default.
- `latest.pth`: latest training state, used for resume training.

If `checkpoint.save_every` is greater than zero, epoch checkpoints are also written as `epoch_<n>.pth`.

## Inference and Evaluation

Run prediction:

```bash
python run_predict.py
```

Run evaluation:

```bash
python run_evaluate.py
```

The evaluation script reports aggregate trajectory reconstruction metrics and per-scene metrics under the configured evaluation directory.

## Data License

The source code is released under the MIT License. The data files under `data/gt` and `data/map_files` are provided for research and reproducibility purposes.

The original video data are distributed separately through the LoRFT dataset page on Hugging Face. Please follow the corresponding dataset terms when using the videos.

## Citation

The related manuscript has not been formally published yet. Please do not cite this repository as a paper publication.

A BibTeX entry will be added after publication. For now, if you use this repository, please refer to the GitHub project URL.

## Contact

For questions about the code or data format, please open an issue in this repository.
