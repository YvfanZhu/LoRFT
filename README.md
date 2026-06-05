<div align="center">

# LoRFT / Map-RSTNet

**Benchmarking Long-Range Vehicle Trajectory Reconstruction from Fixed Highway Cameras**

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

This repository contains the public implementation of Map-RSTNet used in the accompanying paper. Map-RSTNet reconstructs long-range vehicle trajectories from observed tracking fragments by combining residual sequence prediction with map-aware geometric constraints.

LoRFT studies a fixed-camera highway setting where a reliable near-field tracklet is observed, while the distant continuation of the same vehicle becomes difficult to preserve because of perspective compression, scale decay, occlusion, and association instability. Map-RSTNet reconstructs the far-range continuation in a road-aligned state space and projects the result back to the original image plane.

The released code supports preprocessing, model training, prediction, and evaluation on highway trajectory scenes. The default configuration follows the paper setting with `obs_len=60` and `pred_len=125`.

## News

- Public code release for the LoRFT benchmark pipeline and Map-RSTNet.
- The pipeline supports scene-level splits and multi-clip ground-truth folders.
- Original video data are hosted on Hugging Face: [YvfanZhu/LoRFT](https://huggingface.co/datasets/YvfanZhu/LoRFT).

## Highlights

- **Map-aware trajectory reconstruction.** Road centerlines, boundaries, and zone metadata provide geometric context for long-range prediction.
- **Residual Seq2Seq modeling.** The model predicts future trajectory displacement from observed vehicle motion and map-related features.
- **Scene-level evaluation.** The default split separates training, validation, and test road scenes to reduce scene leakage.
- **Multi-clip support.** Multiple GT clips can share the same road-scene map file.
- **Reproducible public pipeline.** Preprocessing, training, inference, and evaluation are exposed through simple entry scripts.

## Benchmark Overview

LoRFT is designed for long-range vehicle trajectory reconstruction from fixed highway cameras. The benchmark contains:

- 22 expressway surveillance scenes.
- 366,109 video frames.
- 6,601 manually verified vehicle trajectories.
- 2,694,889 vehicle bounding boxes.
- Scene-level road-geometry annotations.
- Predefined scene-level splits and evaluation scripts.

On LoRFT, Map-RSTNet achieves `ADE=12.32`, `FDE=21.71`, and `RMSE-5s=27.47` pixels under the default long-range reconstruction setting.

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

The label column is used by the public inference and evaluation flow:

- `label=0`: observed rows used as model input.
- `label=1`: target rows used for evaluation.

Training preprocessing builds supervised sliding windows from the labeled trajectories. The default public configuration uses `obs_len=60`, `pred_len=125`, and 25 FPS data.

## Installation

Create a Python environment and install the dependencies:

```bash
pip install -r requirements.txt
```

The main dependencies are PyTorch, NumPy, pandas, PyYAML, pykalman, Shapely, and TensorBoard.

## Quick Start

Run the full public pipeline from the repository root:

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

If this repository is useful for your research, please cite the accompanying paper. The BibTeX entry will be updated after publication.

```bibtex
@article{maprstnet,
  title   = {LoRFT: Benchmarking Long-Range Vehicle Trajectory Reconstruction from Fixed Highway Cameras},
  author  = {Anonymous},
  journal = {To appear},
  year    = {2026}
}
```

## Contact

For questions about the code or data format, please open an issue in this repository.
