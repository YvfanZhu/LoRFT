<div align="center">

# LoRFT / Map-RSTNet

**Benchmarking Long-Range Vehicle Trajectory Reconstruction from Fixed Highway Cameras**

<p>
  <a href="#news">News</a> |
  <a href="#abstract">Abstract</a> |
  <a href="#lorft-benchmark">LoRFT Benchmark</a> |
  <a href="#map-rstnet">Map-RSTNet</a> |
  <a href="#quick-start">Quick Start</a> |
  <a href="docs/PIPELINE.md">Pipeline</a>
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

This project provides **LoRFT**, a benchmark for long-range vehicle trajectory reconstruction from fixed highway cameras, and **Map-RSTNet**, a map-aware residual Seq2Seq reference model for this task.

To our knowledge, LoRFT is the first benchmark dedicated to reconstructing the far-range continuation of the same vehicle from a reliable near-field tracklet in a fixed highway surveillance view.

LoRFT contains 22 expressway surveillance scenes, 366,109 video frames, 6,601 manually verified vehicle trajectories, 2,694,889 vehicle bounding boxes, scene-level road-geometry annotations, predefined scene-level splits, evaluation scripts, and model code.

## News

- Initial release of the LoRFT benchmark resources and Map-RSTNet reference model.
- The code supports preprocessing, training, inference, and evaluation under the scene-level LoRFT protocol.
- Original videos are hosted separately on Hugging Face: [YvfanZhu/LoRFT](https://huggingface.co/datasets/YvfanZhu/LoRFT).

## Abstract

Long-range highway vehicle trajectories are useful for traffic safety analysis, autonomous driving evaluation, and data-driven traffic management, but collecting them continuously at scale remains difficult. Fixed highway cameras are already deployed for routine monitoring, yet automatic tracking often keeps only the near-field portion of a vehicle trajectory. In distant road regions, perspective compression, scale decay, occlusion, and association instability can cause tracklets to fragment or terminate early, although the same vehicle may still be verified from motion continuity across neighboring frames.

LoRFT formulates this setting as long-range vehicle trajectory reconstruction: given a reliable near-field tracklet, recover the distant continuation of the same vehicle in the original fixed-camera view. The benchmark provides manually verified observed/reference trajectory pairs, vehicle bounding boxes, scene-level road geometry, predefined scene-level splits, and evaluation scripts. Map-RSTNet addresses this task with a road-aligned state representation, direction alignment, residual decoding, and dynamic geometry refresh.

## LoRFT Benchmark

Fixed highway cameras are widely deployed for continuous traffic monitoring, but automatic tracking often preserves only the near-field part of a vehicle trajectory. In distant road regions, perspective compression, scale decay, occlusion, and unstable association can make a tracklet fragment or terminate early, even when the same vehicle remains traceable from neighboring frames.

LoRFT turns this observation into a reconstruction benchmark:

- **Input:** a reliable near-field tracklet of a known vehicle.
- **Target:** the manually verified distant continuation of the same vehicle in the same fixed-camera view.
- **Coordinate system:** original image coordinates, without assuming scene-specific camera calibration.
- **Evaluation unit:** observed/reference trajectory pairs from held-out highway scenes.

This setting is different from standard trajectory forecasting. The task is not to predict future motion from a complete and reliable observation history; it is to recover the distant continuation of the same trajectory after fixed-camera tracking becomes unreliable.

### Dataset Statistics

| Item | Value |
| --- | ---: |
| Surveillance scenes | 22 |
| Video frames | 366,109 |
| Manually verified vehicle trajectories | 6,601 |
| Vehicle bounding boxes | 2,694,889 |
| Frame rate | 25 FPS |
| Video resolution | 352 x 288 |
| Split protocol | 14 train / 4 validation / 4 test scenes |

The videos were collected from expressway sites in Sichuan Province, China, using roadside and gantry-mounted fixed cameras. The scenes include different road layouts, viewpoints, traffic directions, and visible road extents.

### Annotation Protocol

LoRFT uses a semi-automatic annotation pipeline. Candidate boxes and tracklets are first generated with YOLOv11 and ByteTrack, then manually corrected and verified. Annotators correct localization errors, association mistakes, false detections, short occlusion-induced interruptions, and distant track fragmentation.

Each GT row has 10 comma-separated fields:

```text
frame,id,x,y,w,h,c,d,e,label
```

The `label` field defines the reconstruction protocol:

- `label=0`: manually verified observed segment used as model input.
- `label=1`: manually verified distant reference segment used for evaluation.

The two labels describe the input and reference parts of the same fixed-camera trajectory. They do not necessarily indicate chronological order: depending on traffic direction and camera placement, the distant reference segment may appear later or earlier in video time.

### Road Geometry

Each scene has a preprocessed map JSON file with:

- road centerlines,
- left and right road boundaries,
- zone-level traffic-direction metadata.

These annotations support road-aligned trajectory representation and geometry-aware reconstruction in fixed surveillance views.

## Map-RSTNet

Map-RSTNet is a map-aware residual Seq2Seq LSTM for reconstructing distant vehicle trajectories under partial observability.

The model:

- converts image-space boxes into bottom-center trajectory points;
- encodes each point with a 10-dimensional road-aware state, including lateral position, normalized vertical position, box scale, velocity, road-boundary distances, and heading difference;
- aligns opposite traffic directions into a shared reconstruction order;
- predicts residual displacements autoregressively from a 60-frame observation window;
- refreshes local road geometry during decoding so the reconstructed trajectory remains conditioned on the current road position;
- projects the reconstructed states back to the image plane for evaluation.

The default configuration uses 60 observed frames and reconstructs a 125-frame distant segment at 25 FPS.

## Benchmark Results

The following numbers are computed on held-out LoRFT test scenes under the scene-level protocol. Errors are measured in pixels; lower is better.

| Model | ADE | FDE | RMSE@1s | RMSE@2s | RMSE@3s | RMSE@4s | RMSE@5s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CS-LSTM | 13.85 | 26.99 | 8.48 | 16.42 | 25.12 | 35.72 | 48.87 |
| GRIP++ | 17.50 | 33.80 | 8.77 | 16.14 | 23.25 | 30.65 | 38.57 |
| DeepTrack | 29.77 | 44.53 | 31.89 | 44.44 | 53.85 | 63.61 | 68.54 |
| MixNet | 28.04 | 48.75 | 19.58 | 28.24 | 37.60 | 50.42 | 64.74 |
| GNP | 20.50 | 36.36 | 18.61 | 21.57 | 24.75 | 35.15 | 64.12 |
| PRF | 15.53 | 25.67 | 12.20 | 13.58 | 18.15 | 24.40 | 30.70 |
| Map-RSTNet | **12.32** | **21.71** | **8.15** | **13.30** | **17.97** | **22.32** | **27.47** |

Compared with the best baseline for each metric, Map-RSTNet reduces ADE from 13.85 to 12.32, FDE from 25.67 to 21.71, and RMSE@5s from 30.70 to 27.47, corresponding to relative reductions of 11.0%, 15.4%, and 10.5%.

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

## Installation

Create a Python environment and install the dependencies:

```bash
pip install -r requirements.txt
```

Main dependencies include PyTorch, NumPy, pandas, PyYAML, pykalman, Shapely, and TensorBoard.

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

and writes outputs under:

```text
outputs/experiments/map_rstnet
```

## Configuration

The default configuration is assembled from YAML files in `configs/`:

```text
configs/base.yaml      # project paths and scene list
configs/data.yaml      # sliding-window settings and scene split
configs/model.yaml     # model architecture
configs/train.yaml     # optimizer, epochs, checkpointing, device settings
configs/predict.yaml   # inference input/output settings
configs/eval.yaml      # evaluation paths and metrics
```

The default scene split is defined in `configs/data.yaml`.

## Application Scenarios

LoRFT is intended for fixed-camera highway settings where perception quality changes substantially with distance. Beyond the reconstruction protocol, the annotations can support analysis of far-range vehicle detection and tracking failures, including missed detections, identity switches, track fragmentation, and premature termination under the same surveillance views.

Reconstructed trajectories can also provide longer image-space motion records for downstream traffic mining tasks such as trajectory prediction, lane-change analysis, conflict analysis, active risk assessment, and data-driven traffic management. These uses should respect the dataset coordinate system: LoRFT provides image-space trajectories and road-geometry annotations, but not scene-specific camera calibration parameters. Applications requiring metric speed, acceleration, or physical vehicle dynamics need additional calibration or image-to-road mapping.

## Data Availability

The repository includes processed trajectory and road-geometry files used by the released pipeline. Original videos are distributed separately through the LoRFT dataset page on Hugging Face:

- [https://huggingface.co/datasets/YvfanZhu/LoRFT](https://huggingface.co/datasets/YvfanZhu/LoRFT)

Please follow the corresponding dataset terms when using the videos.

## Manuscript Status

The related manuscript has not been formally published yet. Please do not cite this repository as a paper publication.

A formal citation entry will be added after publication. For now, please refer to the GitHub project URL when discussing the released code or benchmark resources.

## Contact

For questions about the code or data format, please open an issue in this repository.
