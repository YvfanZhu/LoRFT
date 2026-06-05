# Data Directory

`data/map_files` contains preprocessed map JSON files. Each map file must include road `centerlines`, `boundaries`, and `zone_meta`. The internal map preprocessing scripts are not part of this public release.

`data/gt` is the canonical GT path for this release. A scene may either contain a single file at `data/gt/<scene>/gt/gt.txt`, or multiple clips at `data/gt/<scene>/<clip>/gt/gt.txt`. All clips under the same `<scene>` share `data/map_files/<scene>.json`.

The GT files are labeled 10-column tracking files:

```text
frame,id,x,y,w,h,c,d,e,label
```

The label column is required for the public inference and evaluation flow:

- `label=0`: observed rows used as model input.
- `label=1`: target rows used for evaluation.

Generated training data, checkpoints, predictions, and evaluation outputs are written to `outputs/` and are ignored by Git.
