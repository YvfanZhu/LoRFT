# Examples

`synthetic_labeled_gt/gt/gt.txt` is a compact 10-column labeled GT example that matches the default sequence lengths: 60 observed rows followed by 125 target rows for one track.

Rows use this schema:

```text
frame,id,x,y,w,h,c,d,e,label
```

`label=0` denotes observed input rows, and `label=1` denotes target rows for evaluation. The coordinates are synthetic and are intended to document the file format rather than to train a meaningful model.
