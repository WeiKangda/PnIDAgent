# Nuclear P&ID Real Test Set (real4)

4 real nuclear P&ID drawings with human-annotated symbol bounding boxes, used as a held-out test set for symbol detection.

## Contents

```
images/   4 P&ID drawings (.jpg)
labels/   4 YOLO-format annotation files (.txt)
```

| Image | Size | Symbols | Plant / System |
|---|---|---|---|
| AFW_NorthANA.jpg | 3300×2550 | 78 | AFW (North Anna) |
| AFW_Surry.jpg | 3304×2556 | 74 | AFW (Surry) |
| APR1400.jpg | 3300×2550 | 85 | CVCS (APR1400) |
| CVCS_Surry.jpg | 3300×2550 | 37 | CVCS (Surry) |
| **Total** | | **274** | |

## Label format (YOLO)

Each `labels/<name>.txt` has one line per symbol:

```
<class> <x_center> <y_center> <width> <height>
```

- All values **normalized to [0,1]** (divide by image width/height).
- Single class: `0 = symbol`.
- Example: `0 0.131515 0.608431 0.021818 0.018431`

To get pixel coordinates: multiply `x_center`, `width` by image width; `y_center`, `height` by image height.

## Notes

- Held-out evaluation set (not used for training).
- Symbols = valves, instruments, pumps, tanks, and other P&ID components.
- Annotations are human-verified ground truth.
