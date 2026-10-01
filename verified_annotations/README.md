# Human-verified annotations

Symbol boxes and pipe lines on real nuclear P&ID pages, pre-labeled by the detectors
(symbols: soup5; lines: lineseg U-Net) and then manually verified/corrected in the browser
editor.

- `labels/<page>.txt` — symbols, YOLO format: `0 cx cy w h` (normalized, single class).
- `lines/<page>.json` — pipes: `{"solid": [[x1,y1,x2,y2],...], "dashed": [...]}` in image pixels.
- `overlay/<page>.jpg` — the verified result drawn on the page (red = symbol, green = solid
  pipe, orange = dashed), for quick visual review.

Pages are the real sheets from `datasets/yolo_pseudo_v2`. This is an in-progress set.
