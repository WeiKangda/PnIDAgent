# Human-verified annotations

Symbol boxes, pipe lines and text on real nuclear P&ID pages, pre-labeled automatically
(symbols: soup5; lines: lineseg U-Net; text: PP-OCRv5) and then manually verified/corrected in a
browser editor.

- `labels/<page>.txt`: symbols, YOLO format `0 cx cy w h` (normalized, single class).
- `lines/<page>.json`: pipes, `{"solid": [[x1,y1,x2,y2],...], "dashed": [...]}` in image pixels.
- `overlay/<page>.jpg`: the verified result drawn on the page (red = symbol, green = solid
  pipe, orange = dashed), **for quick visual review**.
- `text/<page>.json`: text lines, `{"name", "width", "height", "items": [{"id", "box": [x0,y0,x1,y1], "text", "rot"}]}`,
  boxes in image pixels (see "Text" below).
- `text_overlay/<page>.jpg`: the text boxes drawn on the page (blue), for quick visual review.
- `eval_soup5_f1.py`: computes the per-page symbol F1 table below for a YOLO model on these pages.
- `topology/<page>.json`: which symbols are connected by pipes (5 pages **verified**, the rest **unverified**; see "Connectivity" below).
- `topology_verified/`: everything for the 5 pages whose connectivity is verified: drawings, symbols, pipe lines, text, connectivity, review images, method and code.
- `symbol_detection/`: symbol detection, original method (soup5) vs new method (cascade relational re-scoring). Code, per-page results on all 65 pages, test-set (B half) predictions and comparison images. B-half macro F1: soup5 0.728 → new method 0.863.
- `topology_overlay/<page>.jpg`: the connections drawn on the page, one colour per connection.

Pages are the real sheets from `datasets/yolo_pseudo_v2`. **65 pages** (all 65 sheets of that set, 3522 symbols).

## Labeling rules

- One box per symbol. A valve together with its actuator (M, solenoid) and LO/LC mark is one box.
  Tag circles and instrument circles get their own box.
- Large equipment is boxed: pumps, heat exchangers, tanks, steam generators, turbines, condensers,
  heaters, off-page connectors.
- Not boxed: plain text, dashed signal boxes, segment-boundary marks, flow arrows, legend, frame/title block.
- Lines follow pipe centerlines. Dashed/dash-dot lines (signal lines, area and seismic boundaries) are not labeled.
- Lines stop at the edge of a symbol box instead of running through it (a box that holds several other
  symbols, e.g. a tank with spargers inside, is not cut).

The 35 pages marked † below were labeled with the last two rules in place. The first 30 pages were
verified before those rules, so some of their lines still run through symbols (and one page has a dashed line).

## Text

6064 text lines on the 65 pages, one box per line of text.

- `text` is copied exactly as printed (case, punctuation, spacing; printed typos are kept).
  `###` marks text that is visible but unreadable (7 lines, all faded handwriting in title blocks).
- `rot` is how the line has to be turned to read it upright: `0` normal, `cw` the text runs bottom-to-top
  (rotate the crop 90° clockwise), `ccw` top-to-bottom, `180` upside down.
- Included: labels, tags, line numbers, notes, captions, title-block text, and letters inside symbols
  (M, LO, FC, EH, ...). Not included: symbol strokes, arrows and hatching.

How it was made: PP-OCRv5 (server det + rec) at two scales and three orientations, merged; then every
line was checked and corrected by hand against an upright crop, missing text was added, and non-text
boxes were removed. Pages that have a vector PDF with a text layer were also compared word by word
against that layer.

## Connectivity: verified pages

**5 pages are verified** (`"status": "verified"`). [`topology_verified/`](topology_verified/) has everything for these 5 pages in one place: the drawings, all four annotation layers (symbols, pipe lines, text, connectivity), review images for each layer, the method used to build and verify the connectivity, and code that rebuilds the connectivity from the annotations and re-runs the checks.

| Page | Symbols | Connections | Data | Overlay | Explained view |
|---|---|---|---|---|---|
| AFW Davis-Besse reliability analysis, p7 (Third Train Configuration) | 25 | 32 | [json](topology_verified/pseudo_AFW_%28DAVIS-BESSE%29%20AFW%20system%20Reliability%20Analysis_Final%20Report_p7.json) | [overlay](topology_verified/pseudo_AFW_%28DAVIS-BESSE%29%20AFW%20system%20Reliability%20Analysis_Final%20Report_p7.jpg) | [explained](topology_verified/pseudo_AFW_%28DAVIS-BESSE%29%20AFW%20system%20Reliability%20Analysis_Final%20Report_p7_explained.png) |
| CSS (APR1400) | 72 | 94 | [json](topology_verified/pseudo_CSS%20%28APR1400%29.json) | [overlay](topology_verified/pseudo_CSS%20%28APR1400%29.jpg) | [explained](topology_verified/pseudo_CSS%20%28APR1400%29_explained.png) |
| AFW (Seabrook) | 47 | 56 | [json](topology_verified/pseudo_AFW%20%28Seabrook%29.json) | [overlay](topology_verified/pseudo_AFW%20%28Seabrook%29.jpg) | [explained](topology_verified/pseudo_AFW%20%28Seabrook%29_explained.png) |
| RHR (Surry) FSAR | 14 | 40 | [json](topology_verified/pseudo_RHR_%28Surry%29%20FSAR.json) | [overlay](topology_verified/pseudo_RHR_%28Surry%29%20FSAR.jpg) | [explained](topology_verified/pseudo_RHR_%28Surry%29%20FSAR_explained.png) |
| AFW (San Onofre) design review | 60 | 98 | [json](topology_verified/pseudo_AFW_%28San%20Onofre%29%20AFW%20Design%20Review%20and%20Reliability%20Evaluation.json) | [overlay](topology_verified/pseudo_AFW_%28San%20Onofre%29%20AFW%20Design%20Review%20and%20Reliability%20Evaluation.jpg) | [explained](topology_verified/pseudo_AFW_%28San%20Onofre%29%20AFW%20Design%20Review%20and%20Reliability%20Evaluation_explained.png) |

**Easiest way to look at them:** open the *explained* images. Each shows the drawing on the left (pipes coloured by connected network) and the extracted pipe network on the right (coloured dot = symbol, small black dot = junction, line = one pipe segment); the numbers match on both sides. [`how_to_read.png`](topology_verified/how_to_read.png) explains how to read them with one example, and [`topology_5pages_verified.pptx`](topology_verified/topology_5pages_verified.pptx) has all of it as slides.

How they were verified: every connection was cross-checked against this page's human-verified symbol boxes and pipe lines. All verified pipe segments are covered by connections (on AFW (Seabrook) one 54 px segment is covered only by a short diagonal to the corner, a drawing artifact), no connection runs off a verified pipe, every pipe end at a symbol has a connection at that symbol, all junctions are real T-junctions or open pipe ends, crossing decisions match the drawing (a dot means connected), and there are no uncertain points, isolated symbols or unjoined pipe corners. No wrong connections were found, and each page was also inspected visually. The checks and their exact output are described in [`topology_verified/README.md`](topology_verified/README.md#how-the-5-pages-were-verified).

The other 60 pages are still unverified (see below).

## Connectivity (unverified)

**Status: unverified.** Built automatically from the verified symbols and lines, then reviewed page by page
by AI agents. The uncertain points (10 on 6 pages) were confirmed by a human, but a full human check
of every connection has not been done yet. Every unverified file carries `"status": "unverified"`; the 5 verified files carry `"status": "verified"`.

Each `topology/<page>.json` has:

- `symbols`: `{"id", "bbox"}`. `id` is the line number (0-based) in `labels/<page>.txt`; `bbox` is in pixels.
- `junctions`: branch and corner points on the pipes, `{"id": "J0", "xy": [x, y]}`.
- `links`: pipe runs, `{"source", "target", "path"}`. Source/target are a symbol id or a junction id;
  `path` is the list of pipe pieces `[x1,y1,x2,y2]` the run follows.
- `neighbors`: for each symbol, the symbols it reaches through pipes (passing junctions, not passing
  other symbols). This is the easiest field to use.
- `crossings`: places where two pipes cross, with `connected` true/false.
- `notes`: points that were uncertain during review, with the reason.

Rules used: a valve in a line connects to what is on both sides of it. At a crossing, the drawing's own
convention decides (a hop or gap means not connected; on drawings that mark junctions with dots, a plain
crossing without a dot is not connected). Dashed signal lines, area boundaries and pump-turbine shafts
are not pipes. Symbols that no pipe reaches (mostly instrument bubbles and legend items) have no neighbors.

Totals: 3522 symbols, 6202 links, 2468 junctions, 205 symbols with no pipe connection.

## Best symbol model measured against these pages

Model: **soup5** (best single model, weight-average of 5 checkpoints). Inference: adaptive
imgsz (longest side scaled to the model's training scale), conf 0.10 then keep score >= 0.50,
evaluated at IoU 0.5 against the verified symbol GT. Script: `eval_soup5_f1.py` in this folder
(it reproduces the earlier 30-page number exactly, 0.7043).

**Overall macro-F1 = 0.678** across the 65 pages (first 30 pages: 0.704, new 35 pages: 0.655;
for reference 0.888 on the clean real4 set).

| Page | GT | F1 | P | R |
|---|---|---|---|---|
| AFW (Davis -Besse) | 57 | 0.957 | 0.933 | 0.982 |
| AFW (Vogtle) | 76 | 0.953 | 0.973 | 0.934 |
| SI NUREG (Surry) † | 44 | 0.935 | 0.896 | 0.977 |
| Charging pump cooling system NUREG-CR-4550 Vol.3 Part1 Analysis of Core Damage F | 91 | 0.929 | 0.924 | 0.934 |
| LPI-LPR NUREG (Surry) † | 42 | 0.920 | 0.889 | 0.952 |
| AFW (APR1400) † | 75 | 0.909 | 0.956 | 0.867 |
| AFW (Farley) | 73 | 0.909 | 0.864 | 0.959 |
| CCWS NUREG (Surry) | 41 | 0.902 | 0.902 | 0.902 |
| AFW (Seabrook) † | 47 | 0.889 | 0.846 | 0.936 |
| AFW (ST. LUCIE) | 55 | 0.885 | 0.806 | 0.982 |
| AFW (SALEM) | 64 | 0.881 | 0.843 | 0.922 |
| AFW (South Texas) | 60 | 0.876 | 0.869 | 0.883 |
| Boron recovery system (Surry) p3 | 30 | 0.871 | 0.844 | 0.900 |
| SWS (Surry) † | 65 | 0.871 | 0.915 | 0.831 |
| AFW (Turkey Point) | 174 | 0.861 | 0.885 | 0.839 |
| Condensate and Feedwater (APR1400) † | 42 | 0.860 | 0.841 | 0.881 |
| SCS2 (APR1400) | 104 | 0.845 | 0.802 | 0.894 |
| AFW (Cristal River) p2 † | 58 | 0.845 | 0.845 | 0.845 |
| AFW CE Cross Training R325C ch.11.5 † | 76 | 0.843 | 0.922 | 0.776 |
| HVAC (APR1400) p1 † | 19 | 0.811 | 0.833 | 0.789 |
| AFW (Haddam Neck) | 51 | 0.800 | 0.719 | 0.902 |
| ESW (APR1400) † | 48 | 0.791 | 0.895 | 0.708 |
| SCS (APR1400) | 21 | 0.791 | 0.773 | 0.810 |
| Control room HVAC (APR1400) | 97 | 0.788 | 0.792 | 0.784 |
| AFW (San Onofre) Design Review † | 60 | 0.784 | 0.754 | 0.817 |
| AFW (Brain Wood) | 67 | 0.780 | 0.857 | 0.716 |
| AFW (Cristal River) p1 † | 36 | 0.775 | 0.705 | 0.861 |
| Feedwater system (Surry) † | 90 | 0.765 | 0.861 | 0.689 |
| Instrument Air (APR1400) | 68 | 0.752 | 0.769 | 0.735 |
| Boron recovery system (Surry) p1 | 25 | 0.750 | 0.783 | 0.720 |
| Steam and Power Conversion (GE-Hitachi ABWR) † | 140 | 0.749 | 0.722 | 0.779 |
| CCWS (Surry) p2 | 9 | 0.737 | 0.700 | 0.778 |
| ECWS (APR1400) | 62 | 0.733 | 0.949 | 0.597 |
| AFW INL System Study † | 49 | 0.729 | 0.861 | 0.633 |
| RCS-POSRV (APR1400) † | 135 | 0.716 | 0.754 | 0.681 |
| CSS NUREG (Surry) | 24 | 0.706 | 0.667 | 0.750 |
| AFW (DAVIS-BESSE) Reliability p1 | 52 | 0.677 | 0.564 | 0.846 |
| CCWS (APR1400) † | 85 | 0.667 | 0.716 | 0.624 |
| CCWS (Surry) p1 | 19 | 0.667 | 0.514 | 0.947 |
| AFW (DAVIS-BESSE) Reliability p3 † | 54 | 0.652 | 0.551 | 0.796 |
| HVAC (APR1400) p2 † | 23 | 0.651 | 0.700 | 0.609 |
| AFW (DAVIS-BESSE) Reliability p8 | 54 | 0.647 | 0.544 | 0.796 |
| AFW (DAVIS-BESSE) Reliability p9 † | 37 | 0.633 | 0.508 | 0.838 |
| Boron recovery system (Surry) p2 | 9 | 0.632 | 0.600 | 0.667 |
| AFW (DAVIS-BESSE) Reliability p5 † | 36 | 0.624 | 0.509 | 0.806 |
| SI (APR1400) | 93 | 0.609 | 0.577 | 0.645 |
| Main Steam (APR1400) † | 66 | 0.595 | 0.655 | 0.545 |
| Boron recovery system (Surry) p4 † | 17 | 0.579 | 0.524 | 0.647 |
| EPS NUREG (Surry) p1 † | 71 | 0.559 | 0.775 | 0.437 |
| HVAC (APR1400) p3 † | 115 | 0.531 | 0.662 | 0.443 |
| CCWS (Surry) p5 | 11 | 0.516 | 0.400 | 0.727 |
| Power conversion system (NuScale) † | 45 | 0.515 | 0.385 | 0.778 |
| AFW (DAVIS-BESSE) Reliability p6 † | 27 | 0.513 | 0.392 | 0.741 |
| RHR (Surry) † | 14 | 0.511 | 0.364 | 0.857 |
| AFW (DAVIS-BESSE) Reliability p7 † | 25 | 0.494 | 0.365 | 0.760 |
| Compressed Air system (Surry) † | 37 | 0.468 | 0.450 | 0.486 |
| AFW (DAVIS-BESSE) Reliability p2 † | 27 | 0.463 | 0.345 | 0.704 |
| CSS (APR1400) † | 72 | 0.398 | 0.360 | 0.444 |
| AFW (DAVIS-BESSE) Reliability p4 † | 26 | 0.386 | 0.281 | 0.615 |
| EPS NUREG (Surry) p2 | 12 | 0.333 | 0.333 | 0.333 |
| AC Power (APR1400) † | 177 | 0.274 | 0.714 | 0.169 |
| CCWS (Surry) p3 | 7 | 0.273 | 0.200 | 0.429 |
| Main Steam system (Surry) † | 8 | 0.214 | 0.150 | 0.375 |
| Battery system (Surry) | 22 | 0.069 | 0.143 | 0.045 |
| CCWS (Surry) p4 | 6 | 0.000 | 0.000 | 0.000 |
| **macro (65)** | | **0.678** | 0.668 | 0.724 |

† = added in this update (35 pages).

### Reading this

The clean real4 number (0.888) overstates generalization. On this broader, messier set
(different plants, resolutions, scan quality) soup5 is about 0.68.

The low pages fall into a few groups, and none of them is a soup5 malfunction:

- **Large equipment pages** (CCWS (Surry) p4 0.00, Battery system 0.07, EPS NUREG p2 0.33). soup5 is a
  small-symbol detector (valves, instruments, ~30-80 px); these pages are mostly tanks, heat
  exchangers and coolers at 150-500 px.
- **Electrical one-line diagrams** (AC Power (APR1400) 0.27). Most symbols are breakers, transformers
  and connectors, which soup5 never saw in training. Precision is fine (0.71), recall is 0.17.
- **Old scanned, hand-drawn sheets** (AFW (DAVIS-BESSE) Reliability p2-p9, 0.39-0.65). Recall is
  OK (0.6-0.8), precision is low: handwritten tags and scan noise produce false boxes.
- **Different box convention on one page** (CSS (APR1400) 0.40). On this page each valve and its tag
  circle are labeled as one box, while soup5 puts them in separate boxes, so many matches fall below IoU 0.5.
- **Very few symbols** (Main Steam system (Surry), 8 GT): a couple of misses swing F1 a lot.

On pages made of normal small symbols it scores 0.80-0.96.

Caveat: this GT was seeded from soup5 prelabels and then human-corrected, so it is somewhat
anchored to what soup5 already found. The takeaway: small-symbol detection on real pages is
usable (~0.8+), large equipment and electrical symbols are the gap, and real labeled data
(this set) is the way to close it.
