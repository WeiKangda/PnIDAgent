# Five fully verified pages: symbols, pipe lines, text and connectivity

These 5 pages have all four annotation layers verified: symbol boxes, pipe lines and text were checked and corrected by hand, and the pipe connectivity between the symbols, built from those annotations, was verified against them (automatic checks plus visual inspection, see below). Everything needed to use or re-check them is in this folder: the drawings, all four annotation layers, review images, the method, and code that rebuilds and checks the connectivity.

| Page | Symbols | Pipe lines | Text lines | Pipe segments | T junctions | Open pipe ends | Data |
|---|---|---|---|---|---|---|---|
| Davis-Besse p7 | 25 | 28 | 69 | 32 | 7 | 0 | [drawing](images/pseudo_AFW_%28DAVIS-BESSE%29%20AFW%20system%20Reliability%20Analysis_Final%20Report_p7.png) · symbols [txt](symbols/pseudo_AFW_%28DAVIS-BESSE%29%20AFW%20system%20Reliability%20Analysis_Final%20Report_p7.txt) / [json](symbols/pseudo_AFW_%28DAVIS-BESSE%29%20AFW%20system%20Reliability%20Analysis_Final%20Report_p7.json) · [pipe lines](lines/pseudo_AFW_%28DAVIS-BESSE%29%20AFW%20system%20Reliability%20Analysis_Final%20Report_p7.json) · [text](text/pseudo_AFW_%28DAVIS-BESSE%29%20AFW%20system%20Reliability%20Analysis_Final%20Report_p7.json) · [connectivity](pseudo_AFW_%28DAVIS-BESSE%29%20AFW%20system%20Reliability%20Analysis_Final%20Report_p7.json) |
| RHR (Surry) | 14 | 48 | 46 | 40 | 14 | 8 | [drawing](images/pseudo_RHR_%28Surry%29%20FSAR.png) · symbols [txt](symbols/pseudo_RHR_%28Surry%29%20FSAR.txt) / [json](symbols/pseudo_RHR_%28Surry%29%20FSAR.json) · [pipe lines](lines/pseudo_RHR_%28Surry%29%20FSAR.json) · [text](text/pseudo_RHR_%28Surry%29%20FSAR.json) · [connectivity](pseudo_RHR_%28Surry%29%20FSAR.json) |
| AFW (Seabrook) | 47 | 62 | 37 | 56 | 6 | 2 | [drawing](images/pseudo_AFW%20%28Seabrook%29.png) · symbols [txt](symbols/pseudo_AFW%20%28Seabrook%29.txt) / [json](symbols/pseudo_AFW%20%28Seabrook%29.json) · [pipe lines](lines/pseudo_AFW%20%28Seabrook%29.json) · [text](text/pseudo_AFW%20%28Seabrook%29.json) · [connectivity](pseudo_AFW%20%28Seabrook%29.json) |
| AFW (San Onofre) | 60 | 80 | 108 | 98 | 19 | 9 | [drawing](images/pseudo_AFW_%28San%20Onofre%29%20AFW%20Design%20Review%20and%20Reliability%20Evaluation.png) · symbols [txt](symbols/pseudo_AFW_%28San%20Onofre%29%20AFW%20Design%20Review%20and%20Reliability%20Evaluation.txt) / [json](symbols/pseudo_AFW_%28San%20Onofre%29%20AFW%20Design%20Review%20and%20Reliability%20Evaluation.json) · [pipe lines](lines/pseudo_AFW_%28San%20Onofre%29%20AFW%20Design%20Review%20and%20Reliability%20Evaluation.json) · [text](text/pseudo_AFW_%28San%20Onofre%29%20AFW%20Design%20Review%20and%20Reliability%20Evaluation.json) · [connectivity](pseudo_AFW_%28San%20Onofre%29%20AFW%20Design%20Review%20and%20Reliability%20Evaluation.json) |
| CSS (APR1400) | 72 | 105 | 182 | 94 | 14 | 1 | [drawing](images/pseudo_CSS%20%28APR1400%29.png) · symbols [txt](symbols/pseudo_CSS%20%28APR1400%29.txt) / [json](symbols/pseudo_CSS%20%28APR1400%29.json) · [pipe lines](lines/pseudo_CSS%20%28APR1400%29.json) · [text](text/pseudo_CSS%20%28APR1400%29.json) · [connectivity](pseudo_CSS%20%28APR1400%29.json) |
| **Total** | **218** | **323** | **442** | **320** | **60** | **20** | |

Review images for each page:

| Page | Images |
|---|---|
| Davis-Besse p7 | [symbols + pipe lines](overlays/pseudo_AFW_%28DAVIS-BESSE%29%20AFW%20system%20Reliability%20Analysis_Final%20Report_p7_symbols_lines.jpg) · [text](overlays/pseudo_AFW_%28DAVIS-BESSE%29%20AFW%20system%20Reliability%20Analysis_Final%20Report_p7_text.jpg) · [connections on the drawing](pseudo_AFW_%28DAVIS-BESSE%29%20AFW%20system%20Reliability%20Analysis_Final%20Report_p7.jpg) · [explained connectivity](pseudo_AFW_%28DAVIS-BESSE%29%20AFW%20system%20Reliability%20Analysis_Final%20Report_p7_explained.png) |
| RHR (Surry) | [symbols + pipe lines](overlays/pseudo_RHR_%28Surry%29%20FSAR_symbols_lines.jpg) · [text](overlays/pseudo_RHR_%28Surry%29%20FSAR_text.jpg) · [connections on the drawing](pseudo_RHR_%28Surry%29%20FSAR.jpg) · [explained connectivity](pseudo_RHR_%28Surry%29%20FSAR_explained.png) |
| AFW (Seabrook) | [symbols + pipe lines](overlays/pseudo_AFW%20%28Seabrook%29_symbols_lines.jpg) · [text](overlays/pseudo_AFW%20%28Seabrook%29_text.jpg) · [connections on the drawing](pseudo_AFW%20%28Seabrook%29.jpg) · [explained connectivity](pseudo_AFW%20%28Seabrook%29_explained.png) |
| AFW (San Onofre) | [symbols + pipe lines](overlays/pseudo_AFW_%28San%20Onofre%29%20AFW%20Design%20Review%20and%20Reliability%20Evaluation_symbols_lines.jpg) · [text](overlays/pseudo_AFW_%28San%20Onofre%29%20AFW%20Design%20Review%20and%20Reliability%20Evaluation_text.jpg) · [connections on the drawing](pseudo_AFW_%28San%20Onofre%29%20AFW%20Design%20Review%20and%20Reliability%20Evaluation.jpg) · [explained connectivity](pseudo_AFW_%28San%20Onofre%29%20AFW%20Design%20Review%20and%20Reliability%20Evaluation_explained.png) |
| CSS (APR1400) | [symbols + pipe lines](overlays/pseudo_CSS%20%28APR1400%29_symbols_lines.jpg) · [text](overlays/pseudo_CSS%20%28APR1400%29_text.jpg) · [connections on the drawing](pseudo_CSS%20%28APR1400%29.jpg) · [explained connectivity](pseudo_CSS%20%28APR1400%29_explained.png) |

## Contents

| Path | Content |
|---|---|
| `images/<page>.png` | The drawing at its original resolution. All coordinates below are pixels in this image. |
| `symbols/<page>.txt` | Symbol boxes, YOLO format `0 cx cy w h` (normalized, single class). The line number (from 0) is the symbol id. |
| `symbols/<page>.json` | The same boxes in pixels: `{"name", "width", "height", "symbols": [{"id", "bbox": [x0, y0, x1, y1]}]}`. These ids are the ones used in the connectivity files. |
| `lines/<page>.json` | Pipe lines, `{"solid": [[x1, y1, x2, y2], ...], "dashed": [...]}`. Only solid lines are pipes; `dashed` is empty on these 5 pages. |
| `text/<page>.json` | Text, one box per line of text: `{"name", "width", "height", "items": [{"id", "box": [x0, y0, x1, y1], "text", "rot"}]}` (format below). |
| `<page>.json` | Connectivity: which symbols are connected by pipes (format below). |
| `manual_edits.json` | The only changes made by hand on top of the symbol and pipe annotations while reviewing connectivity (see "Manual edits"). |
| `overlays/<page>_symbols_lines.jpg` | Symbols (red, with id) and pipe lines (green; pieces added in review in blue) drawn on the drawing. |
| `overlays/<page>_text.jpg` | Left: the drawing with the text boxes. Right: the transcribed text placed in the same boxes, so the two sides can be compared word by word. |
| `<page>.jpg` | Connections drawn on the drawing, one colour per connected pipe network. |
| `<page>_explained.png` | The drawing next to the extracted pipe network, with matching numbers (see "How to read"). |
| `how_to_read.png`, `topology_5pages_verified.pptx` | How to read the explained images; all of it as slides. |
| `code/` | Code that rebuilds the connectivity from the annotations and runs the checks (see "Reproduce"). |

The same symbol, line and text files are also in `../labels`, `../lines` and `../text` together with the other 60 pages; the copies here are identical.

### Text format

- `text` is copied exactly as printed (case, punctuation, spacing; printed typos are kept). `###` marks text that is visible but unreadable.
- `rot` is how the line has to be turned to read it upright: `0` normal, `cw` the text runs bottom-to-top (rotate the crop 90° clockwise), `ccw` top-to-bottom, `180` upside down.
- Included: labels, tags, line numbers, notes, captions, title-block text, and letters inside symbols (M, LO, FC, ...). Not included: symbol strokes, arrows and hatching.

### Connectivity format

- `symbols`: `{"id", "bbox"}`, the same ids and boxes as `symbols/<page>.json`.
- `junctions`: points where pipes meet or end without a symbol: T junctions (3 pipes meet) and open pipe ends (a pipe that leaves the drawing or ends without a symbol). Corners are not nodes. Format `{"id": "J0", "xy": [x, y]}`.
- `links`: one pipe segment between two nodes, `{"source", "target", "path"}`. Source and target are a symbol id (number) or a junction id (`J...`); `path` is the pipe geometry as a list of `[x1, y1, x2, y2]` pieces.
- `neighbors`: for each symbol, the symbols reachable from it along pipes and through junctions, without passing through another symbol. This is the "which symbol connects to which" answer.
- `crossings`: places where two pipes cross in the middle of both, with `connected` true only when a filled dot is drawn there.
- `status`: `verified`; `status_note` says how.

## How the annotations were made

1. **Symbols.** Pre-labeled by a YOLO detector (soup5), then every box was checked and corrected by hand in a browser editor. One box per symbol: a valve together with its actuator (M, solenoid) and LO/LC mark is one box; tag and instrument circles get their own box; large equipment and off-page connectors are boxed; plain text, dashed signal boxes, flow arrows, legend and title block are not.
2. **Pipe lines.** Pre-labeled by a line segmentation network, then corrected by hand. Lines follow pipe centerlines and stop at the edge of a symbol box. Dashed and dash-dot lines (signal lines, area and seismic boundaries) are not pipes and are not labeled. Room and building outlines are not pipes. Electrical bus bars are labeled as lines.
3. **Text.** PP-OCRv5 at two scales and three orientations, merged; then every line was checked and corrected by hand against an upright crop, missing text was added and non-text boxes were removed.
4. **Connectivity.** Built automatically from the verified symbols and pipe lines with the algorithm below, reviewed, then checked automatically and visually (see "How the 5 pages were verified").

## How connectivity is built

Code: [`code/topo.py`](code/topo.py) (function `build`), on top of [`code/pnid_graph_upstream.py`](code/pnid_graph_upstream.py). The inputs are the symbol boxes, the solid pipe lines and the drawing. Distances are in pixels.

1. **Clean up the pipe lines.** The hand-drawn lines are joined where the drawing shows one continuous pipe:
   - collinear pieces that overlap or nearly touch are merged;
   - a pipe broken by a small hop-over arc where it jumps another pipe is joined back (gap ≤ 45 px, the other pipe runs through the gap, no symbol in it); the result is then treated as a crossing;
   - a bus bar cut in two by a breaker-like symbol is joined back into one line;
   - a pipe end with no other line near it is extended onto a pipe within 28 px when ink on the drawing connects them (hand-drawn corners and T junctions often miss by 10-20 px);
   - short pipes that were not labeled are traced on the drawing: from each side of a symbol box, solid ink is followed horizontally or vertically, and the trace is kept only if it reaches a labeled pipe (so text and leader lines are not taken as pipes).
2. **Find the nodes.** Line ends within 12 px of each other are snapped together. Shared ends become corners, an end on another line becomes a T junction, and two lines crossing in the middle of both are an X crossing.
3. **Decide crossings.** Following P&ID convention, an X crossing is connected only if a filled dot is drawn there. A dot is detected when the ink along both diagonals through the crossing is clearly longer than for a plain cross (at least 1.35 × the line thickness + 1.5 px). Crossings inside a symbol box are ignored (they are strokes of the symbol). A decision recorded in `manual_edits.json` overrides the automatic one.
4. **Attach symbols to pipes.**
   - A pipe that runs through a symbol box or ends at it puts the symbol in series on the pipe; a pipe that only touches the edge of the box gets a branch to the symbol.
   - A pipe that ends pointing at a symbol box within 30 px, with ink in between, is attached to it.
   - A pipe passing beside a symbol within 30 px, with a continuous short ink line between them, is attached.
   - Symbols that sit directly next to each other on a straight horizontal or vertical pipe are connected in series.
   - A pipe that only crosses a symbol's own lead is a crossing, not a branch.
5. **Build the graph.** Along each pipe the nodes are ordered and consecutive nodes are linked. Points where a pipe simply continues or turns a corner are removed, so every link runs from a symbol or junction to the next one. The remaining junctions are T junctions and open pipe ends.
6. **Export.** Symbols, junctions, links with their geometry, the `neighbors` list and the crossings are written to `<page>.json`.

## Manual edits

[`manual_edits.json`](manual_edits.json) records everything that was set by hand during the connectivity review, on top of the symbol and pipe annotations:

- `added_lines`: short pipe pieces where two hand-drawn lines should meet but do not reach each other: 1 on AFW (Seabrook) and 3 on AFW (San Onofre), each 13-32 px long.
- `crossing_decisions`: crossings decided during review: 1 on RHR (Surry) at (1917, 666) and 1 on AFW (San Onofre) at (1645, 1375). Both are "not connected", and the drawing shows no dot at either.

## How the 5 pages were verified

1. **Automatic checks** against this page's human-verified symbols and pipe lines ([`code/check_connectivity.py`](code/check_connectivity.py)):
   1. coverage: every verified pipe line, outside symbol boxes, is covered by connections for at least 80% of its length;
   2. on pipe: every connection, outside symbol boxes, runs on verified pipe lines for at least 90% of its length;
   3. attached: every verified pipe end that stops at a symbol box has a connection at that symbol;
   4. isolated: every symbol has at least one connection;
   5. junctions: no junction joins 4 or more pipes without a drawn dot;
   6. crossings: every crossing decision agrees with the drawing (dot = connected);
   7. dead ends: no pipe stops within 25 px of another pipe without joining it.
2. **Visual inspection** of every page with the explained image and the overlays.

Result: all checks pass on all 5 pages, with one flag on AFW (Seabrook) that is a drawing artifact and not a wrong connection. There, the hand-drawn vertical pipe ends 16 px above the end of the horizontal pipe at (1596-1650, 775), so the graph joins them with a short diagonal to the corner. The connection itself (symbol 27 to symbol 44, through the corner) is correct, but the diagonal covers only 42% of that 54 px horizontal line, which the coverage check reports.

## Reproduce

Requirements: Python 3, `numpy`, `opencv-python`. From this folder:

```
python code/build_connectivity.py --check   # rebuilds the 5 connectivity files from images/, symbols/, lines/ and manual_edits.json
python code/check_connectivity.py           # runs the 7 checks above
```

`build_connectivity.py --check` writes `rebuilt/<page>.json` and reports, for each page, that it is identical to the published `<page>.json` (symbols, junctions, links, neighbors and crossings). `check_connectivity.py` reports the one coverage flag on AFW (Seabrook) explained above and passes everything else.

## How to read the explained images

![How to read](how_to_read.png)

The drawing is on the **left** (pipes drawn in colour, each symbol a numbered box) and the extracted pipe network on the **right** (coloured dot = symbol, small black dot = junction, line = one pipe segment). The numbers match on both sides. The right side is the left side simplified: pipes straightened and text removed, with exactly the same connections.

Start from a symbol and follow the pipe. At a junction (black dot), follow every branch, and stop at the next symbol. The symbols you reach are the ones connected to it. Two pipes that only cross (no solid dot on the drawing) are **not** connected.

## How to check a page yourself

- **Symbols:** open `overlays/<page>_symbols_lines.jpg`. Every symbol on the drawing has exactly one red box, and nothing else is boxed.
- **Pipe lines:** in the same image, every solid pipe is covered by a green line, lines stop at symbol boxes, and no dashed line is drawn.
- **Text:** open `overlays/<page>_text.jpg` and compare the two sides: same words, same boxes, nothing missing.
- **Connectivity:** compare the two sides of `<page>_explained.png`:
  1. nothing missing: every pipe on the drawing appears on the right, between the same two symbols;
  2. nothing extra: every line on the right corresponds to a pipe on the drawing;
  3. crossings: a solid dot on the drawing means connected (a junction on the right); without a dot the two lines on the right must not meet;
  4. no symbol missing: every symbol has a dot on the right, and the numbers match.

To report an error, give the page and the symbol numbers, for example "RHR: 0 should not connect to 9".

## The 5 pages

### 1. AFW · Davis-Besse reliability analysis, p7 (Third Train Configuration)

25 symbols · 28 pipe lines · 69 text lines · 32 pipe segments · 7 T junctions · 0 open pipe ends · status: **verified**

![AFW · Davis-Besse reliability analysis, p7 (Third Train Configuration)](pseudo_AFW_%28DAVIS-BESSE%29%20AFW%20system%20Reliability%20Analysis_Final%20Report_p7_explained.png)

- Data: [drawing](images/pseudo_AFW_%28DAVIS-BESSE%29%20AFW%20system%20Reliability%20Analysis_Final%20Report_p7.png) · symbols [txt](symbols/pseudo_AFW_%28DAVIS-BESSE%29%20AFW%20system%20Reliability%20Analysis_Final%20Report_p7.txt) / [json](symbols/pseudo_AFW_%28DAVIS-BESSE%29%20AFW%20system%20Reliability%20Analysis_Final%20Report_p7.json) · [pipe lines](lines/pseudo_AFW_%28DAVIS-BESSE%29%20AFW%20system%20Reliability%20Analysis_Final%20Report_p7.json) · [text](text/pseudo_AFW_%28DAVIS-BESSE%29%20AFW%20system%20Reliability%20Analysis_Final%20Report_p7.json) · [connectivity](pseudo_AFW_%28DAVIS-BESSE%29%20AFW%20system%20Reliability%20Analysis_Final%20Report_p7.json)
- Images: [symbols + pipe lines](overlays/pseudo_AFW_%28DAVIS-BESSE%29%20AFW%20system%20Reliability%20Analysis_Final%20Report_p7_symbols_lines.jpg) · [text](overlays/pseudo_AFW_%28DAVIS-BESSE%29%20AFW%20system%20Reliability%20Analysis_Final%20Report_p7_text.jpg) · [connections on the drawing](pseudo_AFW_%28DAVIS-BESSE%29%20AFW%20system%20Reliability%20Analysis_Final%20Report_p7.jpg) · [explained connectivity](pseudo_AFW_%28DAVIS-BESSE%29%20AFW%20system%20Reliability%20Analysis_Final%20Report_p7_explained.png)

### 2. RHR · Residual heat removal system (Surry FSAR)

14 symbols · 48 pipe lines · 46 text lines · 40 pipe segments · 14 T junctions · 8 open pipe ends · status: **verified**

![RHR · Residual heat removal system (Surry FSAR)](pseudo_RHR_%28Surry%29%20FSAR_explained.png)

- Data: [drawing](images/pseudo_RHR_%28Surry%29%20FSAR.png) · symbols [txt](symbols/pseudo_RHR_%28Surry%29%20FSAR.txt) / [json](symbols/pseudo_RHR_%28Surry%29%20FSAR.json) · [pipe lines](lines/pseudo_RHR_%28Surry%29%20FSAR.json) · [text](text/pseudo_RHR_%28Surry%29%20FSAR.json) · [connectivity](pseudo_RHR_%28Surry%29%20FSAR.json)
- Images: [symbols + pipe lines](overlays/pseudo_RHR_%28Surry%29%20FSAR_symbols_lines.jpg) · [text](overlays/pseudo_RHR_%28Surry%29%20FSAR_text.jpg) · [connections on the drawing](pseudo_RHR_%28Surry%29%20FSAR.jpg) · [explained connectivity](pseudo_RHR_%28Surry%29%20FSAR_explained.png)

### 3. AFW · Auxiliary feedwater system (Seabrook Station, NUREG/CR-5500)

47 symbols · 62 pipe lines · 37 text lines · 56 pipe segments · 6 T junctions · 2 open pipe ends · status: **verified**

![AFW · Auxiliary feedwater system (Seabrook Station, NUREG/CR-5500)](pseudo_AFW%20%28Seabrook%29_explained.png)

- Data: [drawing](images/pseudo_AFW%20%28Seabrook%29.png) · symbols [txt](symbols/pseudo_AFW%20%28Seabrook%29.txt) / [json](symbols/pseudo_AFW%20%28Seabrook%29.json) · [pipe lines](lines/pseudo_AFW%20%28Seabrook%29.json) · [text](text/pseudo_AFW%20%28Seabrook%29.json) · [connectivity](pseudo_AFW%20%28Seabrook%29.json)
- Images: [symbols + pipe lines](overlays/pseudo_AFW%20%28Seabrook%29_symbols_lines.jpg) · [text](overlays/pseudo_AFW%20%28Seabrook%29_text.jpg) · [connections on the drawing](pseudo_AFW%20%28Seabrook%29.jpg) · [explained connectivity](pseudo_AFW%20%28Seabrook%29_explained.png)

### 4. AFW · Auxiliary feedwater system (San Onofre design review)

60 symbols · 80 pipe lines · 108 text lines · 98 pipe segments · 19 T junctions · 9 open pipe ends · status: **verified**

![AFW · Auxiliary feedwater system (San Onofre design review)](pseudo_AFW_%28San%20Onofre%29%20AFW%20Design%20Review%20and%20Reliability%20Evaluation_explained.png)

- Data: [drawing](images/pseudo_AFW_%28San%20Onofre%29%20AFW%20Design%20Review%20and%20Reliability%20Evaluation.png) · symbols [txt](symbols/pseudo_AFW_%28San%20Onofre%29%20AFW%20Design%20Review%20and%20Reliability%20Evaluation.txt) / [json](symbols/pseudo_AFW_%28San%20Onofre%29%20AFW%20Design%20Review%20and%20Reliability%20Evaluation.json) · [pipe lines](lines/pseudo_AFW_%28San%20Onofre%29%20AFW%20Design%20Review%20and%20Reliability%20Evaluation.json) · [text](text/pseudo_AFW_%28San%20Onofre%29%20AFW%20Design%20Review%20and%20Reliability%20Evaluation.json) · [connectivity](pseudo_AFW_%28San%20Onofre%29%20AFW%20Design%20Review%20and%20Reliability%20Evaluation.json)
- Images: [symbols + pipe lines](overlays/pseudo_AFW_%28San%20Onofre%29%20AFW%20Design%20Review%20and%20Reliability%20Evaluation_symbols_lines.jpg) · [text](overlays/pseudo_AFW_%28San%20Onofre%29%20AFW%20Design%20Review%20and%20Reliability%20Evaluation_text.jpg) · [connections on the drawing](pseudo_AFW_%28San%20Onofre%29%20AFW%20Design%20Review%20and%20Reliability%20Evaluation.jpg) · [explained connectivity](pseudo_AFW_%28San%20Onofre%29%20AFW%20Design%20Review%20and%20Reliability%20Evaluation_explained.png)

### 5. CSS · Containment spray system (APR1400 design document)

72 symbols · 105 pipe lines · 182 text lines · 94 pipe segments · 14 T junctions · 1 open pipe ends · status: **verified**

![CSS · Containment spray system (APR1400 design document)](pseudo_CSS%20%28APR1400%29_explained.png)

- Data: [drawing](images/pseudo_CSS%20%28APR1400%29.png) · symbols [txt](symbols/pseudo_CSS%20%28APR1400%29.txt) / [json](symbols/pseudo_CSS%20%28APR1400%29.json) · [pipe lines](lines/pseudo_CSS%20%28APR1400%29.json) · [text](text/pseudo_CSS%20%28APR1400%29.json) · [connectivity](pseudo_CSS%20%28APR1400%29.json)
- Images: [symbols + pipe lines](overlays/pseudo_CSS%20%28APR1400%29_symbols_lines.jpg) · [text](overlays/pseudo_CSS%20%28APR1400%29_text.jpg) · [connections on the drawing](pseudo_CSS%20%28APR1400%29.jpg) · [explained connectivity](pseudo_CSS%20%28APR1400%29_explained.png)
