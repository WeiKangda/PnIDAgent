# Human-verified annotations of 65 nuclear P&ID pages (v3)

65 real P&ID pages of nuclear power plants (APR1400 design control document, Surry FSAR, NUREG/CR-4550 system sketches
of Surry, auxiliary feedwater drawings of US plants, NuScale, GE-Hitachi ABWR), each annotated by hand and verified in
five layers:

- **symbol boxes**, tight around every symbol;
- **symbol classes**: a class for every box, one per component that has its own symbol in the AP1000 / APR1400 P&ID
  legends, the operator of valves, dampers and pumps, the normal position of valves and dampers, and whether the
  component is typically modeled in a PRA;
- **pipe lines** (solid, and the flow pipes drawn dashed or dotted);
- **text lines** with their transcription and reading direction;
- **connectivity**: which symbols are joined by pipes.

**Status:** boxes, lines, text and connectivity are verified. The symbol classes are in review: the 479 boxes that need
a person's decision are listed in [`symbol_classes/review.xlsx`](symbol_classes/README.md#review).

## Numbers

| | |
|---|---:|
| Pages | 65 |
| Symbol boxes | 3493 |
| Symbol classes | 63 (56 occur on the pages) |
| Valves and dampers normally closed | 357 |
| Symbols of components typically modeled in a PRA | 3066 |
| Pipe line segments | 6388 (50 dashed) |
| Text lines | 6062 |
| Connections (pairs of symbols joined by pipes) | 7237 |

`stats.json` has these numbers for every page.

## Symbol classes

| group | classes (boxes) |
|---|---|
| Valves and dampers | `VLV_GATE` 1298, `VLV_GLOBE` 132, `VLV_BUTTERFLY` 92, `VLV_BALL` 0, `VLV_PLUG` 0, `VLV_NEEDLE` 0, `VLV_DIAPHRAGM` 0, `VLV_ANGLE` 5, `VLV_3WAY` 9, `VLV_CHECK` 494, `VLV_STOP_CHECK` 37, `VLV_SAFETY_RELIEF` 39, `VLV_PILOT_RELIEF` 12, `VLV_VACUUM_BREAKER` 0, `VLV_REGULATOR` 2, `DAMPER_MOTOR` 0, `DAMPER_PNEUMATIC` 27, `DAMPER_EH` 30, `DAMPER_NO_OPERATOR` 15 |
| Equipment | `PUMP_CENTRIFUGAL` 113, `PUMP_TURBINE_DRIVEN` 22, `PUMP_VERTICAL` 13, `COMPRESSOR` 12, `FAN` 48, `TURBINE` 22, `TANK` 76, `HX` 124, `CONDENSER` 11, `COIL` 40, `CHILLER` 11, `HUMIDIFIER` 4, `AHU` 9, `STEAM_GEN` 73, `REACTOR` 1, `PRESSURIZER` 1, `FILTER` 68, `STRAINER` 16, `CARBON_ADSORBER` 12, `MOISTURE_SEPARATOR` 12, `AIR_SEPARATOR` 1, `DEMIN` 10, `DRYER` 5, `ORIFICE` 107, `HEATER` 4, `SPRAY` 21, `SUMP` 1, `TRAP` 0, `ENGINE` 1 |
| Instruments | `INSTR` 78 |
| Electrical | `BREAKER` 181, `TRANSFORMER` 37, `GENERATOR` 10, `MOTOR` 17, `BATTERY` 3, `CHARGER` 3, `INVERTER` 6, `SWITCH` 5, `PANEL` 7 |
| Other | `RUPTURE_DISC` 2, `PENETRATION` 18, `OFFPAGE` 58, `BLOCK` 18, `OTHER` 20 |

Valves keep one class per body symbol, with the operator in `actuator` (`MOTOR`, `AIR`, `SOLENOID`, `HYDRAULIC`, `EH`,
`NONE`, `UNSPEC`), so a fault-tree component type (MOV, AOV, SOV, HOV, CKV, XVM, SRV, PORV, VBV, ...) follows from class +
actuator. Dampers are classed by their operator and pumps by their type (the driver of a pump is its actuator). Details,
definitions and the per-box columns: [`symbol_classes/`](symbol_classes/README.md).

## Files

- `labels/<page>.txt`: symbol boxes, YOLO format `0 cx cy w h` (normalized, single class); the line number is the box id.
- `symbol_classes/`: the class of every box (`symbol_classes.csv`, and YOLO labels with the class index in
  `symbol_classes/labels/`), the class list, the review workbook.
- `lines/<page>.json`: pipes, `{"solid": [[x1,y1,x2,y2],...], "dashed": [...]}` in image pixels; `dashed` holds the flow
  pipes drawn dashed or dotted.
- `text/<page>.json`: text lines, `{"name", "width", "height", "items": [{"id", "box": [x0,y0,x1,y1], "text", "rot"}]}`,
  boxes in image pixels, `rot` = reading direction (`0` upright, `cw` the text runs bottom to top, `ccw` top to bottom,
  `180` upside down).
- `topology_verified/`: the drawings (`images/<page>.png`), the connectivity of every page and the code that rebuilds and
  checks it.
- `legend_library/`: the AP1000 and APR1400 P&ID legend PDFs, every legend entry extracted from them, and the legend entry
  each symbol box is tied to.
- `overlay/<page>.jpg`: symbols (red) and pipes (green) drawn on the page; `text_overlay/<page>.jpg`: text boxes (blue).
- `stats.json`: per page, the numbers above.

## Labeling rules

- One box per symbol, tight around the symbol's ink.
- A valve together with its actuator (M, solenoid, diaphragm) is one box. Lock marks next to a valve (LO, LC, LO.FL,
  LC.FL, L.O., L.C.) are text, not part of the symbol: they get a text box, and the valve box is not extended to hold them.
- A valve and the numbered tag circle drawn on a leader from it are one box (e.g. SCS2 (APR1400), RCS-POSRV (APR1400)).
  Instrument circles that stand on their own get their own box.
- The pilot of a pilot-operated safety relief valve, the square marked P drawn above each POSRV together with the three
  short legs under it, is a symbol of its own (`OTHER`), not part of the valve box; the line from it to the valve and the
  pilot lines that continue below its legs are pipes (RCS-POSRV (APR1400)).
- A pump and its variable frequency drive (VFD) are one box (Power conversion system (NuScale)).
- Labelled equipment blocks (heat exchangers, coolers, chillers drawn as a rectangle with a name) are boxed
  (CCWS (APR1400)).
- A dashed frame drawn around several valves is not a symbol: each valve inside gets its own box (AFW (Haddam Neck)).
- Symbols drawn in a legend to explain the notation (e.g. the two valves next to the words NORMALLY OPEN and NORMALLY
  CLOSED on the Davis-Besse reliability analysis pages) are not symbols of the system: they get no box and no
  connection. The words of the legend are text.
- Marks drawn on a pipe are not symbols: code-class break marks (e.g. "1|2" with its leader), flow arrows, interface
  brackets. A containment penetration drawn as a box on the containment boundary **is** a symbol (`PENETRATION`).
- An equipment box ends at the outline of the equipment: the arrows, labels and CCW lines drawn next to a heat
  exchanger are outside its box.
- The class of a symbol is the most specific class whose legend symbol it matches: a stop-check valve is
  `VLV_STOP_CHECK`, not `VLV_CHECK`; a strainer is `STRAINER`, not `FILTER`. A pump whose driver is neither drawn nor named
  on the drawing has actuator `NONE`.
- The normal position of a valve or damper is the one the drawing shows (filled body = closed, or a written NC / LC / LO).
- Lines follow pipe centerlines and stop at the edge of a symbol box. A box that holds 3 or more other boxes (a tank with
  spargers inside) is not cut.
- Dashed, dash-dot and dotted lines are not labeled when they are signal lines (e.g. from an ESF-SIAS START box or a
  control panel) or area / room / seismic / containment boundaries. Pipes that carry flow but are drawn dashed, dash-dot
  or dotted **are** labeled: the ESW emergency supply on AFW (INL system study), the pilot lines of the four POSRVs on
  RCS-POSRV (APR1400), the dotted CCW supply and return lines of the heat exchangers and the dotted lines to and from
  other systems (IRWST, CSS, DVI, external emergency injection) on SCS, SCS2 and CSS (APR1400).
- No line is added where the drawing shows none (e.g. between the filter, coils and fan inside an HVAC unit housing).
- Handwriting that cannot be read (e.g. initials in a revision block) gets a text box like any other text line; its
  transcription is `###`.

## How it was made

1. Symbol boxes. Inside each box, ink was taken from the page (dark-pixel threshold relative to the background).
   Straight thin strokes crossing the box edge were removed as pipe stubs when a labeled pipe meets them. Pieces mostly
   outside the box, dashes of a dashed line passing by, and text inside the box but outside the symbol were removed too.
   The box grows only where a stroke of the symbol continues past the edge.
2. Lines. Each line was snapped to the pipe centerline and extended to the box. It was then cut at the boxes, and
   corners and T-junctions were re-joined. Short pipe pieces were added between boxes and lines (only along ink, never
   through text), and small gaps along ink were closed.
3. Text boxes. Each box was fitted to its letters; outlines around the letters (instrument circles, frames) and dashed
   signal lines were removed. A box was kept wider when PP-OCRv5 read the tightened crop worse.
4. Automatic checks over all pages: lines inside symbol boxes, duplicate lines, lines off the drawn pipe, symbol boxes
   with empty margins, text boxes cutting letters, symbols that a detector found but no box covered, dashed and dotted
   runs that no line covers, straight pipe runs that no line covers, gaps between collinear pipe ends; every finding was
   fixed or reviewed on the drawing.
5. Human verification of every page in the browser editor (symbols and lines, then text, then connectivity).
6. Symbol classes: labeled by eye page by page, checked class by class and by independent reviewers, then refined to
   the classes above from the labelers' notes, the actuator labels and the legend entries (see
   [`symbol_classes/`](symbol_classes/README.md)).

## Connectivity

[`topology_verified/`](topology_verified/README.md) has the pipe connectivity of all 65 pages, built from these
annotations and verified: automatic checks against the symbols and pipe lines, a visual inspection of every connection,
and an independent inspection of 50 of the pages, whose findings were fixed and re-checked. Remaining flags of the
automatic checks are explained per page.

## Legend library

[`legend_library/`](legend_library/README.md) has the AP1000 and APR1400 P&ID legend PDFs, every legend entry extracted
from them (symbol image, section, printed description, and the symbol class it stands for), and the legend entry each
symbol box of the 65 pages is tied to.
