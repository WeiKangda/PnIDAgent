# Review fixes, November 2026

An external review (17 November 2026) built a task on the SCS (APR1400) page and found places where the reference did not
follow the drawing or the labeling rules of [`../README.md`](../README.md) and the class definitions of
[`../symbol_classes/`](../symbol_classes/README.md). This folder fixes them on that page, and the same kinds of problems on
the other 64 pages, and records every change.

## What the review found on SCS (APR1400), and the fix

| Finding | Fix |
|---|---|
| The dotted flow pipes had no line: CCW supply and return of both heat exchangers, the external emergency injection line, and the dotted lines to and from other systems (IRWST, CSS, DVI, RCS hot leg, SI-179 discharge) | 15 dashed pipe lines added (`dashed` in `lines/`); control signal lines (SIAS / CSAS START) and the containment boundary stay unlabeled |
| Box 21 was the ASME code-class break mark "1\|2", its leader was a pipe line, and the topology connected it to valve 4 | box, leader line and their connections removed |
| The containment penetration (square on the boundary between SI-653 and SI-655) had no box, so the connections ran through it | boxed as `PENETRATION` (new box 21); the suction line ends at it on both sides (4 - 21 - 0, 15 - 21) |
| SI-314 (box 2) has the dot at its waist | `VLV_GATE/MOTOR` -> `VLV_GLOBE/MOTOR` |
| SCP-01 (box 9) has no driver drawn | `PUMP/UNSPEC` -> `PUMP/NONE` |
| (found when re-checking the page) The two heat exchanger boxes held the CCW arrowheads below the outline | boxes 18 and 19 tightened to the outline |

## The same problems on the other pages

| Kind | Search | Result |
|---|---|---|
| Dotted / dashed flow pipes | chains of dots and dashes not covered by a line, on all 65 pages, every candidate looked at on the drawing | CSS (APR1400): 16 dotted CCW lines at the four heat exchangers; SCS2 (APR1400): 16 at its four heat exchangers; Battery system (Surry): 3 dashed 480 V MCC buses. All other candidates are area, room, seismic and containment boundaries, division frames and signal lines. |
| Heat exchanger boxes holding their CCW lines | every equipment box compared with the outline it encloses, 79 candidates looked at | the 8 heat exchanger boxes of CSS and SCS2 (APR1400) tightened, like the 2 of SCS |
| Boxes that are not components | notes of all 3488 boxes, all `OTHER` boxes, the issues listed in `topology_verified/README.md` | Farley: flow arrow 68 removed (two steam supply stubs added in its place). Vogtle: interface bracket 71 removed (the pipe through it added); valve bodies 56 and 64 boxed again inside 67 and 70 removed; motor operator 72 merged into the box of butterfly valve 55. Control room HVAC (APR1400): duplicate damper box 20 removed. |
| Missing symbols | gaps between collinear pipe ends with no box (412 candidates), all 266 pipe pieces added in the connectivity review | Davis-Besse reliability analysis p1: check valve AF-15 boxed (52). SWS (Surry): one drawn pipe piece between two T junctions added. |
| Missing pipes | straight ink runs that no line covers, on all 65 pages (536 runs, each page looked at) | AFW (Davis-Besse): the three inlet pipes from the deaerator storage tanks, fire protection and service water; AFW (Haddam Neck): the two recirculation outlets |
| Valve class (dot at the waist = globe) | all 1,432 gate and globe valve boxes looked at, with a measurement of filled ink at the waist | no other gate valve has a dot at its waist; all 131 globe valves have one |
| Pump actuator | all 148 pump boxes | the 80 pumps labeled `UNSPEC` have no driver drawn and none named on their page -> `NONE`; the 46 `MOTOR` / `DIESEL` and 22 `TURBINE` pumps have their driver drawn or named. `actuators.csv` now says that a driver named on the drawing counts. |
| Line annotations vs connectivity edits | the `removed_lines` and `added_lines` of `topology_verified/manual_edits.json` | 179 lines that are not pipes (page frames, page edges, area and room outlines, containment outline, signal lines) or that the review replaced were taken out of `lines/`; 195 drawn pipe pieces that were only in `added_lines` were put into `lines/`. The rest of these edits stay (overshoots, rerouted pieces, bridges across label text). |

The labeling rules in `../README.md`, the labeling scheme in `../symbol_classes/README.md` and the definitions in
`../symbol_classes/actuators.csv` now state these cases (dotted flow pipes, marks are not symbols, penetrations are
symbols, equipment boxes end at the outline, a pump with no driver shown is `NONE`).

## Numbers

| | before | after |
|---|---:|---:|
| Symbol boxes | 3488 | 3483 |
| Pipe line segments | 6306 (0 dashed) | 6381 (50 dashed) |
| Connections (symbol pairs) | 7197 | 7223 |

Connections changed on 6 pages: SCS (APR1400) (the penetration splits 0 - 4 and 0 - 15), AFW (Farley), AFW (Vogtle),
Control room HVAC (APR1400) and Davis-Besse p1 (removed or added boxes), SWS (Surry) (the added pipe piece joins two
headers, 30 new pairs). On the other 46 pages the connections are unchanged.

## Box ids

A removed box shifts the ids of the boxes after it down by one (labels, symbol classes, legend links, topology and the
manual edits use the new ids):

- AFW (Farley): 69-72 -> 68-71.
- AFW (Vogtle): 57-63 -> 56-62, 65-70 -> 63-68, 73-75 -> 69-71.
- Control room HVAC (APR1400): 21-86 -> 20-85.
- SCS (APR1400): 21 is now the containment penetration.
- Added boxes get the next id: Davis-Besse p1 52 (check valve AF-15).

## Lock marks (19 November 2026)

A difficulty self-test of the RCS-POSRV (APR1400) page found that the labeling rules said a valve box holds its LO/LC lock
mark, while the annotations treat the mark as text. Of the 203 lock marks on 17 pages (LO, LC, LO.FL, LC.FL, L.O., L.C.),
25 lay inside a valve box drawn around them, 15 more were cut by a box edge that reached 7 to 14 px past the valve toward
the mark, and the other 163 lay outside the valve box or only touched it. The rules in [`../README.md`](../README.md) now
say that a lock mark is text and that the valve box is not extended to hold it, and the 40 boxes were moved in to the
valve (`relabel` with a new `box` in `fixes.json`):

| Page | Box drawn around the mark | Box edge into the mark |
|---|---|---|
| SI (APR1400) | 81-91 (motor valves with tag circles, LO.FL), 38 (LO) | 4 |
| CSS (APR1400) | 11, 13, 18, 54 (LO), 66 (LO.FL) | |
| Instrument Air (APR1400) | 33, 34, 36 | |
| SCS2 (APR1400) | 23, 28 | 25 |
| AFW (Vogtle) | | 37, 43, 46, 60, 62, 66 |
| AFW (Davis-Besse reliability analysis) p2 | 7, 9 | |
| AFW (Davis-Besse reliability analysis) p3 | | 15, 16, 26, 29 |
| AFW (Davis-Besse reliability analysis) p6 | | 18 |
| AFW (Davis-Besse reliability analysis) p7 | 4 | |
| AFW (Davis-Besse reliability analysis) p8 | | 16, 20 |

Box ids are those of the release; `fixes.json` uses the ids before the review, where Vogtle 60, 62 and 66 are 61, 63 and 68.
Only the edge on the mark's side moved, to the valve's ink plus the margin the box has on the opposite side (on
Davis-Besse p2 the box was also wide on the other side, and both side edges moved to the valve). The mark keeps its text
box. After the fix no valve box holds a lock mark: 119 marks lie clear of every box, and 84 touch or overlap a valve box
because they are written against the valve, 2 of them by half or more (written inside the outline of a Davis-Besse
valve). On CSS, the pipe that left box 66 through its old edge now starts at the new edge. Classes, actuators and the
connected symbol pairs are unchanged, and both checks below give the same result as before. The rule for the pilot box
(`OTHER`) drawn above each POSRV of RCS-POSRV (APR1400), which the annotations already follow, was added at the same time.

## Files

- `fixes.json`: every change, per page, with the reason: `remove_boxes`, `add_boxes`, `relabel` (class, actuator, a
  tightened `box`), `remove_lines`, `move_lines`, `add_solid`, `add_dashed`, `line_sync` (lines moved between
  `lines/` and `topology_verified/manual_edits.json`), `manual_edits` (connectivity edits), `notes`, `status_note`.
- `apply_fixes.py`: applies `fixes.json` to the files of the release before the review (git commit `1d42d1d`) and rewrites
  `labels/`, `lines/`, `symbol_classes/`, `legend_library/derived/symbol_legend_links.json`, `topology_verified/`
  (symbols, lines, manual edits, shaft links, connectivity) and `stats.json`. Running it again gives the same files.
- `render_images.py`: redraws `overlay/`, `topology_verified/overlays/*_symbols_lines.jpg`, `topology_verified/<page>.jpg`,
  `topology_verified/<page>_explained.png` and `symbol_classes/overlay/` for the changed pages. The evidence tiles in
  `symbol_classes/evidence/` and the class reference images were not redrawn; where they differ, `symbol_classes.csv` holds
  the labels.

`topology_verified/code/build_connectivity.py` now reads the `dashed` lines as pipes, and has one new manual edit,
`no_join` (an elongated symbol that is not on a bus: the pipe parts on its two long sides are not joined through it).

## Checks

- `python ../topology_verified/code/build_connectivity.py --check`: all 65 connectivity files are identical to a rebuild.
- `python ../topology_verified/code/check_connectivity.py`: the flags are those before the review, except: AFW (Vogtle)
  now passes (the isolated operator box is merged), and the short dotted CCW legs at the heat exchangers of CSS and SCS2
  (APR1400) end next to the box and carry no connection (8 and 24 flags, listed in the pages' `notes`).
- Every changed page was looked at in the redrawn images.

## ALE reviews of the single-page tasks (20 November 2026)

The automatic reviews of the ALE tasks built on AFW (Vogtle) and on Davis-Besse reliability analysis p8 and p9 found
reference annotations that did not follow the labeling rules or the class definitions, and on p9 two conventions that
the rules did not state. Fixed in `fixes.json` (ids before the review, as everywhere in that file) and in the labeling
rules:

| Page | Finding | Fix |
|---|---|---|
| AFW (Vogtle) | pumps 002 and 003 labeled `MOTOR`, with no driver drawn or named on the page | `PUMP/NONE` (`actuators.csv`) |
| AFW (Vogtle) | the four TO STEAM GENERATOR n INTAKE boxes and the two FROM STEAM GENERATOR arrows were not boxed, although `OFFPAGE` is an off-page connector / system interface | boxed as `OFFPAGE` (new boxes 72-77); the six pipes that ended at them now connect to them |
| Davis-Besse reliability analysis p8 | the MAKEUP FLOW connector box (34) held the lettering above the pentagon | box tight to the pentagon |
| Davis-Besse reliability analysis p8 | the flow elements on the lines of FT4521 and FT4522 were not boxed, although `ORIFICE` includes flow elements | boxed as `ORIFICE` (new boxes 54, 55); the pipe stops at them on both sides |
| Davis-Besse reliability analysis p8 | the plain crossing at (1184,1374), drawn without a dot, was set connected | the crossing decision removed: not connected, as the rule says |
| Davis-Besse reliability analysis p9 | the same kind of crossing, not raised by a review but found when its self-test was checked: the plain crosses at (2257,1258) and (2484,1260) of the FW459 / FW460 / FW448 valve station were set connected | the two crossing decisions removed: not connected, as the rule says |
| Davis-Besse reliability analysis p9 | 11 valve boxes reached past the valve into the tag next to it (FW-32, FW142, FW154, FW162, FW-102, FW440, SV-1, CD169 and CD170, SP7A, SP7B) or into the lock mark L.O.; check valve 21 and connector 23 also held a few pixels of the pipe | the edges moved in to the ink (boxes 1, 4, 6, 10, 13, 19, 20, 21, 23, 25, 32, 33); the two pipes extended to the new edges |
| Davis-Besse reliability analysis p9 | the two valves of the legend (NORMALLY OPEN, NORMALLY CLOSED) are not boxed and the handwritten initials in the revision block are transcribed `###`, two conventions the labeling rules did not state | two lines added to the labeling rules (`../README.md`) |

Numbers: symbol boxes 3483 -> 3491, pipe line segments 6381 -> 6383, connections (symbol pairs) 7223 -> 7223 (Vogtle
100 -> 110; p8 stays at 75: the two flow elements replace 4 pairs by 6, and the crossing removes 2; p9 58 -> 48). The
connectivity checks give no flag on the three pages.

