# Symbol classes

A class for every one of the 3483 symbol boxes of [`../labels/`](../labels/) (same boxes, same ids), with the
operator of valves, dampers and pumps, the normal position of valves and dampers, the fault-tree component type, and
whether the component is typically modeled in a PRA.

**Status: in review.** The classes were labeled by eye page by page, checked class by class and by independent
reviewers, and then refined to the classes below. The 479 boxes that need a person's
decision are listed in `review.xlsx` ([Review](#review)); 52 rows are
marked `uncertain = yes` (drawn ambiguously, no legend on their page).

**Start here: [index.md](index.md)**, one overview image per page with every box, its id and its class.

## Classes

63 classes in five groups; 56 occur on the 65 pages, the others are kept so
the scheme stays complete. A component gets its own class when the AP1000 / APR1400 P&ID legends draw it with its own
symbol. `legend symbol` = no: the legends have no symbol for it (the class rests on the drawing text). `modeled in PRA`
= no: extracted, but typically not modeled in a PRA; these are screened out when the fault trees are built.

### Valves and dampers

| class | definition | legend symbol | modeled in PRA | boxes |
|---|---|---|---|---:|
| `VLV_GATE` | Gate valve / generic two-way valve (plain bowtie) | yes | yes | 1298 (25 uncertain) |
| `VLV_GLOBE` | Globe valve (bowtie with a dot at the waist) | yes | yes | 132 |
| `VLV_BUTTERFLY` | Butterfly valve | yes | yes | 92 |
| `VLV_BALL` | Ball valve | yes | yes | 0 |
| `VLV_PLUG` | Plug valve | yes | yes | 0 |
| `VLV_NEEDLE` | Needle valve | yes | yes | 0 |
| `VLV_DIAPHRAGM` | Diaphragm valve | yes | yes | 0 |
| `VLV_ANGLE` | Angle valve | yes | yes | 5 |
| `VLV_3WAY` | Three-way / four-way valve | yes | yes | 9 |
| `VLV_CHECK` | Check valve (swing, tilting disc, non-return) | yes | yes | 494 (4 uncertain) |
| `VLV_STOP_CHECK` | Stop-check valve (check valve with a stem, manual or powered) | yes | yes | 37 (4 uncertain) |
| `VLV_SAFETY_RELIEF` | Spring safety or relief valve (the symbol does not tell safety from relief; the tag does) | yes | yes | 39 |
| `VLV_PILOT_RELIEF` | Pilot-operated safety relief valve and its pilot (POSRV, PORV) | yes | yes | 12 |
| `VLV_VACUUM_BREAKER` | Vacuum breaker / vacuum relief valve | yes | yes | 0 |
| `VLV_REGULATOR` | Self-actuated pressure regulator | yes | yes | 2 |
| `DAMPER_MOTOR` | Motor-operated damper (any system) | yes | yes | 0 |
| `DAMPER_PNEUMATIC` | Pneumatic (air) operated damper (any system) | yes | yes | 27 |
| `DAMPER_EH` | Electro-hydraulic operated damper (any system) | yes | yes | 30 |
| `DAMPER_NO_OPERATOR` | Damper with no operator drawn (parallel blade, gravity, fire damper, ...) | yes | yes | 15 |

### Equipment

| class | definition | legend symbol | modeled in PRA | boxes |
|---|---|---|---|---:|
| `PUMP_CENTRIFUGAL` | Centrifugal pump (incl. horizontal centrifugal); the driver is the actuator | yes | yes | 113 |
| `PUMP_TURBINE_DRIVEN` | Turbine-driven pump | yes | yes | 22 |
| `PUMP_VERTICAL` | Vertical pump (canned / wet pit / sump) | yes | yes | 13 |
| `COMPRESSOR` | Compressor | yes | yes | 12 |
| `FAN` | Fan / blower | yes | yes | 48 |
| `TURBINE` | Turbine (incl. turbine-generator) | yes | yes | 22 |
| `TANK` | Tank / vessel / accumulator | yes | yes | 76 |
| `HX` | Heat exchanger / cooler / feedwater heater | yes | yes | 124 |
| `CONDENSER` | Condenser | yes | yes | 11 |
| `COIL` | Heating or cooling coil (HVAC) | yes | no | 40 |
| `CHILLER` | Chiller | yes | yes | 11 |
| `HUMIDIFIER` | Humidifier (HVAC) | yes | yes | 4 |
| `AHU` | Air handling unit / cubicle cooler housing | yes | yes | 9 |
| `STEAM_GEN` | Steam generator | yes | no | 73 |
| `REACTOR` | Reactor vessel | yes | no | 1 |
| `PRESSURIZER` | Pressurizer | no | no | 1 |
| `FILTER` | Filter (piping filter, prefilter, postfilter, HEPA filter) | yes | yes | 68 |
| `STRAINER` | Strainer (basket, Y-type, mesh) | yes | yes | 16 |
| `CARBON_ADSORBER` | Carbon (charcoal) adsorber | yes | yes | 12 |
| `MOISTURE_SEPARATOR` | Moisture separator | yes | yes | 12 |
| `AIR_SEPARATOR` | Air separator assembly | yes | yes | 1 |
| `DEMIN` | Demineralizer / ion exchanger | yes | yes | 10 |
| `DRYER` | Air dryer | yes | yes | 5 |
| `ORIFICE` | Orifice / flow element / venturi | yes | no | 103 (10 uncertain) |
| `HEATER` | Electric heater | yes | yes | 4 |
| `SPRAY` | Spray: spray nozzle, spray header, sparger | yes | yes | 21 |
| `SUMP` | Sump | no | yes | 1 |
| `TRAP` | Steam trap | yes | no | 0 |
| `ENGINE` | Diesel engine / prime mover drawn on its own | no | no | 1 |

### Instruments

| class | definition | legend symbol | modeled in PRA | boxes |
|---|---|---|---|---:|
| `INSTR` | Instrument (bubble or box with a tag) | yes | no | 78 |

### Electrical

| class | definition | legend symbol | modeled in PRA | boxes |
|---|---|---|---|---:|
| `BREAKER` | Circuit breaker | yes | yes | 181 |
| `TRANSFORMER` | Transformer | yes | yes | 37 |
| `GENERATOR` | Generator (incl. diesel generator) | yes | no | 10 |
| `MOTOR` | Motor drawn on its own | yes | yes | 17 |
| `BATTERY` | Battery | yes | yes | 3 |
| `CHARGER` | Battery charger / rectifier | yes | yes | 3 |
| `INVERTER` | Inverter / UPS | yes | yes | 6 |
| `SWITCH` | Switch / disconnect / transfer switch | yes | yes | 5 |
| `PANEL` | Distribution panel / MCC / switchgear box | yes | yes | 7 |

### Other

| class | definition | legend symbol | modeled in PRA | boxes |
|---|---|---|---|---:|
| `RUPTURE_DISC` | Rupture disc | yes | no | 2 |
| `PENETRATION` | Containment penetration (box on the containment boundary, e.g. 'PC') | yes | no | 18 |
| `OFFPAGE` | Off-page connector / system interface | yes | no | 52 |
| `BLOCK` | Labelled equipment block (named by its text) | no | no | 18 |
| `OTHER` | Other | no | no | 20 (9 uncertain) |

## Actuators

The operator of valves and dampers and the driver of pumps (`-` for all other classes). A valve keeps one class per body
symbol and its operator is the actuator, so the fault-tree type follows from class + actuator.

| actuator | definition | boxes |
|---|---|---:|
| `NONE` | No operator drawn and none named on the drawing (manual / check / self-acting valve, pump whose driver is not shown) | 1496 |
| `MOTOR` | Motor operator or motor driver, drawn (M) or named on the drawing (e.g. MOTOR DRIVEN, MD, an MOV / MDP tag) | 527 |
| `AIR` | Air / pneumatic operator (diaphragm or piston) | 209 |
| `SOLENOID` | Solenoid operator | 7 |
| `HYDRAULIC` | Hydraulic operator | 1 |
| `EH` | Electro-hydraulic operator | 60 |
| `UNSPEC` | Operator drawn, type neither identifiable from the symbol nor named on the drawing | 16 |
| `TURBINE` | Turbine driver (pumps), drawn or named on the drawing (e.g. TURBINE DRIVEN, TD) | 23 |
| `DIESEL` | Diesel / engine driver (pumps), drawn or named on the drawing | 8 |
| `-` | Not applicable (not a valve, damper or pump) | 1136 |

## Per-box columns

`symbol_classes.csv` (one row per box; `symbol_classes.json`: the same, grouped by page):

| column | content |
|---|---|
| `page`, `box_id` | the page and the box id (line number in `../labels/<page>.txt`) |
| `x0`, `y0`, `x1`, `y1` | the box in image pixels |
| `class`, `actuator` | see above |
| `normal_position` | valves with a stem and dampers: `open`, `closed`, `not shown` (a damper with no position symbol), `unknown` (half-filled body); `n/a` for check, relief and other self-acting valves and all other classes |
| `locked` | `LO` / `LC` where the drawing marks the valve locked open / closed |
| `ft_component_type` | fault-tree component type from class + actuator (`ft_component_types.csv`); empty where there is none yet |
| `pra_modeled` | `yes` / `no`, from the class |
| `uncertain` | `yes` where the symbol is drawn ambiguously and its page has no legend |
| `class_evidence`, `position_evidence` | where the class and the position come from: the labelers' note, the actuator label, the class label, or the automatic legend match |
| `review` | the review sheet(s) the box is on |
| `legend_entry` | the legend entry the box is tied to (`../legend_library/`) |
| `note` | the labelers' note |

Counts: normal position `open` 1269, `closed` 357, `not shown` 15, `unknown` 4; locked `LO` 155, `LC` 20; `pra_modeled = no` 417.

## How the class is decided

1. The labelers' note: "stop-check", "pilot" / "POSRV", "vacuum", "turbine-driven", "vertical", "strainer", "moisture
   separator", "air separator", "adsorber", ... decide the finer class.
2. The actuator label: dampers are classed by their operator (`DAMPER_MOTOR`, `DAMPER_PNEUMATIC`, `DAMPER_EH`,
   `DAMPER_NO_OPERATOR`), a pump with a turbine driver is `PUMP_TURBINE_DRIVEN`.
3. Only where the note says nothing: the legend entry the box was tied to (chosen automatically by image similarity
   among the legend symbols of its kind). Every box whose class rests on this alone is on a review sheet.

The normal position comes from the note ("normally closed", "solid body", "stippled", "LC", "LO", "normally open"), else
from the open / closed symbol of the legend entry (filled body = closed); a hollow body with no other evidence is `open`.
Every `closed` and `unknown` box is on a review sheet.

Fault-tree types: valves MOV, AOV, SOV, HOV (electro-hydraulic and hydraulic), XVM (manual), CKV (check and manual
stop-check), SRV (safety or relief: the symbol does not tell them apart, the tag does), PORV (pilot-operated), VBV (vacuum
breaker); pumps MDP, TDP, EDP (diesel), PMP (driver not shown); dampers MOD (motor), PND (pneumatic); FLT (filter), STR
(strainer). `DAMPER_EH`, `DAMPER_NO_OPERATOR`, `CARBON_ADSORBER`, `MOISTURE_SEPARATOR`, `AIR_SEPARATOR` and the classes
without a fault-tree role have no type.

## Review

`review.xlsx`: a README sheet, the class list, and one sheet per decision, every row with a crop of the box on its
drawing (red frame = the box), the proposal and empty columns for the decision:

| sheet | boxes | decision |
|---|---:|---|
| `fan_or_pump` | 48 | fans and pumps can share the circular symbol on HVAC drawings: fan, or which pump |
| `normal_position` | 361 | the normally closed valves and dampers, and the ones whose position could not be read |
| `auto_split` | 9 | the class rests only on the automatic legend match |
| `other` | 20 | boxes of class `OTHER`: a class where one fits |
| `uncertain` | 52 | the rows marked uncertain |

The decisions will be applied to `symbol_classes.csv` and the files of this folder rebuilt.

## Files

- `classes.csv` (class, definition, group, legend_png, pra_modeled, count), `classes.txt` (the class names in the order
  of the class index), `actuators.csv`, `ft_component_types.csv` (class|actuator -> fault-tree type).
- `symbol_classes.csv` / `.json`: one row per box (above).
- `labels/<page>.txt`: YOLO labels `class cx cy w h` with the class index of `classes.txt` (same boxes and order as
  `../labels/<page>.txt`).
- `overlay/<page>.jpg`, `index.md`: one overview per page.
- `review.xlsx`: the review workbook.
- `code/`: `derive_classes.py` (the classes, normal positions and the files above from the earlier class labels and the
  legend links), `update_legend_classes.py` (the classes in `../legend_library/derived/`), `render_overlays.py`
  (`overlay/`, `index.md`), `make_review.py` (`review.xlsx`).
