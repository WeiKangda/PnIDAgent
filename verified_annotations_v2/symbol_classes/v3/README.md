# Symbol classes v3 (in review)

A finer class for every one of the 3483 symbol boxes, derived from the v2 labels in [`../`](../README.md), plus the normal
position of valves and dampers. The boxes, their ids and the v2 files are unchanged. This branch is for the symbol
classifier; the v2 classes stay the reference elsewhere.

**Status: proposal, in human review.** 479 boxes are on a review sheet in `review_v3.xlsx` (below); the rest follow the
rules below from the v2 labels and their notes.

## What changes from v2

| v2 class | v3 classes | how the v3 class is decided |
|---|---|---|
| `VLV_CHECK` | `VLV_CHECK`, `VLV_STOP_CHECK` | the note ("stop-check"), else the legend entry the box was tied to |
| `VLV_RELIEF` | `VLV_SAFETY_RELIEF`, `VLV_PILOT_RELIEF`, `VLV_VACUUM_BREAKER` | the note ("pilot", "POSRV", "vacuum"), else the legend entry |
| `DAMPER` | `DAMPER_MOTOR`, `DAMPER_PNEUMATIC`, `DAMPER_EH`, `DAMPER_NO_OPERATOR` | the actuator label (dampers are not limited to HVAC) |
| `PUMP` | `PUMP_CENTRIFUGAL`, `PUMP_TURBINE_DRIVEN`, `PUMP_VERTICAL` | the actuator label / note (turbine-driven, vertical), else the legend entry; the driver stays in `actuator` |
| `FILTER` | `FILTER`, `STRAINER`, `CARBON_ADSORBER`, `MOISTURE_SEPARATOR`, `AIR_SEPARATOR` | the note, else the legend entry |
| `SPARGER` | `SPRAY` (spray nozzle, spray header, sparger) | renamed |

Valves keep one class per body symbol of the legends; the operator stays in `actuator` (as in v2), so a fault-tree type
(MOV, AOV, SOV, HOV, CKV, XVM, SRV, PORV, VBV, ...) follows from class + actuator (`ft_component_types_v3.csv`).
Fault-tree types that change: `FILTER` is now FLT and `STRAINER` STR (v2 mapped all filters to STR); the pilot-operated
relief valves are PORV and the vacuum breakers VBV. `DAMPER_EH`, `DAMPER_NO_OPERATOR`, `CARBON_ADSORBER` and the two
separators have no type yet.

The legend entry of each box (`../../legend_library/derived/symbol_legend_links.json`) was chosen automatically by image
similarity among the entries of its v2 class, so it is used only where the note says nothing, and every box whose v3 class
or normal position rests on it alone is on a review sheet.

## New columns

- `normal_position` (valves with a stem, and dampers): `open`, `closed`, `not shown` (dampers with no position symbol),
  `unknown` (half-filled body), `n/a` (check, relief and other self-acting valves, and all other classes). From the note
  ("normally closed", "solid body", "stippled", "LC", "LO", "normally open"), else from the open / closed variant of the
  legend symbol (filled body = closed); a hollow body with no other evidence is `open`. Counts: open 1269, closed 357,
  not shown 15, unknown 4. Every `closed` and `unknown` box is on the review sheet `normal_position`.
- `locked`: `LO` (155) or `LC` (20) where the drawing marks the valve locked open / closed.
- `pra_modeled`: `no` for the components that are extracted but typically not modeled in a PRA (417 boxes; the classes
  marked `no` below). They are screened out later.
- `legend_png` (in `classes_v3.csv`): `no` for the classes that have no symbol in the AP1000 / APR1400 legends.
- `class_evidence`, `position_evidence`, `legend_entry`, `class_v2`: where the v3 values come from.

## Review

`review_v3.xlsx`: a README sheet, the class list, and one sheet per decision, every row with a crop of the box on its
drawing and empty columns for the decision:

| sheet | boxes | decision |
|---|---:|---|
| `fan_or_pump` | 48 | fans and pumps can share the circular symbol on HVAC drawings: fan or which pump |
| `normal_position` | 361 | the normally closed valves and dampers, and the 4 whose position could not be read |
| `auto_split` | 9 | the v3 class rests only on the automatic legend match (vertical pumps) |
| `other` | 20 | boxes of class `OTHER`: a class where one fits |
| `uncertain_v2` | 52 | the rows marked uncertain in v2 |

The decisions will be applied by a script and the files of this folder rebuilt.

## Classes

| class | definition | group | legend symbol | modeled in PRA | boxes | v2 class |
|---|---|---|---|---|---:|---|
| `VLV_GATE` | Gate valve / generic two-way valve (plain bowtie) | valve | yes | yes | 1298 | VLV_GATE |
| `VLV_GLOBE` | Globe valve (bowtie with a dot at the waist) | valve | yes | yes | 132 | VLV_GLOBE |
| `VLV_BUTTERFLY` | Butterfly valve | valve | yes | yes | 92 | VLV_BUTTERFLY |
| `VLV_BALL` | Ball valve | valve | yes | yes | 0 | VLV_BALL |
| `VLV_PLUG` | Plug valve | valve | yes | yes | 0 | VLV_PLUG |
| `VLV_NEEDLE` | Needle valve | valve | yes | yes | 0 | VLV_NEEDLE |
| `VLV_DIAPHRAGM` | Diaphragm valve | valve | yes | yes | 0 | VLV_DIAPHRAGM |
| `VLV_ANGLE` | Angle valve | valve | yes | yes | 5 | VLV_ANGLE |
| `VLV_3WAY` | Three-way / four-way valve | valve | yes | yes | 9 | VLV_3WAY |
| `VLV_CHECK` | Check valve (swing, tilting disc, non-return) | valve | yes | yes | 494 | VLV_CHECK |
| `VLV_STOP_CHECK` | Stop-check valve (check valve with a stem, manual or powered) | valve | yes | yes | 37 | VLV_CHECK |
| `VLV_SAFETY_RELIEF` | Spring safety or relief valve (the symbol does not tell safety from relief; the tag does) | valve | yes | yes | 39 | VLV_RELIEF |
| `VLV_PILOT_RELIEF` | Pilot-operated safety relief valve and its pilot (POSRV, PORV) | valve | yes | yes | 12 | VLV_RELIEF |
| `VLV_VACUUM_BREAKER` | Vacuum breaker / vacuum relief valve | valve | yes | yes | 0 | VLV_RELIEF |
| `VLV_REGULATOR` | Self-actuated pressure regulator | valve | yes | yes | 2 | VLV_REGULATOR |
| `DAMPER_MOTOR` | Motor-operated damper (any system) | valve | yes | yes | 0 | DAMPER |
| `DAMPER_PNEUMATIC` | Pneumatic (air) operated damper (any system) | valve | yes | yes | 27 | DAMPER |
| `DAMPER_EH` | Electro-hydraulic operated damper (any system) | valve | yes | yes | 30 | DAMPER |
| `DAMPER_NO_OPERATOR` | Damper with no operator drawn (parallel blade, gravity, fire damper, ...) | valve | yes | yes | 15 | DAMPER |
| `RUPTURE_DISC` | Rupture disc | other | yes | no | 2 | RUPTURE_DISC |
| `PUMP_CENTRIFUGAL` | Centrifugal pump (incl. horizontal centrifugal); the driver is the actuator | equipment | yes | yes | 113 | PUMP |
| `PUMP_TURBINE_DRIVEN` | Turbine-driven pump | equipment | yes | yes | 22 | PUMP |
| `PUMP_VERTICAL` | Vertical pump (canned / wet pit / sump) | equipment | yes | yes | 13 | PUMP |
| `COMPRESSOR` | Compressor | equipment | yes | yes | 12 | COMPRESSOR |
| `FAN` | Fan / blower | equipment | yes | yes | 48 | FAN |
| `TURBINE` | Turbine (incl. turbine-generator) | equipment | yes | yes | 22 | TURBINE |
| `TANK` | Tank / vessel / accumulator | equipment | yes | yes | 76 | TANK |
| `HX` | Heat exchanger / cooler / feedwater heater | equipment | yes | yes | 124 | HX |
| `CONDENSER` | Condenser | equipment | yes | yes | 11 | CONDENSER |
| `COIL` | Heating or cooling coil (HVAC) | equipment | yes | no | 40 | COIL |
| `CHILLER` | Chiller | equipment | yes | yes | 11 | CHILLER |
| `HUMIDIFIER` | Humidifier (HVAC) | equipment | yes | yes | 4 | HUMIDIFIER |
| `AHU` | Air handling unit / cubicle cooler housing | equipment | yes | yes | 9 | AHU |
| `STEAM_GEN` | Steam generator | equipment | yes | no | 73 | STEAM_GEN |
| `REACTOR` | Reactor vessel | equipment | yes | no | 1 | REACTOR |
| `PRESSURIZER` | Pressurizer | equipment | no | no | 1 | PRESSURIZER |
| `FILTER` | Filter (piping filter, prefilter, postfilter, HEPA filter) | equipment | yes | yes | 68 | FILTER |
| `STRAINER` | Strainer (basket, Y-type, mesh) | equipment | yes | yes | 16 | FILTER |
| `CARBON_ADSORBER` | Carbon (charcoal) adsorber | equipment | yes | yes | 12 | FILTER |
| `MOISTURE_SEPARATOR` | Moisture separator | equipment | yes | yes | 12 | FILTER |
| `AIR_SEPARATOR` | Air separator assembly | equipment | yes | yes | 1 | FILTER |
| `DEMIN` | Demineralizer / ion exchanger | equipment | yes | yes | 10 | DEMIN |
| `DRYER` | Air dryer | equipment | yes | yes | 5 | DRYER |
| `ORIFICE` | Orifice / flow element / venturi | equipment | yes | no | 103 | ORIFICE |
| `HEATER` | Electric heater | equipment | yes | yes | 4 | HEATER |
| `SPRAY` | Spray: spray nozzle, spray header, sparger | equipment | yes | yes | 21 | SPARGER |
| `SUMP` | Sump | equipment | no | yes | 1 | SUMP |
| `TRAP` | Steam trap | equipment | yes | no | 0 | TRAP |
| `INSTR` | Instrument (bubble or box with a tag) | instrument | yes | no | 78 | INSTR |
| `BREAKER` | Circuit breaker | electrical | yes | yes | 181 | BREAKER |
| `TRANSFORMER` | Transformer | electrical | yes | yes | 37 | TRANSFORMER |
| `GENERATOR` | Generator (incl. diesel generator) | electrical | yes | no | 10 | GENERATOR |
| `MOTOR` | Motor drawn on its own | electrical | yes | yes | 17 | MOTOR |
| `ENGINE` | Diesel engine / prime mover drawn on its own | equipment | no | no | 1 | ENGINE |
| `BATTERY` | Battery | electrical | yes | yes | 3 | BATTERY |
| `CHARGER` | Battery charger / rectifier | electrical | yes | yes | 3 | CHARGER |
| `INVERTER` | Inverter / UPS | electrical | yes | yes | 6 | INVERTER |
| `SWITCH` | Switch / disconnect / transfer switch | electrical | yes | yes | 5 | SWITCH |
| `PANEL` | Distribution panel / MCC / switchgear box | electrical | yes | yes | 7 | PANEL |
| `PENETRATION` | Containment penetration (box on the containment boundary, e.g. 'PC') | other | yes | no | 18 | PENETRATION |
| `OFFPAGE` | Off-page connector / system interface | other | yes | no | 52 | OFFPAGE |
| `BLOCK` | Labelled equipment block (named by its text) | other | no | no | 18 | BLOCK |
| `OTHER` | Other | other | no | no | 20 | OTHER |

## Files

- `relabel_v3.py`: derives everything below from the v2 files (`python relabel_v3.py`).
- `make_review_v3.py`: writes `review_v3.xlsx`.
- `classes_v3.csv`, `symbol_classes_v3.csv` / `.json`, `ft_component_types_v3.csv`, `labels/<page>.txt` (YOLO, v3
  class index, same boxes and order as `../../labels/`).
