# Symbol classes (pre-labels, human review in progress)

**Status: pre-labels.** Every one of the 3488 symbol boxes of [`../labels/`](../labels/) has a class and an actuator,
labeled by eye page by page and checked a second time class by class. A human review of every row is in progress;
until it is done these labels are not verified. Rows marked `uncertain = yes` (108) are drawn ambiguously and have no
legend on their page; they are reviewed first.

## Files

- `overlay/<page>.jpg`: check image of each page. Every box is drawn on the drawing with its box id and a short class tag
  (colour = class group); the panel on the right lists every id with its class, actuator and fault-tree type. `*` = uncertain.
- `symbol_classes.csv`: one row per symbol box. Columns: `page`, `box_id` (line index in `../labels/<page>.txt`),
  `x0,y0,x1,y1` (box in image pixels), `class`, `actuator`, `ft_component_type`, `uncertain`, `note`.
- `symbol_classes.json`: the same, grouped by page.
- `classes.csv`: the 51 classes with definitions and counts. `actuators.csv`: the 10 actuator values.
- `ft_component_types.csv`: how `class` + `actuator` map to a fault-tree component type.

## Labeling scheme

Each symbol gets two labels:

1. **class**: the valve body or the kind of equipment (e.g. `VLV_GATE`, `VLV_GLOBE`, `VLV_CHECK`, `VLV_BUTTERFLY`,
   `VLV_RELIEF`, `DAMPER`, `PUMP`, `HX`, `TANK`, `FILTER`, `ORIFICE`, `INSTR`, `BREAKER`, `TRANSFORMER`, `OFFPAGE`, ...).
   Definitions follow the AP1000 and APR1400 P&ID legends and the legends printed on the drawings.
2. **actuator** (valves, dampers, pumps): `MOTOR` (M operator), `AIR` (diaphragm, dome or piston), `SOLENOID` (S),
   `HYDRAULIC`, `EH` (electro-hydraulic, E/H), `NONE` (no operator drawn: manual, check, self-acting), `UNSPEC`
   (an operator is drawn but its type cannot be identified), `TURBINE` / `DIESEL` / `MOTOR` (pump drivers); `-` = not applicable.

A valve and its actuator, lock mark and tag circle are one box (see the labeling rules in [`../README.md`](../README.md)),
so the actuator is a property of the valve box. A solid (black) body means normally closed; this is written in `note`.

`ft_component_type` is derived from class + actuator with the component codes of the Generic PWR SAPHIRE model
basic-event list: `MOV`, `AOV`, `SOV`, `HOV`, `XVM` (manual valve), `CKV`, `SRV`, `MDP`, `TDP`, `EDP`, `PMP`, `HTX`, `TNK`,
`SMP`, `STR`, `FAN`, `CHL`, `ORF`, `CRB`, `TFM`, `DGN`, `BAT`, `BCH`, `INV`, `PND`, `MOD`. It is empty where no standard code
applies (steam generators, instruments, off-page connectors, ...).

## Sources used to resolve ambiguous symbols

- AP1000 and APR1400 P&ID legend sheets (962 legend entries).
- Legends printed on individual drawings (Surry FSAR figures, San Onofre, Davis-Besse, NuScale).
- NUREG/CR-4550 Vol. 3 (Surry) basic events, e.g. LPI-MOV-PG-1864A, for valve operator types on the Surry NUREG sketches.
- INL AFW system study and the Davis-Besse AFW reliability analysis, for valve operator types on their figures.
- NUREG/CR-5500 Vol. 1 (AFW reliability study), Table D-3, for the pump drivers of the AFW reference plants.
