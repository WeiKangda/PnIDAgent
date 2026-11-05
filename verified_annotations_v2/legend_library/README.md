# P&ID legend library (AP1000 and APR1400)

The symbol legends of two P&ID drawing sets, extracted entry by entry: every legend symbol with its printed description and the
section (category) it is listed under. These legends were used as the reference for the symbol classes in
[`../symbol_classes/`](../symbol_classes/README.md): every symbol box there is tied to the legend entry it corresponds to.

- `original/`: the two legend PDFs and the extraction made from them (962 entries: AP1000 631, APR1400 331).
- `derived/`: what was built on top of it for the symbol classes (an index with the class of each legend symbol, the location of each
  symbol on the PDF page, snippets cut from the PDF with the printed description, and the legend entry of every symbol box).

## original/

| Path | Content |
|---|---|
| `AP1000 (PID Legend).pdf` | AP1000 P&ID legend, 6 pages (electrical, instrument and control, mechanical, piping, instrumentation). |
| `APR1400 (PID Legend).pdf` | APR1400 P&ID legend, 4 pages (equipment and valves, HVAC and fire protection, instrumentation). |
| `extracted/<pdf>_p<k>.json` | One file per PDF page (`p0` = page 1). `{"source", "page", "image", "entries": [...]}`; each entry: `n` (entry number on the page), `section` (the legend section = category), `sym_file` (symbol image), `symbol_text` (letters drawn inside the symbol, if any), `description` (the printed description), `notes`. |
| `extracted/sym/`, `extracted/sym600/` | The symbol images (`sym600` = cut from the 600 dpi rendering). `_a` / `_b` = the open / closed (or first / second) variant of one entry. |
| `extracted/PID_Legend_Symbols.xlsx` | The same entries as a workbook (README, AP1000, APR1400 sheets) with the symbol images. |
| `extracted/review.html` | The same entries as a web page (open it next to `sym600/`). |

Where the source PDF cuts a section title or a description at the page edge, the extraction says so ("truncated in source PDF",
"description cut off").

### Entries per page and section

| Legend | PDF page | File | Entries | Sections (entries) |
|---|---|---|---|---|
| AP1000 | 1 | `ap1000_p0.json` | 94 | 1. SWITCHING DEVICES — A. ABOVE 15KV (NOTES 1) (5); 1. SWITCHING DEVICES — B. 15KV & BELOW (NOTE 1) (5); 2. CIRCUIT PROTECTORS (4); 2. CIRCUIT PROTECTORS — C. MOTOR CONTROL-5KV (NOTES 1 & 6) (8); 3. OPERATING SYMBOLS (TO BE USED WITH SWITCHING DEVICES & CIRCUIT PROTECTORS) (5); 4. INSTRUMENT TRANSFORMERS (NOTE 2) (7); 5. TRANSFORMERS & REACTORS (9); 6. TRANSFORMER CONNECTIONS (8); 7. ROTATING MACHINES (NOTES 3) (6); 8. EXCITATION SYSTEMS FOR SYNCHRONOUS MACHINES (5); 9. MISCELLANEOUS POWER CIRCUIT DEVICES (32) |
| AP1000 | 2 | `ap1000_p1.json` | 168 | 10. METERS AND INSTRUMENTS (26); 11. METER & INSTRUMENT SWITCHES (4); 12. CONTROL SWITCHES & PUSHBUTTON STATIONS (12); 13. MISCELLANEOUS DEVICES (24); 14. RELAYS (32); A. CONTROL DEVICES (12); B. OPERATING COIL & COMPONENTS (14); CONTROL SYMBOLS - C. MISCELLANEOUS DEVICES (27); CONTROL SYMBOLS - C. MISCELLANEOUS DEVICES (GRAPHIC SYMBOLS) (12); ABBREVIATIONS (USED WITH POWER TRANSFORMER...) (5) |
| AP1000 | 3 | `ap1000_p2.json` | 82 | CONDUIT & CABLE TRAY DRAWING SYMBOLS (32); LIGHTING DRAWING SYMBOLS (20); COMMUNICATION DRAWING SYMBOLS - A. SPEAKERS (10); COMMUNICATION DRAWING SYMBOLS - B. HANDSETS (6); COMMUNICATION DRAWING SYMBOLS - C. TELEPHONE (6); GROUNDING DRAWING SYMBOLS - A. BELOW GRADE SYSTEMS (6); GROUNDING DRAWING SYMBOLS - B. ABOVE GRADE SYSTEMS (2) |
| AP1000 | 4 | `ap1000_p3.json` | 116 | VALVES (9) (13) (14) (27); DAMPERS (9) (13) (10); OPERATORS & ACCESSORIES (9) (13) (10); EQUIPMENT (37); VALVES AND DAMPERS (1); PLENUM LEGEND (title and right side truncated in source PDF) (10); SPECIFIC MECHANICAL ... (title and right side truncated in source PDF) (14); ARCHITECTURAL PLUMBING ITEMS AND MODULES (7) |
| AP1000 | 5 | `ap1000_p4.json` | 105 | PIPING AND DUCT COMPONENTS (49); LINES (22); LINE AND VALVE / DAMPER SPECIFICATIONS (24); CON... (RIGHTMOST COLUMN, SECTION TITLE TRUNCATED IN SOURCE PDF) (10) |
| AP1000 | 6 | `ap1000_p5.json` | 66 | INSTRUMENTS AND INSTRUMENTATION FUNCTIONS (3); ALARMS (6); INSTRUMENT FUNCTION CODES - MEASURED OR INITIATED VARIABLE (1ST LETTER) (21); INSTRUMENT FUNCTION CODES - FUNCTION CODE (2ND, 3RD, 4TH LETTER) (2) (14); INSTRUMENTATION NUMBERING (1); CONTROL INTERFACE BLOCKS (12); FLOW SENSORS AND OTHE[R ...] (SECTION TITLE TRUNCATED) (9) |
| APR1400 | 1 | `apr1400_p0.json` | 147 | EQUIPMENT SYMBOLS (34); VALVE SYMBOLS - LINE VALVE-BODIES (21); VALVE SYMBOLS (2); VALVE POSITION STATUS (TYPICAL FOR ALL VALVES) (5); SELF-ACTUATED VALVES (5); OPERATOR SYMBOLS (FOR USE WITH VALVE & DAMPER SYMBOL) (7); OTHER OPERATORS (4); OPERATOR FAILURE MODE ABBREVIATION (3); MISCELLANEOUS PIPING SPECIALTY SYMBOLS (30); PIPE LINE SYMBOLS (section header truncated in source: 'PIPE LIN...') (26); Untitled section (header truncated in source: 'F...') (9); Untitled section (header truncated in source: 'MISC...') (1) |
| APR1400 | 2 | `apr1400_p1.json` | 97 | HVAC EQUIPMENT / DAMPERS SYMBOLS (26); TEST CONNECTIONS (8); ABBREVIATIONS (HVAC ONLY) (8); FIRE PROTECTION SYMBOLS * (26); RIGHT COLUMN (HEADER TRUNCATED IN SOURCE PDF) (17); ABBREVIATIONS (FIRE PROTECTION — truncated in source PDF) (12) |
| APR1400 | 3 | `apr1400_p2.json` | 71 | INSTRUMENT & CONTROL DEVICE SYMBOLS (12); COMPUTER INPUT - OUTPUT SIGNAL NUMBER (1); LOCAL PANEL INPUT SYMBOL (3); CONTROL & INSTRUMENTATION FUNCTION SYMBOLS & CODES - FUNCTION SYMBOLS (2); CONTROL & INSTRUMENTATION FUNCTION SYMBOLS & CODES - FUNCTION SYMBOL CODES (2); INSTRUMENT TYPE CODES - FIRST LETTER: VARIABLE (MEASURED OR INITIATING) (24); INSTRUMENT TYPE CODES - SECOND OR SUCCEEDING LETTERS: INSTRUMENT FUNCTION (27) |
| APR1400 | 4 | `apr1400_p3.json` | 16 | TYPICAL INSTRUMENT & CONTROL SYMBOL EXAMPLES - MULTIPLE CONTROL SIGNALS FOR PNEUMATIC OPERATORS (2); TYPICAL INSTRUMENT & CONTROL SYMBOL EXAMPLES - SINGLE CONTROL SIGNALS FOR PNEUMATIC OPERATORS (1); TYPICAL INSTRUMENT & CONTROL SYMBOL EXAMPLES - VALVE / DAMPER POSITION SIGNALS (3); VALNE OPERATOR CONTROLS ( TYPICAL FOR OTHER CONTROL DEVICES ) (3); INSTRUMENT & CONTROL SIGNAL SYMBOLS (7) |

## derived/

| Path | Content |
|---|---|
| `legend_index.json` | One row per legend image (682): `legend_id`, `pdf`, `pdf_page`, `entry`, `variant` (open / closed), `section`, `description`, `image`, `class` (the symbol class of [`../symbol_classes/classes.csv`](../symbol_classes/classes.csv) this legend symbol stands for; empty for legend items that are not P&ID symbols, e.g. line types, codes, abbreviations; 304 rows have a class), `not_applicable_variant` (the "NA" cells of the valve tables, never used), and for the entries used by the symbol labels: `location_600dpi`, `snippet`, `snippet_symbol_only`, `printed_description_ocr`. |
| `snippets/<legend_id>.png` | Cut from the legend PDF page: the symbol (red frame) together with its printed description. `<legend_id>_sym.png`: the symbol alone. |
| `symbol_legend_links.json` | Per drawing page and symbol box (`box_id` as in `../labels/`): its `class`, `actuator`, and the `legend_id` of the legend symbol (`legend_symbol`) and of the operator (`legend_operator`) it corresponds to (`null` = no legend entry). |
| `code/` | The scripts that built these files (paths in them refer to the original workspace). |

How the derived files were made:

1. Each legend PDF page was rendered at 600 dpi; the extracted symbol images (`sym600`) were located on it by template matching
   (page orientation and position in `location_600dpi`; score 1.0 = exact match).
2. The printed descriptions were found on the page with OCR (PP-OCRv5) and matched to the extracted description text; the snippet shows
   the symbol and the description together as printed.
3. Every symbol box of the 65 drawings was tied to the legend entry of its class (the candidates of a class were fixed by hand; among
   them the most similar image was chosen with DINOv2 features), and powered valves also to the entry of their operator.
