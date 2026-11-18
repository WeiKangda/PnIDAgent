"""Symbol classes v3: derive the finer classes and the normal position of every symbol box from the v2 labels.

Inputs (v2, unchanged): ../symbol_classes.csv (class, actuator, note per box), ../ft_component_types.csv,
../../legend_library/derived/{legend_index.json, symbol_legend_links.json} (the legend entry each box was tied to).
Outputs (this folder):
  classes_v3.csv            the v3 classes: definition, group, legend_png, pra_modeled, count, from_v2
  symbol_classes_v3.csv     one row per box: v3 class, actuator, normal position, locked, fault-tree type, PRA flag,
                            the evidence of the class and of the position, the review sheet the box is on, v2 class, note
  symbol_classes_v3.json    the same, grouped by page
  ft_component_types_v3.csv class + actuator -> fault-tree component type
  labels/<page>.txt         YOLO labels with the v3 class index (same boxes and order as ../../labels/<page>.txt)

The class of a box comes first from its note (written by the labelers), then from the actuator label, and only then
from the legend entry the box was tied to automatically (DINOv2 similarity among the candidates of its v2 class).
A box whose v3 class or normal position rests on the automatic legend match alone, or where the two disagree, is put
on a review sheet (make_review_v3.py).

  python relabel_v3.py
"""
import collections, csv, json, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
SC = os.path.dirname(HERE)                         # symbol_classes
V = os.path.dirname(SC)                            # verified_annotations_v2

# ---------------------------------------------------------------- the v3 classes
# (class, definition, group, from_v2); legend_png and pra_modeled are set below
CLASSES = [
    ("VLV_GATE", "Gate valve / generic two-way valve (plain bowtie)", "valve", "VLV_GATE"),
    ("VLV_GLOBE", "Globe valve (bowtie with a dot at the waist)", "valve", "VLV_GLOBE"),
    ("VLV_BUTTERFLY", "Butterfly valve", "valve", "VLV_BUTTERFLY"),
    ("VLV_BALL", "Ball valve", "valve", "VLV_BALL"),
    ("VLV_PLUG", "Plug valve", "valve", "VLV_PLUG"),
    ("VLV_NEEDLE", "Needle valve", "valve", "VLV_NEEDLE"),
    ("VLV_DIAPHRAGM", "Diaphragm valve", "valve", "VLV_DIAPHRAGM"),
    ("VLV_ANGLE", "Angle valve", "valve", "VLV_ANGLE"),
    ("VLV_3WAY", "Three-way / four-way valve", "valve", "VLV_3WAY"),
    ("VLV_CHECK", "Check valve (swing, tilting disc, non-return)", "valve", "VLV_CHECK"),
    ("VLV_STOP_CHECK", "Stop-check valve (check valve with a stem, manual or powered)", "valve", "VLV_CHECK"),
    ("VLV_SAFETY_RELIEF", "Spring safety or relief valve (the symbol does not tell safety from relief; the tag does)", "valve", "VLV_RELIEF"),
    ("VLV_PILOT_RELIEF", "Pilot-operated safety relief valve and its pilot (POSRV, PORV)", "valve", "VLV_RELIEF"),
    ("VLV_VACUUM_BREAKER", "Vacuum breaker / vacuum relief valve", "valve", "VLV_RELIEF"),
    ("VLV_REGULATOR", "Self-actuated pressure regulator", "valve", "VLV_REGULATOR"),
    ("DAMPER_MOTOR", "Motor-operated damper (any system)", "valve", "DAMPER"),
    ("DAMPER_PNEUMATIC", "Pneumatic (air) operated damper (any system)", "valve", "DAMPER"),
    ("DAMPER_EH", "Electro-hydraulic operated damper (any system)", "valve", "DAMPER"),
    ("DAMPER_NO_OPERATOR", "Damper with no operator drawn (parallel blade, gravity, fire damper, ...)", "valve", "DAMPER"),
    ("RUPTURE_DISC", "Rupture disc", "other", "RUPTURE_DISC"),
    ("PUMP_CENTRIFUGAL", "Centrifugal pump (incl. horizontal centrifugal); the driver is the actuator", "equipment", "PUMP"),
    ("PUMP_TURBINE_DRIVEN", "Turbine-driven pump", "equipment", "PUMP"),
    ("PUMP_VERTICAL", "Vertical pump (canned / wet pit / sump)", "equipment", "PUMP"),
    ("COMPRESSOR", "Compressor", "equipment", "COMPRESSOR"),
    ("FAN", "Fan / blower", "equipment", "FAN"),
    ("TURBINE", "Turbine (incl. turbine-generator)", "equipment", "TURBINE"),
    ("TANK", "Tank / vessel / accumulator", "equipment", "TANK"),
    ("HX", "Heat exchanger / cooler / feedwater heater", "equipment", "HX"),
    ("CONDENSER", "Condenser", "equipment", "CONDENSER"),
    ("COIL", "Heating or cooling coil (HVAC)", "equipment", "COIL"),
    ("CHILLER", "Chiller", "equipment", "CHILLER"),
    ("HUMIDIFIER", "Humidifier (HVAC)", "equipment", "HUMIDIFIER"),
    ("AHU", "Air handling unit / cubicle cooler housing", "equipment", "AHU"),
    ("STEAM_GEN", "Steam generator", "equipment", "STEAM_GEN"),
    ("REACTOR", "Reactor vessel", "equipment", "REACTOR"),
    ("PRESSURIZER", "Pressurizer", "equipment", "PRESSURIZER"),
    ("FILTER", "Filter (piping filter, prefilter, postfilter, HEPA filter)", "equipment", "FILTER"),
    ("STRAINER", "Strainer (basket, Y-type, mesh)", "equipment", "FILTER"),
    ("CARBON_ADSORBER", "Carbon (charcoal) adsorber", "equipment", "FILTER"),
    ("MOISTURE_SEPARATOR", "Moisture separator", "equipment", "FILTER"),
    ("AIR_SEPARATOR", "Air separator assembly", "equipment", "FILTER"),
    ("DEMIN", "Demineralizer / ion exchanger", "equipment", "DEMIN"),
    ("DRYER", "Air dryer", "equipment", "DRYER"),
    ("ORIFICE", "Orifice / flow element / venturi", "equipment", "ORIFICE"),
    ("HEATER", "Electric heater", "equipment", "HEATER"),
    ("SPRAY", "Spray: spray nozzle, spray header, sparger", "equipment", "SPARGER"),
    ("SUMP", "Sump", "equipment", "SUMP"),
    ("TRAP", "Steam trap", "equipment", "TRAP"),
    ("INSTR", "Instrument (bubble or box with a tag)", "instrument", "INSTR"),
    ("BREAKER", "Circuit breaker", "electrical", "BREAKER"),
    ("TRANSFORMER", "Transformer", "electrical", "TRANSFORMER"),
    ("GENERATOR", "Generator (incl. diesel generator)", "electrical", "GENERATOR"),
    ("MOTOR", "Motor drawn on its own", "electrical", "MOTOR"),
    ("ENGINE", "Diesel engine / prime mover drawn on its own", "equipment", "ENGINE"),
    ("BATTERY", "Battery", "electrical", "BATTERY"),
    ("CHARGER", "Battery charger / rectifier", "electrical", "CHARGER"),
    ("INVERTER", "Inverter / UPS", "electrical", "INVERTER"),
    ("SWITCH", "Switch / disconnect / transfer switch", "electrical", "SWITCH"),
    ("PANEL", "Distribution panel / MCC / switchgear box", "electrical", "PANEL"),
    ("PENETRATION", "Containment penetration (box on the containment boundary, e.g. 'PC')", "other", "PENETRATION"),
    ("OFFPAGE", "Off-page connector / system interface", "other", "OFFPAGE"),
    ("BLOCK", "Labelled equipment block (named by its text)", "other", "BLOCK"),
    ("OTHER", "Other", "other", "OTHER"),
]
# Components that are extracted but typically not modeled in a PRA (screened out later)
NOT_PRA = {"RUPTURE_DISC", "COIL", "STEAM_GEN", "REACTOR", "PRESSURIZER", "ORIFICE", "TRAP", "INSTR", "GENERATOR", "ENGINE",
           "PENETRATION", "OFFPAGE", "BLOCK", "OTHER"}
# Classes with no symbol in the AP1000 / APR1400 legends
NO_LEGEND = {"SUMP", "PRESSURIZER", "ENGINE", "BLOCK", "OTHER"}
# Valves and dampers whose normal position (open / closed) is shown by the symbol
HAS_POSITION = {"VLV_GATE", "VLV_GLOBE", "VLV_BUTTERFLY", "VLV_BALL", "VLV_PLUG", "VLV_NEEDLE", "VLV_DIAPHRAGM", "VLV_ANGLE",
                "VLV_3WAY", "VLV_STOP_CHECK", "DAMPER_MOTOR", "DAMPER_PNEUMATIC", "DAMPER_EH", "DAMPER_NO_OPERATOR"}

FT_V3 = {  # class|actuator -> fault-tree component type, for the classes that v3 adds or renames
    "VLV_STOP_CHECK|NONE": "CKV", "VLV_STOP_CHECK|MOTOR": "MOV", "VLV_STOP_CHECK|AIR": "AOV", "VLV_STOP_CHECK|SOLENOID": "SOV",
    "VLV_STOP_CHECK|EH": "HOV", "VLV_STOP_CHECK|HYDRAULIC": "HOV",
    "VLV_SAFETY_RELIEF|*": "SRV", "VLV_PILOT_RELIEF|*": "PORV", "VLV_VACUUM_BREAKER|*": "VBV",
    "DAMPER_MOTOR|*": "MOD", "DAMPER_PNEUMATIC|*": "PND",
    "PUMP_CENTRIFUGAL|MOTOR": "MDP", "PUMP_CENTRIFUGAL|DIESEL": "EDP", "PUMP_CENTRIFUGAL|TURBINE": "TDP", "PUMP_CENTRIFUGAL|NONE": "PMP",
    "PUMP_CENTRIFUGAL|UNSPEC": "PMP", "PUMP_TURBINE_DRIVEN|*": "TDP",
    "PUMP_VERTICAL|MOTOR": "MDP", "PUMP_VERTICAL|DIESEL": "EDP", "PUMP_VERTICAL|NONE": "PMP", "PUMP_VERTICAL|UNSPEC": "PMP",
    "FILTER|*": "FLT", "STRAINER|*": "STR",
}


def has(n, *words):
    return any(w in n for w in words)


def classify(r, leg_desc):
    """(v3 class, evidence, review sheet or '') of one box."""
    c, a, n, d = r["class"], r["actuator"], r["note"].lower(), (leg_desc or "").upper()
    if c == "VLV_CHECK":
        if has(n, "stop-check", "stop check"): return "VLV_STOP_CHECK", "note", ""
        if d.startswith(("STOP CHECK", "STOP-CHECK")): return "VLV_STOP_CHECK", "legend match (automatic)", "auto_split"
        return "VLV_CHECK", "note" if "check" in n or "non-return" in n else "v2 class", ""
    if c == "VLV_RELIEF":
        if has(n, "pilot", "posrv", "porv"): return "VLV_PILOT_RELIEF", "note", ""
        if has(n, "vacuum"): return "VLV_VACUUM_BREAKER", "note", ""
        if "PILOT" in d: return "VLV_PILOT_RELIEF", "legend match (automatic)", "auto_split"
        if "VACUUM" in d: return "VLV_VACUUM_BREAKER", "legend match (automatic)", "auto_split"
        return "VLV_SAFETY_RELIEF", "note" if has(n, "safety", "relief") else "v2 class", ""
    if c == "DAMPER":
        m = {"MOTOR": "DAMPER_MOTOR", "AIR": "DAMPER_PNEUMATIC", "EH": "DAMPER_EH", "NONE": "DAMPER_NO_OPERATOR"}
        if a in m: return m[a], "actuator label", ""
        return "DAMPER_NO_OPERATOR", "actuator label", "auto_split"
    if c == "PUMP":
        if a == "TURBINE" or has(n, "turbine-driven", "turbine driven"): return "PUMP_TURBINE_DRIVEN", "actuator label / note", ""
        if "vertical" in n: return "PUMP_VERTICAL", "note", ""
        if "VERTICAL" in d: return "PUMP_VERTICAL", "legend match (automatic)", "auto_split"
        return "PUMP_CENTRIFUGAL", "symbol (centrifugal casing)", ""
    if c == "FILTER":
        for w, k in (("moisture separator", "MOISTURE_SEPARATOR"), ("air separator", "AIR_SEPARATOR"), ("strainer", "STRAINER"),
                     ("adsorber", "CARBON_ADSORBER"), ("filter", "FILTER")):
            if w in n: return k, "note", ""
        for w, k in (("MOISTURE", "MOISTURE_SEPARATOR"), ("AIR SEPARATOR", "AIR_SEPARATOR"), ("STRAINER", "STRAINER"), ("ADSORBER", "CARBON_ADSORBER")):
            if w in d: return k, "legend match (automatic)", "auto_split"
        return "FILTER", "legend match (automatic)" if d else "v2 class", "auto_split"
    if c == "SPARGER": return "SPRAY", "v2 class", ""
    if c == "FAN": return "FAN", "v2 class", "fan_or_pump"
    if c == "OTHER": return "OTHER", "v2 class", "other"
    return c, "v2 class", ""


CLOSED_NOTE = re.compile(r"normally closed|solid body|(?<!half-)stippled|locked closed|\(lc\)|\blc\b|\bnc\b")
OPEN_NOTE = re.compile(r"normally open|locked open|\(lo\)|\blo\b|\bno\b(?=\))")


def position(cls, r, leg):
    """(normal position, evidence, locked, review) of a valve or damper; 'n/a' for the others."""
    if cls not in HAS_POSITION: return "n/a", "", "", ""
    n = r["note"].lower(); d = (leg.get("description") or "").upper(); var = leg.get("variant") or ""
    locked = "LC" if re.search(r"locked closed|\(lc\)|\blc\b", n) else "LO" if re.search(r"locked open|\(lo\)|\blo\b", n) else ""
    if "half-stippled" in n: return "unknown", "note (half-stippled body)", locked, "normal_position"
    by_note = "closed" if CLOSED_NOTE.search(n) else "open" if OPEN_NOTE.search(n) else ""
    by_leg = ("closed" if "NORMALLY CLOSED" in d else "open" if "NORMALLY OPEN" in d else "") if cls.startswith("DAMPER") else \
             {"closed": "closed", "open": "open"}.get(var, "")
    if by_note and by_leg and by_note != by_leg:
        return by_note, f"note says {by_note}, legend match says {by_leg}", locked, "normal_position"
    if by_note:
        return by_note, "note" + (" and legend match" if by_leg else ""), locked, "normal_position" if by_note == "closed" else ""
    if by_leg:
        return by_leg, "legend match (automatic)", locked, "normal_position" if by_leg == "closed" else ""
    if cls.startswith("DAMPER"): return "not shown", "", locked, ""
    return "open", "drawn open (hollow body)", locked, ""


def main():
    rows = list(csv.DictReader(open(f"{SC}/symbol_classes.csv", encoding="utf-8")))
    idx = {e["legend_id"]: e for e in json.load(open(f"{V}/legend_library/derived/legend_index.json"))}
    links = json.load(open(f"{V}/legend_library/derived/symbol_legend_links.json"))
    link = {(p, e["box_id"]): e for p, es in links.items() for e in es}
    ft2 = {r["class|actuator"]: r["ft_component_type"] for r in csv.DictReader(open(f"{SC}/ft_component_types.csv", encoding="utf-8"))}
    names = [c[0] for c in CLASSES]
    from_v2 = collections.defaultdict(list)
    for c in CLASSES: from_v2[c[3]].append(c[0])

    def ft(cls, act):
        for k in (f"{cls}|{act}", f"{cls}|*"):
            if k in FT_V3: return FT_V3[k]
        v2 = next(c[3] for c in CLASSES if c[0] == cls)
        if cls in ("FILTER", "STRAINER") or v2 not in (cls,): return ""
        return ft2.get(f"{v2}|{act}", ft2.get(f"{v2}|*", ""))

    out = []
    for r in rows:
        e = link.get((r["page"], int(r["box_id"])), {})
        leg = idx.get(e.get("legend_symbol"), {})
        cls, ev, review = classify(r, leg.get("description"))
        pos, pos_ev, locked, pos_review = position(cls, r, leg)
        sheets = [s for s in (review, pos_review) if s]
        if r["uncertain"].strip(): sheets.append("uncertain_v2")
        out.append({"page": r["page"], "box_id": r["box_id"], "x0": r["x0"], "y0": r["y0"], "x1": r["x1"], "y1": r["y1"],
                    "class": cls, "actuator": r["actuator"], "normal_position": pos, "locked": locked,
                    "ft_component_type": ft(cls, r["actuator"]), "pra_modeled": "no" if cls in NOT_PRA else "yes",
                    "class_evidence": ev, "position_evidence": pos_ev, "review": ";".join(sheets),
                    "legend_entry": f"{leg.get('pdf', '')[:7].strip()} #{leg.get('entry', '')} {leg.get('description', '')[:60]}" if leg else "",
                    "class_v2": r["class"], "uncertain_v2": r["uncertain"], "note": r["note"]})
    fields = list(out[0])
    with open(f"{HERE}/symbol_classes_v3.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n"); w.writeheader(); w.writerows(out)
    by_page = collections.defaultdict(list)
    for o in out: by_page[o["page"]].append({k: v for k, v in o.items() if k != "page"})
    json.dump(by_page, open(f"{HERE}/symbol_classes_v3.json", "w", encoding="utf-8"), indent=0, ensure_ascii=False)
    count = collections.Counter(o["class"] for o in out)
    with open(f"{HERE}/classes_v3.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n"); w.writerow(["class", "definition", "group", "legend_png", "pra_modeled", "count", "from_v2"])
        for c, d, g, v2 in CLASSES:
            w.writerow([c, d, g, "no" if c in NO_LEGEND else "yes", "no" if c in NOT_PRA else "yes", count.get(c, 0), v2])
    with open(f"{HERE}/ft_component_types_v3.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n"); w.writerow(["class|actuator", "ft_component_type"])
        for k, v in sorted({f"{o['class']}|{o['actuator']}": o["ft_component_type"] for o in out}.items()): w.writerow([k, v])
    os.makedirs(f"{HERE}/labels", exist_ok=True)
    for page, lst in by_page.items():
        yolo = [l for l in open(f"{V}/labels/{page}.txt") if l.strip()]
        assert len(yolo) == len(lst), page
        with open(f"{HERE}/labels/{page}.txt", "w") as f:
            for l, o in zip(yolo, sorted(lst, key=lambda o: int(o["box_id"]))):
                f.write(f"{names.index(o['class'])} " + " ".join(l.split()[1:]) + "\n")
    sheets = collections.Counter(s for o in out for s in o["review"].split(";") if s)
    print(f"{len(out)} boxes, {len(count)} classes in use; review sheets: {dict(sheets)}")
    print("normal position:", dict(collections.Counter(o["normal_position"] for o in out)))


if __name__ == "__main__":
    main()
