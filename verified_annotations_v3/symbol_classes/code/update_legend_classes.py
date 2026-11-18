"""Put the symbol classes of this folder into the legend library:

  ../../legend_library/derived/symbol_legend_links.json   the class and actuator of every box (from ../symbol_classes.csv)
  ../../legend_library/derived/legend_index.json          the class each legend symbol stands for

The legend symbols of the classes that are split by symbol (check, relief, pump, filter, spray) get the finer class from
their printed description; a damper symbol gets the class of the dampers tied to it (else the operator its description
names). Legend symbols of kinds that do not occur on the 65 pages and have no class of their own (gear pump, cyclone separator, ...)
get no class.

  python update_legend_classes.py
"""
import collections, csv, json, os

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.dirname(HERE); V = os.path.dirname(OUT)
D = f"{V}/legend_library/derived"
SPLIT = {"VLV_CHECK", "VLV_RELIEF", "DAMPER", "PUMP", "FILTER", "SPARGER"}


def legend_class(c, d):
    d = d.upper()
    if c not in SPLIT or c is None: return c
    if c == "VLV_CHECK": return "VLV_STOP_CHECK" if "STOP" in d else "VLV_CHECK"
    if c == "VLV_RELIEF":
        if "RUPTURE" in d: return "RUPTURE_DISC"
        if "PILOT" in d: return "VLV_PILOT_RELIEF"
        if "VACUUM" in d: return "VLV_VACUUM_BREAKER"
        return "VLV_SAFETY_RELIEF"
    if c == "DAMPER":
        if "PNEUMATIC" in d: return "DAMPER_PNEUMATIC"
        if "ELECTRO-HYDRAULIC" in d: return "DAMPER_EH"
        if "MOTOR" in d: return "DAMPER_MOTOR"
        return "DAMPER_NO_OPERATOR"
    if c == "PUMP":
        if "TURBINE DRIVEN" in d: return "PUMP_TURBINE_DRIVEN"
        if "VERTICAL" in d or "SUMP" in d or "DEEP WELL" in d or "SUBMERSIBLE" in d: return "PUMP_VERTICAL"
        if "CENTRIFUGAL" in d or "CANNED MOTOR" in d: return "PUMP_CENTRIFUGAL"
        return None
    if c == "FILTER":
        if "MOISTURE" in d: return "MOISTURE_SEPARATOR"
        if "AIR SEPARATOR" in d: return "AIR_SEPARATOR"
        if "STRAINER" in d: return "STRAINER"
        if "SEPARATOR" in d: return None
        return "FILTER"
    if c == "SPARGER": return "SPRAY"


def main():
    rows = {(r["page"], int(r["box_id"])): r for r in csv.DictReader(open(f"{OUT}/symbol_classes.csv", encoding="utf-8"))}
    links = json.load(open(f"{D}/symbol_legend_links.json"))
    for p, es in links.items():
        for e in es:
            r = rows[(p, e["box_id"])]; e["class"], e["actuator"] = r["class"], r["actuator"]
    json.dump(links, open(f"{D}/symbol_legend_links.json", "w"), indent=1, ensure_ascii=False)
    idx = json.load(open(f"{D}/legend_index.json"))
    linked = collections.defaultdict(collections.Counter)      # legend symbol -> classes of the boxes tied to it
    for es in links.values():
        for e in es:
            if e.get("legend_symbol") is not None: linked[e["legend_symbol"]][e["class"]] += 1
    for e in idx:
        desc = (e.get("description") or "").upper()
        if "DAMPER" in desc and e.get("class") in ("DAMPER", "ACT_AIR", "ACT_EH"):
            # a damper symbol: the class of the dampers tied to it, else the operator its description names
            c = linked[e["legend_id"]].most_common(1)
            e["class"] = c[0][0] if c and c[0][0].startswith("DAMPER") else legend_class("DAMPER", desc)
        elif e.get("class") in SPLIT: e["class"] = legend_class(e["class"], desc)
        if (e.get("description") or "").upper().startswith("CARBON ADSORBER") and e.get("class") is None and "NOZZLE" not in e["description"].upper():
            e["class"] = "CARBON_ADSORBER"
    json.dump(idx, open(f"{D}/legend_index.json", "w"), indent=1, ensure_ascii=False)
    classes = {r["class"] for r in csv.DictReader(open(f"{OUT}/classes.csv", encoding="utf-8"))}
    used = {e["class"] for e in idx if e.get("class") and not e["class"].startswith("ACT_")}
    print("legend classes not in classes.csv:", sorted(used - classes))
    print("classes without a legend symbol:", sorted(classes - used))


if __name__ == "__main__":
    main()
