"""Link every symbol to the legend PDF (AP1000 / APR1400 P&ID legend sheets, 962 entries): the legend entry that defines
its class (valve body or equipment) and, for powered valves, the legend entry of its operator.
APR1400 pages use the APR1400 legend (it is the legend of those drawings); other pages use the AP1000 legend as the standard
symbol convention, except where only one legend has the symbol. Within the allowed entries the choice follows the note
(stop-check, normally closed, cavitating venturi, ...) and then visual similarity (DINOv2). Legend images marked "NA"
(variant does not exist) are never used. Output legend_link.json {idx: {"body", "op", "family"}} (indices into legend.json)."""
import json, re, numpy as np
HERE = "/home/suxinqi666/code/PnIDAgent_project/symbol_class"
LG = json.load(open(f"{HERE}/legend.json")); S = json.load(open(f"{HERE}/symbols.json")); L = json.load(open(f"{HERE}/labels_current.json"))
IT = json.load(open(f"{HERE}/inside_text.json")); NA = set(json.load(open(f"{HERE}/legend_na.json")))
E = np.load(f"{HERE}/emb.npy"); LE = np.load(f"{HERE}/legend_emb.npy"); sim = E @ LE.T
APR = {"VLV_GATE": [450, 451], "VLV_GLOBE": [452, 453], "VLV_BUTTERFLY": [454, 455], "VLV_CHECK": [481], "VLV_ANGLE": [466, 467], "VLV_3WAY": [458, 459],
       "VLV_RELIEF": [494, 493], "VLV_REGULATOR": [495], "RUPTURE_DISC": [521], "PUMP": [419, 420], "COMPRESSOR": [434], "FAN": [578, 579, 591], "TANK": [427],
       "HX": [425, 426], "COIL": [581, 582], "CHILLER": [590], "HUMIDIFIER": [580], "STEAM_GEN": [417], "REACTOR": [418], "FILTER": [534, 513, 583, 584, 586, 587, 539],
       "ORIFICE": [523, 525], "HEATER": [538], "SPARGER": [537], "TRAP": [512], "INSTR": [656], "TURBINE": [416]}
AP = {"VLV_GATE": [213, 214], "VLV_GLOBE": [241, 242], "VLV_BUTTERFLY": [201, 202], "VLV_CHECK": [203], "VLV_ANGLE": [199, 200], "VLV_3WAY": [237, 238],
      "VLV_RELIEF": [236], "VLV_BALL": [456], "VLV_PLUG": [229], "VLV_NEEDLE": [227], "VLV_DIAPHRAGM": [207], "RUPTURE_DISC": [324], "PUMP": [273], "COMPRESSOR": [295, 434],
      "FAN": [300, 578], "TANK": [283, 427], "HX": [288, 286], "CONDENSER": [285], "AHU": [304], "DEMIN": [291], "DRYER": [293], "FILTER": [345, 347],
      "ORIFICE": [328, 332], "HEATER": [306], "SPARGER": [537], "INSTR": [396], "BREAKER": [10], "TRANSFORMER": [34, 134], "GENERATOR": [51], "MOTOR": [440],
      "BATTERY": [76], "CHARGER": [78], "INVERTER": [77], "SWITCH": [8, 5], "PANEL": [133, 131], "PENETRATION": [344], "OFFPAGE": [388], "DAMPER": [257, 487],
      "STEAM_GEN": [417], "TURBINE": [416], "REACTOR": [418], "COIL": [581, 582], "CHILLER": [590], "HUMIDIFIER": [302], "TRAP": [352]}
DAMPER_OP = {"AIR": [592, 593], "EH": [594, 595], "NONE": [487, 257], "UNSPEC": [487, 257]}
OPS = {"MOTOR": [500, 267], "AIR": [263, 505], "SOLENOID": [268], "EH": [506], "HYDRAULIC": [26]}
def family(n): return "apr" if "APR1400" in n else "ap"
out = {}
for i, s in enumerate(S):
    c, a = L[str(i)][:2]; note = (L[str(i)][2] if len(L[str(i)]) > 2 else "").lower(); fam = family(s["page"]); t = IT[str(i)].upper()
    if c == "DAMPER": cand = DAMPER_OP.get(a, [487])
    elif c == "PUMP" and a == "TURBINE": cand = [276]
    elif c == "VLV_CHECK" and "stop-check" in note: cand = [482, 483] if fam == "apr" else [217, 218]
    elif c == "VLV_RELIEF" and ("posrv" in note or "pilot" in note): cand = [497]
    elif c == "ORIFICE" and "venturi" in note: cand = [525] if fam == "apr" else [332]
    elif c == "COIL": cand = [581] if "HC" in t or "heating" in note else [582] if "CC" in t or "cooling" in note else [581, 582]
    elif c == "FILTER" and fam == "apr":
        kw = [("prefilter", 583), ("hepa", 584), ("postfilter", 586), ("moisture", 587), ("separator", 539), ("strainer", 513), ("adsorber", 534)]
        cand = [j for k, j in kw if k in note] or APR["FILTER"]
    elif c == "OFFPAGE" and "AC Power" in s["page"] and s["id"] in (6, 7, 8, 10, 16, 171): cand = []   # circle connectors: no legend entry
    elif c == "OFFPAGE" and "AC Power" in s["page"] and "arrow" not in note: cand = [90]      # electrical hexagon: AP1000 MATCH HEXAGON
    elif c == "VLV_3WAY": cand = [237]     # AP1000 THREE WAY VALVE (three triangles); the APR1400 3-way entries are ball / plug valves
    elif c == "SWITCH": cand = [8] if "transfer" in note else [5]
    else: cand = (APR.get(c) if fam == "apr" else None) or AP.get(c) or APR.get(c) or []
    cand = [j for j in cand if j not in NA]
    if len(cand) == 2 and LG[cand[0]]["n"] == LG[cand[1]]["n"] and LG[cand[0]]["src"] == LG[cand[1]]["src"]:     # open / closed variants of one entry
        closed = "solid body" in note or "normally closed" in note or "locked closed" in note
        cand = [cand[1] if closed else cand[0]]
    body = max(cand, key=lambda j: sim[i, j]) if cand else None
    op = None
    if a in OPS and c.startswith("VLV"):
        oc = OPS[a]
        if a == "MOTOR": oc = [501] if ("dc motor" in note or "DC" in t.split()) else [500] if fam == "apr" else [267]
        if a == "AIR": oc = [672] if ("cylinder" in note or "piston" in note) else [263]          # diaphragm (dome) unless a cylinder is drawn
        op = max(oc, key=lambda j: sim[i, j])
    if c == "BREAKER" and "AC Power" in s["page"]: op = 13      # draw-out breaker: breaker between two disconnect devices
    out[str(i)] = {"body": body, "op": op, "family": fam}
json.dump(out, open(f"{HERE}/legend_link.json", "w"))
import collections
print("linked:", sum(1 for v in out.values() if v["body"] is not None), "| with operator:", sum(1 for v in out.values() if v["op"] is not None))
print("not linked:", collections.Counter(L[i][0] for i, v in out.items() if v["body"] is None))
