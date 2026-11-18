"""Source (evidence) for every class label: which legend entry, which text on the drawing, which document.
Output: sources.json {global idx: {"level": A|B|C, "items": [str, ...]}} and legend_refs/ (reference symbol images)."""
import json, re, glob, os, shutil, collections
R = "/home/suxinqi666/code/PnIDAgent_project"; HERE = f"{R}/symbol_class"
S = json.load(open(f"{HERE}/symbols.json")); L = json.load(open(f"{HERE}/labels_current.json"))
LEG = {}
for f in sorted(glob.glob(f"{R}/PID_Legend_Symbols_data/*.json")):
    src = os.path.basename(f)[:-5]
    for e in json.load(open(f))["entries"]: LEG[(src, e["n"])] = e
def leg(src, n):
    e = LEG[(src, n)]; pdf = "AP1000 (PID Legend).pdf" if src.startswith("ap1000") else "APR1400 (PID Legend).pdf"
    d = re.sub(r"\s+", " ", (e.get("description") or "")).strip().replace("\u2014", "-").replace("\u2013", "-")
    d = d if len(d) <= 80 else d[:77].rstrip() + "..."
    return f'{pdf} page {int(src[-1]) + 1}, entry {n} "{d}"'
CLS_REF = {"VLV_GATE": [("apr1400_p0", 35), ("ap1000_p3", 8)], "VLV_GLOBE": [("apr1400_p0", 36), ("ap1000_p3", 22)], "VLV_BUTTERFLY": [("apr1400_p0", 37), ("ap1000_p3", 2)],
    "VLV_BALL": [("apr1400_p0", 38)], "VLV_PLUG": [("apr1400_p0", 40)], "VLV_NEEDLE": [("apr1400_p0", 47)], "VLV_DIAPHRAGM": [("apr1400_p0", 45)], "VLV_ANGLE": [("apr1400_p0", 43), ("ap1000_p3", 1)],
    "VLV_3WAY": [("ap1000_p3", 20), ("apr1400_p0", 39)], "VLV_CHECK": [("apr1400_p0", 52), ("apr1400_p0", 53), ("ap1000_p3", 3)], "VLV_RELIEF": [("apr1400_p0", 64), ("apr1400_p0", 63), ("ap1000_p3", 19)],
    "VLV_REGULATOR": [("apr1400_p0", 65)], "RUPTURE_DISC": [("apr1400_p0", 84)], "DAMPER": [("apr1400_p1", 15), ("apr1400_p1", 17), ("ap1000_p3", 33)], "PUMP": [("apr1400_p0", 4), ("ap1000_p3", 49)],
    "COMPRESSOR": [("apr1400_p0", 19), ("ap1000_p3", 74)], "FAN": [("apr1400_p1", 1), ("apr1400_p1", 2)], "TURBINE": [("apr1400_p0", 1)], "TANK": [("apr1400_p0", 12), ("ap1000_p3", 59)],
    "HX": [("apr1400_p0", 10), ("apr1400_p0", 11), ("ap1000_p3", 64)], "CONDENSER": [("ap1000_p3", 61)], "COIL": [("apr1400_p1", 4), ("apr1400_p1", 5)], "CHILLER": [("apr1400_p1", 13)],
    "HUMIDIFIER": [("apr1400_p1", 3)], "AHU": [("ap1000_p3", 80)], "STEAM_GEN": [("apr1400_p0", 2)], "REACTOR": [("apr1400_p0", 3)], "FILTER": [("ap1000_p4", 30), ("ap1000_p4", 32), ("apr1400_p0", 76)],
    "DEMIN": [("ap1000_p3", 67)], "DRYER": [("ap1000_p3", 69)], "ORIFICE": [("apr1400_p0", 86), ("apr1400_p0", 88), ("ap1000_p4", 13)], "HEATER": [("apr1400_p0", 101)], "SPARGER": [("apr1400_p0", 100)],
    "SUMP": [("apr1400_p0", 34)], "TRAP": [("apr1400_p0", 75)], "INSTR": [("apr1400_p2", 6), ("ap1000_p5", 1)], "BREAKER": [("ap1000_p0", 11)], "TRANSFORMER": [("ap1000_p0", 35)],
    "GENERATOR": [("ap1000_p0", 52)], "MOTOR": [("ap1000_p0", 53)], "BATTERY": [("ap1000_p0", 77)], "CHARGER": [("ap1000_p0", 79)], "INVERTER": [("ap1000_p0", 78)], "SWITCH": [("ap1000_p0", 6)],
    "PANEL": [("ap1000_p2", 3)], "OFFPAGE": [("ap1000_p4", 97)], "PENETRATION": [("ap1000_p4", 29)]}
ACT_REF = {"MOTOR": [("apr1400_p0", 70), ("ap1000_p3", 43)], "AIR": [("apr1400_p0", 72), ("ap1000_p3", 39)], "SOLENOID": [("apr1400_p0", 74), ("ap1000_p3", 44)],
           "EH": [("apr1400_p0", 106), ("ap1000_p3", 41)], "HYDRAULIC": [("ap1000_p0", 27)], "NONE": [("apr1400_p0", 107)], "TURBINE": [("ap1000_p3", 52)]}
# legends printed on the page itself
PAGE_LEG = {"San Onofre": [("act", "MOTOR", "- Motor VALVE"), ("act", "AIR", "- Pneumatic VALVE"), ("act", "EH", "- Electro-Hydraulic"), ("act", "HYDRAULIC", "- Hydraulic VALVE")],
            "Feedwater system_(Surry)": [("act", "MOTOR", "Motor-Operated Valve"), ("cls", "VLV_CHECK", "Stop-Check Valve")],
            "RHR_(Surry)": [("act", "MOTOR", "MOTOR OPERATED VALVE")], "SWS_(Surry)": [("act", "MOTOR", "MOV - MOTOR OPERATED VALVE"), ("cls", "FILTER", "S - STRAINER"), ("cls", "PUMP", "PP - PUMP"), ("cls", "HX", "HXS - HEAT EXCHANGERS")],
            "Compressed Air": [("cls", "MOTOR", "M - MOTOR")], "Main Steam system_(Surry)": [("cls", "STEAM_GEN", "SG - STEAM GENERATOR"), ("cls", "TURBINE", "HP / LP - HIGH / LOW PRESSURE")],
            "Battery system": [("cls", "BREAKER", "2P / 3P - 2 POLE / 3 POLE")]}
ABBR_LEG = {"Boron recovery system_(Surry) FSAR_p1": {"HCV": "HAND CONTROL VALVE", "LCV": "LEVEL CONTROL VALVE", "PCV": "PRESSURE CONTROL VALVE"},
            "RHR_(Surry)": {"HCV": "HAND CONTROL VALVE", "FCV": "FLOW CONTROL VALVE"}}
CLS_TXT = {"VLV_CHECK": r"\bCV ?\d|\bCKV|CHECK", "VLV_RELIEF": r"\bSRV|\bRV[- ]?\d|SAFET|RELIEF|MSSV|POSRV|\bPSV", "PUMP": r"PUMP|\bPP ?\d|\bP-\d|MDP|TDP|AFP\b|\bPMP|\bRCP",
    "TANK": r"TANK|\bTK ?\d|\bTK\b|\bCST|RWST|IRWST|RECEIVER|\bDST|ACCUM", "HX": r"\bHX|HXS|\bHE ?\d|HEAT EXCH|COOLER|HEATER|MFHX|\bHTX|\bMSR\b|^MS$", "CONDENSER": r"CONDENSER|\bGSC\b",
    "STEAM_GEN": r"\bSG|S/G|STEAM GEN|STM\.? ?GEN|ST\. ?GEN|STEAM\s*GENERATOR", "FILTER": r"STRAINER|FILTER|\bST-? ?\d|\bSTR ?\d|ADSORBER|\bAD\b|HEPA|SEPARATOR|SILENCER|^S\d?$|^F$|DEBRIS",
    "DEMIN": r"DEMIN|POLISH|ION EXCH", "ORIFICE": r"\bFE\b|\bFE[- ]?\d|ORIFICE|VENTURI|FLOW NOZZLE", "TURBINE": r"TURBINE|AFPT|\bHP\b|\bLP\b|\bK-\d|^CIV$", "FAN": r"FAN|BLOWER|EXHAUSTER|\bAH\d",
    "COMPRESSOR": r"COMPRESSOR|\bCM ?\d|VACUUM", "COIL": r"^HC$|^CC$|^EL$|^CW$|COIL", "CHILLER": r"CHILLER|\bCH ?\d", "HUMIDIFIER": r"\bHU ?\d|HUMIDIF", "AHU": r"AIR HANDLING|\bAHU|CUBICLE|COOLING UNIT|\bACU",
    "REACTOR": r"REACTOR", "PRESSURIZER": r"PRESSURIZER|\bPZ", "SPARGER": r"SPARGER|SPRAY|\bSZ ?\d", "SUMP": r"SUMP", "INSTR": r"^\(?(FT|FE|PT|LT|TE|TI|PI|LI|FI|FIC|PDIT|PDT|RT|RIT|XS|R|L|T|F|RM|P)\b",
    "BREAKER": r"BRK|BREAKER|\bCB\b|\b52\b|^\d?P$|^NO$|^NC$", "TRANSFORMER": r"TRANSF|XFMR|XFM|\bTR\b|KVA|\d+ ?-? ?\d+ ?V\b|\d+/\d+ ?V", "GENERATOR": r"GEN|\bDG ?\d|GTG",
    "MOTOR": r"^M$|MOTOR", "BATTERY": r"BATTER", "CHARGER": r"CHARGER|RECTIF", "INVERTER": r"INVERTER|^INV$|\bUPS", "SWITCH": r"SWITCH|TRANSFER", "PANEL": r"PANEL|\bBUS\b|MCC|SWGR|EXCITATION|VITAL|SWITCHYARD",
    "PENETRATION": r"^PC$|PENETRATION", "DRYER": r"DRYER", "ENGINE": r"DIESEL|ENGINE", "OFFPAGE": r"^TO\b|^FROM\b|\bTO\b|\bFROM\b|SYSTEM|SUPPLY|\bSWGR|NOZZLE|^\d?[A-Z]{1,2}$|^\d$",
    "BLOCK": r"[A-Z]{3,}", "HEATER": r"HEATER|HTRS?\b"}
ACT_TXT = {"MOTOR": r"^M$|\bMOV|\bMO-|MOTOR|^DC$|\bM\.?O\.?\b", "AIR": r"\bAOV|\bA\.O\.|\bHCV|\bFCV|\bLCV|\bPCV|\bTCV|\bTV-|\bHV\d|AIR|^P$", "SOLENOID": r"^S$|\bSOV|SOLENOID",
           "EH": r"E ?/ ?H|^EH$", "HYDRAULIC": r"HYDRAUL", "NONE": r"^\(?XV|MANUAL|HAND|\bL\.?O\.?\b|\bL\.?C\.?\b|LOCKED", "TURBINE": r"TURBINE|\bTD\b|TDP|AFPT", "DIESEL": r"DIESEL|\bDD\b|EDP|ENGINE"}
LINK = json.load(open(f"{HERE}/legend_link.json")); LGI = json.load(open(f"{HERE}/legend.json"))
def legx(j):
    e = LGI[j]; v = os.path.basename(e["file"])[:-4]; var = " (closed variant)" if v.endswith("_b") else " (open variant)" if v.endswith("_a") else ""
    return leg(e["src"], e["n"]) + var
TX = {}
def texts(i):
    s = S[i]; n = s["page"]
    if n not in TX: TX[n] = json.load(open(f"{R}/line_results/text_verified/{n}.json"))["items"]
    x0, y0, x1, y1 = s["box"]; sz = max(x1 - x0, y1 - y0); g = max(12, 0.5 * sz); out = []
    for t in TX[n]:
        a0, b0, a1, b1 = t["box"]; inside = a0 >= x0 - 3 and b0 >= y0 - 3 and a1 <= x1 + 3 and b1 <= y1 + 3
        gap = max(0, a0 - x1, x0 - a1, b0 - y1, y0 - b1)
        if inside or gap <= g: out.append(("inside" if inside else "next to the box", t["text"].strip()))
    return out
src_out = {}; lev = collections.Counter()
for i, s in enumerate(S):
    c, a = L[str(i)][:2]; note = L[str(i)][2] if len(L[str(i)]) > 2 else ""; n = s["page"]; items = []; strong_c = strong_a = False
    for key, ents in PAGE_LEG.items():
        if key in n:
            for kind, val, txt in ents:
                if (kind == "cls" and c == val) or (kind == "act" and a == val): items.append(f'page legend: "{txt}"'); strong_c |= kind == "cls"; strong_a |= kind == "act"
    tx = texts(i)
    for where, t in tx:
        tu = t.upper()
        for key, ab in ABBR_LEG.items():
            if key in n:
                for k_, v_ in ab.items():
                    if re.search(rf"\b{k_}\b", tu): items.append(f'page legend: "{k_} - {v_}" with drawing text "{t}"'); strong_a = True
        if c == "BLOCK" and where != "inside": continue
        if c in CLS_TXT and re.search(CLS_TXT[c], tu): items.append(f'drawing text {where}: "{t}"'); strong_c = True
        elif a in ACT_TXT and re.search(ACT_TXT[a], tu): items.append(f'drawing text {where}: "{t}"'); strong_a = True
    if re.search(r"NUREG|INL AFW|Davis-Besse AFW|legend on page", note):
        items.append("document: " + note); strong_c = strong_a = True
    lk = LINK[str(i)]
    if lk["body"] is not None:
        own = "the legend of this drawing set" if lk["family"] == "apr" and LGI[lk["body"]]["src"].startswith("apr") else "standard symbol, this drawing has no own legend for it" if not any(k in n for k in PAGE_LEG) else "standard symbol, not defined by the legend on this page"
        items.append(f"legend PDF symbol: {legx(lk['body'])} ({own})")
    if lk["op"] is not None: items.append(f"legend PDF operator: {legx(lk['op'])}")
    if a == "NONE" and c.startswith("VLV") and c not in ("VLV_CHECK", "VLV_RELIEF", "VLV_REGULATOR"): items.append("no operator drawn: " + leg("apr1400_p0", 107))
    if note and not note.startswith("CHECK") and "document:" not in " ".join(items): items.append("note: " + note)
    if note.startswith("CHECK"): level = "C"; items.insert(0, "uncertain: " + note[7:])
    elif strong_c or strong_a or c in ("BLOCK",) and tx: level = "A"
    elif lk["body"] is not None: level = "B"
    else: level = "B" if not tx else "A"
    # de-duplicate, keep order
    seen = set(); items = [x for x in items if not ((x.split(": ", 1)[-1] if x.startswith("drawing text") else x) in seen or seen.add(x.split(": ", 1)[-1] if x.startswith("drawing text") else x))]
    src_out[str(i)] = {"level": level, "items": items}; lev[level] += 1
json.dump(src_out, open(f"{HERE}/sources.json", "w"))
print(dict(lev))
noref = collections.Counter(L[str(i)][0] for i in range(len(S)) if src_out[str(i)]["level"] == "C" and not L[str(i)][2:3] or (src_out[str(i)]["level"] == "C" and not L[str(i)][2].startswith("CHECK")) if len(L[str(i)]) > 0)
print("C without CHECK note:", [(S[i]["page"][7:25], S[i]["id"], L[str(i)][:2]) for i in range(len(S)) if src_out[str(i)]["level"] == "C" and not (len(L[str(i)]) > 2 and L[str(i)][2].startswith("CHECK"))][:30])
print(collections.Counter((L[str(i)][0], src_out[str(i)]["level"]) for i in range(len(S))).most_common(60))
