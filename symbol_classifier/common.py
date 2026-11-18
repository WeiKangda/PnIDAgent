"""Shared pieces of the symbol classifier: the labels of verified_annotations_v3, the document families used for
cross-validation, the symbol crops and the context features of a box.

A box is described by
  - its crop: the symbol with a small margin, padded to a square, grey (crop());
  - its context (context_features()): the text lines inside and around the box (hashed tokens: operator letters such
    as M / S / E/H, position marks NC / LO / LC / FO / FC, tag prefixes such as FT / PT / PP / TK), its size relative
    to the other symbols of the page and its aspect ratio (orientation-free) and, when the connectivity is known, the number of pipe connections
    at the symbol (its ports).
"""
import csv, json, math, os, re, zlib
from collections import Counter, defaultdict

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "verified_annotations_v3")
CROP = 128                     # crop side in pixels
HASH_DIM = 512                 # hashed text tokens
CTX_DIM = 2 * HASH_DIM + 4 + 5  # tokens inside, tokens around, 4 size features, 5 port-count bins


def family(page):
    """Document family of a page: the unit left out in cross-validation (pages of one family share a drawing style)."""
    p = page[7:] if page.startswith("pseudo_") else page
    if "APR1400" in p: return "APR1400 DCD"
    if "DAVIS-BESSE) AFW system Reliability" in p: return "Davis-Besse reliability report"
    if "NUREG-CR-4550" in p: return "NUREG/CR-4550 (Surry)"
    if "Surry" in p: return "Surry FSAR"
    if p.startswith("AFW"): return "AFW, other plants"
    if "GE-Hitachi" in p: return "GE-Hitachi ABWR"
    if "NuScale" in p: return "NuScale"
    return "other"


def load_classes():
    rows = list(csv.DictReader(open(os.path.join(DATA, "symbol_classes", "classes.csv"), encoding="utf-8")))
    return [r["class"] for r in rows], {r["class"]: r for r in rows}


VALVE_FT = {"MOTOR": "MOV", "AIR": "AOV", "SOLENOID": "SOV", "EH": "HOV", "HYDRAULIC": "HOV", "NONE": "XVM"}


def ft_types():
    """class + actuator -> fault-tree component type (symbol_classes/ft_component_types.csv: every pair on the pages)."""
    return {r["class|actuator"]: r["ft_component_type"]
            for r in csv.DictReader(open(os.path.join(DATA, "symbol_classes", "ft_component_types.csv"), encoding="utf-8"))}


def ft_type(cls, act, table):
    """The fault-tree component type of a predicted class + actuator: from the table, and for a valve + actuator pair that
    does not occur on the pages, from the actuator (MOV, AOV, SOV, HOV, XVM). '' = no component type."""
    k = f"{cls}|{act}"
    if k in table: return table[k]
    if cls.startswith("VLV_") and cls not in ("VLV_CHECK", "VLV_STOP_CHECK", "VLV_SAFETY_RELIEF", "VLV_PILOT_RELIEF",
                                              "VLV_VACUUM_BREAKER"):
        return VALVE_FT.get(act, "")
    same = {v for kk, v in table.items() if kk.split("|")[0] == cls}        # a class with one type whatever the actuator
    return same.pop() if len(same) == 1 else ""


def load_dataset():
    """One dict per symbol box with its labels, page, family and box."""
    rows = list(csv.DictReader(open(os.path.join(DATA, "symbol_classes", "symbol_classes.csv"), encoding="utf-8")))
    for r in rows:
        r["box_id"] = int(r["box_id"]); r["bbox"] = [float(r[k]) for k in ("x0", "y0", "x1", "y1")]
        r["family"] = family(r["page"])
    return rows


def page_image(page):
    return cv2.imread(os.path.join(DATA, "topology_verified", "images", f"{page}.png"), cv2.IMREAD_GRAYSCALE)


def crop(gray, bbox, side=CROP, margin=0.08):
    """The symbol with a margin of 8% of its longer side (+3 px), padded to a square with white, resized to side x side."""
    H, W = gray.shape
    x0, y0, x1, y1 = bbox
    m = margin * max(x1 - x0, y1 - y0) + 3
    a, b, c, d = int(max(0, x0 - m)), int(max(0, y0 - m)), int(min(W, x1 + m)), int(min(H, y1 + m))
    im = gray[b:d, a:c]
    h, w = im.shape
    s = max(h, w)
    sq = np.full((s, s), 255, np.uint8)
    sq[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = im
    return cv2.resize(sq, (side, side), interpolation=cv2.INTER_AREA if s > side else cv2.INTER_CUBIC)


# ------------------------------------------------------------------ context
TOKEN = re.compile(r"[A-Z0-9/]+")


def norm_tokens(text):
    """Upper-case tokens with the digits collapsed, so that tags keep their letters: 'MOV-1018' -> 'MOV', '#';
    'FT345D' -> 'FT#D' and its prefix 'FT#'; 'SI-314 (NC)' -> 'SI', '#', 'NC'."""
    out = []
    for t in TOKEN.findall(text.upper()):
        t2 = re.sub(r"\d+", "#", t)
        out.append(t2)
        m = re.match(r"([A-Z]{1,4})#", t2)
        if m: out.append(m.group(1) + "#")
    return out


def _h(s):
    return zlib.crc32(s.encode()) % HASH_DIM


def context_features(page_rows, texts, topo=None):
    """Context vectors (len(page_rows) x CTX_DIM) for the boxes of one page.

    texts: [{'box': [x0,y0,x1,y1], 'text': str}]; topo: the connectivity file of the page, whose 'links' join symbol ids
    (the box ids) and junctions (optional; without it the port features are 0)."""
    sides = [max(r["bbox"][2] - r["bbox"][0], r["bbox"][3] - r["bbox"][1]) for r in page_rows]
    med = float(np.median(sides)) if sides else 1.0
    ports = Counter()
    if topo is not None:
        for l in topo["links"]:
            for e in (l["source"], l["target"]):
                if isinstance(e, int): ports[e] += 1
    out = np.zeros((len(page_rows), CTX_DIM), np.float32)
    for i, r in enumerate(page_rows):
        x0, y0, x1, y1 = r["bbox"]; w, h = max(1.0, x1 - x0), max(1.0, y1 - y0); s = max(w, h)
        ins, near = Counter(), Counter()
        for t in texts:
            tx0, ty0, tx1, ty1 = t["box"]; cx, cy = (tx0 + tx1) / 2, (ty0 + ty1) / 2
            if x0 - 0.1 * w <= cx <= x1 + 0.1 * w and y0 - 0.1 * h <= cy <= y1 + 0.1 * h:
                for tok in norm_tokens(t["text"]): ins[_h("in:" + tok)] += 1
            else:
                dx = max(x0 - cx, 0, cx - x1); dy = max(y0 - cy, 0, cy - y1)
                if math.hypot(dx, dy) <= 1.0 * s:
                    for tok in norm_tokens(t["text"]): near[_h("near:" + tok)] += 1
        for k, v in ins.items(): out[i, k] = min(v, 3)
        for k, v in near.items(): out[i, HASH_DIM + k] = min(v, 3)
        lo, sh = max(w, h), min(w, h)                         # orientation-free: a symbol may be drawn turned
        out[i, 2 * HASH_DIM:2 * HASH_DIM + 4] = [math.log(lo / med), math.log(sh / med), math.log(lo / sh), math.log(lo / med) ** 2]
        if topo is not None:
            out[i, 2 * HASH_DIM + 4 + min(ports[r["box_id"]], 4)] = 1.0      # 0, 1, 2, 3, 4+ ports
    return out


def page_context(page, page_rows, use_topology=True):
    t = json.load(open(os.path.join(DATA, "text", f"{page}.json")))
    texts = [{"box": it["box"], "text": it["text"]} for it in t["items"]]
    topo = None
    if use_topology:
        p = os.path.join(DATA, "topology_verified", f"{page}.json")
        topo = json.load(open(p)) if os.path.exists(p) else None
    return context_features(page_rows, texts, topo)


def by_page(rows):
    d = defaultdict(list)
    for i, r in enumerate(rows): d[r["page"]].append(i)
    return d


# ------------------------------------------------------------------ evaluation
def load_cache(cache):
    d = np.load(os.path.join(cache, "dataset.npz"))
    meta = json.load(open(os.path.join(cache, "meta.json")))
    return {k: d[k] for k in d.files}, meta


def scores(d, meta, pred_cls, pred_act=None, pred_pos=None, mask=None):
    """Accuracy and macro-F1 of the class (macro over the classes with at least 5 boxes), actuator accuracy on the boxes
    that have an actuator, class+actuator accuracy, accuracy of the fault-tree component type that follows from them (on
    the boxes that have one), normal-position accuracy on the open / closed boxes.
    mask: the boxes to score (default all)."""
    from sklearn.metrics import f1_score
    n = len(d["cls"]); mask = np.ones(n, bool) if mask is None else mask
    y = d["cls"][mask]; p = pred_cls[mask]
    cnt = np.bincount(d["cls"], minlength=len(meta["classes"]))
    labels = np.where(cnt >= 5)[0]
    out = {"n": int(mask.sum()), "class_acc": float((y == p).mean()),
           "class_macro_f1": float(f1_score(y, p, labels=labels, average="macro", zero_division=0))}
    has_act = mask & (d["act"] != meta["actuators"].index("-"))
    if pred_act is not None:
        out["act_acc"] = float((d["act"][has_act] == pred_act[has_act]).mean())
        out["class_act_acc"] = float(((d["cls"] == pred_cls) & (d["act"] == pred_act))[mask].mean())
    if pred_act is not None and "ft" in d:
        table = ft_types(); C, A = meta["classes"], meta["actuators"]
        pft = np.array([ft_type(C[c], A[a], table) for c, a in zip(pred_cls, pred_act)])
        m = mask & (d["ft"] != "")
        out["ft_acc"] = float((pft[m] == d["ft"][m]).mean()); out["ft_n"] = int(m.sum())
    if pred_pos is not None:
        m = mask & (d["pos"] >= 0)
        out["pos_acc"] = float((d["pos"][m] == pred_pos[m]).mean()); out["pos_n"] = int(m.sum())
    return out


def family_folds(d):
    fams = sorted(set(d["family"].tolist()))
    return [(f, np.where(d["family"] != f)[0], np.where(d["family"] == f)[0]) for f in fams]


def legend_crops(classes):
    """Crops of the legend symbols that stand for one of the classes (legend_library/derived/legend_index.json): the
    symbol-only snippet where there is one, else the extracted 600 dpi image, trimmed to its ink."""
    leg = os.path.join(DATA, "legend_library", "derived")
    out, cls, src, ids = [], [], [], []
    for e in json.load(open(os.path.join(leg, "legend_index.json"))):
        c = e.get("class")
        if c not in classes or e.get("not_applicable_variant"): continue
        sym = os.path.join(leg, "snippets", f"{e['legend_id']}_sym.png")
        g = cv2.imread(sym if os.path.exists(sym) else os.path.normpath(os.path.join(leg, e["image"])), cv2.IMREAD_GRAYSCALE)
        if g is None: continue
        ys, xs = np.where(g < 160)
        if len(xs) == 0: continue
        out.append(crop(g, [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]))
        cls.append(classes.index(c)); src.append(e["pdf"].split()[0]); ids.append(e["legend_id"])
    return np.array(out, np.uint8), np.array(cls), np.array(src), np.array(ids)
