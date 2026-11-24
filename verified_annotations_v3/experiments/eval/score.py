#!/usr/bin/env python3
"""Score predictions on the verified pages of this release with the metrics of the paper.

  python score.py symbols --pred FILE [--classes]
  python score.py text    --pred DIR
  python score.py lines   --pred DIR
  python score.py graph   --symbols FILE|gt --lines DIR|gt [--no-dot]
  python score.py pairs   --pred FILE [--match iou|center]
  python score.py export-lines --dir DIR [--pages all|test|train]

Common options: --pages test|train|all (default test: the B_test pages of ../split.json), --v3 DIR (default: the
dataset folder two levels up from this file), --out FILE (JSON with the per-page and mean scores).
Scores are macro averages over pages unless stated. Formats and definitions: README.md in this folder.
"""
import argparse
import csv
import json
import os
import re
import struct
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PREFIX = "pseudo_"


# ------------------------------------------------------------------ metrics of the paper
# Line metric: count_scores of the paper's rescoring script (copied from eval_tools/eval_lines.py). A reference segment is
# found when predictions within 8 px and 10 degrees cover at least half of it; a prediction is valid when reference
# segments cover at least half of it.

def _ang(s): return np.degrees(np.arctan2(s[3] - s[1], s[2] - s[0])) % 180.0


def _span(a, b, tol):
    L = float(np.hypot(a[2] - a[0], a[3] - a[1]))
    if L < 1 or min(abs(_ang(a) - _ang(b)), 180 - abs(_ang(a) - _ang(b))) > 10.0: return L, 0.0, 0.0
    ux, uy = (a[2] - a[0]) / L, (a[3] - a[1]) / L
    for px, py in ((b[0], b[1]), (b[2], b[3])):
        if abs(-uy * (px - a[0]) + ux * (py - a[1])) > tol: return L, 0.0, 0.0
    t1 = ux * (b[0] - a[0]) + uy * (b[1] - a[1]); t2 = ux * (b[2] - a[0]) + uy * (b[3] - a[1])
    return L, max(0.0, min(t1, t2)), min(L, max(t1, t2))


def _union_frac(a, others, tol):
    iv = []
    for b in others:
        L, lo, hi = _span(a, b, tol)
        if hi > lo: iv.append((lo, hi))
    if not iv: return 0.0, 0
    iv.sort(); tot, ce = 0.0, -1.0; L = float(np.hypot(a[2] - a[0], a[3] - a[1]))
    for lo, hi in iv:
        lo = max(lo, ce)
        if hi > lo: tot += hi - lo; ce = hi
    return tot / L, len(iv)


def count_scores(gt, pred, tol=8, min_overlap=0.5):
    rec = [(_union_frac(g, pred, tol)[0] >= min_overlap) for g in gt]
    prec = [(_union_frac(p, gt, tol)[0] >= min_overlap) for p in pred]
    R = float(np.mean(rec)) if rec else 0.0; P = float(np.mean(prec)) if prec else 0.0
    return P, R, (2 * P * R / (P + R) if P + R else 0.0)


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1])); I = ix * iy
    if not I: return 0.0
    U = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - I
    return I / U if U > 0 else 0.0


def match_iou(pr_boxes, gt_boxes, thr=0.5):
    """Symbol matching of graph_score: all (prediction, reference) pairs in order of decreasing IoU, one to one, IoU >= thr.
    pr_boxes, gt_boxes: {id: box}. Returns {prediction id: reference id}."""
    m, used = {}, set()
    cand = sorted(((iou(pr_boxes[p], gt_boxes[g]), p, g) for p in pr_boxes for g in gt_boxes), key=lambda t: -t[0])
    for v, p, g in cand:
        if v < thr: break
        if p in m or g in used: continue
        m[p] = g; used.add(g)
    return m


def match_center(pr_boxes, gt_boxes):
    """Lenient matching: the predicted box centre lies inside the reference box; nearest centres first, one to one."""
    ctr = lambda b: ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)
    cand = []
    for p, pb in pr_boxes.items():
        (px, py) = ctr(pb)
        for g, gb in gt_boxes.items():
            if gb[0] <= px <= gb[2] and gb[1] <= py <= gb[3]:
                gx, gy = ctr(gb); cand.append((float(np.hypot(px - gx, py - gy)), p, g))
    cand.sort(key=lambda t: t[0])
    m, used = {}, set()
    for d, p, g in cand:
        if p in m or g in used: continue
        m[p] = g; used.add(g)
    return m


def graph_score(gt_nodes, gt_edges, gt_boxes, pr_nodes, pr_edges, pr_boxes, sym_iou=0.5, jct_tol=20.0):
    """Edge metric of the paper's rescoring script (eval_tools/eval_graph.score): symbols matched at IoU >= 0.5, junctions
    within 20 px; an edge is correct when both ends are matched and the matched pair is a reference edge.
    Returns P, R, F1 and sheet-exact (1.0 when every reference edge is found and there is no extra edge)."""
    m = {("sym", p): ("sym", g) for p, g in match_iou(pr_boxes, gt_boxes, sym_iou).items()}; used = set(m.values())
    gj = [(k, xy) for k, xy in gt_nodes.items() if k[0] == "jct"]; pj = [(k, xy) for k, xy in pr_nodes.items() if k[0] == "jct"]
    cand = []
    for pk, (px, py) in pj:
        for gk, (gx, gy) in gj:
            d = float(np.hypot(px - gx, py - gy))
            if d <= jct_tol: cand.append((d, pk, gk))
    cand.sort(key=lambda t: t[0])
    for d, pk, gk in cand:
        if pk in m or gk in used: continue
        m[pk] = gk; used.add(gk)
    G = {tuple(sorted([u, v], key=str)) for u, v in gt_edges if u != v}
    Pe = {tuple(sorted([u, v], key=str)) for u, v in pr_edges if u != v}
    tp, mapped = 0, set()
    for u, v in Pe:
        if u in m and v in m:
            e = tuple(sorted([m[u], m[v]], key=str))
            if e in G and e not in mapped: mapped.add(e); tp += 1
    P = tp / len(Pe) if Pe else 0.0; R = tp / len(G) if G else 0.0; F = 2 * P * R / (P + R) if P + R else 0.0
    return P, R, F, float(tp == len(G) == len(Pe))


def match_by_score(pred, ref, thr=0.5, ok=None):
    """Matching of the paper's symbol scores (paper_numbers.py): predictions in order of decreasing score each take the
    unmatched reference box with the highest IoU >= thr. ok(i, j) can forbid a pair (class-aware matching).
    pred: [{"box", "score"}], ref: [box]. Returns [(prediction index, reference index)]."""
    order = sorted(range(len(pred)), key=lambda i: -pred[i]["score"])
    used = [False] * len(ref); out = []
    for i in order:
        p = pred[i]["box"]; best, bj = thr, -1
        for j, q in enumerate(ref):
            if not used[j] and (ok is None or ok(i, j)):
                v = iou(p, q)
                if v >= best: best, bj = v, j
        if bj >= 0: used[bj] = True; out.append((i, bj))
    return out


def prf(tp, fp, fn):
    P = tp / max(1, tp + fp); R = tp / max(1, tp + fn)
    return P, R, 2 * P * R / max(1e-9, P + R)


def counts(tp, fp, fn):
    P, R, F = prf(tp, fp, fn)
    return {"tp": tp, "fp": fp, "fn": fn, "P": P, "R": R, "F1": F}


def levenshtein(a, b):
    if len(a) < len(b): a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


# ------------------------------------------------------------------ dataset

def image_size(path):
    """(width, height), from the PNG header when possible."""
    with open(path, "rb") as f: head = f.read(24)
    if head[:8] == b"\x89PNG\r\n\x1a\n" and head[12:16] == b"IHDR": return struct.unpack(">II", head[16:24])
    import cv2
    g = cv2.imread(path, cv2.IMREAD_UNCHANGED); return g.shape[1], g.shape[0]


class Dataset:
    def __init__(self, root):
        self.root = os.path.abspath(root); self.tv = os.path.join(self.root, "topology_verified")
        if not os.path.isfile(os.path.join(self.tv, "manual_edits.json")): sys.exit(f"not a dataset folder (no topology_verified/): {root}")
        self._code = None; self._edits = None

    def path(self, *p): return os.path.join(self.root, *p)

    def pages(self, which):
        sp = json.load(open(self.path("experiments", "split.json")))
        return sorted({"test": sp["B_test"], "train": sp["A_train"], "all": sp["A_train"] + sp["B_test"]}[which])

    def code(self):
        """topo, pnid_graph_upstream and build_connectivity of topology_verified/code."""
        if self._code is None:
            sys.dont_write_bytecode = True      # no __pycache__ in the dataset folder
            sys.path.insert(0, os.path.join(self.tv, "code"))
            import topo, pnid_graph_upstream, build_connectivity
            self._code = (topo, pnid_graph_upstream, build_connectivity)
        return self._code

    def edits(self):
        if self._edits is None: self._edits = json.load(open(os.path.join(self.tv, "manual_edits.json")))
        return self._edits

    def gray(self, n):
        import cv2
        return cv2.imread(os.path.join(self.tv, "images", n + ".png"), 0)

    def symbol_boxes(self, n):
        """Reference symbol boxes in page pixels (floats, as paper_numbers.py); id = line number in labels/<page>.txt.
        Checks that ids and (truncated) boxes agree with the connectivity file topology_verified/<page>.json."""
        W, H = image_size(os.path.join(self.tv, "images", n + ".png")); out = []
        for l in open(self.path("labels", n + ".txt")):
            p = l.split()
            if len(p) >= 5:
                cx, cy, w, h = map(float, p[1:5]); out.append([(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H])
        ref = {s["id"]: s["bbox"] for s in json.load(open(os.path.join(self.tv, n + ".json")))["symbols"]}
        if sorted(ref) != list(range(len(out))) or any([int(v) for v in b] != ref[i] for i, b in enumerate(out)):
            print(f"warning: labels/{n}.txt and topology_verified/{n}.json disagree (ids or boxes)", file=sys.stderr)
        return out

    def classes(self):
        """{(page, box id): (class, actuator, fault-tree type)} and the table {(class, actuator): fault-tree type}."""
        ref = {(r["page"], int(r["box_id"])): (r["class"], r["actuator"], r["ft_component_type"])
               for r in csv.DictReader(open(self.path("symbol_classes", "symbol_classes.csv")))}
        ft = {}
        for r in csv.DictReader(open(self.path("symbol_classes", "ft_component_types.csv"))):
            c, a = r["class|actuator"].split("|"); ft[(c, a)] = r["ft_component_type"]
        return ref, ft

    def graph(self, n):
        """Reference graph as in the paper's rescoring script, plus the unordered symbol pairs of `neighbors`."""
        d = json.load(open(os.path.join(self.tv, n + ".json")))
        nodes = {("sym", s["id"]): None for s in d["symbols"]}; nodes.update({("jct", j["id"]): tuple(j["xy"]) for j in d["junctions"]})
        key = lambda x: ("jct", x) if isinstance(x, str) else ("sym", int(x))
        edges = [(key(l["source"]), key(l["target"])) for l in d["links"]]
        boxes = {s["id"]: tuple(s["bbox"]) for s in d["symbols"]}
        pairs = {tuple(sorted((int(a), int(b)))) for a, lst in d["neighbors"].items() for b in lst if int(a) != int(b)}
        return nodes, edges, boxes, pairs

    def gt_lines(self, n):
        """The pipe-line layer the connectivity was built from (build_connectivity.build): lines/<page>.json solid and dashed,
        without removed_lines, labeled lines clipped out of vector_regions and replaced by vector_lines, plus added_lines.
        lines/ does not contain these corrections, so they are applied here once. Integer page pixels."""
        topo, _, bc = self.code(); e = self.edits().get(n, {})
        L = json.load(open(self.path("lines", n + ".json"))); drop = e.get("removed_lines", [])
        segs = [tuple(map(int, s)) for s in L["solid"] + L.get("dashed", []) if not any(bc.same_line(s, d) for d in drop)]
        if e.get("vector_regions"):
            kept = []
            for sg in segs:
                for part in bc.clip_out(list(sg), e["vector_regions"]):
                    if topo.seg_len(part) >= 6: kept.append(tuple(int(round(v)) for v in part))
            segs = kept + [tuple(int(v) for v in s) for s in e.get("vector_lines", [])]
        segs += [tuple(int(round(v)) for v in a) for a in e.get("added_lines", [])]
        return segs

    def gt_lines_typed(self, n):
        """gt_lines() with the solid / dashed type of every segment kept; added pieces and vector-region lines are solid."""
        topo, _, bc = self.code(); e = self.edits().get(n, {})
        L = json.load(open(self.path("lines", n + ".json"))); drop = e.get("removed_lines", []); out = {}
        for t in ("solid", "dashed"):
            segs = [tuple(map(int, s)) for s in L.get(t, []) if not any(bc.same_line(s, d) for d in drop)]
            if e.get("vector_regions"):
                kept = []
                for sg in segs:
                    for part in bc.clip_out(list(sg), e["vector_regions"]):
                        if topo.seg_len(part) >= 6: kept.append(tuple(int(round(v)) for v in part))
                segs = kept
            out[t] = segs
        if e.get("vector_regions"): out["solid"] += [tuple(int(v) for v in s) for s in e.get("vector_lines", [])]
        out["solid"] += [tuple(int(round(v)) for v in a) for a in e.get("added_lines", [])]
        return out


# ------------------------------------------------------------------ input files

def alt_names(n):
    return [n, n[len(PREFIX):]] if n.startswith(PREFIX) else [n, PREFIX + n]


def find_page(d, n):
    for k in alt_names(n):
        if k in d: return d[k]
    return None


def find_file(folder, n):
    for k in alt_names(n):
        f = os.path.join(folder, k + ".json")
        if os.path.exists(f): return f
    return None


def norm_label(x):
    if x is None: return None
    x = str(x).strip().upper()
    return x or None


def load_symbol_preds(path):
    """{page: [{"box", "score", "class", "actuator"}]} from {"pages": {page: [...]}}; a box is [x0,y0,x1,y1,score(,class
    (,actuator))] or {"bbox", "score", "class", "actuator"}; a page entry may also be {"symbols": [...]} (graph output)."""
    d = json.load(open(path)); P = d.get("pages", d); out = {}
    for k, lst in P.items():
        if isinstance(lst, dict): lst = lst.get("symbols", [])
        boxes = []
        for p in lst:
            if isinstance(p, dict):
                b, s, c, a = p.get("bbox", p.get("box")), p.get("score"), p.get("class"), p.get("actuator")
            else:
                b, s = p[:4], (p[4] if len(p) > 4 else 1.0); c = p[5] if len(p) > 5 else None; a = p[6] if len(p) > 6 else None
            boxes.append({"box": [float(v) for v in b[:4]], "score": float(1.0 if s is None else s), "class": norm_label(c), "actuator": norm_label(a)})
        out[k] = boxes
    return out


def load_lines(path):
    if path is None: return []
    L = json.load(open(path))
    return [tuple(int(round(v)) for v in s[:4]) for s in L.get("solid", []) + L.get("dashed", [])]


# ------------------------------------------------------------------ output

def fmt(v, w=8):
    if v is None: return f"{'-':>{w}s}"
    if isinstance(v, (int, np.integer)) and not isinstance(v, bool): return f"{v:>{w}d}"
    return f"{v:>{w}.3f}"


def print_table(title, header, rows, first=12):
    print(title)
    print(" " * first + "".join(f"{h:>8s}" for h in header))
    for name, vals in rows: print(f"{name:<{first}s}" + "".join(fmt(v) for v in vals))


def mean_of(values):
    vals = [v for v in values if v is not None]
    return (float(np.mean(vals)) if vals else None), len(vals)


def finish(a, task, per_page, mean, totals, missing, extra=None):
    if missing:
        print(f"warning: no prediction for {len(missing)} page(s), scored as empty: " + "; ".join(missing), file=sys.stderr)
    if a.out:
        res = {"task": task, "pages": a.pages, "n_pages": len(per_page), "missing_pages": missing, "mean": mean,
               "totals": totals, "per_page": per_page}
        res.update(extra or {})
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=1)
        print(f"wrote {a.out}")


# ------------------------------------------------------------------ subcommands

def cmd_symbols(a, ds):
    preds = load_symbol_preds(a.pred); per, missing = {}, []
    if a.classes:
        ref_cls, ft = ds.classes()
        by_class = {}
        for (c, act), t in ft.items(): by_class.setdefault(c, set()).add(t)

        def ft_type(c, act):
            if c is None: return None
            if (c, act) in ft: return ft[(c, act)]
            return next(iter(by_class[c])) if len(by_class.get(c, ())) == 1 else None
    no_class = 0
    for n in ds.pages(a.pages):
        ref = ds.symbol_boxes(n); pr = find_page(preds, n)
        if pr is None: missing.append(n); pr = []
        variants = [("box", None)]
        if a.classes:
            rc = [ref_cls.get((n, j), (None, None, None)) for j in range(len(ref))]
            pt = [ft_type(p["class"], p["actuator"]) for p in pr]; no_class += sum(p["class"] is None for p in pr)
            variants += [("class", lambda i, j: pr[i]["class"] is not None and pr[i]["class"] == rc[j][0]),
                         ("ft_type", lambda i, j: pt[i] is not None and pt[i] == rc[j][2])]
        r = {}
        for name, ok in variants:
            tp = len(match_by_score(pr, ref, 0.5, ok)); r[name] = counts(tp, len(pr) - tp, len(ref) - tp)
        per[n] = r
    if no_class: print(f"warning: {no_class} predicted boxes have no class", file=sys.stderr)
    names = list(next(iter(per.values())).keys())
    mean = {v: {k: mean_of([r[v][k] for r in per.values()])[0] for k in ("P", "R", "F1")} for v in names}
    tot = {v: {k: int(sum(r[v][k] for r in per.values())) for k in ("tp", "fp", "fn")} for v in names}
    print_table(f"symbols, {a.pages} pages ({len(per)}), IoU >= 0.5, macro over pages; TP/FP/FN summed",
                ["P", "R", "F1", "TP", "FP", "FN"],
                [(v, [mean[v]["P"], mean[v]["R"], mean[v]["F1"], tot[v]["tp"], tot[v]["fp"], tot[v]["fn"]]) for v in names])
    finish(a, "symbols", per, mean, tot, missing, {"pred": os.path.basename(a.pred)})


ORIENT = {"0": "horizontal", "180": "horizontal", "cw": "vertical", "ccw": "vertical"}
SUBSETS = ("all", "horizontal", "vertical")


def norm_rot(r):
    if r is None: return "0"
    r = str(r).strip().lower()
    return "0" if r in ("", "0.0") else r


def norm_text(t): return re.sub(r"\s+", "", t or "").upper()


def cover(p, r):
    """Share of the reference box r covered by the box p."""
    ix = max(0, min(p[2], r[2]) - max(p[0], r[0])); iy = max(0, min(p[3], r[3]) - max(p[1], r[1])); A = (r[2] - r[0]) * (r[3] - r[1])
    return ix * iy / A if A > 0 else 0.0


def text_page(ref_items, pred_items):
    R = [{"box": [float(v) for v in it["box"][:4]], "text": norm_text(it["text"]), "rot": norm_rot(it["rot"]),
          "care": it["text"].strip() != "###"} for it in ref_items]
    P = [{"box": [float(v) for v in it["box"][:4]], "score": float(1.0 if it.get("score") is None else it["score"]),
          "text": norm_text(it.get("text")), "rot": norm_rot(it.get("rot"))} for it in pred_items]
    pairs = match_by_score(P, [r["box"] for r in R], 0.5); matched = dict(pairs)
    out = {}
    for sub in SUBSETS:
        ins = lambda rot: sub == "all" or ORIENT.get(rot) == sub
        refs = [j for j, r in enumerate(R) if r["care"] and ins(r["rot"])]
        if not refs: out[sub] = None; continue
        tp = [(i, j) for i, j in pairs if R[j]["care"] and ins(R[j]["rot"])]
        fp = sum(1 for i in range(len(P)) if i not in matched and ins(P[i]["rot"]))
        r = counts(len(tp), fp, len(refs) - len(tp))
        nchar = sum(len(R[j]["text"]) for _, j in tp)
        r.update(pairs=len(tp), exact_n=sum(P[i]["text"] == R[j]["text"] for i, j in tp),
                 edits=sum(levenshtein(P[i]["text"], R[j]["text"]) for i, j in tp), ref_chars=nchar,
                 direction_n=sum(P[i]["rot"] == R[j]["rot"] for i, j in tp),
                 read_n=sum(any(p["text"] == R[j]["text"] and cover(p["box"], R[j]["box"]) >= 0.5 for p in P) for j in refs), refs=len(refs))
        r.update(exact=r["exact_n"] / len(tp) if tp else None, cer=r["edits"] / max(1, nchar) if tp else None,
                 direction=r["direction_n"] / len(tp) if tp else None, read_rate=r["read_n"] / len(refs))
        out[sub] = r
    return out


TEXT_KEYS = ("P", "R", "F1", "exact", "cer", "direction", "read_rate")


def cmd_text(a, ds):
    per, missing = {}, []
    for n in ds.pages(a.pages):
        f = find_file(a.pred, n); pred = json.load(open(f)) if f else []
        if f is None: missing.append(n)
        per[n] = text_page(json.load(open(ds.path("text", n + ".json")))["items"], pred["items"] if isinstance(pred, dict) else pred)
    mean, tot = {}, {}
    for sub in SUBSETS:
        rows = [r[sub] for r in per.values() if r[sub] is not None]
        mean[sub] = {k: mean_of([r[k] for r in rows])[0] for k in TEXT_KEYS}
        mean[sub]["n_pages"] = len(rows); mean[sub]["n_pages_with_pairs"] = mean_of([r["exact"] for r in rows])[1]
        tot[sub] = {k: int(sum(r[k] for r in rows)) for k in ("tp", "fp", "fn", "pairs", "exact_n", "edits", "ref_chars", "direction_n", "read_n", "refs")}
    print_table(f"text, {a.pages} pages ({len(per)}), detection at IoU >= 0.5; exact, CER, direction over matched pairs; macro over pages",
                ["det P", "det R", "det F1", "exact", "CER", "dir", "read", "pages"],
                [(s, [mean[s][k] for k in TEXT_KEYS] + [mean[s]["n_pages"]]) for s in SUBSETS])
    finish(a, "text", per, mean, tot, missing, {"pred": os.path.basename(os.path.normpath(a.pred))})


def cmd_lines(a, ds):
    per, missing = {}, []
    for n in ds.pages(a.pages):
        f = find_file(a.pred, n)
        if f is None: missing.append(n)
        gt, ps = ds.gt_lines(n), load_lines(f)
        P, R, F = count_scores(gt, ps); per[n] = {"P": P, "R": R, "F1": F, "n_gt": len(gt), "n_pred": len(ps)}
    mean = {k: mean_of([r[k] for r in per.values()])[0] for k in ("P", "R", "F1")}
    tot = {k: int(sum(r[k] for r in per.values())) for k in ("n_gt", "n_pred")}
    print_table(f"lines, {a.pages} pages ({len(per)}), count F1 at 8 px / 10 deg / half the length, macro over pages",
                ["P", "R", "F1", "ref", "pred"], [("lines", [mean["P"], mean["R"], mean["F1"], tot["n_gt"], tot["n_pred"]])])
    finish(a, "lines", per, mean, tot, missing, {"pred": os.path.basename(os.path.normpath(a.pred))})


def pairs_from_edges(bc, syms, edges):
    """Unordered symbol pairs of a built graph, as `neighbors` is defined (build_connectivity.neighbors): reachable along
    links through junctions without passing another symbol."""
    nid = lambda u: u[1] if u[0] == "sym" else f"J{u[1]}"
    nb = bc.neighbors([{"id": i} for i, _ in syms], [{"source": nid(u), "target": nid(v)} for u, v, *_ in edges])
    return {tuple(sorted((int(x), y))) for x, lst in nb.items() for y in lst if int(x) != y}


def pair_counts(pred_pairs, m, gold):
    """pred_pairs over predicted ids, m: predicted id -> reference id (one to one), gold: reference pairs.
    A pair with an unmatched symbol is a false positive."""
    mapped = {tuple(sorted((m[x], m[y]))) for x, y in pred_pairs if x in m and y in m}
    tp = len(mapped & gold)
    return counts(tp, len(pred_pairs) - tp, len(gold) - tp)


def cmd_graph(a, ds):
    topo, up, bc = ds.code()
    spred = None if a.symbols == "gt" else load_symbol_preds(a.symbols)
    per, missing = {}, []; t_all = time.time()
    for n in ds.pages(a.pages):
        t0 = time.time()
        g = ds.gray(n); H, W = g.shape
        gn, ge, gb, gpairs = ds.graph(n)
        if spred is None:
            syms = bc.load_symbols(n, W, H)
        else:
            pr = find_page(spred, n)
            if pr is None: missing.append(n + " (symbols)"); pr = []
            syms = [(i, tuple(int(round(v)) for v in p["box"])) for i, p in enumerate(pr)]
        if a.lines == "gt":
            segs = ds.gt_lines(n)
        else:
            f = find_file(a.lines, n)
            if f is None: missing.append(n + " (lines)")
            segs = load_lines(f)
        S = [(s, "solid") for s in segs]
        if a.no_dot: nodes, edges = up.build_topology(S, syms, tol=12)
        else: nodes, edges, *_ = topo.build(S, syms, g)
        boxes = {i: b for i, b in syms}
        P, R, F, X = graph_score(gn, ge, gb, nodes, [(u, v) for u, v, _ in edges], boxes)
        m = match_iou(boxes, gb)
        per[n] = {"edge": {"P": P, "R": R, "F1": F, "exact": X},
                  "pair": pair_counts(pairs_from_edges(bc, syms, edges), m, gpairs),
                  "symbol": counts(len(m), len(boxes) - len(m), len(gb) - len(m)), "seconds": round(time.time() - t0, 2)}
        print(f"{time.time() - t_all:6.0f}s  {n[len(PREFIX):][:60]:60s} edge F1 {F:.3f}  pair F1 {per[n]['pair']['F1']:.3f}", file=sys.stderr, flush=True)
    mean = {"edge": {k: mean_of([r["edge"][k] for r in per.values()])[0] for k in ("P", "R", "F1", "exact")}}
    for v in ("pair", "symbol"): mean[v] = {k: mean_of([r[v][k] for r in per.values()])[0] for k in ("P", "R", "F1")}
    tot = {"exact_pages": int(sum(r["edge"]["exact"] for r in per.values())), "seconds": round(time.time() - t_all, 1)}
    for v in ("pair", "symbol"): tot[v] = {k: int(sum(r[v][k] for r in per.values())) for k in ("tp", "fp", "fn")}
    asm = "pnid_graph_upstream.build_topology (no dot test)" if a.no_dot else "topo.build"
    print_table(f"graph, {a.pages} pages ({len(per)}), symbols {a.symbols if a.symbols == 'gt' else os.path.basename(a.symbols)}, "
                f"lines {a.lines if a.lines == 'gt' else os.path.basename(os.path.normpath(a.lines))}, {asm}; macro over pages",
                ["P", "R", "F1", "exact"],
                [("edges", [mean["edge"][k] for k in ("P", "R", "F1", "exact")]),
                 ("symbol pairs", [mean["pair"][k] for k in ("P", "R", "F1")] + [None]),
                 ("symbols", [mean["symbol"][k] for k in ("P", "R", "F1")] + [None])], first=14)
    print(f"sheet-exact pages: {tot['exact_pages']} of {len(per)}; time {tot['seconds']:.0f} s")
    finish(a, "graph", per, mean, tot, missing, {"symbols": a.symbols if a.symbols == "gt" else os.path.basename(a.symbols),
                                                "lines": a.lines if a.lines == "gt" else os.path.basename(os.path.normpath(a.lines)),
                                                "assembler": asm})


def cmd_pairs(a, ds):
    P = json.load(open(a.pred)); P = P.get("pages", P); per, missing = {}, []
    for n in ds.pages(a.pages):
        _, _, gb, gpairs = ds.graph(n); e = find_page(P, n)
        if e is None: missing.append(n); e = {"symbols": [], "connections": []}
        conns = []
        for c in e.get("connections", []):
            x, y = (c["source"], c["target"]) if isinstance(c, dict) else (c[0], c[1])
            if str(x) != str(y): conns.append(tuple(sorted((str(x), str(y)))))
        conns = set(conns)
        if "symbols" in e:
            boxes = {}
            for s in e["symbols"]:
                boxes.setdefault(str(s["id"]), [float(v) for v in s.get("bbox", s.get("box"))[:4]])
            m = match_iou(boxes, gb) if a.match == "iou" else match_center(boxes, gb)
            sym = counts(len(m), len(boxes) - len(m), len(gb) - len(m))
        else:      # connections given in reference symbol ids (e.g. symbols marked on the page)
            m = {str(i): i for i in gb}; sym = None
        per[n] = {"pair": pair_counts(conns, m, gpairs), "symbol": sym}
    mean = {"pair": {k: mean_of([r["pair"][k] for r in per.values()])[0] for k in ("P", "R", "F1")},
            "symbol": {k: mean_of([r["symbol"][k] if r["symbol"] else None for r in per.values()])[0] for k in ("P", "R", "F1")}}
    tot = {v: {k: int(sum(r[v][k] for r in per.values() if r[v])) for k in ("tp", "fp", "fn")} for v in ("pair", "symbol")}
    print_table(f"pairs, {a.pages} pages ({len(per)}), symbols matched by {a.match}; macro over pages; TP/FP/FN summed",
                ["P", "R", "F1", "TP", "FP", "FN"],
                [(v, [mean[v]["P"], mean[v]["R"], mean[v]["F1"], tot[v]["tp"], tot[v]["fp"], tot[v]["fn"]]) for v in ("pair", "symbol")])
    finish(a, "pairs", per, mean, tot, missing, {"pred": os.path.basename(a.pred), "match": a.match})


def cmd_export_lines(a, ds):
    """Write the reference pipe-line layer (gt_lines) of the chosen pages as <page>.json in the format of lines/."""
    os.makedirs(a.dir, exist_ok=True); pages = ds.pages(a.pages); n_seg = 0
    for n in pages:
        t = ds.gt_lines_typed(n)
        assert sorted(t["solid"] + t["dashed"]) == sorted(ds.gt_lines(n)), n
        with open(os.path.join(a.dir, n + ".json"), "w") as f:
            json.dump({"solid": [list(s) for s in t["solid"]], "dashed": [list(s) for s in t["dashed"]]}, f)
        n_seg += len(t["solid"]) + len(t["dashed"])
    print(f"{len(pages)} pages, {n_seg} segments written to {a.dir}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--pages", choices=("test", "train", "all"), default="test", help="B_test (default), A_train or all pages of ../split.json")
    common.add_argument("--v3", default=os.path.dirname(os.path.dirname(HERE)), help="dataset folder (default: two levels up)")
    common.add_argument("--out", help="write per-page and mean scores to this JSON file")
    s = sub.add_parser("symbols", parents=[common], help="symbol boxes: P/R/F1 at IoU >= 0.5")
    s.add_argument("--pred", required=True, help='JSON {"pages": {page: [[x0,y0,x1,y1,score], ...]}}')
    s.add_argument("--classes", action="store_true", help="also class-aware and fault-tree-type-aware F1")
    s = sub.add_parser("text", parents=[common], help="text lines: detection, exact match, CER, direction, read rate")
    s.add_argument("--pred", required=True, help="folder with <page>.json in the format of text/<page>.json")
    s = sub.add_parser("lines", parents=[common], help="pipe lines: count F1 at 8 px and 10 degrees")
    s.add_argument("--pred", required=True, help='folder with <page>.json {"solid": [...], "dashed": [...]}')
    s = sub.add_parser("graph", parents=[common], help="assemble a graph and score edges and symbol pairs")
    s.add_argument("--symbols", required=True, help="'gt' or a JSON file in the format of the symbols subcommand")
    s.add_argument("--lines", required=True, help="'gt' or a folder in the format of the lines subcommand")
    s.add_argument("--no-dot", action="store_true", help="assemble with pnid_graph_upstream.build_topology (no dot test)")
    s = sub.add_parser("pairs", parents=[common], help="connected symbol pairs of a graph without junctions")
    s.add_argument("--pred", required=True, help='JSON {"pages": {page: {"symbols": [{"id", "bbox"}], "connections": [[id1, id2], ...]}}}')
    s.add_argument("--match", choices=("iou", "center"), default="iou", help="symbol matching: IoU >= 0.5 (default) or box centre inside")
    s = sub.add_parser("export-lines", help="write the reference pipe-line layer (lines/ with the corrections applied)")
    s.add_argument("--dir", required=True, help="output folder for <page>.json")
    s.add_argument("--pages", choices=("test", "train", "all"), default="all")
    s.add_argument("--v3", default=os.path.dirname(os.path.dirname(HERE)), help="dataset folder (default: two levels up)")
    a = ap.parse_args(argv)
    ds = Dataset(a.v3)
    {"symbols": cmd_symbols, "text": cmd_text, "lines": cmd_lines, "graph": cmd_graph, "pairs": cmd_pairs,
     "export-lines": cmd_export_lines}[a.cmd](a, ds)


if __name__ == "__main__":
    main()
