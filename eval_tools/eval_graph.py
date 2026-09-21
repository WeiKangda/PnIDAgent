#!/usr/bin/env python3
"""Connectivity evaluator: predicted pipe topology vs dpid.derive_graph().

Metric
  Nodes are matched first: symbol nodes by box IoU >= `sym_iou` (identity when
  GT symbols are fed), junction nodes by nearest position within `jct_tol`.
  A predicted edge (u, v) is a true positive when both endpoints matched and
  the matched pair is a GT edge.  Reported per sheet, then averaged:

    edge P / R / F1      unordered node pairs; line kind ignored
    exact                1 when every GT edge is recovered and no extra edge
    deg_err              mean |deg_pred - deg_gt| over GT symbols (unmatched
                         GT symbols count as degree 0)
    deg_exact            fraction of GT symbols with exactly the right degree
    sym-closure F1       symbol-to-symbol adjacency (derive_edges view), for
                         comparison only -- see FINDINGS.md for why it is not
                         the target

Ablation ladder (--symbols / --lines each 'gt' or a prediction source):

    python tools/eval_graph.py --symbols gt --lines gt                  # digitize_pnid, topology
    python tools/eval_graph.py --symbols gt --lines gt --assembler repo-chains   # legacy
    python tools/eval_graph.py --symbols gt --lines pred
    python tools/eval_graph.py --symbols yolo:/path/preds.json --lines gt
    python tools/eval_graph.py --symbols yolo:/path/preds.json --lines pred

Prediction sources
  lines 'pred'        the tuned classical line stage (line_variants.detect)
  lines 'json:PATH'   {sheet_idx: {"solid": [[x1,y1,x2,y2],..], "dashed": [...]}}
  symbols 'yolo:PATH' {sheet_idx: [{"bbox": [x1,y1,x2,y2], "cls": int, "conf": f}]}
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys as _sys; _sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from _paths import ROOT as _ROOT  # <ROOT>/PnIDAgent/eval_tools layout
import sys
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dpid  # noqa: E402
import graph_assembly as GA  # noqa: E402

Box = Tuple[int, int, int, int]


def _iou(a: Box, b: Box) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if not inter:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua else 0.0


# ---------------------------------------------------------------------------
# node matching
# ---------------------------------------------------------------------------

def match_nodes(gt: dpid.Graph, pred: dpid.Graph, gt_boxes: Dict[str, Box],
                pred_boxes: Dict[str, Box], sym_iou: float, jct_tol: float
                ) -> Dict[object, object]:
    """pred node -> gt node, greedy best-first."""
    m: Dict[object, object] = {}
    # symbols: identity if ids coincide with GT ids and boxes match, else IoU
    cand = []
    for pn in pred.nodes:
        if pn[0] != "sym":
            continue
        pb = pred_boxes.get(pn[1])
        for gn in gt.nodes:
            if gn[0] != "sym":
                continue
            gb = gt_boxes.get(gn[1])
            if pb is None or gb is None:
                continue
            v = _iou(pb, gb)
            if v >= sym_iou:
                cand.append((v, pn, gn))
    cand.sort(reverse=True)
    used = set()
    for _, pn, gn in cand:
        if pn in m or gn in used:
            continue
        m[pn] = gn
        used.add(gn)

    cand = []
    for pn, (px, py) in pred.nodes.items():
        if pn[0] != "jct":
            continue
        for gn, (gx, gy) in gt.nodes.items():
            if gn[0] != "jct":
                continue
            d = float(np.hypot(px - gx, py - gy))
            if d <= jct_tol:
                cand.append((d, pn, gn))
    cand.sort()
    for _, pn, gn in cand:
        if pn in m or gn in used:
            continue
        m[pn] = gn
        used.add(gn)
    return m


def closure(g: dpid.Graph) -> set:
    """Symbol-to-symbol adjacency through junctions (derive_edges view)."""
    adj = collections.defaultdict(list)
    for u, v, _ in g.edges:
        adj[u].append(v)
        adj[v].append(u)
    out = set()
    for n in g.nodes:
        if n[0] != "sym":
            continue
        seen, stack = {n}, list(adj[n])
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            if cur[0] == "sym":
                out.add(tuple(sorted([n, cur], key=str)))
                continue
            stack.extend(x for x in adj[cur] if x not in seen)
    return out


def _prf(tp: int, np_: int, ng: int) -> Tuple[float, float, float]:
    p = tp / np_ if np_ else 0.0
    r = tp / ng if ng else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def score(gt: dpid.Graph, pred: dpid.Graph, gt_boxes: Dict[str, Box],
          pred_boxes: Dict[str, Box], sym_iou: float = 0.5, jct_tol: float = 20.0
          ) -> Dict[str, float]:
    m = match_nodes(gt, pred, gt_boxes, pred_boxes, sym_iou, jct_tol)
    gt_edges = {tuple(sorted([u, v], key=str)) for u, v, _ in gt.edges}
    pred_edges = {tuple(sorted([u, v], key=str)) for u, v, _ in pred.edges}
    mapped = set()
    tp = 0
    for u, v in pred_edges:
        if u in m and v in m:
            e = tuple(sorted([m[u], m[v]], key=str))
            if e in gt_edges and e not in mapped:
                mapped.add(e)
                tp += 1
    p, r, f1 = _prf(tp, len(pred_edges), len(gt_edges))

    # degree over GT symbols
    gdeg, pdeg = gt.degree(), pred.degree()
    inv = {g: pn for pn, g in m.items()}
    errs, exact = [], 0
    for gn in gt.nodes:
        if gn[0] != "sym":
            continue
        pd = pdeg.get(inv[gn], 0) if gn in inv else 0
        errs.append(abs(pd - gdeg.get(gn, 0)))
        exact += errs[-1] == 0
    n_sym = len(errs)

    gc, pc = closure(gt), closure(pred)
    pc_m = {tuple(sorted([m[u], m[v]], key=str)) for u, v in pc if u in m and v in m}
    cp, cr, cf1 = _prf(len(pc_m & gc), len(pc), len(gc))

    n_gt_jct = sum(1 for n in gt.nodes if n[0] == "jct")
    n_pr_jct = sum(1 for n in pred.nodes if n[0] == "jct")
    return {"edge_p": p, "edge_r": r, "edge_f1": f1, "tp": tp,
            "n_pred_edges": len(pred_edges), "n_gt_edges": len(gt_edges),
            "exact": float(tp == len(gt_edges) == len(pred_edges)),
            "deg_err": float(np.mean(errs)) if errs else 0.0,
            "deg_exact": exact / n_sym if n_sym else 0.0,
            "closure_p": cp, "closure_r": cr, "closure_f1": cf1,
            "n_gt_jct": n_gt_jct, "n_pred_jct": n_pr_jct,
            "jct_matched": sum(1 for pn in m if pn[0] == "jct"),
            "sym_matched": sum(1 for pn in m if pn[0] == "sym"),
            "n_gt_sym": n_sym}


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------

def gt_inputs(sheet: dpid.Sheet):
    segs = [(l.seg, l.kind) for l in sheet.lines]
    syms = [(s.id, s.box) for s in sheet.symbols]
    return segs, syms


def pred_lines_classical(root: str, idx: int, max_dim, max_trans):
    """The repo's line stage as configured in SOLID_CFG (overrides optional)."""
    import cv2
    import line_variants as LV
    ptl = LV.import_line_stage()
    LV.configure(ptl, max_dim=max_dim, max_transitions=max_trans)
    img = cv2.imread(os.path.join(root, "image_2", f"{idx}.jpg"))
    out = LV.detect(ptl, img, solid_source="canny")["merged"]
    return [(tuple(int(v) for v in s), "solid") for s in out]


def _load_json_source(spec: str) -> Dict[str, object]:
    with open(spec.split(":", 1)[1]) as fh:
        return json.load(fh)


def run_sheet(job) -> Dict:
    (root, idx, sym_src, line_src, assembler, tol, pad, jct_tol, sym_iou,
     max_dim, max_trans, max_line_distance, gt_tol, sym_conf) = job
    sheet = dpid.load_sheet(root, idx)
    gt = dpid.derive_graph(sheet, tol=gt_tol)
    gt_boxes = {s.id: s.box for s in sheet.symbols}

    gsegs, gsyms = gt_inputs(sheet)
    if line_src == "gt":
        segs = gsegs
    elif line_src == "pred":
        segs = pred_lines_classical(root, idx, max_dim, max_trans)
    else:
        d = _load_json_source(line_src).get(str(idx), {})
        segs = [(tuple(int(v) for v in s), "solid") for s in d.get("solid", [])] + \
               [(tuple(int(v) for v in s), "dashed") for s in d.get("dashed", [])]

    if sym_src == "gt":
        syms = gsyms
    else:
        rows = [r for r in _load_json_source(sym_src).get(str(idx), [])
                if float(r.get("conf", 1.0)) >= sym_conf]
        syms = [(f"p{i}", tuple(int(round(v)) for v in r["bbox"])) for i, r in enumerate(rows)]
    pred_boxes = dict(syms)

    if assembler == "repo":
        pred = GA.repo_links(segs, syms, max_line_distance=max_line_distance,
                             assembler="topology", snap_tol=tol, symbol_pad=pad)
    elif assembler == "repo-chains":
        pred = GA.repo_links(segs, syms, max_line_distance=max_line_distance,
                             assembler="chains")
    else:
        pred = GA.assemble(segs, syms, tol=tol, pad=pad,
                           drop_inbox_ends=(line_src != "gt"))

    res = score(gt, pred, gt_boxes, pred_boxes, sym_iou=sym_iou, jct_tol=jct_tol)
    res["idx"] = idx
    return res


def summarize(rows: List[Dict]) -> Dict[str, float]:
    m = lambda k: float(np.mean([r[k] for r in rows]))
    out = {k: m(k) for k in ("edge_p", "edge_r", "edge_f1", "exact", "deg_err",
                             "deg_exact", "closure_p", "closure_r", "closure_f1",
                             "n_pred_edges", "n_gt_edges", "n_gt_jct", "n_pred_jct")}
    out["sheets"] = len(rows)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(
        _ROOT, "dataset"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ids", default=None, help="comma list or file of sheet ids")
    ap.add_argument("--symbols", default="gt", help="gt | yolo:PATH")
    ap.add_argument("--lines", default="gt", help="gt | pred | json:PATH")
    ap.add_argument("--assembler", default="repo", choices=("topo", "repo", "repo-chains"),
                    help="repo = digitize_pnid default (topology); repo-chains = legacy; "
                         "topo = pnid_graph.build_topology called directly")
    ap.add_argument("--tol", type=int, default=None,
                    help="assembler endpoint snap px (default 3 for gt lines, 12 for pred)")
    ap.add_argument("--pad", type=int, default=6, help="symbol box pad px")
    ap.add_argument("--jct-tol", type=float, default=20.0, help="junction match px")
    ap.add_argument("--sym-iou", type=float, default=0.5)
    ap.add_argument("--sym-conf", type=float, default=0.25, help="confidence cut for predicted symbols")
    ap.add_argument("--gt-tol", type=int, default=3, help="derive_graph tol")
    ap.add_argument("--max-dim", type=int, default=None, help="override SOLID_CFG target_max_dim")
    ap.add_argument("--max-trans", type=int, default=None,
                    help="override max_transitions (re-enables the solidity filter)")
    ap.add_argument("--max-line-distance", type=float, default=50.0,
                    help="digitize_pnid max_line_distance (repo assembler)")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    if args.ids:
        if os.path.exists(args.ids):
            ids = [int(x) for x in open(args.ids).read().split()]
        else:
            ids = [int(x) for x in args.ids.split(",")]
    else:
        ids = dpid.sheet_ids(root)
    if args.limit:
        ids = ids[:args.limit]
    tol = args.tol if args.tol is not None else (3 if args.lines == "gt" else 12)

    jobs = [(root, i, args.symbols, args.lines, args.assembler, tol, args.pad,
             args.jct_tol, args.sym_iou, args.max_dim, args.max_trans,
             args.max_line_distance, args.gt_tol, args.sym_conf) for i in ids]
    rows: List[Dict] = []
    with ProcessPoolExecutor(args.workers) as ex:
        for r in ex.map(run_sheet, jobs):
            rows.append(r)
            if len(rows) % 25 == 0 or len(rows) == len(jobs):
                print(f"  [{len(rows)}/{len(jobs)}]", flush=True)

    s = summarize(rows)
    print(f"\nsymbols={args.symbols} lines={args.lines} assembler={args.assembler} "
          f"tol={tol} pad={args.pad} jct_tol={args.jct_tol} sheets={s['sheets']}")
    print(f"  edge   P {s['edge_p']:.3f}  R {s['edge_r']:.3f}  F1 {s['edge_f1']:.3f}   "
          f"exact-sheet {s['exact']:.3f}   pred/gt edges {s['n_pred_edges']:.0f}/{s['n_gt_edges']:.0f}")
    print(f"  degree err {s['deg_err']:.3f}  deg-exact {s['deg_exact']:.3f}   "
          f"junctions pred/gt {s['n_pred_jct']:.0f}/{s['n_gt_jct']:.0f}")
    print(f"  sym-closure (comparison only)  P {s['closure_p']:.3f}  R {s['closure_r']:.3f}  "
          f"F1 {s['closure_f1']:.3f}")
    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"config": vars(args), "summary": s, "rows": rows}, fh, indent=1)
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
