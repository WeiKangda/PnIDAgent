#!/usr/bin/env python3
"""Sheet-level symbol detection metrics against Dataset-P&ID symbols.npy.

Scores full-sheet predictions (after tile merging), not tiles, so the numbers
mean what the graph stage will see.  COCO-style AP (101-point interpolation)
at IoU 0.50 and averaged over 0.50:0.95, both class-agnostic and 32-way, plus
per-class AP so the hard classes are visible.  Also P/R/F1 at a working
confidence for the connectivity ladder.

Predictions: {sheet_idx: [{"bbox": [x1,y1,x2,y2], "cls": 0..31, "conf": f}]}
(what finetune_yolo_symbols.py --mode predict_sheets writes).

Usage:
    python tools/eval_symbols.py --pred preds_val.json
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys as _sys; _sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from _paths import ROOT as _ROOT  # <ROOT>/PnIDAgent/eval_tools layout
import sys
from typing import Dict, List, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dpid  # noqa: E402

IOUS = np.round(np.arange(0.5, 0.96, 0.05), 2)


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + ab[None, :] - inter + 1e-9)


def _ap(rec: np.ndarray, prec: np.ndarray) -> float:
    """COCO 101-point interpolated AP: at each recall threshold, the best
    precision at any recall >= it (zero once the curve ends)."""
    mpre = np.array(prec, dtype=float)
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.searchsorted(rec, np.linspace(0, 1, 101), side="left")
    vals = np.where(idx < len(mpre), mpre[np.minimum(idx, len(mpre) - 1)], 0.0)
    return float(np.mean(vals))


def evaluate(gt: Dict[int, List[Tuple[int, Tuple]]], pred: Dict[int, List[Tuple[int, Tuple, float]]],
             agnostic: bool) -> Dict:
    """gt: idx -> [(cls, box)]; pred: idx -> [(cls, box, conf)]."""
    classes = [0] if agnostic else sorted({c for v in gt.values() for c, _ in v})
    # per class, per IoU: collect (conf, tp) across sheets
    ap = {c: {} for c in classes}
    n_gt = collections.Counter()
    for c in classes:
        recs = {t: [] for t in IOUS}
        for idx in gt:
            g = np.array([b for cc, b in gt[idx] if agnostic or cc == c], dtype=float).reshape(-1, 4)
            p = sorted([(conf, b) for cc, b, conf in pred.get(idx, []) if agnostic or cc == c],
                       key=lambda t: -t[0])
            n_gt[c] += len(g)
            pb = np.array([b for _, b in p], dtype=float).reshape(-1, 4)
            iou = _iou_matrix(pb, g)
            for t in IOUS:
                used = np.zeros(len(g), bool)
                for pi, (conf, _) in enumerate(p):
                    tp = 0
                    if len(g):
                        cand = np.where((iou[pi] >= t) & ~used)[0]
                        if len(cand):
                            j = cand[np.argmax(iou[pi][cand])]
                            used[j] = True
                            tp = 1
                    recs[t].append((conf, tp))
        for t in IOUS:
            if n_gt[c] == 0:
                ap[c][t] = float("nan")
                continue
            r = sorted(recs[t], key=lambda x: -x[0])
            tps = np.cumsum([x[1] for x in r])
            fps = np.cumsum([1 - x[1] for x in r])
            rec = tps / n_gt[c]
            prec = tps / np.maximum(tps + fps, 1e-9)
            ap[c][t] = _ap(rec, prec) if len(r) else 0.0
    out = {"per_class": {}, "n_gt": dict(n_gt)}
    ap50, ap5095 = [], []
    for c in classes:
        a50 = ap[c][0.5]
        a = float(np.nanmean([ap[c][t] for t in IOUS]))
        out["per_class"][c] = {"ap50": a50, "ap50_95": a, "n_gt": n_gt[c]}
        if not np.isnan(a50):
            ap50.append(a50)
            ap5095.append(a)
    out["map50"] = float(np.mean(ap50)) if ap50 else 0.0
    out["map50_95"] = float(np.mean(ap5095)) if ap5095 else 0.0
    return out


def prf_at(gt, pred, conf_thr: float, iou_thr: float = 0.5, agnostic: bool = False) -> Dict[str, float]:
    tp = fp = fn = 0
    for idx in gt:
        g = gt[idx]
        p = [x for x in pred.get(idx, []) if x[2] >= conf_thr]
        gb = np.array([b for _, b in g], dtype=float).reshape(-1, 4)
        pb = np.array([b for _, b, _ in p], dtype=float).reshape(-1, 4)
        iou = _iou_matrix(pb, gb)
        used = np.zeros(len(g), bool)
        for pi in np.argsort([-x[2] for x in p]):
            ok = (iou[pi] >= iou_thr) & ~used
            if not agnostic:
                ok &= np.array([c == p[pi][0] for c, _ in g], bool) if len(g) else ok
            cand = np.where(ok)[0]
            if len(cand):
                used[cand[np.argmax(iou[pi][cand])]] = True
                tp += 1
            else:
                fp += 1
        fn += int((~used).sum())
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": P, "recall": R, "f1": 2 * P * R / (P + R) if P + R else 0.0,
            "tp": tp, "fp": fp, "fn": fn}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(
        _ROOT, "dataset"))
    ap.add_argument("--pred", required=True)
    ap.add_argument("--conf", type=float, default=0.25, help="working threshold for P/R/F1")
    ap.add_argument("--gt", default="npy", choices=("npy", "mask", "auto"),
                    help="ground truth: symbols.npy, mask png boxes (IoU 0.97 with npy), or "
                         "npy where complete else mask")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    with open(args.pred) as fh:
        raw = json.load(fh)
    root = os.path.abspath(args.root)
    sys.path.insert(0, os.path.join(_ROOT, "PnIDAgent"))
    gt, pred = {}, {}
    n_mask = 0
    for k, rows in raw.items():
        idx = int(k)
        use_npy = args.gt == "npy" or (args.gt == "auto" and dpid.is_complete(root, idx))
        if use_npy:
            if not dpid.is_complete(root, idx):
                continue
            sheet = dpid.load_sheet(root, idx)
            gt[idx] = [(s.cls - 1, s.box) for s in sheet.symbols]
        else:
            from finetune_yolo_symbols import load_symbols_from_mask
            gt[idx] = [(c, (x1, y1, x2, y2)) for c, x1, y1, x2, y2 in
                       load_symbols_from_mask(os.path.join(root, "mask", f"{idx}_mask.png"))]
            n_mask += 1
        pred[idx] = [(int(r["cls"]), tuple(r["bbox"]), float(r["conf"])) for r in rows]
    if n_mask:
        print(f"note: {n_mask} sheets scored against mask-derived boxes")

    agn = evaluate(gt, pred, agnostic=True)
    cls = evaluate(gt, pred, agnostic=False)
    prf_a = prf_at(gt, pred, args.conf, agnostic=True)
    prf_c = prf_at(gt, pred, args.conf, agnostic=False)
    print(f"sheets={len(gt)}  gt boxes={sum(len(v) for v in gt.values())}  "
          f"pred boxes={sum(len(v) for v in pred.values())}")
    print(f"  class-agnostic  mAP50 {agn['map50']:.3f}  mAP50-95 {agn['map50_95']:.3f}   "
          f"@conf{args.conf}: P {prf_a['precision']:.3f} R {prf_a['recall']:.3f} F1 {prf_a['f1']:.3f}")
    print(f"  32-way          mAP50 {cls['map50']:.3f}  mAP50-95 {cls['map50_95']:.3f}   "
          f"@conf{args.conf}: P {prf_c['precision']:.3f} R {prf_c['recall']:.3f} F1 {prf_c['f1']:.3f}")
    print("  per-class (dataset id = cls+1):")
    rows = sorted(cls["per_class"].items(), key=lambda kv: kv[1]["ap50"])
    for c, v in rows:
        print(f"    c{c + 1:02d}  n={v['n_gt']:>5}  AP50 {v['ap50']:.3f}  AP50-95 {v['ap50_95']:.3f}")
    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"config": vars(args), "sheets": len(gt), "agnostic": agn, "classwise": cls,
                       "prf_agnostic": prf_a, "prf_classwise": prf_c}, fh, indent=1)
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
