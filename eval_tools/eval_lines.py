#!/usr/bin/env python3
"""Score the repo's solid-line detector against Dataset-P&ID line annotations.

Primary metric is the buffer method (Wiedemann): rasterize both sets, then ask
what fraction of GT length falls within `tol` of a prediction (completeness)
and what fraction of predicted length falls within `tol` of GT (correctness).
Length-weighted, so it does not collapse when one GT pipe is detected as three
fragments -- which classical Hough output does constantly.

Count-based F1 with greedy one-to-one matching is reported alongside it, plus a
fragmentation figure, because the graph stage cares about segment identity and
not just ink coverage.

Usage:
    python tools/eval_lines.py --limit 20                    # baseline
    python tools/eval_lines.py --limit 20 --max-dim 4096     # L1 A/B
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
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dpid  # noqa: E402
import line_variants as LV  # noqa: E402

Seg = Tuple[int, int, int, int]


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def _raster(segs: Sequence[Seg], shape: Tuple[int, int]) -> np.ndarray:
    m = np.zeros(shape, np.uint8)
    for x1, y1, x2, y2 in segs:
        cv2.line(m, (int(x1), int(y1)), (int(x2), int(y2)), 255, 1)
    return m


def buffer_scores(gt: Sequence[Seg], pred: Sequence[Seg], shape: Tuple[int, int],
                  tol: int) -> Dict[str, float]:
    gt_m, pr_m = _raster(gt, shape), _raster(pred, shape)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol + 1, 2 * tol + 1))
    gt_buf = cv2.dilate(gt_m, k)
    pr_buf = cv2.dilate(pr_m, k)
    gt_n, pr_n = int((gt_m > 0).sum()), int((pr_m > 0).sum())
    comp = float(((gt_m > 0) & (pr_buf > 0)).sum()) / gt_n if gt_n else 0.0
    corr = float(((pr_m > 0) & (gt_buf > 0)).sum()) / pr_n if pr_n else 0.0
    f1 = 2 * comp * corr / (comp + corr) if comp + corr else 0.0
    return {"completeness": comp, "correctness": corr, "buffer_f1": f1,
            "gt_px": gt_n, "pred_px": pr_n}


def _ang(s: Seg) -> float:
    return np.degrees(np.arctan2(s[3] - s[1], s[2] - s[0])) % 180.0


def _span(a: Seg, b: Seg, tol: int) -> Tuple[float, float, float]:
    """Project b onto a's axis. Returns (length_of_a, lo, hi), empty if unrelated.

    Both endpoints of b must sit within `tol` of a's infinite line and the two
    must be near-collinear, so this only reports genuine overlap along a pipe.
    """
    L = float(np.hypot(a[2] - a[0], a[3] - a[1]))
    if L < 1 or min(abs(_ang(a) - _ang(b)), 180 - abs(_ang(a) - _ang(b))) > 10.0:
        return L, 0.0, 0.0
    ux, uy = (a[2] - a[0]) / L, (a[3] - a[1]) / L
    for px, py in ((b[0], b[1]), (b[2], b[3])):
        if abs(-uy * (px - a[0]) + ux * (py - a[1])) > tol:
            return L, 0.0, 0.0
    t1 = ux * (b[0] - a[0]) + uy * (b[1] - a[1])
    t2 = ux * (b[2] - a[0]) + uy * (b[3] - a[1])
    return L, max(0.0, min(t1, t2)), min(L, max(t1, t2))


def _union_frac(a: Seg, others: Sequence[Seg], tol: int) -> Tuple[float, int]:
    """Fraction of a covered by the union of `others`, and how many touch it."""
    iv: List[Tuple[float, float]] = []
    for b in others:
        L, lo, hi = _span(a, b, tol)
        if hi > lo:
            iv.append((lo, hi))
    if not iv:
        return 0.0, 0
    iv.sort()
    tot, ce = 0.0, -1.0
    for lo, hi in iv:
        lo = max(lo, ce)
        if hi > lo:
            tot += hi - lo
            ce = hi
    return tot / L, len(iv)


def count_scores(gt: Sequence[Seg], pred: Sequence[Seg], tol: int,
                 min_overlap: float = 0.5) -> Dict[str, float]:
    """Per-segment detection rates, matched many-to-many.

    One-to-one matching is wrong here: GT splits every straight pipe run at each
    junction, while the detector's merge step happily emits one segment spanning
    the whole run. So a GT segment counts as detected when the union of
    predictions covers min_overlap of it, and a prediction counts as valid when
    GT covers min_overlap of it. `fragmentation` is the mean number of distinct
    predictions landing on one GT segment.
    """
    rec_hits, frags = [], []
    for g in gt:
        f, n = _union_frac(g, pred, tol)
        rec_hits.append(f >= min_overlap)
        if n:
            frags.append(n)
    prec_hits = [_union_frac(p, gt, tol)[0] >= min_overlap for p in pred]

    rec = float(np.mean(rec_hits)) if rec_hits else 0.0
    prec = float(np.mean(prec_hits)) if prec_hits else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"count_precision": prec, "count_recall": rec, "count_f1": f1,
            "n_gt": len(gt), "n_pred": len(pred),
            "fragmentation": float(np.mean(frags)) if frags else 0.0}


# ---------------------------------------------------------------------------

def run_sheet(job) -> Dict:
    (root, idx, max_dim, tol, notes_keep, minlen_mult, hough, variant,
     min_density, max_gap, max_trans, pred_json) = job
    ptl = LV.import_line_stage()
    LV.configure(ptl, max_dim=max_dim, minlen_mult=minlen_mult, hough=hough,
                 notes_keep=notes_keep, min_density=min_density,
                 max_gap=max_gap, max_transitions=max_trans)

    img = cv2.imread(os.path.join(root, "image_2", f"{idx}.jpg"))
    sheet = dpid.load_sheet(root, idx)
    H, W = img.shape[:2]

    if pred_json:
        with open(pred_json) as fh:
            d = json.load(fh).get(str(idx), {})
        pred = [tuple(int(v) for v in s) for s in d.get("solid", []) + d.get("dashed", [])]
    elif variant == "repo":
        pred = [tuple(int(v) for v in s[:4])
                for s in ptl._step4_core(img, None)["solid"]]
    elif variant == "unet":
        out = ptl.step4_extract_lines(img, None, method="unet")
        pred = [tuple(int(v) for v in s[:4]) for s in out["solid"] + out["dashed"]]
    else:
        pred = LV.detect(ptl, img, solid_source=variant)["merged"]

    res = {"idx": idx, "H": H, "W": W}
    for name, gt in (("solid", [l.seg for l in sheet.lines if l.kind == "solid"]),
                     ("all", [l.seg for l in sheet.lines])):
        r = buffer_scores(gt, pred, (H, W), tol)
        r.update(count_scores(gt, pred, tol))
        res[name] = r

    # how much GT the detector's own masking makes unreachable
    ratio = ptl.SOLID_CFG["notes_keep_ratio"]
    if ratio == "auto":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cap = ptl.SOLID_CFG["target_max_dim"] / max(H, W)
        gc = cv2.resize(gray, (int(W * cap), int(H * cap)), interpolation=cv2.INTER_AREA) if cap < 1 else gray
        b = ptl.detect_notes_boundary(gc)
        keep = W if b is None else b / min(cap, 1.0)
        res["notes_boundary_frac"] = keep / W
    else:
        keep = ratio * W
    seg_all = [l.seg for l in sheet.lines]
    tot = sum(np.hypot(s[2] - s[0], s[3] - s[1]) for s in seg_all) or 1
    right = sum(np.hypot(s[2] - s[0], s[3] - s[1])
                for s in seg_all if min(s[0], s[2]) > keep)
    res["gt_len_right_of_notes_crop"] = float(right / tot)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(
        _ROOT, "dataset"))
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--tol", type=int, default=8,
                    help="buffer radius in original-image px")
    ap.add_argument("--max-dim", type=int, default=None,
                    help="override SOLID_CFG target_max_dim (baseline 2200)")
    ap.add_argument("--notes-keep", default=None,
                    help="override notes_keep_ratio: a ratio (as-shipped 0.78) or 'auto'")
    ap.add_argument("--pred-json", default=None,
                    help="score externally produced segments: {idx: {solid: [...], dashed: [...]}}")
    ap.add_argument("--ids", default=None, help="sheet id list file or 'split.json:val'")
    ap.add_argument("--minlen-mult", type=float, default=1.0,
                    help="scale min_line_length on top of the cap ratio")
    ap.add_argument("--hough", type=int, default=None,
                    help="override hough_threshold (baseline 120)")
    ap.add_argument("--variant", default="repo",
                    choices=("repo", "unet", "canny", "ink", "band", "none"),
                    help="line source; 'repo' = classical _step4_core, 'unet' = the default "
                         "learned stage (needs torch), "
                         "'canny' mirrors it, 'ink' samples an ink mask, "
                         "'none' skips the filter")
    ap.add_argument("--min-density", type=float, default=None,
                    help="override is_solid_line min_density (baseline 0.68)")
    ap.add_argument("--max-gap", type=int, default=None,
                    help="override max_gap, counted in samples (baseline 4/80)")
    ap.add_argument("--max-trans", type=int, default=None,
                    help="override max_transitions (baseline 6)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    ap.add_argument("--out", default=None, help="write per-sheet JSON here")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    ids = dpid.sheet_ids(root)
    if args.ids:
        if ":" in args.ids and args.ids.rsplit(":", 1)[0].endswith(".json"):
            path, key = args.ids.rsplit(":", 1)
            want = set(json.load(open(path))[key])
        else:
            want = set(int(x) for x in open(args.ids).read().split())
        ids = [i for i in ids if i in want]
    if args.limit:
        ids = ids[:args.limit]
    jobs = [(root, i, args.max_dim, args.tol, args.notes_keep,
             args.minlen_mult, args.hough, args.variant, args.min_density,
             args.max_gap, args.max_trans, args.pred_json) for i in ids]

    rows: List[Dict] = []
    with ProcessPoolExecutor(args.workers) as ex:
        for r in ex.map(run_sheet, jobs):
            rows.append(r)
            print(f"  [{len(rows):>3}/{len(jobs)}] sheet {r['idx']:<4} "
                  f"comp {r['solid']['completeness']:.3f} "
                  f"corr {r['solid']['correctness']:.3f} "
                  f"cntF1 {r['solid']['count_f1']:.3f} "
                  f"pred {r['solid']['n_pred']}/gt {r['solid']['n_gt']}",
                  flush=True)

    cap = args.max_dim or LV.import_line_stage().SOLID_CFG["target_max_dim"]
    src = f"pred_json={args.pred_json}" if args.pred_json else f"variant={args.variant}"
    print(f"\n{src}  target_max_dim={cap}  tol={args.tol}px  sheets={len(rows)}")
    for name in ("solid", "all"):
        m = lambda k: float(np.mean([r[name][k] for r in rows]))
        print(f"  [{name:5}] completeness {m('completeness'):.3f}   "
              f"correctness {m('correctness'):.3f}   "
              f"buffer-F1 {m('buffer_f1'):.3f}")
        print(f"          count P {m('count_precision'):.3f}  "
              f"R {m('count_recall'):.3f}  F1 {m('count_f1'):.3f}   "
              f"frag {m('fragmentation'):.2f}   "
              f"pred/sheet {m('n_pred'):.0f}  gt/sheet {m('n_gt'):.0f}")
    print(f"  GT length right of the notes crop: "
          f"{100*float(np.mean([r['gt_len_right_of_notes_crop'] for r in rows])):.1f}%"
          "  (unreachable by construction)")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"config": vars(args), "rows": rows}, fh, indent=1)
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
