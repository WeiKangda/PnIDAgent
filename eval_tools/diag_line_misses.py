#!/usr/bin/env python3
"""Find out which GT pipe segments the line stage misses, and where they go.

Line recall sat at ~0.80 across every Hough threshold and min_line_length tried,
so the loss is not in candidate generation. This attributes each miss to a
stage: candidates that Hough never proposed, candidates the is_solid_line
density filter rejected, and candidates lost in merge_segments.

Usage:
    python tools/diag_line_misses.py --limit 10 --max-dim 4096
"""
from __future__ import annotations

import argparse
import collections
import os
import sys as _sys; _sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from _paths import ROOT as _ROOT  # <ROOT>/PnIDAgent/eval_tools layout
import sys
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dpid          # noqa: E402
import eval_lines as EL  # noqa: E402
import line_variants as LV  # noqa: E402

Seg = Tuple[int, int, int, int]


def _recalled(gt: Sequence[Seg], pred: Sequence[Seg], tol: int,
              th: float = 0.5) -> List[bool]:
    return [EL._union_frac(g, pred, tol)[0] >= th for g in gt]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(
        _ROOT, "dataset"))
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--max-dim", type=int, default=4096)
    ap.add_argument("--tol", type=int, default=8)
    ap.add_argument("--variant", default="canny",
                    choices=("canny", "ink", "band", "none"))
    args = ap.parse_args()

    ptl = LV.import_line_stage()
    LV.configure(ptl, max_dim=args.max_dim)

    root = os.path.abspath(args.root)
    stage_rec = collections.defaultdict(list)
    len_buckets = collections.defaultdict(lambda: [0, 0])   # bucket -> [hit, total]
    kind_buckets = collections.defaultdict(lambda: [0, 0])
    near_sym = [0, 0]
    far_sym = [0, 0]

    for i in dpid.sheet_ids(root)[:args.limit]:
        img = cv2.imread(os.path.join(root, "image_2", f"{i}.jpg"))
        sheet = dpid.load_sheet(root, i)
        sets = LV.detect(ptl, img, solid_source=args.variant)
        gt = [l.seg for l in sheet.lines]

        for stage in ("hough", "post_solid", "merged"):
            r = _recalled(gt, sets[stage], args.tol)
            stage_rec[stage].append(float(np.mean(r)))
            if stage == "merged":
                final = r

        boxes = [x.box for x in sheet.symbols]
        for l, ok in zip(sheet.lines, final):
            L = float(np.hypot(l.seg[2] - l.seg[0], l.seg[3] - l.seg[1]))
            b = "<100" if L < 100 else "100-300" if L < 300 else \
                "300-800" if L < 800 else ">=800"
            len_buckets[b][1] += 1
            len_buckets[b][0] += ok
            kind_buckets[l.kind][1] += 1
            kind_buckets[l.kind][0] += ok
            touches = any(EL._span(l.seg, (bx[0], (bx[1] + bx[3]) // 2,
                                           bx[2], (bx[1] + bx[3]) // 2), 40)[2] > 0
                          or (bx[0] - 40 <= l.seg[0] <= bx[2] + 40
                              and bx[1] - 40 <= l.seg[1] <= bx[3] + 40)
                          for bx in boxes)
            tgt = near_sym if touches else far_sym
            tgt[1] += 1
            tgt[0] += ok
        print(f"  sheet {i:<4} hough {stage_rec['hough'][-1]:.3f} -> "
              f"solid-filter {stage_rec['post_solid'][-1]:.3f} -> "
              f"merged {stage_rec['merged'][-1]:.3f}   "
              f"cands {len(sets['hough'])}->{len(sets['post_solid'])}"
              f"->{len(sets['merged'])}", flush=True)

    print(f"\nrecall by stage (variant={args.variant}, max_dim={args.max_dim}, "
          f"tol={args.tol}px, {args.limit} sheets, all GT lines):")
    for stage in ("hough", "post_solid", "merged"):
        print(f"  {stage:14} {float(np.mean(stage_rec[stage])):.3f}")
    print("\nfinal recall by GT segment length:")
    for b in ("<100", "100-300", "300-800", ">=800"):
        h, t = len_buckets[b]
        if t:
            print(f"  {b:>8} px  {h/t:.3f}  (n={t})")
    print("\nfinal recall by line kind:")
    for k, (h, t) in sorted(kind_buckets.items()):
        print(f"  {k:8} {h/t:.3f}  (n={t})")
    print("\nfinal recall vs symbol proximity:")
    for name, (h, t) in (("near a symbol", near_sym), ("clear of symbols", far_sym)):
        if t:
            print(f"  {name:18} {h/t:.3f}  (n={t})")


if __name__ == "__main__":
    main()
