#!/usr/bin/env python3
"""Tag-aware OCR post-processing, scored offline against raw prediction dumps.

P&ID text on Dataset-P&ID is 90% a handful of shapes (56 sheets, 14.5k words):

    AA-99999   25%   instrument / equipment tag      CD-31021
    999        11%   bubble number                   101
    AAA         9%   bubble function letters         SDL
    9"-AA-9999  6%   pipe label (all vertical text)  5"-EK-2648
    AA-99       4%   LG-10
    9" / 99"    5%   nominal size                    3"  12"
    AAA-9-99    2%   ERV-8-20
    9"x9"       2%   reducer                         9"x7"

The recognizer's errors are mostly within-slot confusions (O/0, I/1, S/5,
B/8, Z/2, G/6) and dropped or mangled inch marks.  `correct()` tries the
slot-aware substitutions and accepts a candidate only when it turns a
non-conforming string into one that matches a tag grammar; conforming input is
left alone, and prose is left alone.

    python tools/ocr_post.py --dump results/ocr_rot90_dev_dump.json           # A/B: raw vs corrected
"""
from __future__ import annotations

import argparse
import json
import os
import sys as _sys; _sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from _paths import ROOT as _ROOT  # <ROOT>/PnIDAgent/eval_tools layout
import re
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dpid  # noqa: E402
import eval_ocr as EO  # noqa: E402

sys.path.insert(0, os.path.join(_ROOT, "PnIDAgent"))
from tag_grammar import GRAMMAR, TO_DIGIT, TO_ALPHA, conforms, normalize_punct, correct  # noqa: E402,F401


# ---------------------------------------------------------------------------

def score(root: str, dump: Dict[str, List[Dict]], post) -> Dict[str, Dict[str, float]]:
    rows = []
    for k, items in dump.items():
        sheet = dpid.load_sheet(root, int(k))
        pred = [(tuple(int(v) for v in it["bbox"]), post(it["text"])) for it in items]
        res = {}
        for name, gt in EO._strata(sheet).items():
            r = EO.match(gt, pred, 0.5)
            r["det_f1_iou30"] = EO.match(gt, pred, 0.3)["det_f1"]
            r["read_rate"] = EO.read_rate(gt, pred)
            r["n_gt"] = len(gt)
            res[name] = r
        rows.append(res)
    out = {}
    for name in ("all", "tag_h", "tag_v", "prose"):
        out[name] = {k: float(np.mean([r[name][k] for r in rows]))
                     for k in ("n_gt", "det_precision", "det_recall", "det_f1", "det_f1_iou30",
                               "rec_exact", "rec_cer", "read_rate")}
    out["sheets"] = len(rows)
    return out


def print_table(title: str, s: Dict) -> None:
    print(f"\n{title}  (sheets={s['sheets']})")
    print(f"  {'stratum':8} {'gt':>5} {'detP':>6} {'detR':>6} {'detF1':>6} {'F1@.3':>6} "
          f"{'exact':>6} {'CER':>6} {'read':>6}")
    for name in ("all", "tag_h", "tag_v", "prose"):
        m = s[name]
        print(f"  {name:8} {m['n_gt']:5.0f} {m['det_precision']:6.3f} {m['det_recall']:6.3f} "
              f"{m['det_f1']:6.3f} {m['det_f1_iou30']:6.3f} {m['rec_exact']:6.3f} "
              f"{m['rec_cer']:6.3f} {m['read_rate']:6.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(
        _ROOT, "dataset"))
    ap.add_argument("--dump", required=True)
    ap.add_argument("--ids", default=None, help="restrict to these sheet ids (file)")
    ap.add_argument("--show-errors", type=int, default=0, help="print N residual tag errors")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    with open(args.dump) as fh:
        dump = json.load(fh)
    if args.ids:
        want = set(open(args.ids).read().split())
        dump = {k: v for k, v in dump.items() if k in want}
    root = os.path.abspath(args.root)

    raw = score(root, dump, lambda t: t)
    fixed = score(root, dump, correct)
    print_table("raw", raw)
    print_table("tag-grammar corrected", fixed)

    if args.show_errors:
        shown = 0
        for k, items in dump.items():
            sheet = dpid.load_sheet(root, int(k))
            gt = EO._strata(sheet)["all"]
            pred = [(tuple(int(v) for v in it["bbox"]), correct(it["text"]), it["text"]) for it in items]
            cand = sorted(((EO._iou(gb, pb), gi, pi) for gi, (gb, _, _) in enumerate(gt)
                           for pi, (pb, _, _) in enumerate(pred) if EO._iou(gb, pb) >= 0.5), reverse=True)
            ug, up = set(), set()
            for _, gi, pi in cand:
                if gi in ug or pi in up:
                    continue
                ug.add(gi); up.add(pi)
                g, p, p_raw = gt[gi][1], pred[pi][1], pred[pi][2]
                if EO.norm(g) != EO.norm(p) and len(g) <= EO.TAG_MAX_LEN:
                    print(f"    sheet {k:>4} gt {g!r:16} raw {p_raw!r:18} fixed {p!r}")
                    shown += 1
                    if shown >= args.show_errors:
                        break
            if shown >= args.show_errors:
                break
    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"raw": raw, "corrected": fixed}, fh, indent=1)


if __name__ == "__main__":
    main()
