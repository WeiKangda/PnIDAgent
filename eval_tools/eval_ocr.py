#!/usr/bin/env python3
"""Score the repo's PaddleOCR text stage against Dataset-P&ID word annotations.

Reported separately because they fail for different reasons:

  detection    box P/R/F1 at IoU 0.5 and 0.3
  recognition  on matched pairs: exact-match rate and CER
  end-to-end   fraction of GT words whose text is recovered at all

The stage merges neighbouring boxes (merge_close_text), so a GT pair like
"SDL" + "101" can come back as one "SDL 101" box. That is a segmentation
difference, not an OCR error, so a containment-tolerant score is reported next
to exact match: a GT word counts as read if its text appears inside the text of
an overlapping detection.

Results are broken out by rotation, because map_boxes_90_to_0() exists in the
repo but is never called -- vertical pipe labels are expected to score ~0.

Usage:
    python tools/eval_ocr.py --limit 5
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys as _sys; _sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from _paths import ROOT as _ROOT  # <ROOT>/PnIDAgent/eval_tools layout
import re
import sys
from typing import Dict, List, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dpid  # noqa: E402

REPO = os.path.join(_ROOT,
                    "PnIDAgent")

Box = Tuple[int, int, int, int]
TAG_MAX_LEN = 15          # above this a GT word is note prose, not a tag


def _iou(a: Box, b: Box) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if not inter:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua else 0.0


def _contained(inner: Box, outer: Box) -> float:
    """Fraction of `inner` inside `outer` -- catches merged detections."""
    ix = max(0, min(inner[2], outer[2]) - max(inner[0], outer[0]))
    iy = max(0, min(inner[3], outer[3]) - max(inner[1], outer[1]))
    area = (inner[2] - inner[0]) * (inner[3] - inner[1])
    return (ix * iy) / area if area else 0.0


def cer(ref: str, hyp: str) -> float:
    """Character error rate via Levenshtein distance."""
    if not ref:
        return 0.0 if not hyp else 1.0
    prev = list(range(len(hyp) + 1))
    for i, rc in enumerate(ref, 1):
        cur = [i]
        for j, hc in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (rc != hc)))
        prev = cur
    return prev[-1] / len(ref)


def norm(t: str) -> str:
    return re.sub(r"\s+", "", t).upper()


def match(gt: Sequence[Tuple[Box, str, int]],
          pred: Sequence[Tuple[Box, str]], iou_th: float) -> Dict[str, float]:
    """Greedy one-to-one box matching at `iou_th`, then score the text."""
    cand = []
    for gi, (gb, _, _) in enumerate(gt):
        for pi, (pb, _) in enumerate(pred):
            v = _iou(gb, pb)
            if v >= iou_th:
                cand.append((v, gi, pi))
    cand.sort(reverse=True)
    pairs: List[Tuple[int, int]] = []
    ug, up = set(), set()
    for _, gi, pi in cand:
        if gi in ug or pi in up:
            continue
        ug.add(gi)
        up.add(pi)
        pairs.append((gi, pi))

    prec = len(pairs) / len(pred) if pred else 0.0
    rec = len(pairs) / len(gt) if gt else 0.0
    out = {"det_precision": prec, "det_recall": rec,
           "det_f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0}
    if pairs:
        ex = [norm(gt[g][1]) == norm(pred[p][1]) for g, p in pairs]
        ce = [cer(norm(gt[g][1]), norm(pred[p][1])) for g, p in pairs]
        out["rec_exact"] = float(np.mean(ex))
        out["rec_cer"] = float(np.mean(ce))
    else:
        out["rec_exact"] = out["rec_cer"] = 0.0
    return out


def read_rate(gt: Sequence[Tuple[Box, str, int]],
              pred: Sequence[Tuple[Box, str]], overlap: float = 0.5) -> float:
    """Fraction of GT words whose text appears in an overlapping detection.

    Tolerates the stage's box merging: only asks whether the characters were
    recovered somewhere, not whether the box boundaries agree.
    """
    hits = 0
    for gb, gt_text, _ in gt:
        want = norm(gt_text)
        for pb, pt in pred:
            if _contained(gb, pb) >= overlap and want in norm(pt):
                hits += 1
                break
    return hits / len(gt) if gt else 0.0


def _strata(sheet) -> Dict[str, List[Tuple[Box, str, int]]]:
    words = [(w.box, w.text, w.rot) for w in sheet.words]
    return {
        "all": words,
        "tag_h": [w for w in words if w[2] == 0 and len(w[1]) <= TAG_MAX_LEN],
        "tag_v": [w for w in words if w[2] == 90 and len(w[1]) <= TAG_MAX_LEN],
        "prose": [w for w in words if len(w[1]) > TAG_MAX_LEN],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(
        _ROOT, "dataset"))
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--out", default=None)
    ap.add_argument("--gpu", action="store_true",
                    help="run PaddleOCR on CUDA; needs paddlepaddle-gpu")
    ap.add_argument("--ids", default=None, help="sheet id list file or split.json:val")
    ap.add_argument("--rot90", action="store_true",
                    help="add the 90-degree pass for vertical tags (step1 rotations=(0, 90))")
    ap.add_argument("--preprocess", default="clahe", choices=("clahe", "none", "binary"),
                    help="step1 preprocess mode (as shipped: clahe; new stage default: none)")
    ap.add_argument("--scales", default="1.0,1.35", help="step1 scales (as shipped 1.0,1.35; new default 1.0)")
    ap.add_argument("--drop-border", type=int, default=0,
                    help="drop detections within this many px of a tile border (step1 drop_border)")
    ap.add_argument("--tag-correct", action="store_true",
                    help="apply tag_grammar.correct in the stage (off here so dumps stay raw)")
    ap.add_argument("--dump", default=None,
                    help="write raw predictions {idx: [{bbox, text, score}]} here, so "
                         "post-processing variants can be scored offline (tools/ocr_post.py)")
    args = ap.parse_args()
    dump: Dict[str, List[Dict]] = {}

    import cv2
    sys.path.insert(0, REPO)
    import process_text_lines as ptl
    from paddleocr import PaddleOCR

    # Same construction as process_text_lines.py's __main__; the stage calls
    # ocr.ocr(patch, cls=True), which is the PaddleOCR 2.7.x API.
    ocr = PaddleOCR(lang="en", use_gpu=args.gpu, use_angle_cls=True)
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
    rotations = (0, 90) if args.rot90 else (0,)

    rows: List[Dict] = []
    for i in ids:
        img = cv2.imread(os.path.join(root, "image_2", f"{i}.jpg"))
        img, _ = ptl.resize_keep_aspect(img)       # no-op: sheets are 7168 wide
        sheet = dpid.load_sheet(root, i)
        out = ptl.step1_paddleocr(img, ocr, rotations=rotations, preprocess=args.preprocess,
                                  scales=tuple(float(x) for x in args.scales.split(",")),
                                  drop_border=args.drop_border,
                                  tag_correct=args.tag_correct)   # dict, not a tuple
        pred = [(tuple(int(v) for v in b), t)
                for b, t in zip(out["boxes"], out["texts"])]
        if args.dump:
            dump[str(i)] = [{"bbox": [int(v) for v in b], "text": t, "score": float(sc)}
                            for b, t, sc in zip(out["boxes"], out["texts"], out["scores"])]
            with open(args.dump, "w") as fh:
                json.dump(dump, fh)

        res = {"idx": i, "n_pred": len(pred)}
        for name, gt in _strata(sheet).items():
            r = match(gt, pred, 0.5)
            r["det_f1_iou30"] = match(gt, pred, 0.3)["det_f1"]
            r["read_rate"] = read_rate(gt, pred)
            r["n_gt"] = len(gt)
            res[name] = r
        rows.append(res)
        a = res["all"]
        print(f"  [{len(rows)}/{len(ids)}] sheet {i:<4} detF1 {a['det_f1']:.3f} "
              f"exact {a['rec_exact']:.3f} CER {a['rec_cer']:.3f} "
              f"read {a['read_rate']:.3f} pred {len(pred)}/gt {a['n_gt']}",
              flush=True)

    print(f"\nsheets={len(rows)}")
    print(f"  {'stratum':8} {'gt':>5} {'detP':>6} {'detR':>6} {'detF1':>6} "
          f"{'F1@.3':>6} {'exact':>6} {'CER':>6} {'read':>6}")
    for name in ("all", "tag_h", "tag_v", "prose"):
        m = lambda k: float(np.mean([r[name][k] for r in rows]))
        print(f"  {name:8} {m('n_gt'):5.0f} {m('det_precision'):6.3f} "
              f"{m('det_recall'):6.3f} {m('det_f1'):6.3f} {m('det_f1_iou30'):6.3f} "
              f"{m('rec_exact'):6.3f} {m('rec_cer'):6.3f} {m('read_rate'):6.3f}")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"config": vars(args), "rows": rows}, fh, indent=1)
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
