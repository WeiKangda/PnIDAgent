#!/usr/bin/env python3
"""Solid-line detection variants, sharing the repo's stages so A/Bs are honest.

`detect()` mirrors process_text_lines._step4_core step for step, but exposes the
intermediate candidate sets and lets the solidity test be swapped out. That is
where the line stage loses most of its recall: on Dataset-P&ID at max_dim 4096,
Hough proposes segments covering 0.852 of GT lines and is_solid_line rejects
enough of them to leave 0.690.

  solid_source='canny'  reproduces the repo
  solid_source='ink'    samples an adaptive-threshold ink mask
  solid_source='band'   samples a perpendicular band on the ink mask
  solid_source='none'   skips the filter, to bound its cost

Measured on 30 sheets: 'ink' and 'band' do not recover the loss, and min_density
is never the binding criterion -- 0.68, 0.35 and 0.20 give identical output. The
rejections come from max_gap and max_transitions. Both are counted in samples,
and solid_stats() always takes exactly 80 samples however long the segment is,
so max_gap=4 means "one twentieth of this segment" rather than "4 pixels".

Two smaller oddities worth knowing while reading the stage: solid_stats()
samples the bare centreline with no perpendicular tolerance even though the repo
ships a band sampler for exactly this (_step4_sample_hits, half_width=2, never
called since the dashed path was removed), and the filter is fed a dilated Canny
map, which holds the two flanks of each stroke rather than the stroke itself.
"""
from __future__ import annotations

import os
import sys as _sys; _sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from _paths import ROOT as _ROOT  # <ROOT>/PnIDAgent/eval_tools layout
import sys
import types
from typing import Dict, List, Tuple

import cv2
import numpy as np

REPO = os.path.join(_ROOT,
                    "PnIDAgent")

Seg = Tuple[int, int, int, int]


def import_line_stage():
    """Import process_text_lines without PaddleOCR or scikit-image.

    Both are imported at module scope; the solid-line path uses neither.
    """
    paddle = types.ModuleType("paddleocr")
    paddle.PaddleOCR = object
    sys.modules.setdefault("paddleocr", paddle)
    for name in ("skimage", "skimage.morphology"):
        mod = types.ModuleType(name)
        mod.skeletonize = lambda x: x
        sys.modules.setdefault(name, mod)
    sys.path.insert(0, REPO)
    import process_text_lines as ptl
    return ptl


def configure(ptl, max_dim=None, minlen_mult=1.0, hough=None, notes_keep=None,
              min_density=None, max_gap=None, max_transitions=None) -> None:
    """Apply overrides to the stage's module-level SOLID_CFG."""
    if max_dim is not None:
        ptl.SOLID_CFG["target_max_dim"] = max_dim
        # min_line_length is in capped-image pixels, so raising the cap without
        # raising it too just admits short spurious segments.
        s = (max_dim / 2200.0) * minlen_mult
        ptl.SOLID_CFG["min_line_length_at_scale"] = {
            k: max(12, int(round(v * s)))
            for k, v in {1.0: 90, 0.75: 70, 0.6: 55}.items()}
    if hough is not None:
        ptl.SOLID_CFG["hough_threshold"] = hough
    if notes_keep is not None:
        ptl.SOLID_CFG["notes_keep_ratio"] = (
            "auto" if str(notes_keep) == "auto" else float(notes_keep))
    if min_density is not None:
        ptl.SOLID_CFG["min_density"] = min_density
    # solid_stats takes a fixed 80 samples whatever the segment length, so
    # max_gap is really "this fraction of 80", not a pixel count.
    if max_gap is not None:
        ptl.SOLID_CFG["max_gap"] = max_gap
    if max_transitions is not None:
        ptl.SOLID_CFG["max_transitions"] = max_transitions
        ptl.SOLID_CFG["solidity_filter"] = True      # an explicit value re-enables the filter
    if min_density is not None or max_gap is not None:
        ptl.SOLID_CFG["solidity_filter"] = True


def _ink_mask(gray: np.ndarray) -> np.ndarray:
    """Binary ink map, dilated by 1 so a 1px-off Hough line still samples it."""
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 25, 7)
    return cv2.dilate(bw, np.ones((3, 3), np.uint8), iterations=1)


def _is_solid_band(ptl, bw, x1, y1, x2, y2, min_density: float,
                   max_gap_px: int = 6, max_transitions: int = 6) -> bool:
    """Solidity test over a perpendicular band, with the gap measured in pixels."""
    hits = ptl._step4_sample_hits(bw, x1, y1, x2, y2, step=1, half_width=2)
    if hits.size < 10:
        return False
    if hits.mean() < min_density:
        return False
    if int(np.sum(hits[1:] != hits[:-1])) > max_transitions:
        return False
    gap = run = 0
    for v in hits:
        run = run + 1 if v == 0 else 0
        gap = max(gap, run)
    return gap <= max_gap_px


def detect(ptl, img_bgr: np.ndarray, solid_source: str = "canny"
           ) -> Dict[str, List[Seg]]:
    """Run the solid-line path, returning each stage's candidates."""
    cfg = ptl.SOLID_CFG
    H0, W0 = img_bgr.shape[:2]
    gray0 = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    cap = cfg["target_max_dim"] / max(H0, W0)
    if cap < 1:
        gray_cap = cv2.resize(gray0, (int(W0 * cap), int(H0 * cap)), cv2.INTER_AREA)
    else:
        cap, gray_cap = 1.0, gray0
    inv_cap = 1.0 / cap
    notes_cap = ptl.remove_right_notes_block(gray_cap, keep_ratio=cfg["notes_keep_ratio"])

    raw: List[Seg] = []
    kept: List[Seg] = []
    for sc in cfg["scales"]:
        gray = (gray_cap if sc == 1.0 else
                cv2.resize(gray_cap, (int(gray_cap.shape[1] * sc),
                                      int(gray_cap.shape[0] * sc)), cv2.INTER_AREA))
        edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0),
                          cfg["canny_low"], cfg["canny_high"], L2gradient=True)
        notes = (notes_cap if sc == 1.0 else
                 cv2.resize(notes_cap, (gray.shape[1], gray.shape[0]),
                            interpolation=cv2.INTER_NEAREST))
        keep = cv2.bitwise_and(
            ptl.find_inner_frame_mask(gray, shrink_px=cfg["frame_shrink_px"]), notes)
        edges = cv2.bitwise_and(edges, edges, mask=keep)

        if solid_source in ("ink", "band"):
            samp = cv2.bitwise_and(_ink_mask(gray), keep)
        else:
            samp = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

        min_len = cfg["min_line_length_at_scale"].get(
            sc, int(cfg["min_line_length_at_scale"][1.0] * sc))
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, cfg["hough_threshold"],
                                minLineLength=min_len,
                                maxLineGap=cfg["max_line_gap"])
        if lines is None:
            continue

        f = (1.0 / sc) * inv_cap
        for x1, y1, x2, y2 in lines[:, 0]:
            s = (int(round(x1 * f)), int(round(y1 * f)),
                 int(round(x2 * f)), int(round(y2 * f)))
            raw.append(s)
            if solid_source == "none" or (solid_source == "canny"
                                          and not cfg.get("solidity_filter", True)):
                ok = True
            elif solid_source == "band":
                ok = _is_solid_band(ptl, samp, x1, y1, x2, y2,
                                    min_density=cfg["min_density"],
                                    max_transitions=cfg["max_transitions"])
            else:
                ok = ptl.is_solid_line(
                    samp, x1, y1, x2, y2,
                    samples=cfg["cont_samples"], min_density=cfg["min_density"],
                    max_gap=cfg["max_gap"], max_transitions=cfg["max_transitions"])
            if ok:
                kept.append(s)

    merged = [tuple(int(v) for v in s[:4]) for s in ptl.merge_segments(
        kept, angle_thr=cfg["merge_angle_thr_deg"], gap_thr=cfg["merge_gap_thr"],
        perp_thr=cfg["merge_perp_thr"], dedup_dist=cfg["dedup_dist"])]
    return {"hough": raw, "post_solid": kept, "merged": merged}
