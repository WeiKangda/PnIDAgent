#!/usr/bin/env python3
"""
Single P&ID Image Processing Pipeline

Processes one P&ID image through 3 steps:

1. Text detection + recognition (PaddleOCR on the resized image,
   with tiling and multi-scale passes).
2. Text box filtering / merging to clean detections and prepare
   for line suppression.
3. Line extraction (solid and optionally dashed) using morphology
   + Hough + LSD candidate generation, heuristic classification,
   and collinearity-based merging.

Usage:
    python process_text_lines.py --image path/to/pnid.jpg --out output_dir [options]
"""

import os
import sys
import json
import argparse
import re
import math
from pathlib import Path
from skimage.morphology import skeletonize
import cv2
import numpy as np
from PIL import Image, ImageDraw
from paddleocr import PaddleOCR

# Valid characters for P&ID text
_VALID = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_/().,%:+ '\"")

# ============================================================================
# STEP 1: Text Detection + Recognition (PaddleOCR)
# ============================================================================

def load_image(path):
    """Load image from path."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img

def resize_keep_aspect(img, target_width=7168):
    """Resize image to target width while maintaining aspect ratio."""
    h, w = img.shape[:2]
    if w == target_width:
        return img, 1.0
    scale = target_width / float(w)
    new_h = int(round(h * scale))
    resized = cv2.resize(img, (target_width, new_h), interpolation=cv2.INTER_LINEAR)
    return resized, scale

def quad_to_bbox(quad):
    xs = [p[0] for p in quad]
    ys = [p[1] for p in quad]
    return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]

def _iou(a, b):
    # a,b: [x1,y1,x2,y2]
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    if inter <= 0:
        return 0.0
    area_a = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
    area_b = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
    return inter / max(1e-6, (area_a + area_b - inter))

def _nms_text_items(items, iou_th=0.35):
    """
    items: list of dicts: {"bbox":[...], "score":float, "text":str, "quad":[[x,y]...]}
    Keep higher-score item when overlap is high.
    """
    if not items:
        return items
    items = sorted(items, key=lambda d: d["score"], reverse=True)
    kept = []
    for it in items:
        ok = True
        for k in kept:
            if _iou(it["bbox"], k["bbox"]) > iou_th:
                ok = False
                break
        if ok:
            kept.append(it)
    return kept

def _preprocess_for_ocr(img_bgr, mode="clahe"):
    """
    mode:
      - "none": no preprocessing
      - "clahe": good default for gray BG P&IDs
      - "binary": more aggressive (can help faint text, can also hurt)
    """
    if mode == "none":
        return img_bgr

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # light denoise (keeps edges)
    gray = cv2.fastNlMeansDenoising(gray, h=7, templateWindowSize=7, searchWindowSize=21)

    if mode == "clahe":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        # back to BGR for PaddleOCR
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    if mode == "binary":
        # adaptive binarization (useful when contrast is poor)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        bw = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            35, 7
        )
        return cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)

    return img_bgr

def _sliding_windows(H, W, win=1400, stride=900):
    ys = list(range(0, max(1, H - win + 1), stride))
    xs = list(range(0, max(1, W - win + 1), stride))
    if ys and ys[-1] + win < H:
        ys.append(H - win)
    if xs and xs[-1] + win < W:
        xs.append(W - win)
    for y in ys:
        for x in xs:
            yield x, y, win, win

def is_valid_pid_text(text, bbox):
    """
    Decide whether OCR result is meaningful for P&ID.
    """
    if not text:
        return False

    s = text.strip()
    if len(s) < 2:
        return False

    x1,y1,x2,y2 = bbox
    w = x2 - x1
    h = y2 - y1

    # Kill extremely small junk
    if w * h < 120:
        return False

    # Remove pure symbols or punctuation
    if re.fullmatch(r"[^\w]+", s):
        return False

    # Single letters almost always noise in P&ID
    if len(s) == 1 and s.isalpha():
        return False

    # Allow typical P&ID patterns
    if re.search(r"\d", s):       # has digits → keep
        return True

    if re.search(r"[A-Z]{2,}", s):  # multi-letter tags like INS, STA, PC
        return True

    # Otherwise reject
    return False

def merge_close_text(items, y_tol=6, x_gap=12):
    items = sorted(items, key=lambda d: (d["bbox"][1], d["bbox"][0]))
    merged = []
    used = [False]*len(items)

    for i,a in enumerate(items):
        if used[i]:
            continue
        x1,y1,x2,y2 = a["bbox"]
        text = a["text"]
        used[i] = True

        for j,b in enumerate(items):
            if used[j]:
                continue
            bx1,by1,bx2,by2 = b["bbox"]
            if abs(by1 - y1) < y_tol and 0 < bx1 - x2 < x_gap:
                text += b["text"]
                x2 = bx2
                used[j] = True

        merged.append({
            "bbox":[x1,y1,x2,y2],
            "text":text,
            "score":a["score"],
            "quad":a["quad"]
        })

    return merged

def step1_paddleocr(
    img_resized,
    ocr,
    *,
    use_tiling=True,
    tile=1400,
    stride=900,
    preprocess="clahe",     # "clahe" is best default for P&IDs
    scales=(1.0, 1.35),     # multi-scale: helps tiny text a lot
    min_conf=0.25,          # allow weaker text; filter later in your step2 anyway
    nms_iou=0.35,
    debug=False
):
    """
    Outcome-driven PaddleOCR step:
      - optional preprocessing (clahe / adaptive binary)
      - multi-scale OCR
      - (optional) tiling + overlap merge
      - NMS dedup
    Returns:
      boxes: [x1,y1,x2,y2]
      scores: float
      texts: str
      quads: 4-pt polygon
    """
    print("\n=== STEP 1: Text Detection + Recognition (PaddleOCR, tiled + multiscale) ===")

    H0, W0 = img_resized.shape[:2]
    all_items = []

    for sc in scales:
        if abs(sc - 1.0) < 1e-6:
            img_sc = img_resized
            sc_factor = 1.0
        else:
            img_sc = cv2.resize(
                img_resized,
                (int(round(W0 * sc)), int(round(H0 * sc))),
                interpolation=cv2.INTER_CUBIC
            )
            sc_factor = sc

        img_sc = _preprocess_for_ocr(img_sc, mode=preprocess)

        H, W = img_sc.shape[:2]

        if use_tiling:
            windows = list(_sliding_windows(H, W, win=tile, stride=stride))
        else:
            windows = [(0, 0, W, H)]

        for (x0, y0, ww, hh) in windows:
            patch = img_sc[y0:y0+hh, x0:x0+ww]
            res = ocr.ocr(patch, cls=True)

            # PaddleOCR can return: None, [], [None], [[...]], or sometimes [...]
            if not res:
                continue
            if isinstance(res, list) and len(res) > 0 and res[0] is None:
                continue
            
            items = res[0] if (isinstance(res, list) and len(res) > 0 and isinstance(res[0], list)) else res
            if not items:
                continue
            
            for item in items:
                if item is None:
                    continue
                quad, (text, conf) = item
                if text is None:
                    text = ""
                conf = float(conf)

                if conf < min_conf:
                    continue

                # quad in patch coords -> image_sc coords
                quad_xy = [[float(px + x0), float(py + y0)] for (px, py) in quad]
                b = quad_to_bbox(quad_xy)

                # map back to original img_resized coords
                b0 = [int(round(b[0] / sc_factor)),
                      int(round(b[1] / sc_factor)),
                      int(round(b[2] / sc_factor)),
                      int(round(b[3] / sc_factor))]

                quad0 = [[float(qx / sc_factor), float(qy / sc_factor)] for (qx, qy) in quad_xy]

                # clamp
                b0[0] = max(0, min(W0-1, b0[0]))
                b0[2] = max(0, min(W0-1, b0[2]))
                b0[1] = max(0, min(H0-1, b0[1]))
                b0[3] = max(0, min(H0-1, b0[3]))

                if (b0[2] - b0[0]) < 3 or (b0[3] - b0[1]) < 3:
                    continue

                all_items.append({
                    "bbox": b0,
                    "score": conf,
                    "text": text,
                    "quad": quad0
                })

    # Dedup across overlaps + scales
    all_items = _nms_text_items(all_items, iou_th=nms_iou)

    filtered = []
    for it in all_items:
        if is_valid_pid_text(it["text"], it["bbox"]):
            filtered.append(it)
    
    all_items = filtered
    all_items = merge_close_text(all_items)

    # Pack outputs in your existing format
    boxes  = [it["bbox"] for it in all_items]
    scores = [float(it["score"]) for it in all_items]
    texts  = [it["text"] for it in all_items]
    quads  = [it["quad"] for it in all_items]

    print(f"  Found {len(boxes)} text lines (after tiling+scales+NMS)")

    return {
        "resized_shape": [int(H0), int(W0)],
        "boxes": boxes,
        "scores": scores,
        "texts": texts,
        "quads": quads
    }

# ============================================================================
# STEP 2: Merge Text Regions (IoU-based NMS)
# ============================================================================

def iou(a, b):
    """Calculate Intersection over Union of two boxes."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / float(area_a + area_b - inter + 1e-9)

def nms_merge(boxes, scores, iou_thr=0.3):
    """Merge overlapping boxes using Non-Maximum Suppression."""
    idxs = np.argsort(scores)[::-1]
    merged = []
    used = set()

    for i in idxs:
        if i in used:
            continue
        base = boxes[i]
        group = [i]

        for j in idxs:
            if j in used or j == i:
                continue
            if iou(base, boxes[j]) >= iou_thr:
                group.append(j)

        # Compute union box
        gx1 = min(boxes[k][0] for k in group)
        gy1 = min(boxes[k][1] for k in group)
        gx2 = max(boxes[k][2] for k in group)
        gy2 = max(boxes[k][3] for k in group)

        best = max(group, key=lambda k: scores[k])
        merged.append(([int(gx1), int(gy1), int(gx2), int(gy2)], float(scores[best])))
        used.update(group)

    return merged

def map_boxes_90_to_0(boxes90, W0):
    """Map boxes from 90° rotation back to 0° coordinates."""
    mapped = []
    for b in boxes90:
        x1, y1, x2, y2 = b
        pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
        pts_m = np.stack([pts[:, 1], W0 - pts[:, 0]], axis=1)  # (y, W-x)
        bx1, by1 = np.min(pts_m, axis=0)
        bx2, by2 = np.max(pts_m, axis=0)
        mapped.append([int(bx1), int(by1), int(bx2), int(by2)])
    return mapped

# ============================================================================
# STEP 3: Line Extraction (Morphology + Hough + LSD)
# ============================================================================

def _open_len(bw, klen, thick, orient):
    """Morphological opening with oriented kernel."""
    if orient == 'h':
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (klen, thick))
    else:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (thick, klen))
    return cv2.morphologyEx(bw, cv2.MORPH_OPEN, k, iterations=1)

def components_to_segments(bin_img, orient='h', min_len=12):
    """Convert connected components to line segments."""
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bin_img, connectivity=8)
    segs = []
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if area < 6:
            continue
        if orient == 'h':
            if w < min_len:
                continue
            y0 = int(y + h / 2)
            segs.append((x, y0, x + w, y0))
        else:
            if h < min_len:
                continue
            x0 = int(x + w / 2)
            segs.append((x0, y, x0, y + h))
    return segs


def merge_collinear_segments(segs, gap_px=7, ang_tol=5.5, join_tol=3):
    """Merge collinear line segments."""
    if not segs:
        return []

    H, V = [], []

    def ang(s):
        x1, y1, x2, y2 = s
        return np.degrees(np.arctan2(y2 - y1, (x2 - x1) + 1e-6))

    # Separate horizontal and vertical
    for x1, y1, x2, y2 in segs:
        a = ang((x1, y1, x2, y2))
        if abs(a) <= ang_tol or abs(abs(a) - 180) <= ang_tol:
            y = int(round((y1 + y2) / 2))
            H.append((min(x1, x2), y, max(x1, x2), y))
        elif abs(abs(a) - 90) <= ang_tol:
            x = int(round((x1 + x2) / 2))
            V.append((x, min(y1, y2), x, max(y1, y2)))

    merged = []

    # Merge horizontals
    H.sort(key=lambda s: (s[1], s[0], s[2]))
    cur = None
    for s in H:
        if cur is None:
            cur = list(s)
            continue
        same = abs(s[1] - cur[1]) <= join_tol
        gap = s[0] - cur[2]
        if same and 0 <= gap <= gap_px:
            cur[2] = max(cur[2], s[2])
        else:
            if cur[2] - cur[0] >= 6:
                merged.append(tuple(cur))
            cur = list(s)
    if cur is not None and cur[2] - cur[0] >= 6:
        merged.append(tuple(cur))

    # Merge verticals
    V.sort(key=lambda s: (s[0], s[1], s[3]))
    cur = None
    for s in V:
        if cur is None:
            cur = list(s)
            continue
        same = abs(s[0] - cur[0]) <= join_tol
        gap = s[1] - cur[3]
        if same and 0 <= gap <= gap_px:
            cur[3] = max(cur[3], s[3])
        else:
            if cur[3] - cur[1] >= 6:
                merged.append(tuple(cur))
            cur = list(s)
    if cur is not None and cur[3] - cur[1] >= 6:
        merged.append(tuple(cur))

    return merged

def _ensure_uint8(img):
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def _angle_deg(x1, y1, x2, y2):
    return np.degrees(np.arctan2((y2 - y1), (x2 - x1)))

def _band_ink_score(bw, x1, y1, x2, y2, band=2, samples=80):
    """
    Sample pixels in a band around the segment and compute:
    - fill: fraction of samples that land on ink
    - transitions: number of 0<->1 flips along the center line samples (dashed tends to flip more)
    """
    h, w = bw.shape[:2]
    x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
    xs = np.linspace(x1, x2, samples)
    ys = np.linspace(y1, y2, samples)

    # normal direction
    dx, dy = (x2 - x1), (y2 - y1)
    L = np.hypot(dx, dy) + 1e-6
    nx, ny = -dy / L, dx / L

    vals_center = []
    hits = 0
    total = 0

    for i in range(samples):
        cx, cy = xs[i], ys[i]
        # center sample for transitions
        ix, iy = int(round(cx)), int(round(cy))
        if 0 <= ix < w and 0 <= iy < h:
            v = 1 if bw[iy, ix] > 0 else 0
        else:
            v = 0
        vals_center.append(v)

        # band samples for fill
        for b in range(-band, band+1):
            px = int(round(cx + b * nx))
            py = int(round(cy + b * ny))
            if 0 <= px < w and 0 <= py < h:
                hits += 1 if bw[py, px] > 0 else 0
                total += 1

    # fill ratio
    fill = hits / max(1, total)

    # transitions along center
    trans = 0
    for i in range(1, len(vals_center)):
        if vals_center[i] != vals_center[i-1]:
            trans += 1

    return fill, trans

def _step4_write_dbg(out_dir, name, img):
    if not out_dir:
        return
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(os.path.join(out_dir, name), img)


def _step4_len(x1, y1, x2, y2):
    return float(math.hypot(x2 - x1, y2 - y1))


def _step4_sample_hits(bw, x1, y1, x2, y2, step=1, half_width=2):
    """
    Sample a thin band around the segment on binary ink image.
    bw: uint8, 255=ink, 0=background
    returns: 1D array of 0/1 hits.
    """
    h, w = bw.shape[:2]
    L = max(1, int(_step4_len(x1, y1, x2, y2)))
    n = max(12, L // step)

    xs = np.linspace(x1, x2, n).astype(np.int32)
    ys = np.linspace(y1, y2, n).astype(np.int32)

    dx, dy = (x2 - x1), (y2 - y1)
    norm = math.hypot(dx, dy) + 1e-6
    nx, ny = -dy / norm, dx / norm

    hits = []
    for x, y in zip(xs, ys):
        hit = 0
        for t in range(-half_width, half_width + 1):
            xx = int(round(x + t * nx))
            yy = int(round(y + t * ny))
            if 0 <= xx < w and 0 <= yy < h and bw[yy, xx] > 0:
                hit = 1
                break
        hits.append(hit)
    return np.array(hits, dtype=np.uint8)

# ==============================
# Step 4 (SOLID ONLY): Multiscale Hough + solidness filter + conservative merge
# ==============================
SOLID_CFG = {
    "target_max_dim": 2200,
    "canny_low": 50,
    "canny_high": 150,
    "scales": [1.0, 0.75, 0.6],
    "hough_threshold": 120,
    "max_line_gap": 10,
    "min_line_length_at_scale": {1.0: 90, 0.75: 70, 0.6: 55},
    "frame_shrink_px": 28,
    "notes_keep_ratio": 0.78,   #for our dataset
    "merge_min_len": 30.0,
    "merge_angle_thr_deg": 4.0,
    "merge_end_dist_thr": 12.0,
    "merge_gap_thr": 12.0,
    "cont_samples": 80,
    "min_density": 0.68,
    "max_gap": 4,
    "max_transitions": 6,
}

def find_inner_frame_mask(gray, shrink_px=12):
    h, w = gray.shape
    bw = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        35, 10
    )
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)

    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return np.ones((h, w), dtype=np.uint8) * 255

    best = None
    best_score = -1.0
    for c in cnts:
        x, y, ww, hh = cv2.boundingRect(c)
        area = ww * hh
        if area < 0.20 * w * h:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.01 * peri, True)
        rectish = 1.0 if len(approx) <= 8 else 0.0
        edge_touch = (x < 0.08*w) + (y < 0.08*h) + ((x+ww) > 0.92*w) + ((y+hh) > 0.92*h)
        score = area + rectish * 0.25 * area + edge_touch * 0.10 * area
        if score > best_score:
            best_score = score
            best = (x, y, ww, hh)

    if best is None:
        return np.ones((h, w), dtype=np.uint8) * 255

    x, y, ww, hh = best
    x1 = max(x + shrink_px, 0)
    y1 = max(y + shrink_px, 0)
    x2 = min(x + ww - shrink_px, w)
    y2 = min(y + hh - shrink_px, h)

    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    return mask

def remove_right_notes_block(gray, keep_ratio=0.78):
    h, w = gray.shape
    cut_x = int(w * keep_ratio)
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[:, :cut_x] = 255
    return mask

def seg_len(s):
    x1, y1, x2, y2 = s
    return float(np.hypot(x2 - x1, y2 - y1))

def seg_angle_deg(s):
    x1, y1, x2, y2 = s
    return float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))

def ang_diff_deg(a, b):
    d = abs(a - b) % 360.0
    d = min(d, 360.0 - d)
    return min(d, abs(d - 180.0))

def point_dist(p, q):
    return float(np.hypot(p[0] - q[0], p[1] - q[1]))

def point_line_perp_dist(px, py, x1, y1, x2, y2):
    vx, vy = x2-x1, y2-y1
    wx, wy = px-x1, py-y1
    area2 = abs(vx*wy - vy*wx)
    L = np.hypot(vx, vy) + 1e-9
    return float(area2 / L)

def unit_dir_from_angle_deg(a):
    rad = np.radians(a)
    return float(np.cos(rad)), float(np.sin(rad))

def project_scalar(pt, origin, ux, uy):
    return (pt[0] - origin[0]) * ux + (pt[1] - origin[1]) * uy

def detect_symbol_mask(gray, min_area=600, max_dim=260, max_ar=2.8, dilate=10):
    # binary ink mask
    bw = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 35, 10
    )
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)

    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros_like(gray, dtype=np.uint8)
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if area < min_area:
            continue
        if max(w, h) > max_dim:
            continue
        ar = max(w, h) / (min(w, h) + 1e-6)
        if ar > max_ar:
            continue
        cv2.rectangle(mask, (x, y), (x+w, y+h), 255, -1)

    if dilate > 0:
        k = np.ones((dilate, dilate), np.uint8)
        mask = cv2.dilate(mask, k, iterations=1)

    return mask  # 255 where symbols are

def merge_two_segments(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ang = seg_angle_deg(a)
    ux, uy = unit_dir_from_angle_deg(ang)
    origin = (ax1, ay1)

    pts = [(ax1, ay1), (ax2, ay2), (bx1, by1), (bx2, by2)]
    ts = [project_scalar(p, origin, ux, uy) for p in pts]

    pmin = pts[int(np.argmin(ts))]
    pmax = pts[int(np.argmax(ts))]
    return [int(round(pmin[0])), int(round(pmin[1])), int(round(pmax[0])), int(round(pmax[1]))]

def should_merge(a, b,
                 angle_thr=4.0,
                 end_dist_thr=12.0,
                 gap_thr=12.0,
                 perp_thr=6.0,
                 max_seg_len=350.0):
    la = seg_len(a)
    lb = seg_len(b)
    if la < SOLID_CFG["merge_min_len"] or lb < SOLID_CFG["merge_min_len"]:
        return False
    if max(la, lb) > max_seg_len:
        return False

    aa = seg_angle_deg(a)
    bb = seg_angle_deg(b)
    if ang_diff_deg(aa, bb) > angle_thr:
        return False

    a_pts = [(a[0], a[1]), (a[2], a[3])]
    b_pts = [(b[0], b[1]), (b[2], b[3])]
    mind = min(point_dist(p, q) for p in a_pts for q in b_pts)
    close_enough = (mind <= end_dist_thr)

    ux, uy = unit_dir_from_angle_deg(aa)
    origin = a_pts[0]

    def seg_proj(seg):
        p1 = (seg[0], seg[1])
        p2 = (seg[2], seg[3])
        t1 = project_scalar(p1, origin, ux, uy)
        t2 = project_scalar(p2, origin, ux, uy)
        return (min(t1, t2), max(t1, t2))

    a0, a1 = seg_proj(a)
    b0, b1 = seg_proj(b)
    inter = min(a1, b1) - max(a0, b0)
    overlap_or_small_gap = (inter >= -gap_thr)

    if not (close_enough and overlap_or_small_gap):
        return False

    bmx, bmy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
    if point_line_perp_dist(bmx, bmy, a[0], a[1], a[2], a[3]) > perp_thr:
        return False

    return True

def merge_segments(segments, angle_thr=4.0, end_dist_thr=12.0, gap_thr=12.0):
    segs = [list(map(int, s)) for s in segments]
    segs = [s for s in segs if seg_len(s) >= SOLID_CFG["merge_min_len"]]

    changed = True
    while changed:
        changed = False
        used = [False] * len(segs)
        new_segs = []
        for i in range(len(segs)):
            if used[i]:
                continue
            cur = segs[i]
            used[i] = True
            merged_any = True
            while merged_any:
                merged_any = False
                for j in range(len(segs)):
                    if used[j]:
                        continue
                    if should_merge(cur, segs[j], angle_thr, end_dist_thr, gap_thr, perp_thr=6.0):
                        cur = merge_two_segments(cur, segs[j])
                        used[j] = True
                        merged_any = True
                        changed = True
            new_segs.append(cur)
        segs = new_segs
    return segs

def solid_stats(edge_img, x1, y1, x2, y2, samples=80):
    xs = np.linspace(x1, x2, samples).astype(int)
    ys = np.linspace(y1, y2, samples).astype(int)

    h, w = edge_img.shape
    valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    xs, ys = xs[valid], ys[valid]
    if len(xs) < 10:
        return 0.0, 999, 999

    hits = (edge_img[ys, xs] > 0).astype(np.uint8)
    density = float(hits.mean())
    transitions = int(np.sum(hits[1:] != hits[:-1]))

    max_gap = 0
    run = 0
    for v in hits:
        if v == 0:
            run += 1
            max_gap = max(max_gap, run)
        else:
            run = 0

    return density, max_gap, transitions

def is_solid_line(edge_img, x1, y1, x2, y2,
                  samples=80,
                  min_density=0.68,
                  max_gap=4,
                  max_transitions=6):
    density, mgap, trans = solid_stats(edge_img, x1, y1, x2, y2, samples=samples)
    if density < min_density:
        return False
    if mgap > max_gap:
        return False
    if trans > max_transitions:
        return False
    return True

def _step4_core(
    img_bgr, step2_data,
    suppress_text=False,
    suppress_pad=10,
    notes_right_frac=0.0,
    out_dir=None,
    suppress_symbols=False,
    **kwargs
):
    """
    Solid-only Step 4.
    Returns dict with:
      solid: list[[x1,y1,x2,y2]]
      dashed: []
      notes_xmin: int or None
    """
    H0, W0 = img_bgr.shape[:2]
    cfg = SOLID_CFG

    # optional notes crop info (only for overlay; detector masks notes internally)
    notes_xmin = None
    notes_keep_ratio = cfg["notes_keep_ratio"]

    # cap scale (same as your final file)
    gray0 = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    cap_scale = cfg["target_max_dim"] / max(H0, W0)
    if cap_scale < 1:
        new_w = int(W0 * cap_scale)
        new_h = int(H0 * cap_scale)
        gray_cap = cv2.resize(gray0, (new_w, new_h), cv2.INTER_AREA)
    else:
        cap_scale = 1.0
        gray_cap = gray0
    symbol_mask_cap = None
    if suppress_symbols:
        symbol_mask_cap = detect_symbol_mask(gray_cap)
    inv_cap = 1.0 / cap_scale

    pred_all = []
    text_mask_cap = None
    if suppress_text and isinstance(step2_data, dict):
        boxes = step2_data.get("boxes") or step2_data.get("merged_boxes") or []
        if boxes:
            text_mask_cap = np.ones(gray_cap.shape, dtype=np.uint8) * 255
            for (x1, y1, x2, y2) in boxes:
                cx1 = int(round((x1 - suppress_pad) * cap_scale))
                cy1 = int(round((y1 - suppress_pad) * cap_scale))
                cx2 = int(round((x2 + suppress_pad) * cap_scale))
                cy2 = int(round((y2 + suppress_pad) * cap_scale))
                cx1 = max(0, min(gray_cap.shape[1]-1, cx1))
                cx2 = max(0, min(gray_cap.shape[1]-1, cx2))
                cy1 = max(0, min(gray_cap.shape[0]-1, cy1))
                cy2 = max(0, min(gray_cap.shape[0]-1, cy2))
                text_mask_cap[cy1:cy2+1, cx1:cx2+1] = 0
    for sc in cfg["scales"]:
        if sc != 1.0:
            gh, gw = gray_cap.shape
            gray = cv2.resize(gray_cap, (int(gw * sc), int(gh * sc)), cv2.INTER_AREA)
        else:
            gray = gray_cap
        
        gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(gray_blur, cfg["canny_low"], cfg["canny_high"], L2gradient=True)

        # keep-mask (frame + notes)
        frame_mask = find_inner_frame_mask(gray, shrink_px=cfg["frame_shrink_px"])
        notes_mask = remove_right_notes_block(gray, keep_ratio=notes_keep_ratio)
        keep_mask = cv2.bitwise_and(frame_mask, notes_mask)
        edges = cv2.bitwise_and(edges, edges, mask=keep_mask)

        # Optional text suppression (apply AFTER keep_mask)
        if text_mask_cap is not None:
            if sc != 1.0:
                tm = cv2.resize(text_mask_cap, (edges.shape[1], edges.shape[0]), interpolation=cv2.INTER_NEAREST)
            else:
                tm = text_mask_cap
            edges = cv2.bitwise_and(edges, edges, mask=tm)

        # Optional symbol suppression
        if symbol_mask_cap is not None:
            if sc != 1.0:
                sm = cv2.resize(symbol_mask_cap, (edges.shape[1], edges.shape[0]), interpolation=cv2.INTER_NEAREST)
            else:
                sm = symbol_mask_cap
            edges = cv2.bitwise_and(edges, edges, mask=cv2.bitwise_not(sm))
        
        # sampling edges (for solidity stats)
        edges_samp = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

        min_len = cfg["min_line_length_at_scale"].get(
            sc, int(cfg["min_line_length_at_scale"][1.0] * sc)
        )

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=cfg["hough_threshold"],
            minLineLength=min_len,
            maxLineGap=cfg["max_line_gap"],
        )
        if lines is None:
            continue

        inv_sc = 1.0 / sc

        for x1, y1, x2, y2 in lines[:, 0]:
            if not is_solid_line(
                edges_samp, x1, y1, x2, y2,
                samples=cfg["cont_samples"],
                min_density=cfg["min_density"],
                max_gap=cfg["max_gap"],
                max_transitions=cfg["max_transitions"]
            ):
                continue

            # scaled -> cap
            cx1, cy1 = x1 * inv_sc, y1 * inv_sc
            cx2, cy2 = x2 * inv_sc, y2 * inv_sc

            # cap -> original resized image coords
            ox1, oy1 = int(round(cx1 * inv_cap)), int(round(cy1 * inv_cap))
            ox2, oy2 = int(round(cx2 * inv_cap)), int(round(cy2 * inv_cap))
            pred_all.append([ox1, oy1, ox2, oy2])

    merged = merge_segments(
        pred_all,
        angle_thr=cfg["merge_angle_thr_deg"],
        end_dist_thr=cfg["merge_end_dist_thr"],
        gap_thr=cfg["merge_gap_thr"],
    )

    return {
        "solid": merged,
        "dashed": [],          # dashed dependency removed
        "notes_xmin": notes_xmin,
    }

def step4_extract_lines_solid_only(img_bgr, step2_data, **kwargs):
    return _step4_core(img_bgr, step2_data, **kwargs)

# ============================================================================
# Visualization
# ============================================================================

def draw_text_overlay(img, detections, out_path):
    """Draw text detection overlay."""
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)
        if d.get("text"):
            draw.text((x1, max(0, y1 - 14)), d["text"], fill=(255, 0, 0))
    pil.save(out_path)


def draw_line_overlay(img, solid, dashed, out_path, notes_xmin=None):
    """Draw line detection overlay with direction arrows."""
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    dr = ImageDraw.Draw(pil)

    # Draw solid lines
    for item in solid:
        # Handle both old format [x1,y1,x2,y2] and new format {"line": [...], "direction": ...}
        if isinstance(item, dict):
            x1, y1, x2, y2 = item["line"]
            direction = item.get("direction", "none")
        else:
            x1, y1, x2, y2 = item
            direction = "none"

        dr.line([(x1, y1), (x2, y2)], fill=(0, 255, 0), width=2)

        # Draw direction arrow if specified
        if direction != "none":
            _draw_direction_arrow(dr, x1, y1, x2, y2, direction, (0, 200, 0))

    # Draw dashed lines
    for item in dashed:
        # Handle both old format [x1,y1,x2,y2] and new format {"line": [...], "direction": ...}
        if isinstance(item, dict):
            x1, y1, x2, y2 = item["line"]
            direction = item.get("direction", "none")
        else:
            x1, y1, x2, y2 = item
            direction = "none"

        dr.line([(x1, y1), (x2, y2)], fill=(255, 0, 0), width=2)

        # Draw direction arrow if specified
        if direction != "none":
            _draw_direction_arrow(dr, x1, y1, x2, y2, direction, (200, 0, 0))

    if notes_xmin is not None:
        dr.line([(notes_xmin, 0), (notes_xmin, pil.height)], fill=(255, 0, 0), width=2)

    pil.save(out_path)


def _draw_direction_arrow(draw, x1, y1, x2, y2, direction, color):
    """Draw direction arrow(s) on a line."""
    # Calculate line center and arrow properties
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    dx, dy = x2 - x1, y2 - y1
    length = np.sqrt(dx**2 + dy**2)

    if length < 1:
        return

    # Normalize direction vector
    dx, dy = dx / length, dy / length

    # Arrow size (proportional to line length, max 20 pixels)
    arrow_size = min(20, length * 0.15)

    # Perpendicular vector for arrow wings
    perp_x, perp_y = -dy, dx

    def draw_arrow_at(px, py):
        """Draw a single arrow at position (px, py)"""
        # Arrow tip points in the direction of the line
        tip_x = px + dx * arrow_size * 0.6
        tip_y = py + dy * arrow_size * 0.6

        # Arrow wings
        wing1_x = px - dx * arrow_size * 0.4 + perp_x * arrow_size * 0.4
        wing1_y = py - dy * arrow_size * 0.4 + perp_y * arrow_size * 0.4

        wing2_x = px - dx * arrow_size * 0.4 - perp_x * arrow_size * 0.4
        wing2_y = py - dy * arrow_size * 0.4 - perp_y * arrow_size * 0.4

        # Draw arrow as a filled triangle
        draw.polygon([(tip_x, tip_y), (wing1_x, wing1_y), (wing2_x, wing2_y)],
                    fill=color, outline=color)

    if direction == "forward":
        # Arrow pointing from start to end (at center)
        draw_arrow_at(cx, cy)

    elif direction == "backward":
        # Arrow pointing from end to start (reverse direction at center)
        draw_arrow_at(cx - dx * arrow_size, cy - dy * arrow_size)

    elif direction == "bidirectional":
        # Two arrows, one at 1/3 and one at 2/3
        pos1_x, pos1_y = x1 + dx * length * 0.33, y1 + dy * length * 0.33
        pos2_x, pos2_y = x1 + dx * length * 0.67, y1 + dy * length * 0.67

        draw_arrow_at(pos1_x, pos1_y)
        draw_arrow_at(pos2_x, pos2_y)

TAG_RE = re.compile(r"^(?:[A-Z]{1,3}-?\d{2,5}[A-Z]?|SG\d+|PC|FO|N\d+|E\d+|ATM)$")
WORDY_RE = re.compile(r"^[A-Z][A-Z\s\-]{2,}$") 

def norm_text(t: str) -> str:
    t = t.strip()
    t = re.sub(r"\s+", " ", t)
    return t

def bbox_area(b):
    x1,y1,x2,y2 = b
    return max(0, x2-x1) * max(0, y2-y1)

def x_overlap_ratio(a, b):
    ax1, _, ax2, _ = a
    bx1, _, bx2, _ = b
    inter = max(0, min(ax2, bx2) - max(ax1, bx1))
    denom = max(1, min(ax2 - ax1, bx2 - bx1))  # relative to smaller width
    return inter / denom

def y_overlap_ratio(a, b):
    _, ay1, _, ay2 = a
    _, by1, _, by2 = b
    inter = max(0, min(ay2, by2) - max(ay1, by1))
    denom = max(1, min(ay2 - ay1, by2 - by1))  # relative to smaller height
    return inter / denom

def same_line(a, b, min_y_overlap=0.55):
    return y_overlap_ratio(a, b) >= min_y_overlap

def v_stack(a, b, min_x_overlap=0.55):
    return x_overlap_ratio(a, b) >= min_x_overlap

def merge_two(i, j, joiner=" "):
    # i/j are dicts: {"bbox":[...], "text":..., "score":...}
    ax1,ay1,ax2,ay2 = i["bbox"]
    bx1,by1,bx2,by2 = j["bbox"]
    nb = [min(ax1,bx1), min(ay1,by1), max(ax2,bx2), max(ay2,by2)]
    nt = (i["text"] + joiner + j["text"]).strip()
    ns = max(i["score"], j["score"])
    return {"bbox": nb, "text": nt, "score": ns}

def should_join_horiz(i, j):
    a, b = i["bbox"], j["bbox"]
    if not same_line(a, b): 
        return False
    # ensure left-to-right ordering
    if b[0] < a[0]:
        i, j = j, i
        a, b = i["bbox"], j["bbox"]
    gap = b[0] - a[2]
    h = max(1, min(a[3]-a[1], b[3]-b[1]))
    # small gap: tokens on same label line
    if gap > 1.2 * h:
        return False

    t1, t2 = norm_text(i["text"]), norm_text(j["text"])
    # join if either looks like a split tag or both look like label words
    tagish = (TAG_RE.match(t1.replace(" ","")) or TAG_RE.match(t2.replace(" ","")))
    wordy  = (WORDY_RE.match(t1.upper()) or WORDY_RE.match(t2.upper()))
    short  = (len(t1) <= 2 or len(t2) <= 2)
    return bool(tagish or wordy or short)

def should_join_vert(i, j):
    a, b = i["bbox"], j["bbox"]
    # ensure top-to-bottom ordering
    if b[1] < a[1]:
        i, j = j, i
        a, b = i["bbox"], j["bbox"]
    if not v_stack(a, b):
        return False
    gap = b[1] - a[3]
    w = max(1, min(a[2]-a[0], b[2]-b[0]))
    # allow slightly bigger gaps for multi-line blocks
    if gap > 0.35 * w and gap > 25:
        return False

    t1, t2 = norm_text(i["text"]), norm_text(j["text"])
    # join multiline blocks: uppercase words, or tank labels, etc.
    looks_block = (WORDY_RE.match(t1.upper()) or WORDY_RE.match(t2.upper()))
    return bool(looks_block)

def merge_items(items, img_w=None):
    # normalize text once
    items = [{"bbox": it["bbox"], "text": norm_text(it["text"]), "score": float(it["score"])} for it in items]

    # sort by y then x for stable merging
    items.sort(key=lambda d: (d["bbox"][1], d["bbox"][0]))

    # 1) horizontal merging pass
    changed = True
    while changed:
        changed = False
        out = []
        used = [False]*len(items)
        for i in range(len(items)):
            if used[i]: 
                continue
            cur = items[i]
            for j in range(i+1, len(items)):
                if used[j]:
                    continue
                if should_join_horiz(cur, items[j]):
                    # prevent runaway merges across the diagram
                    if img_w is not None:
                        ax1, ay1, ax2, ay2 = cur["bbox"]
                        bx1, by1, bx2, by2 = items[j]["bbox"]
                        new_w = max(ax2, bx2) - min(ax1, bx1)
                        if new_w > 0.22 * img_w:   # tune 0.18–0.28 as needed
                            continue
                
                    # joiner: no space for tag fragments, space for words
                    t1 = cur["text"].replace(" ", "")
                    t2 = items[j]["text"].replace(" ", "")
                    joiner = "" if (TAG_RE.match(t1) or TAG_RE.match(t2) or (len(cur["text"]) <= 2 or len(items[j]["text"]) <= 2)) else " "
                    cur = merge_two(cur, items[j], joiner=joiner)
                    used[j] = True
                    changed = True

            out.append(cur)
            used[i] = True
        items = sorted(out, key=lambda d: (d["bbox"][1], d["bbox"][0]))

    # 2) vertical merging pass (multi-line blocks)
    changed = True
    while changed:
        changed = False
        out = []
        used = [False]*len(items)
        for i in range(len(items)):
            if used[i]:
                continue
            cur = items[i]
            for j in range(i+1, len(items)):
                if used[j]:
                    continue
                if should_join_vert(cur, items[j]):
                    cur = merge_two(cur, items[j], joiner=" ")
                    used[j] = True
                    changed = True
            out.append(cur)
            used[i] = True
        items = sorted(out, key=lambda d: (d["bbox"][1], d["bbox"][0]))

    # 3) cleanup tiny junk after merge
    cleaned = []
    for it in items:
        t = it["text"].strip()
        if not t:
            continue
        area = bbox_area(it["bbox"])
        if len(t) == 1 and not t.isalnum():
            continue
        if area < 120 and it["score"] < 0.5:
            continue
        cleaned.append(it)

    return cleaned

def has_long_line(comp_mask, min_len=60):
    # comp_mask is uint8 {0,255}
    edges = cv2.Canny(comp_mask, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30,
                            minLineLength=min_len, maxLineGap=8)
    return lines is not None

def suppress_symbol_blobs_safe(
    bw,
    min_area=1200,
    max_ar=2.2,
    max_dim=220,
    extent_thr=0.25,
    edge_touch_thr=0.10,
    hough_min_len=80
):
    num, lab, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    out = bw.copy()
    removed = 0

    def has_long_line(comp_mask255):
        edges = cv2.Canny(comp_mask255, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30,
                                minLineLength=hough_min_len, maxLineGap=8)
        return lines is not None

    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue

        ar = max(w, h) / float(max(1, min(w, h)))
        if ar > max_ar or max(w, h) > max_dim:
            continue

        extent = area / float(max(1, w*h))
        if extent < extent_thr:
            continue  # too sparse -> likely line-ish

        # Edge-touch veto: likely a line/junction, not an isolated symbol
        comp = (lab == i).astype(np.uint8)
        roi = comp[y:y+h, x:x+w]
        band = 2
        top = roi[:band, :].sum()
        bot = roi[-band:, :].sum()
        lef = roi[:, :band].sum()
        rig = roi[:, -band:].sum()
        touch_score = max(top, bot, lef, rig) / float(max(1, area))
        if touch_score > edge_touch_thr:
            continue

        # Long-line veto: don’t remove components containing a long straight segment
        comp255 = roi * 255
        if has_long_line(comp255):
            continue

        out[lab == i] = 0
        removed += 1

    print(f"  suppress_symbol_blobs_safe: removed={removed}")
    return out
def neighbors8(y, x):
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            yield y + dy, x + dx

def detect_symbol_boxes(img_bgr, text_boxes_xyxy=None, pad=6, min_area=60, max_area=50000):
    """
    Returns symbol candidate bboxes in XYXY (x1,y1,x2,y2) in the SAME space as img_bgr.
    text_boxes_xyxy: list of [x1,y1,x2,y2] (optional) to erase text before symbol detection.
    """
    H, W = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # binarize dark ink
    _, bw = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # erase text regions so we don't treat text as symbols
    if text_boxes_xyxy:
        for x1,y1,x2,y2 in text_boxes_xyxy:
            x1 = max(0, int(x1-pad)); y1 = max(0, int(y1-pad))
            x2 = min(W-1, int(x2+pad)); y2 = min(H-1, int(y2+pad))
            bw[y1:y2, x1:x2] = 0

    # clean
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)

    # connected components => boxes
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    boxes = []
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if area < min_area or area > max_area:
            continue
        # avoid full-width/height junk
        if w > 0.8*W or h > 0.8*H:
            continue
        boxes.append([float(x), float(y), float(x+w), float(y+h)])
    return boxes

def segment_coverage_in_box(seg, box, step=4):
    """
    Returns fraction of segment length that lies inside the box.
    """
    x1, y1, x2, y2 = seg
    bx1, by1, bx2, by2 = box

    L = ((x2-x1)**2 + (y2-y1)**2) ** 0.5
    if L < 1e-6:
        return 0.0

    n = max(5, int(L / step))
    inside = 0

    for i in range(n + 1):
        t = i / n
        px = x1 + t * (x2 - x1)
        py = y1 + t * (y2 - y1)

        if bx1 <= px <= bx2 and by1 <= py <= by2:
            inside += 1

    return inside / (n + 1)

def post_filter_lines_strict(
    lines,
    text_boxes=None,
    symbol_boxes=None,
    text_cov_thr=0.15,
    symbol_cov_thr=0.20,
    pad=10,
):
    text_boxes = text_boxes or []
    symbol_boxes = symbol_boxes or []

    kept = []

    for seg in lines:
        reject = False

        # text suppression
        for b in text_boxes:
            bx1, by1, bx2, by2 = b
            box = [bx1-pad, by1-pad, bx2+pad, by2+pad]
            if segment_coverage_in_box(seg, box) >= text_cov_thr:
                reject = True
                break

        if reject:
            continue

        # symbol suppression
        for b in symbol_boxes:
            bx1, by1, bx2, by2 = b
            box = [bx1-pad, by1-pad, bx2+pad, by2+pad]
            if segment_coverage_in_box(seg, box) >= symbol_cov_thr:
                reject = True
                break

        if not reject:
            kept.append(seg)

    return kept
    
# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Process a single P&ID image through complete pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required
    parser.add_argument("--image", required=True, help="Path to input P&ID image")
    parser.add_argument("--out", required=True, help="Output directory")

    # Step 1: EasyOCR detection
    parser.add_argument("--target-width", type=int, default=7168,
                       help="Resize image to this width")
    parser.add_argument("--win", type=int, default=800,
                       help="Sliding window size")
    parser.add_argument("--stride", type=int, default=400,
                       help="Sliding window stride")
    parser.add_argument("--lang", default="en",
                       help="EasyOCR language")
    parser.add_argument("--gpu", action="store_true",
                       help="Use GPU for PaddleOCR")

    # Step 2: Merge
    parser.add_argument("--nms-iou", type=float, default=0.3,
                       help="IoU threshold for merging boxes")

    # Interactive editing
    parser.add_argument("--interactive", action="store_true",
                       help="Launch interactive editors after text and line extraction")

    # Step 3: Line extraction
    parser.add_argument("--suppress-symbols", action="store_true",
                    help="Remove large compact symbols before line detection")
    parser.add_argument("--suppress-text", action="store_true",
                    help="Remove text regions before line detection")
    parser.add_argument("--suppress-pad", type=int, default=2,
                       help="Padding for text suppression")
    

    args = parser.parse_args()

    # Validate input
    if not os.path.exists(args.image):
        print(f"Error: Image not found: {args.image}")
        sys.exit(1)

    # Create output directory
    os.makedirs(args.out, exist_ok=True)

    # Get image name for output files
    img_name = Path(args.image).stem

    print(f"\n{'='*70}")
    print(f"Processing P&ID Image: {args.image}")
    print(f"Output directory: {args.out}")
    print(f"{'='*70}")

    # Load and resize image
    print("\nLoading and resizing image...")
    img = load_image(args.image)
    img_resized, scale = resize_keep_aspect(img, target_width=args.target_width)
    print(f"  Original size: {img.shape[1]}x{img.shape[0]}")
    print(f"  Resized to: {img_resized.shape[1]}x{img_resized.shape[0]} (scale={scale:.3f})")

    # =========================
    # Initialize PaddleOCR
    # =========================
    print(f"\nInitializing PaddleOCR (gpu={args.gpu})...")
    ocr = PaddleOCR(
        lang=args.lang,             # "en"
        use_gpu=args.gpu,           # True on GPU node
        use_angle_cls=True          # keep for 2.7.3; later can switch to orientation flags
    )

    # STEP 1: PaddleOCR detection + recognition
    step1_data = step1_paddleocr(
        img_resized, ocr,
        use_tiling=True,
        tile=1400,
        stride=900,
        preprocess="clahe",
        scales=(1.0, 1.35),
        min_conf=0.22,
        nms_iou=0.35
    )
    step1_data["image_path"] = args.image
    step1_data["target_width"] = args.target_width
    step1_data["scale"] = float(scale)

    step1_json = os.path.join(args.out, f"{img_name}_step1_paddleocr.json")
    with open(step1_json, "w", encoding="utf-8") as f:
        json.dump(step1_data, f, ensure_ascii=False, indent=2)
    print(f"\n  Saved: {step1_json}")

    # STEP 2: Filter / clean text boxes.
    # We start from raw PaddleOCR boxes and:
    #   - drop very tiny / extreme aspect boxes
    #   - drop boxes that are too large (likely non-text regions)
    #   - drop low-score boxes
    # This cleaned set is used both for Tesseract (Step 3)
    # and for text suppression in the line detector (Step 4).
    step2_data = {
        "resized_shape": step1_data["resized_shape"],
        "nms_iou": args.nms_iou,
        "boxes": step1_data["boxes"],
        "scores": step1_data["scores"],
    }

    # --- filter logic ---
    H, W = img_resized.shape[:2]
    img_area = H * W

    f_boxes, f_scores = [], []
    for b, s in zip(step2_data["boxes"], step2_data["scores"]):
        x1, y1, x2, y2 = b
        bw, bh = x2-x1, y2-y1
        area = bw * bh

        if bw < 28 and bh < 14:
            continue
        if (bw / max(1,bh) > 20) or (bh / max(1,bw) > 20):
            continue
        if area > 0.02 * img_area:
            continue
        if s < 0.25:
            continue

        f_boxes.append(b)
        f_scores.append(s)

    step2_data["boxes"] = f_boxes
    step2_data["scores"] = f_scores

    step2_json = os.path.join(args.out, f"{img_name}_step2_boxes.json")
    with open(step2_json, "w", encoding="utf-8") as f:
        json.dump(step2_data, f, ensure_ascii=False, indent=2)
    print(f"\n  Saved: {step2_json}")

    # STEP 3: build "step3_data" directly from PaddleOCR outputs (NO Tesseract)
    # Align texts with filtered boxes by re-filtering indices
    step3_data = []
    # Build a quick lookup using original (box,score,text)
    orig = list(zip(step1_data["boxes"], step1_data["scores"], step1_data["texts"]))
    # Keep only those that survived the same filter condition (recompute same rules)
    for (b, s, t) in orig:
        x1, y1, x2, y2 = b
        bw, bh = x2-x1, y2-y1
        area = bw * bh
        if bw < 28 and bh < 14: 
            continue
        if (bw / max(1,bh) > 20) or (bh / max(1,bw) > 20):
            continue
        if area > 0.02 * img_area:
            continue
        if s < 0.25:
            continue
        step3_data.append({"bbox": [int(v) for v in b], "text": t, "score": float(s)})
    step3_data = merge_items(step3_data, img_w=W)
    step3_json = os.path.join(args.out, f"{img_name}_step3_text.json")
    with open(step3_json, "w", encoding="utf-8") as f:
        json.dump(step3_data, f, ensure_ascii=False, indent=2)
    print(f"\n  Saved: {step3_json}")

    text_overlay = os.path.join(args.out, f"{img_name}_step3_text_overlay.png")
    draw_text_overlay(img_resized, step3_data, text_overlay)
    print(f"  Saved: {text_overlay}")

    # STEP 4: Extract SOLID lines
    step4_kwargs = dict(
        suppress_text=args.suppress_text,
        suppress_pad=args.suppress_pad,
    )
    
    step4_data = step4_extract_lines_solid_only(img_resized, step2_data, **step4_kwargs)
    
    step4_data["image_path"] = args.image
    step4_data["target_width"] = args.target_width
    step4_data["scale"] = float(scale)
    
    step4_json = os.path.join(args.out, f"{img_name}_step4_lines.json")
    with open(step4_json, "w", encoding="utf-8") as f:
        json.dump(step4_data, f, ensure_ascii=False, indent=2)
    print(f"\n  Saved: {step4_json}")
    
    line_overlay = os.path.join(args.out, f"{img_name}_step4_lines_overlay.png")
    draw_line_overlay(
        img_resized,
        step4_data["solid"],
        step4_data.get("dashed", []),
        line_overlay,
        notes_xmin=step4_data.get("notes_xmin"),
    )
    print(f"  Saved: {line_overlay}")

    # Interactive editing (if requested)
    if args.interactive:
        print(f"\n{'='*70}")
        print("LAUNCHING INTERACTIVE EDITORS")
        print(f"{'='*70}")

        # Stage 1: Text Editor
        print("\n=== Stage 1: Interactive Text Editor ===")
        print(f"Opening text editor for: {step3_json}")
        print("You can:")
        print("  - Delete text boxes by selecting them and pressing Delete")
        print("  - Combine text boxes by selecting multiple and pressing 'C'")
        print("  - Edit text by selecting a box and pressing 'E'")
        print("Close the editor window when done to proceed to line editing.\n")

        try:
            # Import and run text editor
            from interactive_text_editor import TextEditor
            text_editor = TextEditor(json_path=step3_json)
            text_editor.run()
            print("\nText editing complete.")
        except Exception as e:
            print(f"\nWarning: Text editor failed: {e}")
            print("Continuing to line editor...")

        # Stage 2: Line Editor
        print("\n=== Stage 2: Interactive Line Editor ===")
        print(f"Opening line editor for: {step4_json}")
        print("You can:")
        print("  - Delete lines by selecting them and pressing Delete")
        print("  - Add new lines by pressing 'D' and drawing")
        print("  - Toggle line type (solid/dashed) in the toolbar")
        print("Close the editor window when done.\n")

        try:
            # Import and run line editor
            from interactive_line_editor import LineEditor
            line_editor = LineEditor(json_path=step4_json)
            line_editor.run()
            print("\nLine editing complete.")
        except Exception as e:
            print(f"\nWarning: Line editor failed: {e}")

        print(f"\n{'='*70}")
        print("INTERACTIVE EDITING COMPLETE!")
        print(f"{'='*70}")
        print("\nEdited results have been saved in-place:")
        print(f"  - {step3_json}")
        print(f"  - {step4_json}")
        print("\nYou may want to regenerate the overlays to visualize the edited results.")
        print()

    # Print summary
    print(f"\n{'='*70}")
    print("PIPELINE COMPLETE!")
    print(f"{'='*70}")
    print(f"\nResults summary:")
    print(f"  Text regions detected: {len(step2_data['boxes'])}")
    # print(f"  Text boxes with OCR: {len(step3_data)}")
    print(f"  Solid lines: {len(step4_data['solid'])}")
    print(f"  Dashed lines: {len(step4_data['dashed'])}")
    print(f"\nAll outputs saved to: {args.out}")
    print(f"  - {img_name}_step1_paddleocr.json (raw PaddleOCR detections)")
    print(f"  - {img_name}_step2_boxes.json (filtered text boxes for OCR/line suppression)")
    print(f"  - {img_name}_step3_text.json (extracted text)")
    print(f"  - {img_name}_step3_text_overlay.png (text visualization)")
    print(f"  - {img_name}_step4_lines.json (line geometry)")
    print(f"  - {img_name}_step4_lines_overlay.png (line visualization)")
    if args.interactive:
        print(f"\nNote: Interactive editing was performed. JSON files contain edited results.")
    print()


if __name__ == "__main__":
    main()