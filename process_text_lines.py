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


# ------------------------------------------------------------
# Dash / solid classification helpers
# ------------------------------------------------------------

def _runs_01(hits):
    runs1, runs0 = [], []
    if hits is None:
        return runs1, runs0

    # Normalize to 1D list of 0/1
    if isinstance(hits, np.ndarray):
        if hits.size == 0:
            return runs1, runs0
        h = (hits.reshape(-1) > 0).astype(np.uint8).tolist()
    else:
        if len(hits) == 0:
            return runs1, runs0
        h = [1 if int(v) > 0 else 0 for v in hits]

    cur = h[0]
    ln = 1
    for v in h[1:]:
        if v == cur:
            ln += 1
        else:
            (runs1 if cur == 1 else runs0).append(ln)
            cur = v
            ln = 1
    (runs1 if cur == 1 else runs0).append(ln)
    return runs1, runs0

def _classify_dash_from_hits(hits, dash_min_bit=6):
    """
    Very simple and robust dashed vs solid classifier.

    hits: 0/1 array sampled along the segment.
          1 = ink, 0 = background.

    Returns:
        "dashed" or "solid"
    """
    if hits is None:
        return "solid"

    # Normalize to 0/1 list
    if isinstance(hits, np.ndarray):
        if hits.size == 0:
            return "solid"
        h = (hits.reshape(-1) > 0).astype(np.uint8).tolist()
    else:
        if len(hits) == 0:
            return "solid"
        h = [1 if int(v) > 0 else 0 for v in hits]

    if len(h) < dash_min_bit:
        # too few samples to conclude "dashed"
        return "solid"

    runs1, runs0 = _runs_01(h)

    # Heuristic:
    # - if we see at least 3 separate background runs (0-runs),
    #   this looks like repeated gaps => dashed.
    if len(runs0) >= 3:
        return "dashed"

    return "solid"

def _cv(vals):
    if not vals:
        return 999.0
    v = np.array(vals, dtype=np.float32)
    m = float(v.mean())
    if m < 1e-6:
        return 999.0
    return float(v.std() / m)

def _step4_seg_angle(x1, y1, x2, y2):
    """
    Orientation of a single segment in degrees in [-180, 180].
    """
    return float(math.degrees(math.atan2((y2 - y1), (x2 - x1))))

def _axis_aligned(x1, y1, x2, y2, tol_deg=4.0):
    """
    True if the segment is really close to horizontal or vertical.

    Instead of using only the angle, we also enforce that almost all
    of the length lies along a single axis. This kills diagonal junk.
    """
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)

    # Ignore tiny segments – they’re usually noise
    L = math.hypot(dx, dy)
    if L < 8:
        return False

    # Main component (x or y) should carry almost all of the length
    main = max(dx, dy)
    off  = min(dx, dy)

    # How much "off-axis" component we allow based on tol_deg
    # (for tol_deg=4, this is about 7% of the main component)
    max_off = math.tan(math.radians(tol_deg)) * main
    return off <= max_off

def _step4_merge_collinear(segments, angle_tol=6, dist_tol=18):
    """
    Greedy merge: group by angle similarity + endpoint proximity,
    then merge by projecting endpoints along direction.
    segments: list of [x1,y1,x2,y2]
    """
    if not segments:
        return []

    segs = [list(map(float, s)) for s in segments]
    used = [False] * len(segs)
    out = []

    def near(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        d2 = min(
            (ax1 - bx1) ** 2 + (ay1 - by1) ** 2,
            (ax1 - bx2) ** 2 + (ay1 - by2) ** 2,
            (ax2 - bx1) ** 2 + (ay2 - by1) ** 2,
            (ax2 - bx2) ** 2 + (ay2 - by2) ** 2,
        )
        return d2 <= dist_tol * dist_tol

    for i, s in enumerate(segs):
        if used[i]:
            continue
        used[i] = True
        group = [s]
        a0 = _step4_seg_angle(*s)

        changed = True
        while changed:
            changed = False
            for j, t in enumerate(segs):
                if used[j]:
                    continue
                a1 = _step4_seg_angle(*t)
                da = abs(((a1 - a0 + 90) % 180) - 90)  # modulo 180
                if da <= angle_tol and any(near(t, g) for g in group):
                    used[j] = True
                    group.append(t)
                    changed = True

        ang = math.radians(a0)
        ux, uy = math.cos(ang), math.sin(ang)

        pts = []
        for x1, y1, x2, y2 in group:
            pts.append((x1, y1))
            pts.append((x2, y2))

        projs = [p[0] * ux + p[1] * uy for p in pts]
        pmin = pts[int(np.argmin(projs))]
        pmax = pts[int(np.argmax(projs))]
        out.append([float(pmin[0]), float(pmin[1]), float(pmax[0]), float(pmax[1])])

    return out

def _point_to_seg_dist(px, py, x1, y1, x2, y2):
    """
    Euclidean distance from point (px,py) to line segment (x1,y1)-(x2,y2).
    """
    vx, vy = (x2 - x1), (y2 - y1)
    wx, wy = (px - x1), (py - y1)
    seg_len2 = vx * vx + vy * vy
    if seg_len2 <= 1e-6:
        # degenerate segment
        return math.hypot(px - x1, py - y1)

    t = (wx * vx + wy * vy) / seg_len2
    t = max(0.0, min(1.0, t))
    projx = x1 + t * vx
    projy = y1 + t * vy
    return math.hypot(px - projx, py - projy)


def _remove_cross_short_artifacts(solid_segs, dashed_segs, len_th=55.0, end_tol=6.0):
    """
    Remove tiny 'cross-hair' artefacts:
      - segment length < len_th
      - BOTH endpoints lie very close to some other segment.
    These are the little + shaped junk lines around symbols/junctions.

    We use BOTH solid + dashed as neighbours so we don't miss mixed cases.
    """
    all_segs = list(solid_segs) + list(dashed_segs)
    n = len(all_segs)
    if n == 0:
        return solid_segs, dashed_segs

    keep = [True] * n

    for i, (x1, y1, x2, y2) in enumerate(all_segs):
        L = _step4_len(x1, y1, x2, y2)
        if L >= len_th:
            continue  # long pipes are always kept

        p1 = (x1, y1)
        p2 = (x2, y2)
        min1 = 1e9
        min2 = 1e9

        for j, (u1, v1, u2, v2) in enumerate(all_segs):
            if i == j:
                continue
            d1 = _point_to_seg_dist(p1[0], p1[1], u1, v1, u2, v2)
            d2 = _point_to_seg_dist(p2[0], p2[1], u1, v1, u2, v2)
            if d1 < min1:
                min1 = d1
            if d2 < min2:
                min2 = d2

        # both endpoints are essentially “sitting on” other segments → cross junk
        if (min1 < end_tol) and (min2 < end_tol):
            keep[i] = False

    # split back into solid / dashed
    new_solid, new_dashed = [], []
    for idx, seg in enumerate(all_segs):
        if not keep[idx]:
            continue
        if idx < len(solid_segs):
            new_solid.append(seg)
        else:
            new_dashed.append(seg)

    return new_solid, new_dashed
    
# ------------------------------------------------------------
# Main step4
# ------------------------------------------------------------
def _step4_core(
    img_bgr, step2_data,
    suppress_text=False,
    suppress_pad=10,
    notes_right_frac=0.0,
    sat_th=35,          # kept for compatibility; not used
    v_keep_max=0.85,    # kept for compatibility; not used
    kernel_pct=0.0010,  # kept for compatibility; not used
    min_len=30,
    merge_gap=18,
    angle_tol=6,
    canny1=50,
    canny2=150,
    hough_minlen_frac=0.03,
    hough_maxgap=12,
    dash_min_bit=6,
    dash_close_gap=1,   # kept for compatibility; not used
    out_dir=None,
    suppress_symbols=False,
    solid_only=False,   # <<< NEW: if True, drop dashed in final output
):
    """
    Shared Step 4 core:

      1) grayscale + equalize
      2) build binary ink mask (bw_samp) for sampling / dash detection
      3) optionally suppress text boxes & symbol blobs
      4) thicken lines slightly, Canny on grayscale for Hough
      5) Hough + LSD to get candidate segments
      6) keep only axis-aligned candidates
      7) classify solid vs dashed based on bw_samp hits
      8) merge collinear fragments
      9) if solid_only=True => discard dashed segments in final result
    """
    H, W = img_bgr.shape[:2]
    work = img_bgr.copy()

    # ----------------------------
    # 0. Optional notes panel crop
    # ----------------------------
    notes_xmin = None
    if notes_right_frac and float(notes_right_frac) > 0:
        notes_xmin = int(W * (1.0 - float(notes_right_frac)))
        work[:, notes_xmin:] = 255
        _step4_write_dbg(out_dir, "dbg_crop_notes.png", work)

    # ----------------------------
    # 1. Grayscale + equalize (boost thin lines)
    # ----------------------------
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    gray_eq = cv2.equalizeHist(gray)
    _step4_write_dbg(out_dir, "dbg_gray_eq.png", gray_eq)

    # ----------------------------
    # 2. Binary ink mask (for sampling/dash detection)
    # ----------------------------
    bw_samp = cv2.adaptiveThreshold(
        gray_eq, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        35, 10
    )
    bw_samp = _ensure_uint8(bw_samp)

    if suppress_symbols:
        bw_samp = suppress_symbol_blobs_safe(bw_samp)
    _step4_write_dbg(out_dir, "dbg_bw_samp_raw.png", bw_samp)

    # ----------------------------
    # 3. Suppress text boxes from BOTH mask and grayscale
    # ----------------------------
    if suppress_text and isinstance(step2_data, dict):
        boxes = step2_data.get("boxes") or step2_data.get("merged_boxes") or []
        if boxes:
            bw_s2 = bw_samp.copy()
            gray2 = gray_eq.copy()
            for (x1, y1, x2, y2) in boxes:
                x1 = max(0, int(x1) - int(suppress_pad))
                y1 = max(0, int(y1) - int(suppress_pad))
                x2 = min(W - 1, int(x2) + int(suppress_pad))
                y2 = min(H - 1, int(y2) + int(suppress_pad))
                bw_s2[y1:y2+1, x1:x2+1] = 0
                gray2[y1:y2+1, x1:x2+1] = 255
            bw_samp = bw_s2
            gray_eq = gray2
            _step4_write_dbg(out_dir, "dbg_bw_samp_notext.png", bw_samp)
            _step4_write_dbg(out_dir, "dbg_gray_eq_notext.png", gray_eq)

    # ----------------------------
    # 4. Thicken lines slightly, then Canny on grayscale
    # ----------------------------
    gray_edges = gray_eq.copy()
    gray_edges[bw_samp == 0] = 255

    gray_thick = cv2.dilate(gray_edges, np.ones((2, 2), np.uint8), iterations=1)
    gray_thick = cv2.GaussianBlur(gray_thick, (3, 3), 0)

    edges = cv2.Canny(gray_thick, int(canny1), int(canny2))
    _step4_write_dbg(out_dir, "dbg_gray_thick.png", gray_thick)
    _step4_write_dbg(out_dir, "dbg_edges.png", edges)

    # ----------------------------
    # 5. Candidate segments: Hough + LSD
    # ----------------------------
    cand = []
    minLineLength = int(max(int(min_len), float(hough_minlen_frac) * max(H, W)))
    maxLineGap = int(hough_maxgap)

    # 5a. Hough
    hl = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=80,
        minLineLength=minLineLength,
        maxLineGap=maxLineGap
    )
    if hl is not None:
        for x1, y1, x2, y2 in hl[:, 0]:
            if _step4_len(x1, y1, x2, y2) >= float(min_len):
                cand.append([float(x1), float(y1), float(x2), float(y2)])

    # 5b. LSD on inverted ink mask
    try:
        lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
        lsd_lines = lsd.detect(255 - bw_samp)[0]
        if lsd_lines is not None:
            for l in lsd_lines:
                x1, y1, x2, y2 = l[0]
                if _step4_len(x1, y1, x2, y2) >= float(min_len):
                    cand.append([float(x1), float(y1), float(x2), float(y2)])
    except Exception:
        pass

    # Keep only near-axis-aligned pipes (0° / 90°)
    cand_axis = []
    for x1, y1, x2, y2 in cand:
        if _axis_aligned(x1, y1, x2, y2, tol_deg=4.0):
            cand_axis.append([x1, y1, x2, y2])
    cand = cand_axis

    if out_dir:
        vis = work.copy()
        for x1, y1, x2, y2 in cand:
            cv2.line(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
        _step4_write_dbg(out_dir, "dbg_candidates.png", vis)

        # ----------------------------
    # 6. Classify each segment: solid vs dashed
    # ----------------------------
    solid = []
    dashed = []

    for x1, y1, x2, y2 in cand:
        L = _step4_len(x1, y1, x2, y2)

        # sample on the raw binary ink mask (preserves dash gaps)
        hits = _step4_sample_hits(
            bw_samp, x1, y1, x2, y2,
            step=1,
            half_width=3
        )

        # use simple heuristic classifier
        lab = _classify_dash_from_hits(hits, dash_min_bit=dash_min_bit)

        # safety: dashed must be near-axis-aligned
        if lab == "dashed" and not _axis_aligned(x1, y1, x2, y2, tol_deg=4.0):
            lab = "solid"

        if lab == "solid":
            solid.append([x1, y1, x2, y2])
        else:  # "dashed"
            # keep modest minimum to avoid tiny noise
            if L >= max(0.4 * float(min_len), 10.0):
                dashed.append([x1, y1, x2, y2])

    # ----------------------------
    # 7. Merge collinear fragments
    # ----------------------------
    solid_m = _step4_merge_collinear(
        solid, angle_tol=float(angle_tol), dist_tol=float(merge_gap)
    )
    dashed_m = _step4_merge_collinear(
        dashed, angle_tol=float(angle_tol), dist_tol=float(merge_gap) * 2.0
    )

    # ----------------------------
    # 7.5 Remove tiny cross artefacts
    # ----------------------------
    solid_m, dashed_m = _remove_cross_short_artifacts(
        solid_m, dashed_m,
        len_th=55.0,   # you can tune between ~45–70 if needed
        end_tol=6.0    # radius (in px) for "touching" another segment
    )

    # ----------------------------
    # 8. Solid-only vs full mode
    # ----------------------------
    if solid_only:
        # In AFW-style mode we just ignore dashed completely
        dashed_m = []

    # (IMPORTANT: no global DASH_MIN_KEEP here in full mode,
    #  so dashed lines on SAMPLE diagrams are not wiped.)

    # ----------------------------
    # 9. Debug overlay
    # ----------------------------
    if out_dir:
        ov = work.copy()
        for x1, y1, x2, y2 in solid_m:
            cv2.line(ov, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        for x1, y1, x2, y2 in dashed_m:
            cv2.line(ov, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        _step4_write_dbg(out_dir, "dbg_step4_overlay_internal.png", ov)
    # Final safety pass: drop anything that is not really axis-aligned
    solid_lines = [s for s in solid_m if _axis_aligned(*s, tol_deg=2.0)]
    dash_lines  = [d for d in dashed_m  if _axis_aligned(*d, tol_deg=5.0)]

    return {
        "solid": solid_lines,
        "dashed": dash_lines,
        "notes_xmin": notes_xmin,
    }


def step4_extract_lines_solid_only(
    img_bgr, step2_data, **kwargs
):
    """
    Public API: detect only solid lines.
    Any dashed segments found are dropped in the final output.
    """
    return _step4_core(
        img_bgr, step2_data,
        solid_only=True,
        **kwargs
    )


def step4_extract_lines_solid_dashed(
    img_bgr, step2_data, **kwargs
):
    """
    Public API: detect both solid and dashed lines.
    """
    return _step4_core(
        img_bgr, step2_data,
        solid_only=False,
        **kwargs
    )

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

def suppress_symbol_blobs_safe(bw, min_area=1200, max_ar=2.2, max_dim=220):
    """
    Remove only large, compact-ish blobs (symbols).
    Keeps small/broken pipe parts and dashed segments.
    """
    num, lab, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    out = bw.copy()
    removed = 0

    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue

        ar = max(w, h) / float(max(1, min(w, h)))

        # large + compact-ish + not too huge
        if ar <= max_ar and max(w, h) <= max_dim:
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
    parser.add_argument("--notes-right-frac", type=float, default=0.0,
                       help="Fraction of right side to ignore (e.g., 0.23)")
    parser.add_argument("--sat-th", type=int, default=70,
                       help="Saturation threshold for ink isolation")
    parser.add_argument("--v-keep-max", type=int, default=245,
                       help="Max value to keep in HSV for ink isolation")
    parser.add_argument("--kernel-pct", type=float, default=0.0012,
                       help="Morphology kernel size as fraction of image")
    parser.add_argument("--min-len", type=int, default=22,
                       help="Minimum line segment length")
    parser.add_argument("--merge-gap", type=int, default=12,
                       help="Max gap to merge collinear segments")
    parser.add_argument("--angle-tol", type=float, default=5.5,
                       help="Angle tolerance for axis alignment")
    parser.add_argument("--canny1", type=int, default=35,
                       help="Canny edge detection threshold 1")
    parser.add_argument("--canny2", type=int, default=85,
                       help="Canny edge detection threshold 2")
    parser.add_argument("--hough-minlen-frac", type=float, default=0.022,
                       help="Hough min line length as fraction of image")
    parser.add_argument("--hough-maxgap", type=int, default=7,
                       help="Hough max gap between segments")
    parser.add_argument("--dash-min-bit", type=int, default=9,
                       help="Minimum dash segment length")
    parser.add_argument("--dash-close-gap", type=int, default=10,
                       help="Max gap between dashes to join")
    parser.add_argument(
        "--line-mode",
        choices=["solid", "solid_dashed"],
        default="solid",
        help="solid: detect only solid lines (AFW-style); "
             "solid_dashed: detect both solid and dashed lines"
    )
    

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

    # STEP 4: Extract lines
    step4_kwargs = dict(
        suppress_text=args.suppress_text,
        suppress_pad=args.suppress_pad,
        notes_right_frac=args.notes_right_frac,
        sat_th=args.sat_th,
        v_keep_max=args.v_keep_max,
        kernel_pct=args.kernel_pct,
        min_len=args.min_len,
        merge_gap=args.merge_gap,
        angle_tol=args.angle_tol,
        canny1=args.canny1,
        canny2=args.canny2,
        hough_minlen_frac=args.hough_minlen_frac,
        hough_maxgap=args.hough_maxgap,
        dash_min_bit=args.dash_min_bit,
        dash_close_gap=args.dash_close_gap,
        out_dir=args.out,
        suppress_symbols=args.suppress_symbols,
    )

    if args.line_mode == "solid":
        step4_data = step4_extract_lines_solid_only(
            img_resized, step2_data, **step4_kwargs
        )
    else:
        step4_data = step4_extract_lines_solid_dashed(
            img_resized, step2_data, **step4_kwargs
        )

    step4_data["image_path"] = args.image
    step4_data["target_width"] = args.target_width
    step4_data["scale"] = float(scale)

    # Save Step 4 output
    step4_json = os.path.join(args.out, f"{img_name}_step4_lines.json")
    with open(step4_json, "w", encoding="utf-8") as f:
        json.dump(step4_data, f, ensure_ascii=False, indent=2)
    print(f"\n  Saved: {step4_json}")

    # Draw line overlay
    line_overlay = os.path.join(args.out, f"{img_name}_step4_lines_overlay.png")
    draw_line_overlay(img_resized, step4_data["solid"], step4_data["dashed"],
                     line_overlay, notes_xmin=step4_data.get("notes_xmin"))
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