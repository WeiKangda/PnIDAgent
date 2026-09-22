#!/usr/bin/env python3
"""Learned pipe-line extractor: small U-Net over native-resolution tiles.

Default line detector of the pipeline since 2026-09-22.  Measured on the 50
held-out Dataset-P&ID validation sheets (all GT lines, 8 px, many-to-many):
as shipped 0.817, tuned classical Hough 0.931, this model 0.986 count F1;
connectivity edge F1 with detected symbols 0.759 (classical) -> 0.991.

Three classes: background / solid pipe / dashed pipe.  Trained on GT line
segments drawn 3 px wide; frame rulings, tables, text and symbol outlines are
negatives.  Mask -> HoughLinesP -> collinear merge gives straight segments in
the same {"solid": [[x1,y1,x2,y2],...], "dashed": [...]} format as the
classical stage.  Training lives in eval_tools/line_seg.py.

Needs torch; PaddleOCR 2.7.3 cannot share an env with it, so
process_text_lines.py runs this through a subprocess when torch is missing:

    python pnid_lineseg.py --image sheet.jpg --out sheet_step4_lines.json
"""
from __future__ import annotations

import argparse
import collections
import json
import os
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CKPT = os.path.join(HERE, "weights", "lineseg_unet_450.pt")
Seg = Tuple[int, int, int, int]


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------

def build_model(base: int = 32, n_classes: int = 3):
    import torch
    import torch.nn as nn

    def block(i, o):
        return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                             nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))

    class UNet(nn.Module):
        def __init__(self):
            super().__init__()
            c = [base, base * 2, base * 4, base * 8, base * 16]
            self.enc = nn.ModuleList([block(1, c[0])] + [block(c[i], c[i + 1]) for i in range(4)])
            self.pool = nn.MaxPool2d(2)
            self.up = nn.ModuleList([nn.ConvTranspose2d(c[i + 1], c[i], 2, stride=2) for i in range(4)])
            self.dec = nn.ModuleList([block(c[i] * 2, c[i]) for i in range(4)])
            self.head = nn.Conv2d(c[0], n_classes, 1)

        def forward(self, x):
            skips = []
            for i, e in enumerate(self.enc):
                x = e(x)
                if i < 4:
                    skips.append(x)
                    x = self.pool(x)
            for i in range(3, -1, -1):
                x = self.up[i](x)
                x = self.dec[i](torch.cat([x, skips[i]], 1))
            return self.head(x)

    return UNet()


def load_model(ckpt: str = DEFAULT_CKPT, device=None):
    import torch
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(ckpt, map_location="cpu")
    m = build_model(ck.get("base", 32)).to(device)
    if device.type == "cuda":
        m = m.to(memory_format=torch.channels_last)
    m.load_state_dict(ck["model"])
    m.eval()
    return m, device


def _to_tensor(x: np.ndarray, device):
    import torch
    t = torch.from_numpy(x.astype(np.float32) / 255.0).unsqueeze(1)   # N,1,H,W
    return (1.0 - t).to(device, non_blocking=True)                     # ink = 1


# ---------------------------------------------------------------------------
# inference
# ---------------------------------------------------------------------------

def predict_mask(model, gray: np.ndarray, device, tile: int = 1024, overlap: int = 96,
                 batch: int = 8) -> np.ndarray:
    """Full-sheet argmax mask (0 bg, 1 solid, 2 dashed) by tiled inference."""
    import torch
    H, W = gray.shape
    stride = tile - overlap
    ys = sorted(set(list(range(0, max(1, H - tile), stride)) + [max(0, H - tile)]))
    xs = sorted(set(list(range(0, max(1, W - tile), stride)) + [max(0, W - tile)]))
    prob = np.zeros((3, H, W), np.float32)
    cnt = np.zeros((H, W), np.float32)
    coords = [(y, x) for y in ys for x in xs]
    pad_img = np.full((max(H, tile), max(W, tile)), 255, np.uint8)
    pad_img[:H, :W] = gray
    use_amp = device.type == "cuda"
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
        for k in range(0, len(coords), batch):
            chunk = coords[k:k + batch]
            xb = np.stack([pad_img[y:y + tile, x:x + tile] for y, x in chunk])
            x_t = _to_tensor(xb, device)
            if use_amp:
                x_t = x_t.contiguous(memory_format=torch.channels_last)
            p = model(x_t).float().softmax(1).cpu().numpy()
            for (y, x), pp in zip(chunk, p):
                h_, w_ = min(tile, H - y), min(tile, W - x)
                prob[:, y:y + h_, x:x + w_] += pp[:, :h_, :w_]
                cnt[y:y + h_, x:x + w_] += 1
    return (prob / np.maximum(cnt, 1)).argmax(0).astype(np.uint8)


# ---------------------------------------------------------------------------
# mask -> segments
# ---------------------------------------------------------------------------

def _seg_len(s): return float(np.hypot(s[2] - s[0], s[3] - s[1]))
def _ang(s): return float(np.degrees(np.arctan2(s[3] - s[1], s[2] - s[0])) % 180.0)
def _ang_diff(a, b):
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def _project(p, s):
    L = _seg_len(s)
    if L < 1e-6:
        return 0.0, float(np.hypot(p[0] - s[0], p[1] - s[1]))
    ux, uy = (s[2] - s[0]) / L, (s[3] - s[1]) / L
    dx, dy = p[0] - s[0], p[1] - s[1]
    return ux * dx + uy * dy, abs(-uy * dx + ux * dy)


def merge_collinear(segs: Sequence[Seg], ang_tol: float = 2.0, perp_tol: float = 4.0,
                    gap_tol: float = 8.0) -> List[Seg]:
    """Union of collinear segments that overlap or nearly abut (same as pnid_graph)."""
    segs = [tuple(int(v) for v in s) for s in segs if _seg_len(s) >= 1]
    n = len(segs)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        a = segs[i]
        for j in range(i + 1, n):
            b = segs[j]
            if _ang_diff(_ang(a), _ang(b)) > ang_tol:
                continue
            (ta, da), (tb, db) = _project(b[:2], a), _project(b[2:], a)
            if max(da, db) > perp_tol:
                continue
            lo, hi = min(ta, tb), max(ta, tb)
            if hi < -gap_tol or lo > _seg_len(a) + gap_tol:
                continue
            parent[find(i)] = find(j)
    groups: Dict[int, List[Seg]] = collections.defaultdict(list)
    for i in range(n):
        groups[find(i)].append(segs[i])
    out: List[Seg] = []
    for members in groups.values():
        if len(members) == 1:
            out.append(members[0])
            continue
        ref = max(members, key=_seg_len)
        ts = [_project(s[:2], ref)[0] for s in members] + [_project(s[2:], ref)[0] for s in members]
        L = _seg_len(ref)
        ux, uy = (ref[2] - ref[0]) / L, (ref[3] - ref[1]) / L
        out.append((int(round(ref[0] + ux * min(ts))), int(round(ref[1] + uy * min(ts))),
                    int(round(ref[0] + ux * max(ts))), int(round(ref[1] + uy * max(ts)))))
    return out


def mask_to_segments(mask: np.ndarray, cls: int, min_len: int = 100, hough_thr: int = 60,
                     max_gap: int = 8) -> List[Seg]:
    """Straight segments of one class. min_len / hough_thr chosen on the train split."""
    m = (mask == cls).astype(np.uint8) * 255
    if not m.any():
        return []
    lines = cv2.HoughLinesP(m, 1, np.pi / 180, hough_thr, minLineLength=min_len, maxLineGap=max_gap)
    if lines is None:
        return []
    return merge_collinear([tuple(int(v) for v in l) for l in lines.reshape(-1, 4)])


def detect_lines(img_bgr: np.ndarray, ckpt: str = DEFAULT_CKPT, model=None, device=None,
                 frame_mask: bool = True, min_len: int = 100) -> Dict:
    """Full line stage: {"solid", "dashed", "method": "unet"} in image pixels.

    frame_mask: zero the prediction outside the inner drawing frame (the
    classical stage's find_inner_frame_mask).  Synthetic sheets never needed it;
    on real scans the U-Net traces the frame ruling, so it is on by default.
    """
    if model is None:
        model, device = load_model(ckpt, device)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    mask = predict_mask(model, gray, device)
    if frame_mask:
        # Same two masks as the classical stage: inner drawing frame, and the
        # ruled notes / title-block column when one exists.  The U-Net otherwise
        # traces the frame ruling on real scans and the notes-column divider on
        # Dataset-P&ID (a 1300 px vertical false line at 0.79 W).
        try:
            from process_text_lines import find_inner_frame_mask, detect_notes_boundary
            keep = find_inner_frame_mask(gray, shrink_px=int(28 * gray.shape[1] / 4096))
            cap = 4096.0 / max(gray.shape)
            gcap = cv2.resize(gray, None, fx=cap, fy=cap, interpolation=cv2.INTER_AREA) if cap < 1 else gray
            b = detect_notes_boundary(gcap)
            if b is not None:
                keep[:, int((b - 6) / min(cap, 1.0)):] = 0
            mask = np.where(keep > 0, mask, 0).astype(np.uint8)
        except Exception:
            pass
    return {"solid": mask_to_segments(mask, 1, min_len), "dashed": mask_to_segments(mask, 2, min_len),
            "method": "unet", "ckpt": os.path.basename(ckpt)}


def main() -> None:
    ap = argparse.ArgumentParser(description="U-Net pipe-line extractor (default line stage)")
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True, help="JSON with solid/dashed segments")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--target-width", type=int, default=0, help="resize to this width first (0 = as is)")
    ap.add_argument("--no-frame-mask", action="store_true")
    a = ap.parse_args()
    img = cv2.imread(a.image)
    if a.target_width and img.shape[1] != a.target_width:
        s = a.target_width / img.shape[1]
        img = cv2.resize(img, (a.target_width, int(round(img.shape[0] * s))), interpolation=cv2.INTER_CUBIC)
    res = detect_lines(img, a.ckpt, frame_mask=not a.no_frame_mask)
    res["resized_shape"] = [int(img.shape[0]), int(img.shape[1])]
    with open(a.out, "w") as f:
        json.dump(res, f)
    print(f"unet lines: {len(res['solid'])} solid, {len(res['dashed'])} dashed -> {a.out}")


if __name__ == "__main__":
    main()
