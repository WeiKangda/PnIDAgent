#!/usr/bin/env python3
"""Learned pipe-centreline extractor: a small U-Net over native-resolution tiles.

The classical path tops out at Hough recall 0.852 (max_dim 4096) because a 1px
pipe survives the downscale only as a faint anti-aliased trace, and symbols and
text on the pipe break Canny continuity.  This trains a 3-class segmenter
(background / solid pipe / dashed pipe) on GT lines rasterised at 3px, at full
resolution, then turns the predicted mask into straight segments with Hough +
collinear merge so tools/eval_lines.py scores it on the same metric.

Targets are the GT `lines` only, so frame rulings, title-block tables, text and
symbol outlines are negatives: the model learns the notes crop and the symbol
suppression that the classical stage does with masks and heuristics.

    python tools/line_seg.py train   --epochs 20 --gpu 0
    python tools/line_seg.py predict --ckpt results/line_seg/best.pt --ids results/split.json:val \
                                     --out results/line_seg/pred_val.json
    python tools/eval_lines.py --pred-json results/line_seg/pred_val.json --ids results/split.json:val
"""
from __future__ import annotations

import argparse
import json
import os
import sys as _sys; _sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from _paths import ROOT as _ROOT  # <ROOT>/PnIDAgent/eval_tools layout
import random
import sys
import time
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dpid  # noqa: E402

ROOT = os.path.join(_ROOT, "dataset")
Seg = Tuple[int, int, int, int]


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def rasterize(sheet: dpid.Sheet, shape=(4561, 7168), thick: int = 3) -> np.ndarray:
    m = np.zeros(shape, np.uint8)
    for l in sheet.lines:                       # dashed drawn last so it wins ties
        if l.kind == "solid":
            cv2.line(m, l.seg[:2], l.seg[2:], 1, thick)
    for l in sheet.lines:
        if l.kind == "dashed":
            cv2.line(m, l.seg[:2], l.seg[2:], 2, thick)
    return m


def load_ids(spec: str) -> List[int]:
    if ":" in spec and spec.rsplit(":", 1)[0].endswith(".json"):
        p, k = spec.rsplit(":", 1)
        return list(json.load(open(p))[k])
    return [int(x) for x in open(spec).read().split()]


class TileDataset:
    """Random `size` crops from cached grayscale sheets and their masks."""

    def __init__(self, root: str, ids: Sequence[int], size: int = 768, ink_bias: float = 0.8):
        self.size, self.ink_bias = size, ink_bias
        self.imgs, self.masks = [], []
        for i in ids:
            g = cv2.imread(os.path.join(root, "image_2", f"{i}.jpg"), cv2.IMREAD_GRAYSCALE)
            self.imgs.append(g)
            self.masks.append(rasterize(dpid.load_sheet(root, i), g.shape))

    def sample(self, rng: random.Random) -> Tuple[np.ndarray, np.ndarray]:
        k = rng.randrange(len(self.imgs))
        g, m = self.imgs[k], self.masks[k]
        H, W = g.shape
        s = self.size
        for _ in range(8):
            y, x = rng.randrange(0, H - s), rng.randrange(0, W - s)
            mm = m[y:y + s, x:x + s]
            if rng.random() > self.ink_bias or mm.any():
                break
        return g[y:y + s, x:x + s], m[y:y + s, x:x + s]

    def batch(self, n: int, rng: random.Random):
        xs, ys = zip(*(self.sample(rng) for _ in range(n)))
        return np.stack(xs), np.stack(ys)


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


def _to_tensor(x: np.ndarray, device):
    import torch
    t = torch.from_numpy(x.astype(np.float32) / 255.0).unsqueeze(1)   # N,1,H,W
    return (1.0 - t).to(device, non_blocking=True)                     # ink = 1


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------

def train(args) -> None:
    import torch
    import torch.nn.functional as F
    device = torch.device(f"cuda:{args.gpu}")
    sp = json.load(open(args.split))
    tr_ids = [i for i in sp["train"] if dpid.is_complete(args.root, i)]
    va_ids = [i for i in sp["val"] if dpid.is_complete(args.root, i)][:args.val_sheets]
    if args.train_sheets:
        tr_ids = tr_ids[:args.train_sheets]
    print(f"train sheets {len(tr_ids)}  val sheets {len(va_ids)}", flush=True)
    t0 = time.time()
    ds = TileDataset(args.root, tr_ids, size=args.tile)
    vds = TileDataset(args.root, va_ids, size=args.tile)
    print(f"cached in {time.time() - t0:.0f}s", flush=True)

    model = build_model(args.base).to(device).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * args.steps_per_epoch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler()
    # thin classes: weight them up, and add a soft dice term
    cw = torch.tensor([1.0, 8.0, 8.0], device=device)
    rng = random.Random(0)
    os.makedirs(args.out, exist_ok=True)
    best = -1.0

    def loss_fn(logits, y):
        ce = F.cross_entropy(logits, y, weight=cw)
        p = logits.softmax(1)[:, 1:]
        oh = F.one_hot(y, 3).permute(0, 3, 1, 2)[:, 1:].float()
        inter = (p * oh).sum((0, 2, 3))
        dice = 1 - (2 * inter + 1) / (p.sum((0, 2, 3)) + oh.sum((0, 2, 3)) + 1)
        return ce + dice.mean()

    def validate(n_batches=24) -> Dict[str, float]:
        model.eval()
        vr = random.Random(1)
        tp = np.zeros(3); fp = np.zeros(3); fn = np.zeros(3)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for _ in range(n_batches):
                xb, yb = vds.batch(args.batch, vr)
                pred = model(_to_tensor(xb, device).contiguous(memory_format=torch.channels_last)).argmax(1).cpu().numpy()
                for c in (1, 2):
                    tp[c] += ((pred == c) & (yb == c)).sum()
                    fp[c] += ((pred == c) & (yb != c)).sum()
                    fn[c] += ((pred != c) & (yb == c)).sum()
        model.train()
        f1 = {c: 2 * tp[c] / max(1, 2 * tp[c] + fp[c] + fn[c]) for c in (1, 2)}
        return {"f1_solid": float(f1[1]), "f1_dashed": float(f1[2])}

    step = 0
    for ep in range(args.epochs):
        t0, run = time.time(), 0.0
        for _ in range(args.steps_per_epoch):
            xb, yb = ds.batch(args.batch, rng)
            x = _to_tensor(xb, device).contiguous(memory_format=torch.channels_last)
            y = torch.from_numpy(yb.astype(np.int64)).to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = loss_fn(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            run += loss.item()
            step += 1
        v = validate()
        score = (v["f1_solid"] + v["f1_dashed"]) / 2
        print(f"epoch {ep + 1}/{args.epochs}  loss {run / args.steps_per_epoch:.4f}  "
              f"val pixel-F1 solid {v['f1_solid']:.3f} dashed {v['f1_dashed']:.3f}  "
              f"{time.time() - t0:.0f}s", flush=True)
        torch.save({"model": model.state_dict(), "base": args.base, "epoch": ep + 1, "val": v},
                   os.path.join(args.out, "last.pt"))
        if score > best:
            best = score
            torch.save({"model": model.state_dict(), "base": args.base, "epoch": ep + 1, "val": v},
                       os.path.join(args.out, "best.pt"))
    print(f"best val pixel-F1 (mean) {best:.3f}")


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------

def predict_mask(model, gray: np.ndarray, device, tile: int = 1024, overlap: int = 96,
                 batch: int = 8) -> np.ndarray:
    """Full-sheet argmax mask by tiled inference (centre crops of overlapping tiles)."""
    import torch
    H, W = gray.shape
    stride = tile - overlap
    ys = list(range(0, max(1, H - tile), stride)) + [max(0, H - tile)]
    xs = list(range(0, max(1, W - tile), stride)) + [max(0, W - tile)]
    ys, xs = sorted(set(ys)), sorted(set(xs))
    prob = np.zeros((3, H, W), np.float32)
    cnt = np.zeros((H, W), np.float32)
    coords = [(y, x) for y in ys for x in xs]
    pad_img = np.full((max(H, tile), max(W, tile)), 255, np.uint8)
    pad_img[:H, :W] = gray
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for k in range(0, len(coords), batch):
            chunk = coords[k:k + batch]
            xb = np.stack([pad_img[y:y + tile, x:x + tile] for y, x in chunk])
            p = model(_to_tensor(xb, device).contiguous(memory_format=torch.channels_last)).float().softmax(1).cpu().numpy()
            for (y, x), pp in zip(chunk, p):
                h_, w_ = min(tile, H - y), min(tile, W - x)
                prob[:, y:y + h_, x:x + w_] += pp[:, :h_, :w_]
                cnt[y:y + h_, x:x + w_] += 1
    return (prob / np.maximum(cnt, 1)).argmax(0).astype(np.uint8)


def mask_to_segments(mask: np.ndarray, cls: int, min_len: int = 100, hough_thr: int = 60,
                     max_gap: int = 8) -> List[Seg]:
    import graph_assembly as GA
    m = (mask == cls).astype(np.uint8) * 255
    if not m.any():
        return []
    lines = cv2.HoughLinesP(m, 1, np.pi / 180, hough_thr, minLineLength=min_len, maxLineGap=max_gap)
    if lines is None:
        return []
    segs = [tuple(int(v) for v in l) for l in lines.reshape(-1, 4)]
    return GA.merge_collinear(segs, ang_tol=2.0, perp_tol=4.0, gap_tol=max_gap)


def predict(args) -> None:
    import torch
    device = torch.device(f"cuda:{args.gpu}")
    ck = torch.load(args.ckpt, map_location="cpu")
    model = build_model(ck["base"]).to(device).to(memory_format=torch.channels_last)
    model.load_state_dict(ck["model"])
    model.eval()
    ids = load_ids(args.ids) if args.ids else dpid.sheet_ids(args.root)
    ids = [i for i in ids if os.path.exists(os.path.join(args.root, "image_2", f"{i}.jpg"))]
    out: Dict[str, Dict[str, List[Seg]]] = {}
    t0 = time.time()
    for n, i in enumerate(ids, 1):
        g = cv2.imread(os.path.join(args.root, "image_2", f"{i}.jpg"), cv2.IMREAD_GRAYSCALE)
        m = predict_mask(model, g, device, tile=args.tile)
        out[str(i)] = {"solid": mask_to_segments(m, 1, args.min_len),
                       "dashed": mask_to_segments(m, 2, args.min_len)}
        if n % 10 == 0 or n == len(ids):
            print(f"  [{n}/{len(ids)}] {(time.time() - t0) / n:.1f}s/sheet", flush=True)
        if args.save_masks:
            os.makedirs(args.save_masks, exist_ok=True)
            cv2.imwrite(os.path.join(args.save_masks, f"{i}.png"), m * 100)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh)
    print(f"wrote {args.out} ({len(out)} sheets)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("train", "predict"))
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--split", default=os.path.join(os.path.dirname(ROOT), "results", "split.json"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(ROOT), "results", "line_seg"))
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--tile", type=int, default=768)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--steps-per-epoch", type=int, default=400)
    ap.add_argument("--train-sheets", type=int, default=0)
    ap.add_argument("--val-sheets", type=int, default=10)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--ids", default=None)
    ap.add_argument("--min-len", type=int, default=100)
    ap.add_argument("--save-masks", default=None)
    args = ap.parse_args()
    if args.mode == "train":
        train(args)
    else:
        if args.mode == "predict" and args.out.endswith("line_seg"):
            args.out = os.path.join(args.out, "pred.json")
        predict(args)


if __name__ == "__main__":
    main()
