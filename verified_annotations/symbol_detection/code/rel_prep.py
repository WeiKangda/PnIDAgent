"""Data preparation for the page relational model: three element (token) types per page
  candidates: detection boxes of 7 views (soup5 x5, large equipment big4, tiling tile4; all cross-fitted on the A side) merged into a candidate pool; each stores a 64x64 crop + per-view scores + box
  segments: horizontal/vertical segments auto-extracted from the image (structure.segments); A pages also store an "is a human-verified pipe" label (auxiliary task)
  text: boxes + character counts of raw OCR results (score>=0.5)
Labels (used only for the loss): one-to-one positive candidates + regression targets (matched annotation boxes).
Output: results/rel_data/<page>.npz"""
import glob, json, os, sys
import cv2, numpy as np
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from selfsim import SRC, CACHE, sp, gt_of
import rescore_pool as rp
from rescore_pool import build_pool, iou_mat
from structure import segments
from crop_cnn import crop
OUT = f"{HERE}/results/rel_data"; os.makedirs(OUT, exist_ok=True)
RAW = "/home/suxinqi666/code/PnIDAgent_project/ft_verified/text_ocr/raw"; LV = "/home/suxinqi666/code/PnIDAgent_project/line_results/lines_verified"
EXTRA = ["big4", "tile4"]; rp.VIEWS.extend(EXTRA)
D = json.load(open(f"{CACHE}/soup5.json")); T = json.load(open(f"{CACHE}/soup5_tta.json")); X = {v: json.load(open(f"{CACHE}/{v}.json")) for v in EXTRA}


def one2one(B, G):
    y = np.zeros(len(B), np.float32); tgt = np.zeros((len(B), 4), np.float32); has = np.zeros(len(B), np.float32)
    if not len(B) or not len(G): return y, tgt, has
    io = iou_mat(B, G)[0]
    for gi in np.argsort(-io.max(0)):
        j = int(np.argmax(np.where(y == 1, -1, io[:, gi])))
        if io[j, gi] >= 0.5: y[j] = 1
    j = io.argmax(1); ok = io.max(1) >= 0.3; tgt[ok] = G[j[ok]]; has[ok] = 1     # regression target: nearest annotation box with IoU>=0.3
    return y, tgt, has


def work(n):
    g = cv2.imread(glob.glob(f"{SRC}/{glob.escape(n)}.*")[0], 0); H, W = g.shape; sc = 3300 / max(H, W)
    B, V = build_pool({"base": D[n], **T[n], **{v: X[v][n] for v in EXTRA}})
    G = np.array(gt_of(n, W, H), float).reshape(-1, 4)
    crops = np.array([(crop(g, b)[0] * 255).astype(np.uint8) for b in B]) if len(B) else np.zeros((0, 64, 64), np.uint8)
    y, tgt, has = one2one(B, G)
    Hs, Vs, _ = segments(g, sc)
    L = np.concatenate([np.column_stack([Hs[:, 0], Hs[:, 2] - Hs[:, 3] / 2, Hs[:, 1], Hs[:, 2] + Hs[:, 3] / 2, np.zeros(len(Hs))]),
                        np.column_stack([Vs[:, 2] - Vs[:, 3] / 2, Vs[:, 0], Vs[:, 2] + Vs[:, 3] / 2, Vs[:, 1], np.ones(len(Vs))])]) if len(Hs) + len(Vs) else np.zeros((0, 5))
    ly = np.full(len(L), -1, np.float32)                                    # -1 = no pipe annotation (B pages)
    if os.path.exists(f"{LV}/{n}.json") and n in sp["A_train"]:
        mg = np.zeros_like(g, np.uint8)
        for x0, y0, x1, y1 in json.load(open(f"{LV}/{n}.json"))["solid"]: cv2.line(mg, (int(x0), int(y0)), (int(x1), int(y1)), 1, max(5, int(8 / sc)))
        ly = np.array([mg[int(max(0, (b + d) / 2 - 1)):int((b + d) / 2 + 2), int(a):int(c) + 1].mean() if t == 0 else
                       mg[int(b):int(d) + 1, int(max(0, (a + c) / 2 - 1)):int((a + c) / 2 + 2)].mean() for a, b, c, d, t in L], np.float32)
        ly = (ly >= 0.5).astype(np.float32)
    tx = [t["box"] + [sum(ch.isalnum() for ch in t["text"]), sum(ch.isdigit() for ch in t["text"])] for t in json.load(open(f"{RAW}/{n}.json")) if t["score"] >= 0.5]
    TX = np.array(tx, float).reshape(-1, 6)
    np.savez_compressed(f"{OUT}/{n}.npz", B=np.asarray(B, float).reshape(-1, 4), V=V, crops=crops, y=y, tgt=tgt, has=has, L=L, ly=ly, TX=TX, G=G, WH=np.array([W, H]))
    return n, len(B), int(y.sum()), len(G), len(L), int((ly == 1).sum()), len(TX)


if __name__ == "__main__":
    with Pool(20) as p: res = p.map(work, sp["A_train"] + sp["B_test"])
    a = np.array([r[1:] for r in res]); print("Totals candidates/positives/annotations/segments/pipe segments/text:", a.sum(0), "| max tokens per page:", (a[:, 0] + a[:, 3] + a[:, 5]).max())
