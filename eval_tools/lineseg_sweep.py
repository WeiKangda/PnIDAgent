import os, sys as _sys; _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import ROOT as _ROOT
#!/usr/bin/env python3
"""Sweep mask->segment post-processing on saved segmenter masks (no GPU).

    python tools/lineseg_sweep.py --masks results/line_seg/masks_dev --ids results/dev_train_ids.txt
"""
import argparse, json, os, sys
from concurrent.futures import ProcessPoolExecutor
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dpid, eval_lines as EL, line_seg as LS, graph_assembly as GA, eval_graph as EG

def job(a):
    root, masks, idx, min_len, thr, gap, tol = a
    m = cv2.imread(os.path.join(masks, f"{idx}.png"), cv2.IMREAD_UNCHANGED) // 100
    segs = {"solid": LS.mask_to_segments(m, 1, min_len, thr, gap),
            "dashed": LS.mask_to_segments(m, 2, min_len, thr, gap)}
    sheet = dpid.load_sheet(root, idx)
    pred = segs["solid"] + segs["dashed"]
    gt = [l.seg for l in sheet.lines]
    r = EL.count_scores(gt, pred, 8)
    g = dpid.derive_graph(sheet)
    pg = GA.assemble([(s, "solid") for s in segs["solid"]] + [(s, "dashed") for s in segs["dashed"]],
                     [(s.id, s.box) for s in sheet.symbols], tol=tol, pad=6)
    e = EG.score(g, pg, {s.id: s.box for s in sheet.symbols}, {s.id: s.box for s in sheet.symbols})
    return r["count_precision"], r["count_recall"], r["count_f1"], e["edge_p"], e["edge_r"], e["edge_f1"], r["n_pred"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(_ROOT, "dataset"))
    ap.add_argument("--masks", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--tol", type=int, default=12)
    ap.add_argument("--grid", default="30:40:8,60:40:8,100:40:8,150:40:8,60:60:8,100:60:8,60:40:16,100:40:16")
    a = ap.parse_args()
    ids = [int(x) for x in open(a.ids).read().split()]
    print(f"{'min_len':>7} {'thr':>4} {'gap':>4} | {'lineP':>6} {'lineR':>6} {'lineF1':>6} | {'edgeP':>6} {'edgeR':>6} {'edgeF1':>6} | pred/sheet   (sheets={len(ids)})")
    for g in a.grid.split(","):
        ml, thr, gap = (int(v) for v in g.split(":"))
        with ProcessPoolExecutor(min(48, len(ids))) as ex:
            rows = list(ex.map(job, [(a.root, a.masks, i, ml, thr, gap, a.tol) for i in ids]))
        m = np.mean(rows, axis=0)
        print(f"{ml:7d} {thr:4d} {gap:4d} | {m[0]:6.3f} {m[1]:6.3f} {m[2]:6.3f} | {m[3]:6.3f} {m[4]:6.3f} {m[5]:6.3f} | {m[6]:6.0f}", flush=True)

if __name__ == "__main__":
    main()
