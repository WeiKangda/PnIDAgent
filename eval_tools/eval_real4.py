import os, sys as _sys; _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import ROOT as _ROOT
#!/usr/bin/env python3
"""Symbol detection on the real4 set (PnIDAgent/nuclear_pid_real4_dataset).

GT is single-class YOLO txt, so everything is class-agnostic.  Runs the tiled
detector at the sheet's native resolution and resized to 7168 wide (what the
pipeline does), and reports COCO AP + P/R at several confidences.

    python tools/eval_real4.py --model PnIDAgent/runs/results/yolo_runs/y11m_1280/weights/best.pt
"""
import argparse, glob, json, os, sys
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_symbols as ES
REPO = os.path.join(_ROOT, "PnIDAgent")
sys.path.insert(0, REPO)
D = os.path.join(REPO, "nuclear_pid_real4_dataset")

def load_gt(name, W, H):
    out = []
    for line in open(os.path.join(D, "labels", name + ".txt")):
        p = line.split()
        if len(p) != 5:
            continue
        cx, cy, w, h = (float(v) for v in p[1:])
        out.append((0, (int((cx - w / 2) * W), int((cy - h / 2) * H), int((cx + w / 2) * W), int((cy + h / 2) * H))))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from finetune_yolo_symbols import YOLOSymbolDetector
    from process_text_lines import resize_keep_aspect
    det = YOLOSymbolDetector(model_path=a.model)
    names = sorted(os.path.splitext(os.path.basename(f))[0] for f in glob.glob(os.path.join(D, "images", "*.jpg")))
    res = {}
    for mode in ("native", "7168"):
        gt, pred = {}, {}
        for i, n in enumerate(names):
            img = cv2.imread(os.path.join(D, "images", n + ".jpg")); H, W = img.shape[:2]
            g = load_gt(n, W, H)
            if mode == "7168":
                img, s = resize_keep_aspect(img)
                g = [(c, tuple(int(v * s) for v in b)) for c, b in g]
            dets = det.detect(img, conf_threshold=a.conf, agnostic_nms=True)
            gt[i] = g; pred[i] = [(0, d.bbox, d.confidence) for d in dets]
        ev = ES.evaluate(gt, pred, agnostic=True)
        row = {"mAP50": ev["map50"], "mAP50_95": ev["map50_95"], "n_gt": sum(len(v) for v in gt.values())}
        for c in (0.05, 0.1, 0.25, 0.5):
            r = ES.prf_at(gt, pred, c, agnostic=True); row[f"P@{c}"] = r["precision"]; row[f"R@{c}"] = r["recall"]; row[f"F1@{c}"] = r["f1"]
        per = []
        for i, n in enumerate(names):
            r = ES.prf_at({i: gt[i]}, {i: pred[i]}, 0.25, agnostic=True); per.append((n, len(gt[i]), r["precision"], r["recall"]))
        res[mode] = row
        print(f"[{mode}] GT boxes {row['n_gt']}  mAP50 {row['mAP50']:.3f}  mAP50-95 {row['mAP50_95']:.3f}  " +
              "  ".join(f"conf{c}: P {row[f'P@{c}']:.3f} R {row[f'R@{c}']:.3f}" for c in (0.05, 0.1, 0.25, 0.5)))
        for n, k, p, r in per:
            print(f"     {n:14} gt {k:3d}  P@0.25 {p:.3f}  R@0.25 {r:.3f}")
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)

if __name__ == "__main__":
    main()
