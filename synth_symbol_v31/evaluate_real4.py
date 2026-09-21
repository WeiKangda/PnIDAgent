#!/usr/bin/env python3
"""Standard-protocol real4 evaluation without the gpt_detect/llm_baseline module (not in this repo).

Same protocol as evaluate.py: predict the whole drawing at imgsz=1280, conf=0.10,
iou=0.5, max_det=400; keep boxes with score >= 0.65; per-drawing F1 at IoU 0.5;
macro-average over the four drawings.  GT comes from
nuclear_pid_real4_dataset/labels/*.txt (single-class YOLO format).

    python PnIDAgent/synth_symbol_v31/evaluate_real4.py --weights best.pt
    python PnIDAgent/synth_symbol_v31/evaluate_real4.py --weights best.pt --tiled   # tiled 1280 inference instead
"""
import argparse, glob, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
D = os.path.join(REPO, "nuclear_pid_real4_dataset")


def load_gt(name, W, H):
    out = []
    for line in open(os.path.join(D, "labels", name + ".txt")):
        p = line.split()
        if len(p) == 5:
            cx, cy, w, h = (float(v) for v in p[1:])
            out.append([(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H])
    return np.array(out, float).reshape(-1, 4)


def f1_at(pred, gt, thr=0.5):
    if len(pred) == 0 or len(gt) == 0:
        return 0.0
    ix1 = np.maximum(pred[:, None, 0], gt[None, :, 0]); iy1 = np.maximum(pred[:, None, 1], gt[None, :, 1])
    ix2 = np.minimum(pred[:, None, 2], gt[None, :, 2]); iy2 = np.minimum(pred[:, None, 3], gt[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    ap = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1]); ag = (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])
    iou = inter / (ap[:, None] + ag[None, :] - inter + 1e-9)
    used = np.zeros(len(gt), bool); tp = 0
    for i in range(len(pred)):                     # greedy, as in a standard F1 matcher
        cand = np.where((iou[i] >= thr) & ~used)[0]
        if len(cand):
            used[cand[np.argmax(iou[i][cand])]] = True; tp += 1
    P = tp / len(pred); R = tp / len(gt)
    return 2 * P * R / (P + R) if P + R else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--keep", type=float, default=0.65)
    ap.add_argument("--device", default="0")
    ap.add_argument("--tiled", action="store_true", help="tiled native-resolution inference (finetune_yolo_symbols.detect)")
    a = ap.parse_args()
    import cv2
    from ultralytics import YOLO
    names = sorted(os.path.splitext(os.path.basename(f))[0] for f in glob.glob(os.path.join(D, "images", "*.jpg")))
    per = []
    if a.tiled:
        sys.path.insert(0, REPO)
        from finetune_yolo_symbols import YOLOSymbolDetector
        det = YOLOSymbolDetector(model_path=a.weights)
    else:
        m = YOLO(a.weights)
    for n in names:
        ip = os.path.join(D, "images", n + ".jpg"); img = cv2.imread(ip); H, W = img.shape[:2]
        gt = load_gt(n, W, H)
        if a.tiled:
            ds = det.detect(img, conf_threshold=a.conf, iou_threshold=0.5, imgsz=a.imgsz, agnostic_nms=True)
            pred = np.array([d.bbox for d in ds if d.confidence >= a.keep], float).reshape(-1, 4)
        else:
            r = m.predict(ip, imgsz=a.imgsz, conf=a.conf, iou=0.5, verbose=False, max_det=400, device=a.device)[0]
            b = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0, 4))
            c = r.boxes.conf.cpu().numpy() if r.boxes is not None else np.zeros(0)
            pred = b[c >= a.keep]
        per.append((n, f1_at(pred, gt), len(pred), len(gt)))
    macro = float(np.mean([f for _, f, _, _ in per]))
    print(f"\n=== standard protocol (imgsz {a.imgsz}, conf {a.conf}, keep >= {a.keep}, IoU 0.5{', tiled' if a.tiled else ''}) ===")
    for n, f, npred, ngt in per:
        print(f"  {n:14s} F1 = {f:.3f}   (pred {npred}, gt {ngt})")
    print(f"  >> macro-F1 = {macro:.4f}")
    print("  reference (Xinqi): v31 0.883 (0.915/0.931/0.873/0.813); release soup5 0.888; rebuild ~0.853; supervised baseline 0.821")


if __name__ == "__main__":
    main()
