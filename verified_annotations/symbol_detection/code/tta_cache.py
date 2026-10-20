"""多视角检测缓存 (给二次打分当一致性特征): 1.25x / 0.8x 分辨率, 左右翻转, 上下翻转. conf 0.05.
  python tta_cache.py <gpu> <name> <weights>"""
import glob, json, os, sys
import cv2, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from selfsim import SRC, CACHE, PAGES, adaptive
from ultralytics import YOLO
dev, mn, wp = int(sys.argv[1]), sys.argv[2], sys.argv[3]
m = YOLO(wp); out = {}
for n in PAGES:
    im = cv2.imread(glob.glob(f"{SRC}/{glob.escape(n)}.*")[0]); H, W = im.shape[:2]; base = adaptive(max(W, H)); out[n] = {}
    for tag, s, fl in (("s1.25", 1.25, None), ("s0.8", 0.8, None), ("flipx", 1.0, 1), ("flipy", 1.0, 0)):
        x = cv2.flip(im, fl) if fl is not None else im
        r = m.predict(x, imgsz=int(round(base * s / 32)) * 32, conf=0.05, iou=0.5, max_det=2000, verbose=False, device=dev)[0]
        bx = r.boxes.xyxy.cpu().numpy()
        if fl == 1: bx[:, [0, 2]] = W - bx[:, [2, 0]]
        if fl == 0: bx[:, [1, 3]] = H - bx[:, [3, 1]]
        out[n][tag] = [b.tolist() + [float(c)] for b, c in zip(bx, r.boxes.conf.cpu().numpy())]
    print(n, {k: len(v) for k, v in out[n].items()}, flush=True)
json.dump(out, open(f"{CACHE}/{mn}_tta.json", "w")); print("TTA_DONE")
