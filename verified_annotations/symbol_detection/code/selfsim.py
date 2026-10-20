"""Within-page self-similarity recovery (training-free algorithmic improvement).
The same symbol repeats many times on a P&ID page (e.g. dozens of breakers in a single-line diagram). The detector is confident on some, and scores others low or misses them.
Method:
  1. Detector outputs boxes at a low threshold (0.05).
  2. High-scoring boxes (>= SEED) serve as templates; cluster by appearance, keep a few representatives per cluster.
  3. Run normalized cross-correlation of templates over the whole page (same scale, plus 90/180/270 degree rotations); where correlation >= MATCH:
     - if a low-score detection overlaps it -> "rescue" that low-score box;
     - if there is no box -> add a new box (size = template size).
Parameters are chosen on the A half only and reported on the B half.
  python selfsim.py cache <gpu> <name=weights>...     # cache detection results first
  python selfsim.py tune  <name>                      # tune on A, report on B
"""
import glob, json, os, sys, itertools, pickle
import cv2, numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
R = "/home/suxinqi666/code/PnIDAgent_project"; HERE = os.path.dirname(os.path.abspath(__file__))
LAB = f"{R}/line_results/labels_verified"; SRC = f"{R}/datasets/yolo_pseudo_v2/images/train"
sp = json.load(open(f"{HERE}/split.json")); PAGES = sp["A_train"] + sp["B_test"]
CACHE = f"{HERE}/results/selfsim_cache"; os.makedirs(CACHE, exist_ok=True)
adaptive = lambda l: max(1280, min(2048, int(round(1280 * l / 3300 / 32)) * 32))


def iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    if x1 <= x0 or y1 <= y0: return 0.0
    i = (x1 - x0) * (y1 - y0); return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)


def f1(pred, gt):
    used, tp = set(), 0
    for g in gt:
        best = max([(iou(g, p), j) for j, p in enumerate(pred) if j not in used], default=(0, -1))
        if best[0] >= 0.5: used.add(best[1]); tp += 1
    return 2 * tp / max(1, 2 * tp + (len(pred) - tp) + (len(gt) - tp))


def gt_of(n, W, H):
    return [[(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H] for cx, cy, w, h in
            (list(map(float, l.split()[1:5])) for l in open(f"{LAB}/{n}.txt") if l.strip())]


def cache(dev, models):
    from ultralytics import YOLO
    for mn, wp in models.items():
        m = YOLO(wp); out = {}
        for n in PAGES:
            ip = glob.glob(f"{SRC}/{glob.escape(n)}.*")[0]; W, H = Image.open(ip).size
            r = m.predict(ip, imgsz=adaptive(max(W, H)), conf=0.05, iou=0.5, max_det=2000, verbose=False, device=dev)[0]
            out[n] = [b.tolist() + [float(c)] for b, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy())]
        json.dump(out, open(f"{CACHE}/{mn}.json", "w")); print("cached", mn, flush=True)


def ink_img(n, L=25):
    """Ink image with long straight lines (pipes/buses) removed, keeping only symbol strokes, so templates cannot match by sliding along pipes."""
    g = cv2.imread(glob.glob(f"{SRC}/{glob.escape(n)}.*")[0], 0)
    ink = (g < 160).astype(np.uint8)
    lines = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((1, L), np.uint8)) | cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((L, 1), np.uint8))
    return (ink & (1 - cv2.dilate(lines, np.ones((3, 3), np.uint8)))).astype(np.float32)


def propagate(n, dets, seed=0.6, match=0.75, keep=0.5, maxt=24, ink=None):
    """Return the final box list. dets: [x0,y0,x1,y1,score]."""
    ink = ink_img(n) if ink is None else ink; H, W = ink.shape
    final = [d for d in dets if d[4] >= keep]
    seeds = sorted([d for d in dets if d[4] >= seed], key=lambda d: -d[4])
    if not seeds: return [d[:4] for d in final]
    # templates: de-duplicate by size + appearance, at most maxt
    temps = []
    for d in seeds:
        x0, y0, x1, y1 = [int(round(v)) for v in d[:4]]
        if x1 - x0 < 10 or y1 - y0 < 10 or (x1 - x0) * (y1 - y0) > 250 * 250: continue
        t = ink[y0:y1, x0:x1]
        if t.mean() < 0.04: continue
        dup = False
        for T in temps:
            if abs(T.shape[0] - t.shape[0]) <= 3 and abs(T.shape[1] - t.shape[1]) <= 3:
                tt = cv2.resize(t, (T.shape[1], T.shape[0]))
                if cv2.matchTemplate(tt, T, cv2.TM_CCOEFF_NORMED).max() > 0.85: dup = True; break
        if not dup: temps.append(t)
        if len(temps) >= maxt: break
    low = [d for d in dets if d[4] < keep]
    added = []
    for T in temps:
        for k in range(4):
            Tr = np.rot90(T, k).copy() if k else T
            th, tw = Tr.shape
            if th >= H or tw >= W: continue
            res = cv2.matchTemplate(ink, Tr, cv2.TM_CCOEFF_NORMED)
            ys, xs = np.where(res >= match)
            if len(xs) == 0: continue
            order = np.argsort(-res[ys, xs])[:400]
            for i in order:
                x, y = int(xs[i]), int(ys[i]); box = [x, y, x + tw, y + th]
                if any(iou(box, f) > 0.3 for f in final) or any(iou(box, a) > 0.3 for a in added): continue
                # prefer rescuing an overlapping low-score detection (its position/size is more reliable)
                cand = [a for a in low if iou(box, a) > 0.5]
                added.append(max(cand, key=lambda a: a[4])[:4] if cand else box)
    return [d[:4] for d in final] + added


_G = {}


def _init(mn):
    _G["dets"] = json.load(open(f"{CACHE}/{mn}.json"))


def _page(args):
    n, kw = args
    ink = ink_img(n); H, W = ink.shape
    return f1(propagate(n, _G["dets"][n], ink=ink, **kw), gt_of(n, W, H))


def tune(mn, workers=24):
    from multiprocessing import Pool
    pool = Pool(workers, initializer=_init, initargs=(mn,)); _init(mn)
    def score(part, **kw): return float(np.mean(pool.map(_page, [(n, kw) for n in sp[part]])))
    dets = _G["dets"]
    base = {}
    for p in ("A_train", "B_test"):
        v = []
        for n in sp[p]:
            H, W = cv2.imread(glob.glob(f"{SRC}/{glob.escape(n)}.*")[0], 0).shape
            v.append(f1([d[:4] for d in dets[n] if d[4] >= 0.5], gt_of(n, W, H)))
        base[p] = float(np.mean(v))
    print(f"{mn} baseline keep0.5: A {base['A_train']:.4f} B {base['B_test']:.4f}", flush=True)
    rows = []
    for seed, match, keep in itertools.product([0.4, 0.5, 0.6, 0.7], [0.7, 0.75, 0.8, 0.85], [0.5]):
        a = score("A_train", seed=seed, match=match, keep=keep); rows.append((a, seed, match, keep))
        print(f"  seed {seed} match {match} keep {keep}: A {a:.4f}", flush=True)
    best = max(rows)
    b = score("B_test", seed=best[1], match=best[2], keep=best[3])
    print(f"{mn} selected on A: seed {best[1]} match {best[2]} keep {best[3]} -> A {best[0]:.4f} B {b:.4f} (baseline A {base['A_train']:.4f} B {base['B_test']:.4f})", flush=True)
    json.dump({"base": base, "rows": rows, "best": best, "B": b}, open(f"{HERE}/results/selfsim_{mn}.json", "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "cache": cache(int(sys.argv[2]), dict(a.split("=", 1) for a in sys.argv[3:]))
    else: tune(sys.argv[2])
