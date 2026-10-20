"""Tiled inference: 640 tiles at native resolution, stride 480; boxes touching an inner tile edge (not a page edge) are dropped (the symbol is cut, it is whole in a neighboring tile);
NMS after merging (IoU 0.5 or >=80% covered by a higher-scoring box). Cached to selfsim_cache/<name>.json.
  python tile_predict.py <gpu> <name> <weights> <pages: A1|A2|A|B|all> [rot]   (rot: also merge in 90-degree rotated inference)"""
import glob, json, os, sys
import cv2, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from selfsim import SRC, CACHE, sp
from rescore_pool import iou_mat
T, STEP, EDGE = 640, 480, 4


def origins(L):
    o = list(range(0, max(1, L - T), STEP))
    if o[-1] + T < L: o.append(L - T)
    return o


def merge(dets, thr=0.5, ct=0.8):
    if not dets: return []
    d = np.array(sorted(dets, key=lambda x: -x[4])); io, cov = iou_mat(d[:, :4], d[:, :4]); keep = []
    for i in range(len(d)):
        if all(io[i, k] < thr and cov[i, k] < ct for k in keep): keep.append(i)
    return d[keep].tolist()


def predict_page(m, path, dev, rot=False):
    im = cv2.imread(path); H, W = im.shape[:2]; tiles, org = [], []
    for y0 in origins(H):
        for x0 in origins(W):
            t = im[y0:y0 + T, x0:x0 + T]
            if t.shape[0] < T or t.shape[1] < T: t = cv2.copyMakeBorder(t, 0, T - t.shape[0], 0, T - t.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255))
            tiles.append(t); org.append((x0, y0, 0))
            if rot: tiles.append(np.ascontiguousarray(np.rot90(t))); org.append((x0, y0, 1))
    out = []
    for i in range(0, len(tiles), 16):
        rs = m.predict(tiles[i:i + 16], imgsz=T, conf=0.05, iou=0.5, max_det=300, verbose=False, device=dev)
        for r, (x0, y0, k) in zip(rs, org[i:i + 16]):
            for b, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                if k: b = np.array([T - b[3], b[0], T - b[1], b[2]])          # rotate back: inverse of counterclockwise rot90
                if (b[0] < EDGE and x0 > 0) or (b[1] < EDGE and y0 > 0) or (b[2] > T - EDGE and x0 + T < W) or (b[3] > T - EDGE and y0 + T < H): continue
                out.append([b[0] + x0, b[1] + y0, min(W, b[2] + x0), min(H, b[3] + y0), float(c)])
    return merge(out)


if __name__ == "__main__":
    from ultralytics import YOLO
    dev, name, wp, which = int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]; rot = "rot" in sys.argv[5:]
    folds = {**json.load(open(f"{HERE}/tile_folds.json")), **json.load(open(f"{HERE}/folds4.json"))}
    pages = {**{k: v for k, v in folds.items()}, "A": sp["A_train"], "B": sp["B_test"], "all": sp["A_train"] + sp["B_test"]}[which]
    m = YOLO(wp); p = f"{CACHE}/{name}.json"; out = json.load(open(p)) if os.path.exists(p) else {}
    for n in pages:
        out[n] = predict_page(m, glob.glob(f"{SRC}/{glob.escape(n)}.*")[0], dev, rot)
    json.dump(out, open(p, "w")); print("TILEPRED_DONE", name, which, len(pages), flush=True)
