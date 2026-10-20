"""High-resolution tile dataset: cut full pages at native resolution into 640x640 tiles (stride 480, overlap 160) so small symbols keep their original pixel size (full-page detection shrinks them to ~0.39x).
- Cut symbols: if >=60% visible, clip to the tile as a label; if <60%, paint that part of the tile white (so the model does not learn half symbols as background)
- Each tile is also saved rotated by 90 degrees (symbols appear both horizontal and vertical; up-down/left-right flips are left to training augmentation)
- Keep only 30% of tiles without symbols
Datasets: exp65_tile_A (all 40 A pages), exp65_tile_A1 / A2 (A split in two halves by document, for cross-fitting: each predicts the other, giving A-side features for rescoring)
  python tiles.py"""
import glob, json, os, re, random
import cv2, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
R = "/home/suxinqi666/code/PnIDAgent_project"; LAB = f"{R}/line_results/labels_verified"; SRC = f"{R}/datasets/yolo_pseudo_v2/images/train"
sp = json.load(open(f"{HERE}/split.json")); T, STEP = 640, 480
VAL = sp["A_train"][::8]          # the 5 validation pages used by the large-equipment model; here also only for monitoring training


def doc(n): return re.sub(r"_p\d+$", "", n).replace("SCS2_", "SCS_")


def origins(L):
    o = list(range(0, max(1, L - T), STEP))
    if o[-1] + T < L: o.append(L - T)
    return o


def gt(n, W, H):
    return [[(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H] for cx, cy, w, h in
            (list(map(float, l.split()[1:5])) for l in open(f"{LAB}/{n}.txt") if l.strip())]


def make(name, train_pages, val_pages):
    out = f"{R}/datasets/exp65_tile_{name}"; rnd = random.Random(0); cnt = {"train": 0, "val": 0}
    for split, pages in (("train", train_pages), ("val", val_pages)):
        os.makedirs(f"{out}/images/{split}", exist_ok=True); os.makedirs(f"{out}/labels/{split}", exist_ok=True)
        for n in pages:
            im = cv2.imread(glob.glob(f"{SRC}/{glob.escape(n)}.*")[0]); H, W = im.shape[:2]; G = gt(n, W, H)
            for y0 in origins(H):
                for x0 in origins(W):
                    t = im[y0:y0 + T, x0:x0 + T].copy(); labs = []
                    for b in G:
                        ix0, iy0, ix1, iy1 = max(b[0], x0), max(b[1], y0), min(b[2], x0 + T), min(b[3], y0 + T)
                        if ix1 <= ix0 or iy1 <= iy0: continue
                        vis = (ix1 - ix0) * (iy1 - iy0) / ((b[2] - b[0]) * (b[3] - b[1]))
                        if vis >= 0.6: labs.append([ix0 - x0, iy0 - y0, ix1 - x0, iy1 - y0])
                        else: t[int(iy0 - y0):int(np.ceil(iy1 - y0)), int(ix0 - x0):int(np.ceil(ix1 - x0))] = 255
                    if not labs and rnd.random() > 0.3: continue
                    vs = [(t, labs)]
                    if split == "train":    # 90-degree rotation (counterclockwise): (x, y) -> (y, T - x)
                        vs.append((np.ascontiguousarray(np.rot90(t)), [[b[1], T - b[2], b[3], T - b[0]] for b in labs]))
                    for k, (img, bl) in enumerate(vs):
                        stem = f"{n}__{x0}_{y0}_r{k}"
                        cv2.imwrite(f"{out}/images/{split}/{stem}.png", img)
                        with open(f"{out}/labels/{split}/{stem}.txt", "w") as f:
                            for b in bl: f.write(f"0 {(b[0] + b[2]) / 2 / T:.6f} {(b[1] + b[3]) / 2 / T:.6f} {(b[2] - b[0]) / T:.6f} {(b[3] - b[1]) / T:.6f}\n")
                        cnt[split] += 1
    open(f"{out}/data.yaml", "w").write(f"path: {out}\ntrain: images/train\nval: images/val\nnames:\n  0: item\n")
    print(name, cnt, flush=True)


if __name__ == "__main__":
    A = sp["A_train"]; nsym = {n: sum(1 for l in open(f"{LAB}/{n}.txt") if l.strip()) for n in A}
    groups = {}
    for n in A: groups.setdefault(doc(n), []).append(n)
    A1, A2 = [], []
    for g in sorted(groups, key=lambda g: (-sum(nsym[n] for n in groups[g]), g)):
        (A1 if sum(nsym[n] for n in A1) <= sum(nsym[n] for n in A2) else A2).extend(groups[g])
    json.dump({"A1": sorted(A1), "A2": sorted(A2)}, open(f"{HERE}/tile_folds.json", "w"), indent=1)
    print("A1", len(A1), sum(nsym[n] for n in A1), "| A2", len(A2), sum(nsym[n] for n in A2))
    make("A1", A1, [n for n in VAL if n in A2] or A2[:3])    # monitor with pages from the other half (unseen)
    make("A2", A2, [n for n in VAL if n in A1] or A1[:3])
    make("A", A, VAL)
