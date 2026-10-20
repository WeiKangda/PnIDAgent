"""Split the A half into 4 folds by document (balanced symbol counts); build tile data exp65_tile_F0..F3 (train = A minus fold k) and large-equipment data exp65_big_F0..F3.
The fold-k model predicts only fold-k pages -> A-side cross-fitted predictions come from models trained on 30 pages (close to the 40-page all-A model used for B)."""
import json, os, re, glob, shutil, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from tiles import make, doc, LAB, sp, R
A = sp["A_train"]; nsym = {n: sum(1 for l in open(f"{LAB}/{n}.txt") if l.strip()) for n in A}
groups = {}
for n in A: groups.setdefault(doc(n), []).append(n)
F = [[] for _ in range(4)]
for g in sorted(groups, key=lambda g: (-sum(nsym[n] for n in groups[g]), g)):
    min(F, key=lambda f: sum(nsym[n] for n in f)).extend(groups[g])
json.dump({f"F{k}": sorted(f) for k, f in enumerate(F)}, open(f"{HERE}/folds4.json", "w"), indent=1)
print("4 folds:", [(len(f), sum(nsym[n] for n in f)) for f in F])
src = f"{R}/datasets/exp65_big_A"; lab = {os.path.basename(f)[:-4]: f for sd in ("train", "val") for f in glob.glob(f"{src}/labels/{sd}/*.txt")}
img = {os.path.basename(f)[:-4]: f for sd in ("train", "val") for f in glob.glob(f"{src}/images/{sd}/*")}
for k in range(4):
    tr = [n for n in A if n not in F[k]]; va = F[k][:3]
    make(f"F{k}", tr, va)
    out = f"{R}/datasets/exp65_big_F{k}"
    for sd, pages in (("train", tr), ("val", va)):
        os.makedirs(f"{out}/images/{sd}", exist_ok=True); os.makedirs(f"{out}/labels/{sd}", exist_ok=True)
        for n in pages:
            d = f"{out}/images/{sd}/{n}{os.path.splitext(img[n])[1]}"
            if not os.path.exists(d): os.link(img[n], d)
            shutil.copy(lab[n], f"{out}/labels/{sd}/{n}.txt") if n in lab else open(f"{out}/labels/{sd}/{n}.txt", "w").close()
    open(f"{out}/data.yaml", "w").write(f"path: {out}\ntrain: images/train\nval: images/val\nnames:\n  0: item\n")
