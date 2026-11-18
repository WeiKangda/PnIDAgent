"""Locate every legend image (legend.json) on its legend PDF page: render the page at 600 dpi, try the three orientations,
template-match the extracted image (they were cut from the same rendering). Output legend_loc.json {j: [rot, x, y, w, h, score]}."""
import json, os, fitz, cv2, numpy as np
R = "/home/suxinqi666/code/PnIDAgent_project"; HERE = f"{R}/symbol_class"; LP = f"{HERE}/legend_pages"
PDF = {"ap1000": f"{R}/PID_merged/Symbol - Legend/AP1000 (PID Legend).pdf", "apr1400": f"{R}/PID_merged/Symbol - Legend/APR1400 (PID Legend).pdf"}
G = json.load(open(f"{HERE}/legend.json")); ROT = {0: None, 1: cv2.ROTATE_90_CLOCKWISE, 2: cv2.ROTATE_90_COUNTERCLOCKWISE}
def page(src):
    f = f"{LP}/{src}.png"
    if not os.path.exists(f):
        fam, p = src.split("_p"); pix = fitz.open(PDF[fam])[int(p)].get_pixmap(dpi=600)
        im = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3]; cv2.imwrite(f, cv2.cvtColor(im, cv2.COLOR_RGB2GRAY))
    return cv2.imread(f, 0)
LK = json.load(open(f"{HERE}/legend_link.json")); import sys
want = {v[k] for v in LK.values() for k in ("body", "op") if v[k] is not None} | {int(a) for a in sys.argv[1:]}
out = json.load(open(f"{HERE}/legend_loc.json")) if os.path.exists(f"{HERE}/legend_loc.json") else {}; cache = {}
for j, g in enumerate(G):
    if j not in want or str(j) in out: continue
    if g["src"] not in cache: P = page(g["src"]); cache[g["src"]] = {k: (P if r is None else cv2.rotate(P, r)) for k, r in ROT.items()}
    t = cv2.imread(g["file"], 0); best = None
    for k, P in cache[g["src"]].items():
        r = cv2.matchTemplate(P, t, cv2.TM_CCOEFF_NORMED); _, m, _, loc = cv2.minMaxLoc(r)
        if best is None or m > best[5]: best = [k, loc[0], loc[1], t.shape[1], t.shape[0], round(float(m), 3)]
    out[str(j)] = best
json.dump(out, open(f"{HERE}/legend_loc.json", "w"))
import collections; print(collections.Counter(round(v[5], 1) for v in out.values()), collections.Counter((G[int(j)]["src"], v[0]) for j, v in out.items()))
