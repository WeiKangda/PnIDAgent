"""For every legend entry used by the symbol labels, cut from the legend PDF page (600 dpi, reading orientation) the symbol
and its printed description (found by OCR, legend_ocr/). Compact pairs are cut as one region with the symbol framed in red;
otherwise symbol and description are cut separately and placed side by side.
Output legend_snip/<j>.png, legend_snip.json {j: {"desc_ocr", "match", "mode"}}."""
import json, os, re, sys, difflib, cv2, numpy as np
HERE = "/home/suxinqi666/code/PnIDAgent_project/symbol_class"; LP = f"{HERE}/legend_pages"; OUT = f"{HERE}/legend_snip"; os.makedirs(OUT, exist_ok=True)
G = json.load(open(f"{HERE}/legend.json")); LOC = json.load(open(f"{HERE}/legend_loc.json"))
MAN = json.load(open(f"{HERE}/legend_snip_manual.json")) if os.path.exists(f"{HERE}/legend_snip_manual.json") else {}
ROT = {0: None, 1: cv2.ROTATE_90_CLOCKWISE, 2: cv2.ROTATE_90_COUNTERCLOCKWISE}
norm = lambda t: re.sub(r"[^A-Z0-9]", "", t.upper())
pages = {}; ocr = {}
def page(src, rot):
    if (src, rot) not in pages:
        P = cv2.imread(f"{LP}/{src}.png", 0); pages[(src, rot)] = P if ROT[rot] is None else cv2.rotate(P, ROT[rot])
        ocr[(src, rot)] = json.load(open(f"{HERE}/legend_ocr/{src}_r{rot}.json"))
    return pages[(src, rot)], ocr[(src, rot)]
def tight(B, x0, y0, x1, y1):
    sub = B[y0:y1, x0:x1] < 160; ys, xs = np.nonzero(sub)
    return (x0 + xs.min(), y0 + ys.min(), x0 + xs.max() + 1, y0 + ys.max() + 1) if len(xs) else (x0, y0, x1, y1)
res = {}
for j in sorted(LOC, key=int):
    g = G[int(j)]; rot, x, y, w, h, sc = LOC[j]; P, O = page(g["src"], rot); H, W = P.shape
    sym = tight(P, x, y, x + w, y + h)
    n_, lab, st, _ = cv2.connectedComponentsWithStats((P[max(0, y - h // 2):y + h + h // 2, max(0, x - w // 2):x + w + w // 2] < 160).astype(np.uint8), 8)
    ox_, oy_ = max(0, x - w // 2), max(0, y - h // 2)
    for c in range(1, n_):
        a0, b0, cw, ch = st[c][:4]; a0 += ox_; b0 += oy_
        if a0 < sym[2] and a0 + cw > sym[0] and b0 < sym[3] and b0 + ch > sym[1] and cw < 2 * w and ch < 2 * h:
            sym = (min(sym[0], a0), min(sym[1], b0), max(sym[2], a0 + cw), max(sym[3], b0 + ch)); scx, scy = (sym[0] + sym[2]) / 2, (sym[1] + sym[3]) / 2
    d = norm(re.sub(r"\(\d+\)", "", g.get("description") or "")); best = None
    if j in MAN: lines = [O[k] for k in MAN[j]]; best = (1.0, MAN[j][0])
    elif d and "CUTOFF" not in d:
        for k, o in enumerate(O):
            t = norm(re.sub(r"\(\d+\)", "", o["text"]))
            if len(t) < 2: continue
            pre = d[:max(len(t), 4)] if len(t) < len(d) else d
            r = difflib.SequenceMatcher(None, t[:len(pre)], pre).ratio() if len(t) <= len(d) else difflib.SequenceMatcher(None, t, d).ratio() * 0.9
            if r < 0.75: continue
            cx, cy = (o["box"][0] + o["box"][2]) / 2, (o["box"][1] + o["box"][3]) / 2
            dist = abs(cx - scx) + 2.5 * abs(cy - scy)
            key = (r >= 0.85, -dist)
            if best is None or key > best[0:1] + (best[2],): best = (r >= 0.85, k, -dist)
        lines = []
        if best:
            k = best[1]; lines = [O[k]]; got = norm(O[k]["text"])
            while len(got) < len(d) * 0.9:          # continuation lines directly below, left-aligned
                last = lines[-1]["box"]; lh = last[3] - last[1]
                nx = [o for o in O if 0 < o["box"][1] - last[3] < 0.9 * lh and abs(o["box"][0] - lines[0]["box"][0]) < 1.5 * lh and norm(o["text"]) and norm(o["text"])[:4] in d[len(got) - 2:]]
                if not nx: break
                lines.append(nx[0]); got += norm(nx[0]["text"])
    else: lines = []
    if lines:
        db = (min(o["box"][0] for o in lines), min(o["box"][1] for o in lines), max(o["box"][2] for o in lines), max(o["box"][3] for o in lines))
        ux0, uy0, ux1, uy1 = min(sym[0], db[0]), min(sym[1], db[1]), max(sym[2], db[2]), max(sym[3], db[3])
    m = 14
    if lines and ux1 - ux0 <= 1700 and uy1 - uy0 <= 420:
        C = cv2.cvtColor(P[max(0, uy0 - m):uy1 + m, max(0, ux0 - m):ux1 + m], cv2.COLOR_GRAY2BGR)
        ox, oy = max(0, ux0 - m), max(0, uy0 - m)
        cv2.rectangle(C, (sym[0] - ox - 8, sym[1] - oy - 8), (sym[2] - ox + 8, sym[3] - oy + 8), (0, 0, 230), 3); mode = "together"
    else:
        S_ = cv2.cvtColor(P[max(0, sym[1] - m):sym[3] + m, max(0, sym[0] - m):sym[2] + m], cv2.COLOR_GRAY2BGR)
        cv2.rectangle(S_, (6, 6), (S_.shape[1] - 7, S_.shape[0] - 7), (0, 0, 230), 3)
        if lines:
            D = cv2.cvtColor(P[max(0, db[1] - m):db[3] + m, max(0, db[0] - m):db[2] + m], cv2.COLOR_GRAY2BGR)
            hh = max(S_.shape[0], D.shape[0]); pad = lambda A: cv2.copyMakeBorder(A, (hh - A.shape[0]) // 2, hh - A.shape[0] - (hh - A.shape[0]) // 2, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            gap = np.full((hh, 60, 3), 255, np.uint8); cv2.line(gap, (8, hh // 2), (50, hh // 2), (150, 150, 150), 3); cv2.line(gap, (50, hh // 2), (38, hh // 2 - 10), (150, 150, 150), 3); cv2.line(gap, (50, hh // 2), (38, hh // 2 + 10), (150, 150, 150), 3)
            C = np.hstack([pad(S_), gap, pad(D)]); mode = "side by side"
        else: C = S_; mode = "symbol only"
    cv2.imwrite(f"{OUT}/{j}.png", C); cv2.imwrite(f"{OUT}/{j}_sym.png", P[max(0, sym[1] - 10):sym[3] + 10, max(0, sym[0] - 10):sym[2] + 10])
    res[j] = {"desc_ocr": " / ".join(o["text"] for o in lines), "match": best[0] if best else 0, "mode": mode}
json.dump(res, open(f"{HERE}/legend_snip.json", "w"), indent=0)
import collections; print(collections.Counter(v["mode"] for v in res.values()), [(j, G[int(j)]["description"][:40]) for j, v in res.items() if v["mode"] == "symbol only"])
