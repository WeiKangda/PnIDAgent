"""Automatic checks of a connectivity file against the human-verified symbols and pipe lines of the same page.

Checks (distances are given at a page size of 3300 px on the long side and scaled to the real page size):
  1. coverage      every verified pipe line (the part outside symbol boxes) is covered by connections for >= 80% of its length
  2. on_pipe       every connection (the part outside symbol boxes) runs on verified pipe lines for >= 90% of its length
  3. attached      every verified pipe end that stops at a symbol box has a connection at that symbol
  4. isolated      every symbol has at least one connection
  5. junctions     no junction joins 4 or more pipes (a 4-way point is only allowed where a dot is drawn)
  6. crossings     where two pipes cross, "connected" agrees with the drawing: a filled dot means connected
  7. dead_ends     no pipe stops within 25 px of another pipe without joining it (an unjoined corner)

  python code/check_connectivity.py                # all pages listed in manual_edits.json
  python code/check_connectivity.py RHR            # only pages whose name contains "RHR"
"""
import collections, json, os, sys
import cv2, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def draw(H, W, segs, width):
    m = np.zeros((H, W), np.uint8)
    for s in segs: cv2.line(m, (int(s[0]), int(s[1])), (int(s[2]), int(s[3])), 1, width)
    return m


def seg_point_dist(p, s):
    a, b, q = np.array(s[:2], float), np.array(s[2:], float), np.array(p, float); d = b - a
    t = 0.0 if not d.any() else float(np.clip(np.dot(q - a, d) / np.dot(d, d), 0, 1))
    return float(np.linalg.norm(q - (a + t * d)))


def check(n, edits):
    t = json.load(open(f"{ROOT}/{n}.json")); W, H = t["width"], t["height"]
    g = cv2.imread(f"{ROOT}/images/{n}.png", 0); sc = 3300 / max(W, H); tol = max(6, int(8 / sc))
    pipes = json.load(open(f"{ROOT}/lines/{n}.json"))["solid"] + edits.get("added_lines", [])
    boxes = {s["id"]: s["bbox"] for s in t["symbols"]}
    sym = np.zeros((H, W), np.uint8)
    for x0, y0, x1, y1 in boxes.values(): sym[max(0, y0 - tol):y1 + tol, max(0, x0 - tol):x1 + tol] = 1
    link_mask = draw(H, W, [p for l in t["links"] for p in l["path"]], 2 * tol)
    pipe_mask = draw(H, W, pipes, 2 * tol)
    out = collections.defaultdict(list)

    for s in pipes:                                                    # 1
        m = draw(H, W, [s], 1) & (1 - sym); L = int(m.sum())
        if L * sc >= 15 and (m & link_mask).sum() / L < 0.8: out["coverage"].append([int(v) for v in s])
    for i, l in enumerate(t["links"]):                                 # 2
        m = draw(H, W, l["path"], 1) & (1 - sym); L = int(m.sum())
        if L * sc >= 15 and (m & pipe_mask).sum() / L < 0.9: out["on_pipe"].append([l["source"], l["target"]])
    for s in pipes:                                                    # 3
        for p in (s[:2], s[2:]):
            near = [k for k, (x0, y0, x1, y1) in boxes.items() if x0 - tol <= p[0] <= x1 + tol and y0 - tol <= p[1] <= y1 + tol]
            if near and not any(k in (l["source"], l["target"]) and min(seg_point_dist(p, q) for q in l["path"]) <= 2 * tol
                                for k in near for l in t["links"]):
                out["attached"].append({"pipe_end": [int(v) for v in p], "symbols": near})
    deg = collections.Counter(e for l in t["links"] for e in (l["source"], l["target"]))
    out["isolated"] = [k for k in boxes if deg[k] == 0]                # 4
    ink = (g < 128).astype(np.uint8); r = max(5, int(7 / sc))
    def dot(x, y):
        d = np.zeros_like(ink); cv2.circle(d, (int(x), int(y)), r, 1, -1); return (ink & d).sum() / d.sum() > 0.75
    for j in t["junctions"]:                                           # 5
        if deg[j["id"]] >= 4 and not dot(*j["xy"]): out["junctions"].append(j["id"])
    for c in t["crossings"]:                                           # 6
        if dot(c["x"], c["y"]) != c["connected"]: out["crossings"].append(c)
    for j in t["junctions"]:                                           # 7
        if deg[j["id"]] != 1: continue
        own = [l for l in t["links"] if j["id"] in (l["source"], l["target"])]
        others = [q for l in t["links"] if l not in own for q in l["path"]]
        if others and min(seg_point_dist(j["xy"], q) for q in others) <= 25 / sc: out["dead_ends"].append(j["id"])
    return t, {k: out.get(k, []) for k in ("coverage", "on_pipe", "attached", "isolated", "junctions", "crossings", "dead_ends")}


if __name__ == "__main__":
    args = sys.argv[1:]; edits = json.load(open(f"{ROOT}/manual_edits.json")); total = 0
    for n in [n for n in edits if not args or any(a in n for a in args)]:
        t, res = check(n, edits[n]); bad = sum(len(v) for v in res.values()); total += bad
        print(f"{n}  ({len(t['symbols'])} symbols, {len(t['links'])} links, {len(t['crossings'])} crossings): "
              + ("all checks passed" if not bad else f"{bad} problem(s)"))
        for k, v in res.items():
            if v: print(f"    {k}: {v}")
    print("no problems found" if not total else f"{total} problem(s) found")
