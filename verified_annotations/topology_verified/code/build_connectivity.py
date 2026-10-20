"""Rebuild the pipe connectivity of the verified pages from their annotations.

Inputs (paths relative to the topology_verified folder):
  images/<page>.png       the drawing
  symbols/<page>.txt      human-verified symbol boxes, YOLO format; the line number is the symbol id
  lines/<page>.json       human-verified pipe lines ("solid"; dashed lines are not pipes and are not used)
  manual_edits.json       per page: short pipe pieces added by hand during review, and human decisions for X crossings
Output:
  rebuilt/<page>.json     same format as <page>.json (symbols, junctions, links, neighbors, crossings)

  python code/build_connectivity.py            # all pages listed in manual_edits.json
  python code/build_connectivity.py --check    # also compare with the published <page>.json files
  python code/build_connectivity.py Seabrook   # only pages whose name contains "Seabrook"
"""
import collections, json, os, sys
import cv2
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import topo
ROOT = os.path.dirname(HERE)


def load_symbols(n, W, H):
    """Symbol boxes in pixels, rounded down exactly as when the graphs were built."""
    boxes = [list(map(float, l.split()[1:5])) for l in open(f"{ROOT}/symbols/{n}.txt") if l.strip()]
    return [(i, (int((cx - w / 2) * W), int((cy - h / 2) * H), int((cx + w / 2) * W), int((cy + h / 2) * H))) for i, (cx, cy, w, h) in enumerate(boxes)]


def neighbors(symbols, links):
    """Symbol -> symbols reachable through pipes and branch points, without passing through another symbol."""
    adj = collections.defaultdict(set)
    for l in links: adj[l["source"]].add(l["target"]); adj[l["target"]].add(l["source"])
    out = {}
    for s in symbols:
        seen, stack, found = {s["id"]}, [s["id"]], set()
        while stack:
            for v in adj[stack.pop()]:
                if v in seen: continue
                seen.add(v)
                if isinstance(v, int): found.add(v)       # a symbol: stop here
                else: stack.append(v)                    # a branch point: keep walking
        out[str(s["id"])] = sorted(found)
    return out


def build(n, edits):
    g = cv2.imread(f"{ROOT}/images/{n}.png", 0); H, W = g.shape
    syms = load_symbols(n, W, H)
    L = json.load(open(f"{ROOT}/lines/{n}.json"))
    segs = [(tuple(map(int, s)), "solid") for s in L["solid"]]
    segs += [(tuple(int(round(v)) for v in a), "solid") for a in edits.get("added_lines", [])]
    nodes, edges, crossings, _, _, paths = topo.build(segs, syms, g, decisions=edits.get("crossing_decisions", []), tol=12)
    jid = {}
    for k in nodes:
        if k[0] == "jct": jid[k] = f"J{len(jid)}"
    nid = lambda k: k[1] if k[0] == "sym" else jid[k]
    symbols = [{"id": i, "bbox": list(b)} for i, b in syms]
    links = [{"source": nid(u), "target": nid(v), "path": [[round(c, 1) for c in p] for p in path]} for (u, v, _), path in zip(edges, paths)]
    return {"name": n, "width": W, "height": H, "symbols": symbols,
            "junctions": [{"id": jid[k], "xy": list(nodes[k])} for k in jid],
            "links": links, "neighbors": neighbors(symbols, links),
            "crossings": [{"x": round(c[0], 1), "y": round(c[1], 1), "connected": bool(c[2])} for c in crossings]}


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]; check = "--check" in sys.argv
    edits = json.load(open(f"{ROOT}/manual_edits.json"))
    pages = [n for n in edits if not args or any(a in n for a in args)]
    os.makedirs(f"{ROOT}/rebuilt", exist_ok=True); bad = 0
    for n in pages:
        r = build(n, edits[n]); json.dump(r, open(f"{ROOT}/rebuilt/{n}.json", "w"))
        msg = f"{n}: {len(r['symbols'])} symbols, {len(r['junctions'])} branch points, {len(r['links'])} links"
        if check:
            pub = json.load(open(f"{ROOT}/{n}.json"))
            diff = [k for k in ("symbols", "junctions", "links", "neighbors", "crossings") if json.loads(json.dumps(r[k])) != pub[k]]
            msg += "  | identical to the published file" if not diff else f"  | DIFFERENT: {diff}"; bad += bool(diff)
        print(msg)
    if check: print("all identical" if not bad else f"{bad} page(s) differ")
