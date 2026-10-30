"""Rebuild the pipe connectivity of the verified pages from their annotations.

Inputs (paths relative to the topology_verified folder):
  images/<page>.png       the drawing
  symbols/<page>.txt      human-verified symbol boxes, YOLO format; the line number is the symbol id
  lines/<page>.json       human-verified pipe lines ("solid")
  manual_edits.json       per page, the changes made by hand during the connectivity review:
                            added_lines        pipe pieces drawn on the page but missing from the line annotations
                            removed_lines      annotated lines that are not pipes (a boundary or the page edge)
                            crossing_decisions human decisions for X crossings ({"x", "y", "connected"})
                            cuts               points where a connection that the algorithm made does not exist on the drawing
                            unlinks            [symbol id, x, y]: remove the connection of that symbol passing near (x, y)
                            fine_regions       regions with pipes very close to each other: ends are snapped within
                                               fine_tol px instead of 12 px there
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


def same_line(s, d):
    return max(abs(a - b) for a, b in zip(s, d)) <= 2 or max(abs(a - b) for a, b in zip(s, d[2:] + d[:2])) <= 2


def build(n, e):
    g = cv2.imread(f"{ROOT}/images/{n}.png", 0); H, W = g.shape
    syms = load_symbols(n, W, H)
    L = json.load(open(f"{ROOT}/lines/{n}.json"))
    drop = e.get("removed_lines", [])
    segs = [(tuple(map(int, s)), "solid") for s in L["solid"] if not any(same_line(s, d) for d in drop)]
    segs += [(tuple(int(round(v)) for v in a), "solid") for a in e.get("added_lines", [])]
    fine = (e["fine_regions"], int(e.get("fine_tol", 4))) if e.get("fine_regions") else None
    nodes, edges, crossings, _, _, paths = topo.build(segs, syms, g, decisions=e.get("crossing_decisions", []), tol=12, fine=fine)
    for sid, x, y in e.get("unlinks", []):
        keep = [i for i, (ed, pth) in enumerate(zip(edges, paths)) if not (("sym", sid) in (ed[0], ed[1]) and any(topo.point_seg_dist((x, y), tuple(p)) <= 10 for p in pth))]
        edges = [edges[i] for i in keep]; paths = [paths[i] for i in keep]
    for c in e.get("cuts", []):
        keep = [i for i, pth in enumerate(paths) if not any(topo.point_seg_dist(tuple(c), tuple(p)) <= 6 for p in pth)]
        edges = [edges[i] for i in keep]; paths = [paths[i] for i in keep]
    # a crossing that a cut turned into a T junction (one of its four arms removed) is no longer a crossing
    deg = collections.Counter(x for ed in edges for x in (ed[0], ed[1]))
    def node_deg(x, y):
        return max([deg[k] for k, (nx, ny) in nodes.items() if k[0] == "jct" and abs(nx - x) <= 3 and abs(ny - y) <= 3] or [0])
    crossings = [c for c in crossings if not (c[2] and node_deg(c[0], c[1]) < 4)]
    used = {x for ed in edges for x in (ed[0], ed[1])}
    jid = {}
    for k in nodes:
        if k[0] == "jct" and k in used: jid[k] = f"J{len(jid)}"
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
