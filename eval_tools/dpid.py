#!/usr/bin/env python3
"""Loader for Dataset-P&ID (Paliwal et al., arXiv:2109.03794).

Layout on disk (see setup notes in README):
    dataset/image_2/{N}.jpg          7168x4561 RGB
    dataset/mask/{N}_mask.png        same size, 0=bg, 1..32 = symbol class
    dataset/ann/{N}/{N}_*.npy        7 annotation arrays per sheet

Raw schemas (verified against sheet 0):
    symbols   (n,3)  [symbol_id, [x1,y1,x2,y2], class_id]
    lines     (n,4)  [line_id, [x1,y1,x2,y2], pipe_label, 'solid'|'dashed']
    words     (n,4)  [word_id, [x1,y1,x2,y2], text, rotation(0|90)]
    linker    (n,2)  [symbol_id, [word_id, line_id]]
    lines2    (n,5)  [x1,y1,x2,y2,flag]  -- sheet frame + title-block rulings
    KeyValue  (k,2)  title-block key/value pairs
    Table     (r,6)  revision table

The raw arrays have several inconsistencies that every consumer has to repair,
so load_sheet() normalizes them once:

  1. symbols class column mixes str and int ('21' and 21 both occur)
  2. words bboxes are not consistently y1<y2 (65/237 inverted on sheet 0)
  3. some words rows are zero-area placeholders with empty text
  4. pipe_label is only present on a minority of line segments
  5. folders 245/246/247 contain duplicate uploads of the same files
  6. linker is symbol -> (one word, one line), NOT a symbol-to-symbol edge list
  7. a few solid segments do not sit exactly on drawn ink
  8. a third of linker rows have no usable tag: the word id is either absent from
     words.npy entirely or points at one of the placeholder rows from (3), so
     symbol-to-tag GT covers only ~67% of symbols
"""
from __future__ import annotations

import collections
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

Box = Tuple[int, int, int, int]
Seg = Tuple[int, int, int, int]

N_CLASSES = 32


def _norm_box(b: Sequence) -> Box:
    """Order a bbox as (x1, y1, x2, y2) with x1<=x2 and y1<=y2."""
    x1, y1, x2, y2 = (int(v) for v in b)
    return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)


@dataclass
class Symbol:
    id: str
    box: Box
    cls: int            # 1..32
    word: Optional[str] = None       # tag text, resolved via linker
    line: Optional[str] = None       # id of the line it sits on, via linker


@dataclass
class Line:
    id: str
    seg: Seg            # (x1, y1, x2, y2) as annotated, not reordered
    label: str          # pipe number, '' when unlabelled
    kind: str           # 'solid' | 'dashed'


@dataclass
class Word:
    id: str
    box: Box
    text: str
    rot: int            # 0 or 90


@dataclass
class Sheet:
    idx: int
    symbols: List[Symbol]
    lines: List[Line]
    words: List[Word]
    frame: List[Seg]                        # from lines2
    keyvalue: List[Tuple[str, str]]
    table: List[List[str]]
    dropped: Dict[str, int] = field(default_factory=dict)

    @property
    def size(self) -> Tuple[int, int]:
        return 7168, 4561

    def by_id(self, kind: str) -> Dict[str, object]:
        return {o.id: o for o in getattr(self, kind)}


def _load(path: str) -> np.ndarray:
    return np.load(path, allow_pickle=True)


def load_sheet(root: str, idx: int) -> Sheet:
    """Load and normalize one sheet's annotations.

    `root` is the dataset directory holding ann/, image_2/ and mask/.
    """
    d = os.path.join(root, "ann", str(idx))
    p = lambda suf: os.path.join(d, f"{idx}_{suf}.npy")
    dropped: Dict[str, int] = collections.Counter()

    symbols: List[Symbol] = []
    for sid, box, cls in _load(p("symbols")):
        # gotcha 1: class column mixes '21' and 21
        symbols.append(Symbol(id=str(sid), box=_norm_box(box), cls=int(cls)))

    lines: List[Line] = []
    for lid, seg, label, kind in _load(p("lines")):
        lines.append(Line(id=str(lid),
                          seg=tuple(int(v) for v in seg),
                          label=str(label).strip(),   # gotcha 4: often ''
                          kind=str(kind)))

    words: List[Word] = []
    for wid, box, text, rot in _load(p("words")):
        b = _norm_box(box)                       # gotcha 2: y order inverted
        t = str(text).strip()
        if not t or b[0] == b[2] or b[1] == b[3]:
            dropped["placeholder_words"] += 1    # gotcha 3
            continue
        words.append(Word(id=str(wid), box=b, text=t, rot=int(rot)))

    frame = [tuple(int(v) for v in r[:4]) for r in _load(p("lines2"))]
    keyvalue = [(str(a), str(b)) for a, b in _load(p("KeyValue"))]
    table = [[str(c) for c in row] for row in _load(p("Table"))]

    sheet = Sheet(idx=idx, symbols=symbols, lines=lines, words=words,
                  frame=frame, keyvalue=keyvalue, table=table,
                  dropped=dict(dropped))

    # gotcha 6: linker is symbol -> (word, line); attach it to the symbols
    wmap = sheet.by_id("words")
    smap = sheet.by_id("symbols")
    for sid, targets in _load(p("linker")):
        s = smap.get(str(sid))
        if s is None:
            continue
        for t in targets:
            t = str(t)
            if t.startswith("word"):
                w = wmap.get(t)
                if w is None:
                    # gotcha 8: dangling or placeholder tag reference
                    dropped["linker_word_unresolved"] += 1
                else:
                    s.word = w.text
            elif t.startswith("line"):
                s.line = t
    sheet.dropped = dict(dropped)
    return sheet


# ---------------------------------------------------------------------------
# Symbol-to-symbol connectivity, derived from lines + linker (gotcha 6)
# ---------------------------------------------------------------------------

def _on_segment(px: int, py: int, seg: Seg, tol: int) -> bool:
    """True if (px,py) lies on an axis-aligned segment, endpoints included."""
    x1, y1, x2, y2 = seg
    if abs(x1 - x2) <= tol:                                  # vertical
        return abs(px - x1) <= tol and min(y1, y2) - tol <= py <= max(y1, y2) + tol
    if abs(y1 - y2) <= tol:                                  # horizontal
        return abs(py - y1) <= tol and min(x1, x2) - tol <= px <= max(x1, x2) + tol
    # oblique: fall back to perpendicular distance
    dx, dy = x2 - x1, y2 - y1
    L2 = dx * dx + dy * dy
    if L2 == 0:
        return abs(px - x1) <= tol and abs(py - y1) <= tol
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / L2))
    return (px - (x1 + t * dx)) ** 2 + (py - (y1 + t * dy)) ** 2 <= tol * tol


def _box_hits_segment(box: Box, seg: Seg, pad: int = 0) -> bool:
    """Axis-aligned segment vs. bbox overlap test."""
    bx1, by1, bx2, by2 = box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad
    x1, y1, x2, y2 = seg
    if abs(x1 - x2) <= abs(y1 - y2):                         # treat as vertical
        return bx1 <= x1 <= bx2 and min(y1, y2) <= by2 and max(y1, y2) >= by1
    return by1 <= y1 <= by2 and min(x1, x2) <= bx2 and max(x1, x2) >= bx1


def _num(node_id: str) -> int:
    return int(node_id.split("_")[1])


@dataclass
class Graph:
    """Pipe topology of a sheet.

    Nodes are symbols plus the junction points where pipe runs meet or end;
    edges are the pipe runs between consecutive nodes. Junctions are kept as
    real nodes rather than contracted away: a header with 20 branch valves is
    a star through the header's junctions, not a 20-way clique, so a single
    mispredicted run costs one edge instead of a whole clique.
    """
    nodes: Dict[object, Tuple[float, float]]            # node -> (x, y)
    edges: List[Tuple[object, object, str]]             # (u, v, 'solid'|'dashed')

    def degree(self) -> Dict[object, int]:
        d: Dict[object, int] = collections.Counter()
        for u, v, _ in self.edges:
            d[u] += 1
            d[v] += 1
        return d


def derive_graph(sheet: Sheet, tol: int = 3, include_dashed: bool = True) -> Graph:
    """Build the pipe topology from lines + linker.

    Symbol nodes are ('sym', symbol_id); junction nodes are ('jct', i). Pipe
    runs that merely continue straight through a degree-2 point are merged, so
    every junction node in the result is a genuine branch, corner or dead end.
    """
    segs = [l for l in sheet.lines if include_dashed or l.kind == "solid"]

    # Snap coincident endpoints onto shared points.
    pt_id: Dict[Tuple[int, int], int] = {}
    pt_xy: Dict[int, Tuple[int, int]] = {}

    def point_of(x: int, y: int) -> int:
        for dx in range(-tol, tol + 1):
            for dy in range(-tol, tol + 1):
                k = (x + dx, y + dy)
                if k in pt_id:
                    return pt_id[k]
        pt_id[(x, y)] = len(pt_id)
        pt_xy[pt_id[(x, y)]] = (x, y)
        return pt_id[(x, y)]

    ends = [(point_of(*l.seg[:2]), point_of(*l.seg[2:])) for l in segs]

    # A T-junction is one segment's endpoint landing in another's interior.
    interior: Dict[int, List[int]] = collections.defaultdict(list)
    for si, l in enumerate(segs):
        for pid, (px, py) in pt_xy.items():
            if pid not in ends[si] and _on_segment(px, py, l.seg, tol):
                interior[si].append(pid)

    # Symbols cut the run they sit on. linker gives the line directly; a few
    # symbols need the geometric fallback.
    by_line = {l.id: si for si, l in enumerate(segs)}
    sym_on: Dict[int, List[Symbol]] = collections.defaultdict(list)
    for s in sheet.symbols:
        si = by_line.get(s.line) if s.line else None
        if si is None:
            si = next((j for j, l in enumerate(segs)
                       if _box_hits_segment(s.box, l.seg)), None)
        if si is not None:
            sym_on[si].append(s)

    # Chain everything that lies on each segment, in order along it.
    adj: Dict[object, List[Tuple[object, str]]] = collections.defaultdict(list)
    pos: Dict[object, Tuple[float, float]] = {}
    for si, l in enumerate(segs):
        x1, y1, x2, y2 = l.seg
        dx, dy = x2 - x1, y2 - y1
        L2 = float(dx * dx + dy * dy) or 1.0
        items: List[Tuple[float, object]] = []
        for pid in (*ends[si], *interior[si]):
            px, py = pt_xy[pid]
            items.append((((px - x1) * dx + (py - y1) * dy) / L2, ("jct", pid)))
            pos[("jct", pid)] = (px, py)
        for s in sym_on[si]:
            cx, cy = (s.box[0] + s.box[2]) / 2.0, (s.box[1] + s.box[3]) / 2.0
            items.append((((cx - x1) * dx + (cy - y1) * dy) / L2, ("sym", s.id)))
            pos[("sym", s.id)] = (cx, cy)
        items.sort(key=lambda t: t[0])
        for (_, u), (_, v) in zip(items, items[1:]):
            if u != v:
                adj[u].append((v, l.kind))
                adj[v].append((u, l.kind))

    # Contract degree-2 junctions: they are mid-run points, not topology.
    def keep(n: object) -> bool:
        return n[0] == "sym" or len(adj[n]) != 2

    edges: set = set()
    for u in list(adj):
        if not keep(u):
            continue
        for first, kind in adj[u]:
            prev, cur, k = u, first, kind
            while not keep(cur):
                nxt = [(n, kk) for n, kk in adj[cur] if n != prev]
                if not nxt:
                    break
                prev, (cur, kk) = cur, nxt[0]
                if kk == "dashed":
                    k = "dashed"          # a run is dashed if any part of it is
            if keep(cur) and cur != u:
                a, b = sorted([u, cur], key=lambda n: (n[0], str(n[1])))
                edges.add((a, b, k))

    nodes = {n: pos[n] for n in adj if keep(n) and n in pos}
    return Graph(nodes=nodes, edges=sorted(
        edges, key=lambda e: (str(e[0]), str(e[1]))))


def derive_edges(sheet: Sheet, tol: int = 3, include_dashed: bool = True
                 ) -> List[Tuple[str, str, str]]:
    """Symbol-to-symbol adjacency: symbols joined without a third in between.

    Reported for comparison only. It is clique-heavy -- every symbol hanging off
    a shared header becomes mutually adjacent -- so derive_graph() is the metric
    to optimize against. Returns (symbol_a, symbol_b, kind).
    """
    g = derive_graph(sheet, tol=tol, include_dashed=include_dashed)
    adj: Dict[object, List[Tuple[object, str]]] = collections.defaultdict(list)
    for u, v, k in g.edges:
        adj[u].append((v, k))
        adj[v].append((u, k))

    out: set = set()
    for n in g.nodes:
        if n[0] != "sym":
            continue
        seen = {n}
        stack = list(adj[n])
        while stack:
            cur, k = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            if cur[0] == "sym":
                a, b = sorted([n[1], cur[1]], key=_num)
                if a != b:
                    out.add((a, b, k))
                continue                       # do not walk past a symbol
            stack.extend((x, kk) for x, kk in adj[cur] if x not in seen)
    return sorted(out, key=lambda e: (_num(e[0]), _num(e[1])))


# ---------------------------------------------------------------------------

SUFFIXES = ("symbols", "lines", "words", "linker", "lines2", "KeyValue", "Table")


def is_complete(root: str, idx: int) -> bool:
    d = os.path.join(root, "ann", str(idx))
    return all(os.path.exists(os.path.join(d, f"{idx}_{s}.npy")) for s in SUFFIXES)


def sheet_ids(root: str, complete_only: bool = True) -> List[int]:
    """Sheet indices present under ann/. Partially downloaded sheets are skipped."""
    d = os.path.join(root, "ann")
    ids = sorted(int(n) for n in os.listdir(d) if n.isdigit())
    return [i for i in ids if not complete_only or is_complete(root, i)]


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Inspect Dataset-P&ID annotations")
    ap.add_argument("--root", default=os.path.join(os.path.dirname(__file__),
                                                   "..", "dataset"))
    ap.add_argument("--limit", type=int, default=0, help="0 = all sheets")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    ids = sheet_ids(root)
    if args.limit:
        ids = ids[:args.limit]

    tot = collections.Counter()
    cls_hist = collections.Counter()
    rot_hist = collections.Counter()
    deg_hist = collections.Counter()
    for i in ids:
        s = load_sheet(root, i)
        g = derive_graph(s)
        deg = g.degree()
        tot["sheets"] += 1
        tot["symbols"] += len(s.symbols)
        tot["lines"] += len(s.lines)
        tot["dashed"] += sum(1 for l in s.lines if l.kind == "dashed")
        tot["labelled_lines"] += sum(1 for l in s.lines if l.label)
        tot["words"] += len(s.words)
        tot["placeholder_words"] += s.dropped.get("placeholder_words", 0)
        tot["tagged_symbols"] += sum(1 for x in s.symbols if x.word)
        tot["graph_nodes"] += len(g.nodes)
        tot["junctions"] += sum(1 for n in g.nodes if n[0] == "jct")
        tot["graph_edges"] += len(g.edges)
        tot["closure_edges"] += len(derive_edges(s))
        cls_hist.update(x.cls for x in s.symbols)
        rot_hist.update(w.rot for w in s.words)
        deg_hist.update(deg.get(("sym", x.id), 0) for x in s.symbols)

    n = tot["sheets"]
    print(f"sheets: {n}")
    for k in ("symbols", "tagged_symbols", "lines", "dashed", "labelled_lines",
              "words", "placeholder_words", "graph_nodes", "junctions",
              "graph_edges", "closure_edges"):
        print(f"  {k:<18} total {tot[k]:>7}   per sheet {tot[k]/n:8.1f}")
    print(f"  symbol classes seen: {len(cls_hist)} "
          f"(min id {min(cls_hist)}, max id {max(cls_hist)})")
    print(f"  word rotation: {dict(rot_hist)}")
    print(f"  symbol degree in topology: {dict(sorted(deg_hist.items()))}")
    iso = deg_hist.get(0, 0)
    print(f"  isolated symbols: {iso} ({100.0*iso/max(1,tot['symbols']):.1f}%)")


if __name__ == "__main__":
    main()
