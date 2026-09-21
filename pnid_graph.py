#!/usr/bin/env python3
"""Pipe topology from line segments and symbol boxes.

Builds the graph a P&ID actually encodes: nodes are symbols plus genuine
junctions (branches, corners, dead ends), edges are the pipe runs between
consecutive nodes.  Degree-2 pass-through points are contracted away.

Why not chain lines end-to-end and link consecutive symbols (the previous
digitize_pnid approach)?  On Dataset-P&ID 189 of 205 pipe runs on a sheet touch
a T or X junction, and an endpoint-to-endpoint chain cannot represent a branch
at all: every T splits into separate chains that are never linked.  Measured
with perfect symbol and line inputs on 56 sheets, chaining recovers 7.7% of the
true edges; this assembler recovers 100%.

Steps
  1. merge collinear segments that overlap or abut (Hough duplicates)
  2. junction points: shared endpoints (corners), an endpoint on another
     segment's interior (T), two segments crossing (X)
  3. attach symbols: a segment that enters the box, or whose dead-end endpoint
     stops within `pad` px of the box (Hough segments end at the outline)
  4. order the items along each segment, link neighbours, contract degree-2
     junctions

All geometry is in the caller's pixel space.  Node keys are ("sym", symbol_id)
and ("jct", int); positions are (x, y).
"""
from __future__ import annotations

import collections
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

Seg = Tuple[int, int, int, int]
Box = Tuple[int, int, int, int]
Node = Tuple[str, object]


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------

def seg_len(s: Seg) -> float:
    return float(np.hypot(s[2] - s[0], s[3] - s[1]))


def seg_angle(s: Seg) -> float:
    return float(np.degrees(np.arctan2(s[3] - s[1], s[2] - s[0])) % 180.0)


def angle_diff(a: float, b: float) -> float:
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def project(p: Tuple[float, float], s: Seg) -> Tuple[float, float]:
    """(t along s in px from s[0:2], perpendicular distance)."""
    L = seg_len(s)
    if L < 1e-6:
        return 0.0, float(np.hypot(p[0] - s[0], p[1] - s[1]))
    ux, uy = (s[2] - s[0]) / L, (s[3] - s[1]) / L
    dx, dy = p[0] - s[0], p[1] - s[1]
    return ux * dx + uy * dy, abs(-uy * dx + ux * dy)


def point_seg_dist(p: Tuple[float, float], s: Seg) -> float:
    t, d = project(p, s)
    L = seg_len(s)
    if t < 0:
        return float(np.hypot(p[0] - s[0], p[1] - s[1]))
    if t > L:
        return float(np.hypot(p[0] - s[2], p[1] - s[3]))
    return d


def seg_box_hit(s: Seg, box: Box, pad: int) -> bool:
    """Segment crosses / enters the padded box (Liang-Barsky clip)."""
    x1, y1, x2, y2 = box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad
    dx, dy = s[2] - s[0], s[3] - s[1]
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, s[0] - x1), (dx, x2 - s[0]), (-dy, s[1] - y1), (dy, y2 - s[1])):
        if p == 0:
            if q < 0:
                return False
            continue
        r = q / p
        if p < 0:
            if r > t1:
                return False
            t0 = max(t0, r)
        else:
            if r < t0:
                return False
            t1 = min(t1, r)
    return t0 <= t1


def intersect(a: Seg, b: Seg) -> Optional[Tuple[float, float]]:
    """Intersection of the infinite lines through a and b, or None if parallel."""
    x1, y1, x2, y2 = a
    x3, y3, x4, y4 = b
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-9:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
    return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))


def in_box(p: Tuple[float, float], box: Box, pad: int) -> bool:
    return box[0] - pad <= p[0] <= box[2] + pad and box[1] - pad <= p[1] <= box[3] + pad


# ---------------------------------------------------------------------------
# step 1: collinear merge
# ---------------------------------------------------------------------------

def merge_collinear(segs: Sequence[Seg], ang_tol: float = 3.0, perp_tol: float = 6.0,
                    gap_tol: float = 12.0) -> List[Seg]:
    """Union of segments that are collinear and overlap or nearly abut.

    Never joins segments separated by a gap wider than `gap_tol`: that gap is
    usually a symbol, and the symbol step needs to see the two ends.
    """
    segs = [tuple(int(v) for v in s) for s in segs if seg_len(s) >= 1]
    n = len(segs)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        a = segs[i]
        for j in range(i + 1, n):
            b = segs[j]
            if angle_diff(seg_angle(a), seg_angle(b)) > ang_tol:
                continue
            (ta, da), (tb, db) = project(b[:2], a), project(b[2:], a)
            if max(da, db) > perp_tol:
                continue
            La = seg_len(a)
            lo, hi = min(ta, tb), max(ta, tb)
            if hi < -gap_tol or lo > La + gap_tol:
                continue
            parent[find(i)] = find(j)

    groups: Dict[int, List[Seg]] = collections.defaultdict(list)
    for i in range(n):
        groups[find(i)].append(segs[i])

    out: List[Seg] = []
    for members in groups.values():
        if len(members) == 1:
            out.append(members[0])
            continue
        ref = max(members, key=seg_len)
        ts = [project(s[:2], ref)[0] for s in members] + [project(s[2:], ref)[0] for s in members]
        L = seg_len(ref)
        ux, uy = (ref[2] - ref[0]) / L, (ref[3] - ref[1]) / L
        tmin, tmax = min(ts), max(ts)
        out.append((int(round(ref[0] + ux * tmin)), int(round(ref[1] + uy * tmin)),
                    int(round(ref[0] + ux * tmax)), int(round(ref[1] + uy * tmax))))
    return out


# ---------------------------------------------------------------------------
# steps 2-4: topology
# ---------------------------------------------------------------------------

def build_topology(segs: Sequence[Tuple[Seg, str]], symbols: Sequence[Tuple[object, Box]],
                   tol: int = 12, pad: int = 6, merge: bool = True, min_len: float = 0.0,
                   drop_inbox_ends: bool = True
                   ) -> Tuple[Dict[Node, Tuple[float, float]], List[Tuple[Node, Node, str]]]:
    """Graph from segments [(seg, kind)] and symbols [(id, box)].

    tol   endpoint snapping radius, px (3 for exact geometry, 10-20 for Hough)
    pad   how far outside its box a symbol still claims a dead-end segment
    drop_inbox_ends
          a segment that merely ends inside a symbol box terminates at the
          symbol instead of at a dead-end node.  Set False to keep such nodes
          (Dataset-P&ID ground truth does, when the segment runs past the
          symbol centre).

    Returns (nodes {key: (x, y)}, edges [(u, v, kind)]).
    """
    if merge:
        by_kind: Dict[str, List[Seg]] = collections.defaultdict(list)
        for s, k in segs:
            by_kind[k].append(tuple(int(v) for v in s))
        segs = [(s, k) for k, ss in by_kind.items() for s in merge_collinear(ss)]
    segs = [(tuple(int(v) for v in s), k) for s, k in segs if seg_len(s) >= min_len]

    # --- 2. junction points --------------------------------------------------
    pts: List[Tuple[float, float]] = []

    def point_of(x: float, y: float) -> int:
        best, bd = -1, tol + 1e-9
        for i, (px, py) in enumerate(pts):
            d = np.hypot(px - x, py - y)
            if d <= bd:
                best, bd = i, d
        if best >= 0:
            return best
        pts.append((float(x), float(y)))
        return len(pts) - 1

    ends = [(point_of(*s[:2]), point_of(*s[2:])) for s, _ in segs]

    # proper crossings: GT splits every run at a crossing and shares the point,
    # so a crossing is a degree-4 node; merged or predicted runs do not end
    # there, so the point has to be constructed.
    for i in range(len(segs)):
        a = segs[i][0]
        for j in range(i + 1, len(segs)):
            b = segs[j][0]
            if angle_diff(seg_angle(a), seg_angle(b)) < 5.0:
                continue
            x = intersect(a, b)
            if x is None:
                continue
            (ta, _), (tb, _) = project(x, a), project(x, b)
            if -tol <= ta <= seg_len(a) + tol and -tol <= tb <= seg_len(b) + tol:
                point_of(*x)

    interior: Dict[int, List[int]] = collections.defaultdict(list)
    for si, (s, _) in enumerate(segs):
        L = seg_len(s)
        for pid, p in enumerate(pts):
            if pid in ends[si]:
                continue
            t, d = project(p, s)
            if d <= tol and -tol <= t <= L + tol:
                interior[si].append(pid)

    pt_use: Dict[int, int] = collections.Counter()
    for si in range(len(segs)):
        for pid in set((*ends[si], *interior[si])):
            pt_use[pid] += 1

    # --- 3. symbols ----------------------------------------------------------
    # A segment that enters the box itself, or whose dead-end endpoint stops
    # within `pad` of it.  An endpoint that is already a junction of other
    # segments does not attach: a valve drawn just below a T sits on the
    # branch, not on the header.
    sym_on: Dict[int, List[Tuple[object, Box]]] = collections.defaultdict(list)
    for sid, box in symbols:
        for si, (s, _) in enumerate(segs):
            if seg_box_hit(s, box, 1):
                sym_on[si].append((sid, box))
                continue
            for pid, p in zip(ends[si], (s[:2], s[2:])):
                if pt_use[pid] <= 1 and in_box(p, box, pad):
                    sym_on[si].append((sid, box))
                    break

    # --- 4. chain along each segment ---------------------------------------
    adj: Dict[Node, List[Tuple[Node, str]]] = collections.defaultdict(list)
    pos: Dict[Node, Tuple[float, float]] = {}
    for si, (s, kind) in enumerate(segs):
        items: List[Tuple[float, Node]] = []
        for pid in set((*ends[si], *interior[si])):
            if (drop_inbox_ends and pt_use[pid] <= 1
                    and any(in_box(pts[pid], box, pad) for _, box in sym_on[si])):
                continue
            t, _ = project(pts[pid], s)
            items.append((t, ("jct", pid)))
            pos[("jct", pid)] = pts[pid]
        for sid, box in sym_on[si]:
            c = ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)
            t, _ = project(c, s)
            items.append((t, ("sym", sid)))
            pos[("sym", sid)] = c
        items.sort(key=lambda t: t[0])
        for (_, u), (_, v) in zip(items, items[1:]):
            if u != v and (v, kind) not in adj[u]:
                adj[u].append((v, kind))
                adj[v].append((u, kind))

    for sid, box in symbols:            # isolated symbols stay nodes
        pos.setdefault(("sym", sid), ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0))
        adj.setdefault(("sym", sid), [])

    def keep(n: Node) -> bool:
        return n[0] == "sym" or len(adj[n]) != 2

    edges: set = set()
    for u in list(adj):
        if not keep(u):
            continue
        for first, kind in adj[u]:
            prev, cur, k = u, first, kind
            guard = 0
            while not keep(cur) and guard < 100000:
                guard += 1
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
    return nodes, sorted(edges, key=lambda e: (str(e[0]), str(e[1])))
