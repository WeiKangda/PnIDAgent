"""Pipe connectivity (topology) built on human-verified symbol boxes and pipe lines.

Based on the upstream PnIDAgent/pnid_graph.build_topology (junction aware: corners, T junctions and dead ends become nodes), with one main change:
an X crossing (two pipes crossing in the middle of both) is connected only if a filled dot is drawn there; otherwise the pipes just cross (P&ID convention).
Dot detection (has_dot in build): a filled dot makes the ink along both diagonals through the crossing clearly longer than at a plain cross.
"""
import collections
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np

from pnid_graph_upstream import (angle_diff, in_box, intersect, merge_collinear, point_seg_dist, project,
                                 seg_angle, seg_box_hit, seg_len)


def stroke_width(gray: np.ndarray, segs: Sequence[Tuple[int, int, int, int]]) -> float:
    """Sample along the pipes, measure the ink width perpendicular to each pipe, return the median."""
    ink = gray < 140; H, W = gray.shape; ws = []
    for s in segs:
        L = seg_len(s)
        if L < 40: continue
        ux, uy = (s[2] - s[0]) / L, (s[3] - s[1]) / L; px, py = -uy, ux
        for t in np.linspace(0.2, 0.8, 5):
            x, y = s[0] + ux * L * t, s[1] + uy * L * t; run = 0
            for o in range(-8, 9):
                xi, yi = int(round(x + px * o)), int(round(y + py * o))
                if 0 <= xi < W and 0 <= yi < H and ink[yi, xi]: run += 1
            if 0 < run < 15: ws.append(run)
    return float(np.median(ws)) if ws else 3.0


def dot_mask(gray: np.ndarray, w: float) -> np.ndarray:
    k = max(5, int(round(2.4 * w)) | 1)
    return cv2.morphologyEx((gray < 140).astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))


def bridge_hops(segs, symbols, max_gap=45, tol=12):
    """Hop-over arcs: a pipe is interrupted where it crosses another pipe (drawn as a small jump). If the two parts are collinear, the gap is <= max_gap,
    another pipe runs through the gap and there is no symbol in it -> join them back into one pipe (later treated as an X crossing)."""
    segs = [tuple(int(v) for v in s) for s in segs]; used = set(); out = []
    def boxes_hit(a, b):
        x0, x1 = sorted((a[0], b[0])); y0, y1 = sorted((a[1], b[1]))
        return any(not (bx[2] < x0 - 2 or bx[0] > x1 + 2 or bx[3] < y0 - 2 or bx[1] > y1 + 2) for _, bx in symbols)
    for i, a in enumerate(segs):
        if i in used: continue
        cur = a; merged = True
        while merged:
            merged = False
            for j, b in enumerate(segs):
                if j == i or j in used or angle_diff(seg_angle(cur), seg_angle(b)) > 3: continue
                (tb0, d0), (tb1, d1) = project(b[:2], cur), project(b[2:], cur)
                if max(d0, d1) > 4: continue
                L = seg_len(cur); lo, hi = min(tb0, tb1), max(tb0, tb1)
                gap = lo - L if lo > L else (-hi if hi < 0 else None)
                if gap is None or not (tol < gap <= max_gap): continue
                # the two end points at the gap
                ux, uy = (cur[2] - cur[0]) / L, (cur[3] - cur[1]) / L
                e1 = (cur[0] + ux * (L if lo > L else 0), cur[1] + uy * (L if lo > L else 0))
                e2 = (cur[0] + ux * (lo if lo > L else hi), cur[1] + uy * (lo if lo > L else hi))
                mid = ((e1[0] + e2[0]) / 2, (e1[1] + e2[1]) / 2)
                crossing = any(k not in (i, j) and angle_diff(seg_angle(c), seg_angle(cur)) > 60 and project(mid, c)[1] <= tol
                               and -tol <= project(mid, c)[0] <= seg_len(c) + tol for k, c in enumerate(segs))
                if not crossing or boxes_hit(e1, e2): continue
                ts = [0, L, tb0, tb1]; t0, t1 = min(ts), max(ts)
                cur = (int(round(cur[0] + ux * t0)), int(round(cur[1] + uy * t0)), int(round(cur[0] + ux * t1)), int(round(cur[1] + uy * t1)))
                used.add(j); merged = True
        used.add(i); out.append(cur)
    return out


def trace_stubs(gray, symbols, segs, w, maxlen=320, gap=3):
    """From the four sides of each symbol box, trace continuous solid ink outwards horizontally/vertically (short unlabeled pipe stubs); if the trace reaches a labeled pipe, add the traced piece as a pipe segment.
    Only traces that reach a labeled pipe are accepted (reaching another symbol or breaking off is rejected), so leader lines and text are not taken as pipes."""
    H, W = gray.shape; ink = gray < 150
    seg_arr = [tuple(map(float, s)) for s in segs]
    def on_seg(x, y):
        for s in seg_arr:
            L = seg_len(s)
            if L < 1: continue
            t, d = project((x, y), s)
            if d <= 2.5 and -2 <= t <= L + 2: return s
        return None
    def in_other(x, y, me):
        return any(b[0] - 1 <= x <= b[2] + 1 and b[1] - 1 <= y <= b[3] + 1 for sid, b in symbols if sid != me)
    def thin_at(x, y, dx, dy):
        """Ink width across the trace direction at (x, y) is <= 2w+3 (a line, not a solid blob or text)."""
        px, py = -dy, dx; n = 0
        for o in range(-8, 9):
            xi, yi = int(round(x + px * o)), int(round(y + py * o))
            if 0 <= xi < W and 0 <= yi < H and ink[yi, xi]: n += 1
        return 0 < n <= 2 * w + 3
    out = []
    for sid, b in symbols:
        x0, y0, x1, y1 = b
        sides = [((x1, 0), (1, 0)), ((x0, 0), (-1, 0)), ((0, y1), (0, 1)), ((0, y0), (0, -1))]
        for (fx, fy), (dx, dy) in sides:
            horiz = dx != 0
            lo, hi = (int(y0) + 2, int(y1) - 2) if horiz else (int(x0) + 2, int(x1) - 2)
            starts = []
            for v in range(lo, hi + 1):
                x = (fx + dx * 2) if horiz else v; y = v if horiz else (fy + dy * 2)
                xi, yi = int(round(x)), int(round(y))
                if 0 <= xi < W and 0 <= yi < H and ink[yi, xi]: starts.append(v)
            # group consecutive start points and take the centre of each group (one stub)
            groups = []
            for v in starts:
                if groups and v - groups[-1][-1] <= 1: groups[-1].append(v)
                else: groups.append([v])
            for gr in groups:
                if len(gr) > 2 * w + 3: continue          # too wide, not a pipe
                v = (gr[0] + gr[-1]) / 2.0
                bx, by = ((fx, v) if horiz else (v, fy))
                # a labeled pipe already leaves this side here (within 8 px) -> nothing to add
                if any(point_seg_dist((bx + dx * 4, by + dy * 4), sg) <= 8 for sg in seg_arr): continue
                miss = 0; t = 2; hit = None
                while t < maxlen:
                    x, y = bx + dx * t, by + dy * t
                    xi, yi = int(round(x)), int(round(y))
                    if not (0 <= xi < W and 0 <= yi < H): break
                    if in_other(x, y, sid): break
                    seg_hit = on_seg(x, y)
                    if seg_hit is not None and t > 2: hit = (x, y); break
                    okp = any(0 <= int(round(y + (o if horiz else 0))) < H and 0 <= int(round(x + (0 if horiz else o))) < W and
                              ink[int(round(y + (o if horiz else 0))), int(round(x + (0 if horiz else o)))] for o in (-2, -1, 0, 1, 2))
                    if okp and thin_at(x, y, dx, dy): miss = 0
                    else:
                        miss += 1
                        if miss > gap: break
                    t += 1
                if hit is not None and t >= 8:
                    sx, sy = bx - dx * 3, by - dy * 3          # start 3 px inside the box so the assembler attaches it to this symbol
                    out.append(((int(round(sx)), int(round(sy)), int(round(hit[0])), int(round(hit[1]))), "solid"))
    return out


def extend_dangling(segs, symbols, gray, tol=12, reach=28, skip=()):
    """Dangling pipe ends: no other line near the end point (tol), but another line within reach and ink connects them on the drawing -> extend the end point onto that line.
    (When labeling, corners and T junctions are often 10-20 px apart.)"""
    H, W = gray.shape; ink = gray < 160; segs = [list(map(float, s)) for s in segs]
    def inked(p, q):
        L = float(np.hypot(q[0] - p[0], q[1] - p[1]))
        if L < 2: return True
        ux, uy = (q[0] - p[0]) / L, (q[1] - p[1]) / L; px, py = -uy, ux; n = max(3, int(L)); hit = 0
        for t in np.linspace(0, 1, n):
            x, y = p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t
            if any(0 <= int(round(y + o * py)) < H and 0 <= int(round(x + o * px)) < W and ink[int(round(y + o * py)), int(round(x + o * px))] for o in (-2, -1, 0, 1, 2)): hit += 1
        return hit >= 0.85 * n
    def near_box(p):
        return any(in_box(p, b, 10) for _, b in symbols)
    for i in range(len(segs)):
        for e in (0, 1):
            p = (segs[i][2 * e], segs[i][2 * e + 1])
            if near_box(p) or any(r[0] <= p[0] <= r[2] and r[1] <= p[1] <= r[3] for r in skip): continue
            ok = False
            for j, o in enumerate(segs):
                if j == i: continue
                if min(np.hypot(o[0] - p[0], o[1] - p[1]), np.hypot(o[2] - p[0], o[3] - p[1])) <= tol or point_seg_dist(p, tuple(o)) <= tol:
                    ok = True; break
            if ok: continue
            best = None
            for j, o in enumerate(segs):
                if j == i: continue
                d = point_seg_dist(p, tuple(o))
                if d <= reach and (best is None or d < best[0]):
                    L = seg_len(tuple(o)); t, _ = project(p, tuple(o)); t = max(0.0, min(L, t))
                    q = (o[0] + (o[2] - o[0]) * t / L, o[1] + (o[3] - o[1]) * t / L) if L else (o[0], o[1])
                    best = (d, q)
            if best and inked(p, best[1]):
                segs[i][2 * e], segs[i][2 * e + 1] = best[1]
    return [tuple(int(round(v)) for v in s) for s in segs]


def join_passthrough(segs, symbols, pad=4):
    """Elongated symbols (breakers etc., aspect ratio >= 1.8) that cut a line running through them (a bus bar) into two collinear parts entering from the two long sides
    -> join the parts back into one continuous line (then handled as a side tap: the bus stays continuous and the symbol hangs on it)."""
    segs = [tuple(int(v) for v in s) for s in segs]; used = set(); out = []
    for sid, b in symbols:
        bw, bh = b[2] - b[0], b[3] - b[1]
        if max(bw, bh) < 1.8 * min(bw, bh): continue
        tall = bh > bw
        ends = []   # (segment index, which end point, side it enters from)
        for i, sg in enumerate(segs):
            if i in used: continue
            for e, p, q in ((0, sg[:2], sg[2:]), (1, sg[2:], sg[:2])):
                if not in_box(p, b, pad): continue
                if tall and abs(q[1] - p[1]) <= 3 and abs(q[0] - p[0]) > 5: ends.append((i, e, "L" if q[0] < p[0] else "R", p[1]))
                if not tall and abs(q[0] - p[0]) <= 3 and abs(q[1] - p[1]) > 5: ends.append((i, e, "T" if q[1] < p[1] else "B", p[0]))
        a_side = [x for x in ends if x[2] in ("L", "T")]; b_side = [x for x in ends if x[2] in ("R", "B")]
        for x in a_side:
            for y in b_side:
                if x[0] == y[0] or x[0] in used or y[0] in used or abs(x[3] - y[3]) > 4: continue
                sa, sb = segs[x[0]], segs[y[0]]
                fa = sa[2:] if x[1] == 0 else sa[:2]; fb = sb[2:] if y[1] == 0 else sb[:2]     # far ends of the two parts
                segs[x[0]] = (fa[0], fa[1], fb[0], fb[1]); used.add(y[0])
    return [sg for i, sg in enumerate(segs) if i not in used]


def join_through(segs, boxes, tol=5):
    """Pass-through symbols (condensers etc., pipes only pass through the drawing): two aligned line ends on opposite sides of the box -> join them into one line that passes through."""
    segs = [list(map(float, s)) for s in segs]; used = set(); out = []
    for b in boxes:
        ends = []
        for i, s in enumerate(segs):
            if i in used: continue
            for e in (0, 1):
                x, y = s[2 * e], s[2 * e + 1]
                if not (b[0] - tol <= x <= b[2] + tol and b[1] - tol <= y <= b[3] + tol): continue
                side = "T" if abs(y - b[1]) <= tol else "B" if abs(y - b[3]) <= tol else "L" if abs(x - b[0]) <= tol else "R" if abs(x - b[2]) <= tol else None
                if side: ends.append((i, e, side, x, y))
        paired = set()
        for a in ends:
            if a[0] in paired: continue
            for c in sorted(ends, key=lambda c: abs(c[3] - a[3]) + abs(c[4] - a[4]) if False else 0):
                if a[0] == c[0] or a[0] in used or c[0] in used or c[0] in paired: continue
                if (a[2], c[2]) == ("T", "B") and abs(a[3] - c[3]) <= tol or (a[2], c[2]) == ("L", "R") and abs(a[4] - c[4]) <= tol:
                    sa, sc = segs[a[0]], segs[c[0]]
                    fa = sa[2:] if a[1] == 0 else sa[:2]; fc = sc[2:] if c[1] == 0 else sc[:2]
                    segs[a[0]] = [fa[0], fa[1], fc[0], fc[1]]; used.add(c[0]); paired.add(a[0]); break
    return [tuple(int(round(v)) for v in s) for i, s in enumerate(segs) if i not in used]


def build(segs: Sequence[Tuple[Tuple[int, int, int, int], str]], symbols: Sequence[Tuple[object, Tuple[int, int, int, int]]],
          gray: np.ndarray, tol: int = 12, pad: int = 8, decisions=None, trace: bool = True, vec_dots=None, pass_through=(), fine=None):
    """decisions: [{"x","y","connected"}] human decisions for X crossings; a decision within 8 px overrides the automatic one."""
    """Returns (nodes {key: (x, y)}, edges [(u, v, kind)], crossings [(x, y, connected)])."""
    by_kind: Dict[str, list] = collections.defaultdict(list)
    for s, k in segs: by_kind[k].append(tuple(int(v) for v in s))
    FR, FT = (fine or ([], tol))
    def T(x, y):     # smaller snapping distance inside fine regions (closely spaced parallel lines)
        return FT if any(r[0] <= x <= r[2] and r[1] <= y <= r[3] for r in FR) else tol
    PT = set(pass_through); pt_boxes = [b for sid, b in symbols if sid in PT]
    if pt_boxes: by_kind = {k: join_through(merge_collinear(ss), pt_boxes) for k, ss in by_kind.items()}
    segs = [(s, k) for k, ss in by_kind.items() for s in extend_dangling(bridge_hops(join_passthrough(merge_collinear(ss), symbols), symbols), symbols, gray, tol=tol, skip=FR)]
    w0 = stroke_width(gray, [s for s, _ in segs])
    stubs = trace_stubs(gray, [t for t in symbols if t[0] not in PT], [s for s, _ in segs], w0) if trace else []
    segs = segs + stubs
    w = stroke_width(gray, [s for s, _ in segs]); dots = dot_mask(gray, w); H, W = gray.shape
    r = int(max(4, 2.5 * w))

    inkm = (gray < 140)

    def thick(seg, x, y):
        """Ink thickness of this line just outside the crossing (15-30 px away): median length of the continuous ink run across the line."""
        L = seg_len(seg); ux, uy = (seg[2] - seg[0]) / L, (seg[3] - seg[1]) / L; px, py = -uy, ux; res = []
        for dd in (15, 22, 30):
            for sg in (-1, 1):
                cx, cy = x + sg * ux * dd, y + sg * uy * dd; run = 0
                for o in range(-10, 11):
                    xi, yi = int(round(cx + px * o)), int(round(cy + py * o))
                    if 0 <= xi < W and 0 <= yi < H and inkm[yi, xi]: run += 1
                res.append(run)
        return float(np.median(res))

    def diag_run(xi, yi, dx, dy):
        n_ = 0
        for o in range(-15, 16):
            X, Y = xi + dx * o, yi + dy * o
            if 0 <= X < W and 0 <= Y < H and inkm[Y, X]: n_ += 1
        return n_

    def has_dot(x, y, a=None, b=None):
        """Filled dot at the crossing: a dot makes the ink along both diagonals clearly longer than at a plain cross (about 1.4 x the thicker line)."""
        t = max(thick(a, x, y), thick(b, x, y)) if a is not None else w
        need = 1.35 * t + 1.5
        for ox in (-2, -1, 0, 1, 2):
            for oy in (-2, -1, 0, 1, 2):
                xi, yi = int(round(x)) + ox, int(round(y)) + oy
                if min(diag_run(xi, yi, 1, 1), diag_run(xi, yi, 1, -1)) >= need: return True
        return False

    # crossings inside a symbol box do not count (strokes of the symbol itself)
    def in_any_box(x, y):
        return any(b[0] <= x <= b[2] and b[1] <= y <= b[3] for _, b in symbols)

    pts: List[Tuple[float, float]] = []

    def point_of(x, y):
        best, bd = -1, T(x, y) + 1e-9
        for i, (px, py) in enumerate(pts):
            d = np.hypot(px - x, py - y)
            if d <= bd: best, bd = i, d
        if best >= 0: return best
        pts.append((float(x), float(y))); return len(pts) - 1

    ends = [(point_of(*s[:2]), point_of(*s[2:])) for s, _ in segs]
    crossings = []
    for i in range(len(segs)):
        a = segs[i][0]
        for j in range(i + 1, len(segs)):
            b = segs[j][0]
            if angle_diff(seg_angle(a), seg_angle(b)) < 5.0: continue
            x = intersect(a, b)
            if x is None: continue
            (ta, _), (tb, _) = project(x, a), project(x, b); tx = T(*x)
            if not (-tx <= ta <= seg_len(a) + tx and -tx <= tb <= seg_len(b) + tx): continue
            mid_a, mid_b = tx < ta < seg_len(a) - tx, tx < tb < seg_len(b) - tx
            if mid_a and mid_b:                       # true X crossing: in the middle of both segments
                if in_any_box(*x): continue
                hit = [d for d in (decisions or []) if abs(d["x"] - x[0]) <= 8 and abs(d["y"] - x[1]) <= 8]
                vreg = vec_dots and any(rg[0] <= x[0] <= rg[2] and rg[1] <= x[1] <= rg[3] for rg in vec_dots[0])
                if hit: ok = bool(hit[0]["connected"])
                elif vreg: ok = any(np.hypot(dx - x[0], dy - x[1]) <= 6 for dx, dy in vec_dots[1])   # vector region: look for a filled dot in the PDF
                else: ok = has_dot(x[0], x[1], a, b)
                crossings.append((x[0], x[1], ok))
                if not ok: continue                    # no dot = pipes only cross, not connected
            point_of(*x)                              # T junction / corner / X crossing with a dot

    # below as upstream: interior points, symbol attachment, edges in order along each line, contraction of degree-2 points
    interior = collections.defaultdict(list)
    for si, (s, _) in enumerate(segs):
        L = seg_len(s)
        for pid, p in enumerate(pts):
            if pid in ends[si]: continue
            t, d = project(p, s); tp = T(*p)
            if d <= tp and -tp <= t <= L + tp:
                # if this point is an X crossing judged as "only crossing", it must not become a connection point on this line
                if tp < t < L - tp and any(abs(p[0] - cx) < tp and abs(p[1] - cy) < tp and not ok for cx, cy, ok in crossings):
                    continue
                interior[si].append(pid)
    pt_use = collections.Counter()
    for si in range(len(segs)):
        for pid in set((*ends[si], *interior[si])): pt_use[pid] += 1
    sym_on = collections.defaultdict(list); taps = collections.defaultdict(list)
    SNAP, GAPMAX = 15, 30
    inkm2 = gray < 160

    def box_dist(p, b):
        return float(np.hypot(max(b[0] - p[0], 0, p[0] - b[2]), max(b[1] - p[1], 0, p[1] - b[3])))

    def inked(p, q):
        """Is there a continuous ink line from p to q (+-1 px across, ink at 85% of sample points)?"""
        L = float(np.hypot(q[0] - p[0], q[1] - p[1]))
        if L < 2: return True
        ux, uy = (q[0] - p[0]) / L, (q[1] - p[1]) / L; px, py = -uy, ux; n = max(3, int(L)); hit = 0
        for t in np.linspace(0, 1, n):
            x, y = p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t
            if any(0 <= int(round(y + o * py)) < H and 0 <= int(round(x + o * px)) < W and inkm2[int(round(y + o * py)), int(round(x + o * px))] for o in (-1, 0, 1)): hit += 1
        return hit >= 0.85 * n

    def entry(p, d, b):
        """Distance from p along direction d to where the ray enters box b; None if it does not enter."""
        best = None
        for t in np.linspace(0, GAPMAX, int(GAPMAX) + 1):
            x, y = p[0] + d[0] * t, p[1] + d[1] * t
            if b[0] <= x <= b[2] and b[1] <= y <= b[3]: return (x, y), t
        return best

    for si, (s, _) in enumerate(segs):
        # 1) pipe enters or touches the symbol box (upstream logic, pad 2)
        for sid, box in symbols:
            if seg_box_hit(s, box, 2):
                if sid in PT:      # pass-through symbol: only a line end stopping in/at the box connects; a line passing through does not
                    if any(in_box(e, box, 2) for e in (s[:2], s[2:])): sym_on[si].append((sid, box))
                    continue
                bw, bh = box[2] - box[0], box[3] - box[1]; m = 0.3 * min(bw, bh)
                core = (box[0] + m, box[1] + m, box[2] - m, box[3] - m)       # inner part of the box
                ends_in = any(in_box(e, box, 2) for e in (s[:2], s[2:]))
                if seg_box_hit(s, core, 0) or ends_in: sym_on[si].append((sid, box))   # passes through / ends at the box -> symbol is in series on the line
                else: taps[si].append((sid, box))                                  # only touches the box edge -> a branch from the line to the symbol
        # 2) dead end: the first symbol box hit within GAPMAX straight ahead along the pipe, with ink in between (or distance <= pad) -> attach
        L = seg_len(s)
        for pid, p, q in zip(ends[si], (s[:2], s[2:]), (s[2:], s[:2])):
            if pt_use[pid] > 1 or L < 1: continue
            d = ((p[0] - q[0]) / L, (p[1] - q[1]) / L)
            hits = []
            for sid, b in symbols:
                e = entry(p, d, b)
                if e and (e[1] <= pad or inked(p, e[0])): hits.append((e[1], sid, b))
            if hits:
                _, sid, b = min(hits, key=lambda h: h[0])
                if (sid, b) not in sym_on[si]: sym_on[si].append((sid, b))
                continue
            # not straight ahead: a box within GAPMAX of the end, with ink from the end to its nearest point (pipe turns into a nearby valve) -> attach the nearest
            side = []
            for sid, b in symbols:
                qx, qy = min(max(p[0], b[0]), b[2]), min(max(p[1], b[1]), b[3]); dd = float(np.hypot(qx - p[0], qy - p[1]))
                if dd <= GAPMAX and (dd <= pad or inked(p, (qx, qy))): side.append((dd, sid, b))
            if side:
                _, sid, b = min(side, key=lambda h: h[0])
                if (sid, b) not in sym_on[si]: sym_on[si].append((sid, b))
                continue
            near = [(sid, b) for sid, b in symbols if in_box(p, b, SNAP)]
            if len(near) == 1 and near[0] not in sym_on[si]: sym_on[si].append(near[0])
    # 3) side stub: a pipe passes beside a symbol (<= GAPMAX), the symbol centre faces the pipe and a continuous ink line joins them (a short pipe that was not labeled) -> attach
    for sid, b in symbols:
        if sid in PT: continue
        c = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
        for si, (s, _) in enumerate(segs):
            if (sid, b) in sym_on[si] or (sid, b) in taps[si]: continue
            L = seg_len(s)
            if L < 1: continue
            t, dist = project(c, s)
            if not (0 <= t <= L): continue
            ux, uy = (s[2] - s[0]) / L, (s[3] - s[1]) / L; foot = (s[0] + ux * t, s[1] + uy * t)
            dd = (c[0] - foot[0], c[1] - foot[1]); n_ = float(np.hypot(*dd))
            if n_ < 1: continue
            e = entry(foot, (dd[0] / n_, dd[1] / n_), b)
            if e is None or e[1] < 3: continue
            if (sid, b) in taps[si]: continue
            # the symbol already reaches this position through another line (e.g. a vertical line from the box crossing this bus) -> do not add again
            if any((sid, b) in sym_on[sj] and point_seg_dist(foot, segs[sj][0]) <= tol for sj in range(len(segs)) if sj != si): continue
            if not any(other != (sid, b) and in_box(((foot[0] + e[0][0]) / 2, (foot[1] + e[0][1]) / 2), other[1], 0) for other in symbols) and inked(foot, e[0]):
                taps[si].append((sid, b))
    adj = collections.defaultdict(list); pos = {}   # adj[u] = [(v, kind, segment pieces)]
    for si, (s, kind) in enumerate(segs):
        items = []
        for pid in set((*ends[si], *interior[si])):
            if pt_use[pid] <= 1 and any(in_box(pts[pid], box, SNAP) for _, box in sym_on[si]): continue
            t, _ = project(pts[pid], s); items.append((t, ("jct", pid))); pos[("jct", pid)] = pts[pid]
        for sid, box in sym_on[si]:
            c = ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0); t, _ = project(c, s); items.append((t, ("sym", sid))); pos[("sym", sid)] = c
        Ls = seg_len(s)
        for sid, box in taps[si]:
            if (sid, box) in sym_on[si]: continue
            # this line only crosses the symbol's own lead (the symbol is already connected by a lead crossing this line) -> a crossing, not a branch
            crossing_own_lead = False
            for sj, lst in sym_on.items():
                if sj == si or (sid, box) not in lst: continue
                o = segs[sj][0]
                if angle_diff(seg_angle(o), seg_angle(s)) < 30: continue
                xx = intersect(o, s)
                if xx is None: continue
                (ta, _), (tb, _) = project(xx, o), project(xx, s)
                Lo = seg_len(o)
                if not (0 < tb < Ls) or not (-(2 * tol + 6) <= ta <= Lo + 2 * tol + 6): continue
                if tol < ta < Lo - tol: crossing_own_lead = True; break     # the lead crosses this line in its middle -> crossing
                # lead end point near this line: it is a crossing only if this line lies between the lead end and the symbol (the lead comes from the other side and jumps the line);
                # if the lead lies between the symbol and the line (reaching from the symbol to the line), it is a T junction and connected.
                e = o[:2] if abs(ta) <= abs(ta - Lo) else o[2:]
                c = ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)
                if np.hypot(e[0] - c[0], e[1] - c[1]) > np.hypot(xx[0] - c[0], xx[1] - c[1]) + 1: crossing_own_lead = True; break
            if crossing_own_lead: continue
            c = ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0); t, _ = project(c, s); t = max(0.0, min(Ls, t))
            foot = (s[0] + (s[2] - s[0]) * t / Ls, s[1] + (s[3] - s[1]) * t / Ls) if Ls else (s[0], s[1])
            tk = ("jct", f"tap{si}_{sid}"); pos[tk] = foot; pos[("sym", sid)] = c; items.append((t, tk))
            qx, qy = min(max(foot[0], box[0]), box[2]), min(max(foot[1], box[1]), box[3])
            if all(n != ("sym", sid) for n, _, _ in adj[tk]):
                adj[tk].append((("sym", sid), kind, (*foot, qx, qy))); adj[("sym", sid)].append((tk, kind, (*foot, qx, qy)))
        items.sort(key=lambda t: t[0])
        L = seg_len(s); ux, uy = ((s[2] - s[0]) / L, (s[3] - s[1]) / L) if L else (0, 0)
        P = lambda t: (s[0] + ux * max(0, min(L, t)), s[1] + uy * max(0, min(L, t)))
        for (tu, u), (tv, v) in zip(items, items[1:]):
            if u != v and all(n != v or k != kind for n, k, _ in adj[u]):
                piece = (*P(tu), *P(tv)); adj[u].append((v, kind, piece)); adj[v].append((u, kind, piece))
    for sid, box in symbols:
        pos.setdefault(("sym", sid), ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)); adj.setdefault(("sym", sid), [])
    # adjacent symbols in series (walking along the axis): a pipe enters symbol A from one side; continuing along the same straight line through A, if the next symbol B directly adjacent lies on this line
    # (the axis passes through the middle of B, not a corner), A and B are connected, and the walk continues from B. Horizontal/vertical pipes only.
    GAP = 12
    def walk(seg, sid0, box0):
        L = seg_len(seg)
        if L < 1: return
        ux, uy = (seg[2] - seg[0]) / L, (seg[3] - seg[1]) / L
        if not (abs(ux) > 0.98 or abs(uy) > 0.98): return
        horiz = abs(ux) > 0.98
        # which side the pipe enters A from: the end point nearer A sets the direction (walk away from the pipe)
        c0 = ((box0[0] + box0[2]) / 2, (box0[1] + box0[3]) / 2)
        e = min((seg[:2], seg[2:]), key=lambda p: np.hypot(p[0] - c0[0], p[1] - c0[1]))
        other = seg[2:] if e == seg[:2] else seg[:2]
        if horiz: d = 1 if e[0] > other[0] else -1; axis = e[1]
        else: d = 1 if e[1] > other[1] else -1; axis = e[0]
        cur_sid, cur = sid0, box0
        for _ in range(6):
            edge = (cur[2] if d > 0 else cur[0]) if horiz else (cur[3] if d > 0 else cur[1])
            nxt = None
            for sid, b in symbols:
                if sid == cur_sid: continue
                lo, hi = (b[1], b[3]) if horiz else (b[0], b[2]); span = hi - lo
                if not (lo + 0.08 * span <= axis <= hi - 0.08 * span): continue     # axis passes through B (not a corner)
                near = (b[0] - edge if d > 0 else edge - b[2]) if horiz else (b[1] - edge if d > 0 else edge - b[3])
                if -0.5 * ((b[2] - b[0]) if horiz else (b[3] - b[1])) <= near <= GAP and (nxt is None or near < nxt[0]): nxt = (near, sid, b)
            if nxt is None: return
            _, nsid, nb = nxt; u, v = ("sym", cur_sid), ("sym", nsid)
            if all(n != v for n, _, _ in adj[u]):
                cu = ((cur[0] + cur[2]) / 2, (cur[1] + cur[3]) / 2); cv = ((nb[0] + nb[2]) / 2, (nb[1] + nb[3]) / 2)
                adj[u].append((v, "solid", (*cu, *cv))); adj[v].append((u, "solid", (*cu, *cv))); pos[u], pos[v] = cu, cv
            cur_sid, cur = nsid, nb
    for si, lst in sym_on.items():
        s0 = segs[si][0]
        for sid, box in lst:
            # start only from pipes that end at the edge of box A (pipes through A are already linked by the ordering along the line)
            if any(in_box(p, box, SNAP) for p in (s0[:2], s0[2:])) and not seg_box_hit(s0, (box[0] + 3, box[1] + 3, box[2] - 3, box[3] - 3), 0):
                walk(s0, sid, box)
    keep = lambda n: n[0] == "sym" or len(adj[n]) != 2
    edges = {}
    for u in list(adj):
        if not keep(u): continue
        for first, kind, piece in adj[u]:
            prev, cur, k, g, path = u, first, kind, 0, [piece]
            while not keep(cur) and g < 100000:
                g += 1; nxt = [(n, kk, pp) for n, kk, pp in adj[cur] if n != prev]
                if not nxt: break
                prev, (cur, kk, pp) = cur, nxt[0]; path.append(pp)
                if kk == "dashed": k = "dashed"
            if keep(cur) and cur != u:
                a, b = sorted([u, cur], key=lambda n: (n[0], str(n[1])))
                edges.setdefault((a, b, k), path)
    nodes = {n: pos[n] for n in adj if keep(n) and n in pos}
    deg = collections.Counter()
    for (a, b, k) in edges: deg[a] += 1; deg[b] += 1
    box_of = dict(symbols)
    for (a, b, k) in list(edges):
        for j, o in ((a, b), (b, a)):
            if j[0] == "jct" and deg[j] == 1 and o[0] == "sym" and in_box(nodes[j], box_of[o[1]], pad + 4):
                edges.pop((a, b, k), None); nodes.pop(j, None)
    E = sorted(edges, key=lambda e: (str(e[0]), str(e[1])))
    build.last_stubs = stubs
    return nodes, E, crossings, segs, w, [edges[e] for e in E]
