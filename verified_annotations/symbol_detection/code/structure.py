"""Pipe structure cues (no annotations used, image only):
  1. Pipe extraction: open ink with long horizontal/vertical kernels -> connected components -> horizontal/vertical segments (end points, thickness).
  2. Structure features (per candidate box): how many segment end points stop facing a box edge / how many edges are joined by lines / whether a pair of collinear segments clamps it left-right or top-bottom (in-line symbol)
     / whether a line passes straight through the box (annotation convention "lines do not cross symbols": crossing -> likely not a symbol) / fraction of the box covered by lines.
  3. Gap proposals: two collinear segments face each other across a gap that contains graphic ink -> generate a candidate box at the gap (recovers in-line symbols the detector missed).
Human pipe annotations of the A half are only used to check extraction quality, not in the features."""
import glob, json, os, sys
import cv2, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from selfsim import SRC, sp


def segments(g, sc, Lmm=30, maxth=12, ink=None):
    """Return (H, V): horizontal segments [x0, x1, y, thickness], vertical segments [y0, y1, x, thickness] (original pixels). Lmm: minimum line length (pixels at 3300 scale).
    ink: optional mask from learned pipe segmentation (pipe_seg.py); segments are taken only from pipe pixels."""
    ink = (g < 160).astype(np.uint8) if ink is None else ink.astype(np.uint8); L = max(12, int(Lmm / sc)); th = maxth / sc
    out = []
    for k, ax in (((1, L), 0), ((L, 1), 1)):
        m = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones(k, np.uint8))
        n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
        segs = []
        for i in range(1, n):
            x, y, w, h, a = st[i]
            if ax == 0 and h <= th: segs.append([x, x + w - 1, y + h / 2, h])
            if ax == 1 and w <= th: segs.append([y, y + h - 1, x + w / 2, w])
        out.append(np.array(segs, float).reshape(-1, 4))
    return out[0], out[1], ink


def struct_feats(B, H, V, sc):
    """B: N x 4 boxes. Returns N x 9 structure features."""
    B = np.asarray(B, float).reshape(-1, 4); N = len(B)
    if N == 0: return np.zeros((0, 9))
    w, h = B[:, 2] - B[:, 0], B[:, 3] - B[:, 1]; m = np.maximum(3 / sc, 0.15 * np.minimum(w, h))[:, None]
    def touch(S, lo, hi, cross0, cross1, near_lo):
        """Segments of S span [lo, hi] (along the line) at position c (perpendicular). Returns: counts of end points at the box's 'low side' (entering the box) / 'high side', and the count passing through."""
        if not len(S): z = np.zeros(N); return z, z, z
        s0, s1, c = S[:, 0][None], S[:, 1][None], S[:, 2][None]
        inc = (c >= cross0[:, None] - m) & (c <= cross1[:, None] + m)            # line within the box's cross-section range
        inner = (c >= cross0[:, None] + 0.2 * (cross1 - cross0)[:, None]) & (c <= cross1[:, None] - 0.2 * (cross1 - cross0)[:, None])
        end_lo = inc & (np.abs(s1 - lo[:, None]) <= m + 0.25 * (hi - lo)[:, None]) & (s0 < lo[:, None] - m)   # comes from the low side, ends near the low-side edge
        end_hi = inc & (np.abs(s0 - hi[:, None]) <= m + 0.25 * (hi - lo)[:, None]) & (s1 > hi[:, None] + m)   # comes from the high side
        thru = inner & (s0 < lo[:, None] - m) & (s1 > hi[:, None] + m)
        return end_lo.sum(1), end_hi.sum(1), thru.sum(1)
    hl, hr, ht = touch(H, B[:, 0], B[:, 2], B[:, 1], B[:, 3], True)           # horizontal lines: along x, cross-section y
    vt, vb, vt2 = touch(V, B[:, 1], B[:, 3], B[:, 0], B[:, 2], True)          # vertical lines: along y, cross-section x
    sides = (hl > 0).astype(int) + (hr > 0) + (vt > 0) + (vb > 0)
    inline = ((hl > 0) & (hr > 0)) | ((vt > 0) & (vb > 0))
    return np.column_stack([hl + hr + vt + vb, sides, inline, ht + vt2, (hl > 0) & (hr > 0), (vt > 0) & (vb > 0),
                            (ht + vt2 > 0) & ~inline, np.log1p(w * sc), np.log1p(h * sc)]).astype(float)


def gap_proposals(H, V, ink, sc, tol=3, gmin=8, gmax=260, minink=12):
    """Gap between two collinear segments facing each other -> candidate box. The gap (excluding the line itself) must have enough ink."""
    out = []; Hh, Ww = ink.shape; tol, gmin, gmax, minink = tol / sc, gmin / sc, gmax / sc, minink / sc / sc
    for S, horiz in ((H, True), (V, False)):
        if len(S) < 2: continue
        o = np.argsort(S[:, 2]); S = S[o]
        for i in range(len(S)):
            a = S[i]; j = i + 1
            best = None
            while j < len(S) and S[j, 2] - a[2] <= tol:
                b = S[j]
                for p, q in ((a, b), (b, a)):
                    gap = q[0] - p[1]
                    if gmin <= gap <= gmax and (best is None or gap < best[0]): best = (gap, p, q)
                j += 1
            k = i - 1
            while k >= 0 and a[2] - S[k, 2] <= tol:
                b = S[k]; gap = b[0] - a[1]
                if gmin <= gap <= gmax and (best is None or gap < best[0]): best = (gap, a, b)
                k -= 1
            if best is None: continue
            gap, p, q = best; c = (p[2] + q[2]) / 2; s0, s1 = int(p[1]) + 1, int(q[0])
            R = int(max(gap, 20 / sc)); c0, c1 = int(max(0, c - R)), int(c + R + 1)
            reg = ink[c0:c1, s0:s1] if horiz else ink[s0:s1, c0:c1].T          # rows = cross-section direction, columns = along the line
            if reg.size == 0: continue
            prof = reg.sum(1)                                                   # ink of each row in the cross-section direction
            mid = int(round(c - c0)); lo = hi = min(max(mid, 0), len(prof) - 1)
            while lo > 0 and prof[lo - 1] > 0: lo -= 1
            while hi < len(prof) - 1 and prof[hi + 1] > 0: hi += 1
            if reg[lo:hi + 1].sum() < minink or hi - lo < 4 / sc: continue
            if horiz: out.append([s0, c0 + lo, s1, c0 + hi + 1])
            else: out.append([c0 + lo, s0, c0 + hi + 1, s1])
    if not out: return np.zeros((0, 4))
    out = np.array(out, float); _, u = np.unique(np.round(out / 3), axis=0, return_index=True)
    return out[np.sort(u)]


if __name__ == "__main__":
    # Check: (1) extracted lines vs A-half human pipes (pixel-level recall/precision); (2) recall of labeled symbols by gap proposals (IoU>=0.5), especially those the current candidate pool misses
    import pickle
    from selfsim import gt_of, CACHE
    from rescore_pool import iou_mat
    LV = "/home/suxinqi666/code/PnIDAgent_project/line_results/lines_verified"
    P = pickle.load(open(f"{CACHE}/soup5_rescore_pool_one2one_big4_tile4_cnnp_ss2_prop_deep.pkl", "rb")); R = P["R"]
    for part in ("A_train", "B_test"):
        tot = cov_pool = cov_gap = cov_new = ngap = 0; lr = []
        for n in sp[part]:
            g = cv2.imread(glob.glob(f"{SRC}/{glob.escape(n)}.*")[0], 0); Hh, Ww = g.shape; sc = 3300 / max(Hh, Ww)
            H, V, ink = segments(g, sc); gp = gap_proposals(H, V, ink, sc); ngap += len(gp)
            G = np.array(R[n][1]).reshape(-1, 4); Bp = np.array(R[n][0]).reshape(-1, 4)
            if part == "A_train" and os.path.exists(f"{LV}/{n}.json"):
                lv = json.load(open(f"{LV}/{n}.json")); mg = np.zeros_like(g, np.uint8); me = np.zeros_like(g, np.uint8)
                for x0, y0, x1, y1 in lv["solid"]: cv2.line(mg, (int(x0), int(y0)), (int(x1), int(y1)), 1, max(3, int(6 / sc)))
                for x0, x1, y, t in H: cv2.line(me, (int(x0), int(y)), (int(x1), int(y)), 1, 1)
                for y0, y1, x, t in V: cv2.line(me, (int(x), int(y0)), (int(x), int(y1)), 1, 1)
                lr.append(((me & mg).sum() / max(1, me.sum()), (cv2.dilate(me, np.ones((5, 5), np.uint8)) & mg).sum() / max(1, mg.sum())))
            if not len(G): continue
            tot += len(G)
            ip = iou_mat(G, Bp)[0].max(1) >= 0.5 if len(Bp) else np.zeros(len(G), bool)
            ig = iou_mat(G, gp)[0].max(1) >= 0.5 if len(gp) else np.zeros(len(G), bool)
            cov_pool += ip.sum(); cov_gap += ig.sum(); cov_new += (ig & ~ip).sum()
        msg = f"{part}: labels {tot} | current pool coverage {cov_pool / tot:.3f} | gap proposals {ngap}, coverage {cov_gap / tot:.3f}, of which not in pool {cov_new} boxes"
        if lr: msg += f" | extracted lines vs human pipes: precision {np.mean([a for a, b in lr]):.3f} recall {np.mean([b for a, b in lr]):.3f}"
        print(msg, flush=True)
