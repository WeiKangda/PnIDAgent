"""管线结构线索 (不用任何标注, 只看图像):
  1. 管线提取: 墨迹做横/竖长核开运算 -> 连通块 -> 横/竖线段 (端点, 粗细).
  2. 结构特征 (每个候选框): 有几个线段端点正对着框边终止 / 有几条边被线接上 / 左右或上下是否被一对共线线段夹住 (串联符号)
     / 有没有线直接穿过框 (标注习惯 "线不穿符号": 穿过 -> 多半不是符号) / 框内被线占的比例.
  3. 断点候选: 两条共线线段面对面断开, 断口里有图形墨迹 -> 在断口处生成候选框 (补检测器漏掉的串联符号).
A 半的人工管线标注只用来检查提取质量, 不参与特征."""
import glob, json, os, sys
import cv2, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from selfsim import SRC, sp


def segments(g, sc, Lmm=30, maxth=12, ink=None):
    """返回 (H, V): 横线段 [x0, x1, y, 粗细], 竖线段 [y0, y1, x, 粗细] (原图像素). Lmm: 最短线长 (3300 口径像素).
    ink: 可传入学习式管线分割的掩码 (pipe_seg.py), 只从管线像素里提线段."""
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
    """B: N x 4 框. 返回 N x 9 结构特征."""
    B = np.asarray(B, float).reshape(-1, 4); N = len(B)
    if N == 0: return np.zeros((0, 9))
    w, h = B[:, 2] - B[:, 0], B[:, 3] - B[:, 1]; m = np.maximum(3 / sc, 0.15 * np.minimum(w, h))[:, None]
    def touch(S, lo, hi, cross0, cross1, near_lo):
        """S 的线段在 [lo, hi] (沿线方向) , 位置 c (垂直方向). 返回: 端点贴在框的 '低侧'(进入框) / '高侧' 的个数, 以及穿过的个数."""
        if not len(S): z = np.zeros(N); return z, z, z
        s0, s1, c = S[:, 0][None], S[:, 1][None], S[:, 2][None]
        inc = (c >= cross0[:, None] - m) & (c <= cross1[:, None] + m)            # 线在框的横截范围内
        inner = (c >= cross0[:, None] + 0.2 * (cross1 - cross0)[:, None]) & (c <= cross1[:, None] - 0.2 * (cross1 - cross0)[:, None])
        end_lo = inc & (np.abs(s1 - lo[:, None]) <= m + 0.25 * (hi - lo)[:, None]) & (s0 < lo[:, None] - m)   # 从低侧来, 终止在低侧边附近
        end_hi = inc & (np.abs(s0 - hi[:, None]) <= m + 0.25 * (hi - lo)[:, None]) & (s1 > hi[:, None] + m)   # 从高侧来
        thru = inner & (s0 < lo[:, None] - m) & (s1 > hi[:, None] + m)
        return end_lo.sum(1), end_hi.sum(1), thru.sum(1)
    hl, hr, ht = touch(H, B[:, 0], B[:, 2], B[:, 1], B[:, 3], True)           # 横线: 沿 x, 截面 y
    vt, vb, vt2 = touch(V, B[:, 1], B[:, 3], B[:, 0], B[:, 2], True)          # 竖线: 沿 y, 截面 x
    sides = (hl > 0).astype(int) + (hr > 0) + (vt > 0) + (vb > 0)
    inline = ((hl > 0) & (hr > 0)) | ((vt > 0) & (vb > 0))
    return np.column_stack([hl + hr + vt + vb, sides, inline, ht + vt2, (hl > 0) & (hr > 0), (vt > 0) & (vb > 0),
                            (ht + vt2 > 0) & ~inline, np.log1p(w * sc), np.log1p(h * sc)]).astype(float)


def gap_proposals(H, V, ink, sc, tol=3, gmin=8, gmax=260, minink=12):
    """共线面对面的两段线之间的断口 -> 候选框. 断口里 (去掉线本身) 要有足够墨迹."""
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
            reg = ink[c0:c1, s0:s1] if horiz else ink[s0:s1, c0:c1].T          # 行 = 截面方向, 列 = 沿线方向
            if reg.size == 0: continue
            prof = reg.sum(1)                                                   # 截面方向每行的墨迹
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
    # 检查: (1) 提取的线 vs A 半人工管线 (像素级召回/精度); (2) 断点候选对标注符号的召回 (IoU>=0.5), 尤其是现有候选池没覆盖的
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
        msg = f"{part}: 标注 {tot} | 现候选池覆盖 {cov_pool / tot:.3f} | 断点候选 {ngap} 个, 覆盖 {cov_gap / tot:.3f}, 其中池里没有的 {cov_new} 个"
        if lr: msg += f" | 提取线 vs 人工管线: 精度 {np.mean([a for a, b in lr]):.3f} 召回 {np.mean([b for a, b in lr]):.3f}"
        print(msg, flush=True)
