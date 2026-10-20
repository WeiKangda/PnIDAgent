"""页面上下文二次打分 (算法层面, 不重训检测器):
检测器低阈值 (0.05) 出的每个候选框, 算一组"上下文特征", 用 A 半训练一个梯度提升分类器判断它是不是真符号:
  - 原始分数, 尺寸, 长宽比, 方向, 墨迹密度, 文字占比 (OCR 原始框)
  - 重复框关系: 被更高分框覆盖多少, 自己里面装了几个框
  - 同类一致性: 页内外观相似 (方向归一 + 翻转, 归一化互相关) 且尺寸相近的候选框有多少个, 它们的最高分/平均分
    (同一页里同一种符号重复出现, 一个高分 -> 同类低分的也可信)
  - 页面统计: 本页高分框数, 本页分数分布, 自己在本页的分数排位
A 半按文档分组 5 折交叉验证选 (概率阈值, NMS 阈值), 在全部 A 上训练最终分类器, B 半只报告.
  python rescore.py <cache name> [tta]    (tta: 加多视角一致性特征, 需先跑 tta_cache.py)"""
import glob, json, os, sys, re
import cv2, numpy as np
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from selfsim import f1, gt_of, sp, SRC, CACHE, iou
from textfilter import text_mask, ink_in_text
FEATS = ["s", "lsz", "lasp", "horiz", "dens", "txt", "dup_iou", "dup_s", "covered", "n_in", "s_in",
         "n70", "n80", "n90", "smax80", "smax90", "stop3", "simhi", "page_n50", "page_q90", "page_rank"]
D = 24


def desc(ink, b):
    x0, y0, x1, y1 = [int(round(v)) for v in b[:4]]
    c = ink[max(0, y0):y1, max(0, x0):x1].astype(np.float32)
    if c.size == 0: return np.zeros((4, D * D), np.float32)
    if c.shape[0] > c.shape[1]: c = c.T                          # 方向归一: 长边横放
    c = cv2.resize(c, (D, D), interpolation=cv2.INTER_AREA)
    vs = [c, c[:, ::-1], c[::-1, :], c[::-1, ::-1]]
    out = []
    for v in vs:
        v = v.flatten() - v.mean(); out.append(v / (np.linalg.norm(v) + 1e-6))
    return np.array(out)


def cover(a, b):   # a 被 b 覆盖的比例
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix / max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))


VIEWS = ["s1.25", "s0.8", "flipx", "flipy"]


def view_feats(b, views):
    """多视角一致性: 每个视角里和它 IoU>=0.5 的最高分, 以及和分数>=0.25 的框的最大 IoU."""
    out = []
    for v in VIEWS:
        vs = views[v]; out += [max([d[4] for d in vs if iou(b, d) >= 0.5], default=0.0), max([iou(b, d) for d in vs if d[4] >= 0.25], default=0.0)]
    return out + [float(np.mean(out[0::2]))]


def page_feats(args):
    n, dets = args[:2]; views = args[2] if len(args) > 2 else None
    g = cv2.imread(glob.glob(f"{SRC}/{glob.escape(n)}.*")[0], 0); H, W = g.shape; sc = 3300 / max(W, H)
    ink = (g < 160).astype(np.uint8); tm = text_mask(n, g.shape)
    dets = sorted([d for d in dets if d[2] - d[0] >= 2 and d[3] - d[1] >= 2], key=lambda d: -d[4])
    N = len(dets); S = np.array([d[4] for d in dets]) if N else np.zeros(0)
    if N == 0: return n, np.zeros((0, len(FEATS) + (9 if views else 0))), [], gt_of(n, W, H)
    DS = np.stack([desc(ink, d) for d in dets])                  # N x 4 x DD
    sim = np.max(np.einsum("ik,jfk->ijf", DS[:, 0], DS), axis=2)  # 对方任意翻转的最大相关
    L = np.array([max(d[2] - d[0], d[3] - d[1]) for d in dets]); Sh = np.array([min(d[2] - d[0], d[3] - d[1]) for d in dets])
    szok = (np.abs(np.log(L[:, None] / L[None])) < 0.25) & (np.abs(np.log(Sh[:, None] / Sh[None])) < 0.3)
    np.fill_diagonal(szok, False)
    ious = np.array([[iou(a, b) for b in dets] for a in dets]); np.fill_diagonal(ious, 0)
    sim = np.where(szok & (ious < 0.3), sim, -1)
    page_n50 = float((S >= 0.5).sum()); page_q90 = float(np.quantile(S, 0.9))
    F = []
    for i, d in enumerate(dets):
        w, h = d[2] - d[0], d[3] - d[1]
        a = ink[int(d[1]):int(d[3]), int(d[0]):int(d[2])]
        hi = [j for j in range(i)]
        dup_iou = max([ious[i, j] for j in hi], default=0.0); dup_s = max([S[j] for j in hi if ious[i, j] > 0.3], default=0.0)
        covered = max([cover(d, dets[j]) for j in hi], default=0.0)
        inside = [j for j in range(N) if j != i and cover(dets[j], d) > 0.8 and ious[i, j] < 0.5]
        si = sim[i]
        def nsim(t): return float((si >= t).sum())
        def smax(t): return float(S[si >= t].max()) if (si >= t).any() else 0.0
        top3 = np.argsort(-si)[:3]; top3 = [j for j in top3 if si[j] > 0.5]
        F.append([d[4], np.log(max(w, h) * sc), np.log(max(w, h) / max(1, min(w, h))), float(w >= h),
                  float(a.mean()) if a.size else 0.0, ink_in_text(d, ink, tm), dup_iou, dup_s, covered, len(inside),
                  max([S[j] for j in inside], default=0.0), nsim(0.7), nsim(0.8), nsim(0.9), smax(0.8), smax(0.9),
                  float(np.mean(S[top3])) if top3 else 0.0, float(si[S >= 0.5].max()) if (S >= 0.5).any() else -1.0,
                  page_n50, page_q90, float((S > d[4]).mean())] + (view_feats(d, views) if views else []))
    return n, np.array(F, np.float32), [d[:4] for d in dets], gt_of(n, W, H)


def nms(boxes, p, thr):
    order = np.argsort(-p); keep = []
    for i in order:
        if all(iou(boxes[i], boxes[k]) < thr for k in keep): keep.append(i)
    return keep


def page_f1(P, probs, t, nt):
    F, B, G = P; m = probs >= t
    idx = np.where(m)[0]; k = nms([B[i] for i in idx], probs[idx], nt)
    return f1([B[idx[i]] for i in k], G)


if __name__ == "__main__":
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import GroupKFold
    mn = sys.argv[1]; dets = json.load(open(f"{CACHE}/{mn}.json"))
    TTA = "tta" in sys.argv[2:]; tta = json.load(open(f"{CACHE}/{mn}_tta.json")) if TTA else {}; suf = "_tta" if TTA else ""
    args = [(n, dets[n], tta[n]) if TTA else (n, dets[n]) for n in sp["A_train"] + sp["B_test"]]
    with Pool(24) as pool: R = {n: (F, B, G) for n, F, B, G in pool.map(page_feats, args)}
    def lab(n):
        F, B, G = R[n]; return np.array([max([iou(b, g) for g in G], default=0) >= 0.5 for b in B], int)
    doc = lambda n: re.sub(r"_p\d+$", "", n).replace("SCS2_", "SCS_")
    A, Bt = sp["A_train"], sp["B_test"]
    X = np.concatenate([R[n][0] for n in A]); y = np.concatenate([lab(n) for n in A])
    grp = np.concatenate([[doc(n)] * len(R[n][0]) for n in A]); pg = np.concatenate([[n] * len(R[n][0]) for n in A])
    mk = lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40, l2_regularization=1.0, random_state=0)
    # 重复分组交叉验证 (5 次, 每次随机打乱文档分组): 平均 F1 曲线后再选阈值, 减少挑参数的运气成分
    grid = [(t, nt) for t in (0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6) for nt in (0.15, 0.2, 0.25, 0.3, 0.4, 0.5)]
    docs = sorted(set(grp)); acc = np.zeros(len(grid)); po = {}
    for rep in range(5):
        perm = np.random.RandomState(rep).permutation(len(docs)); fold = {d: perm[i] % 5 for i, d in enumerate(docs)}
        fk = np.array([fold[d] for d in grp]); oof = np.zeros(len(y))
        for f in range(5): oof[fk == f] = mk().fit(X[fk != f], y[fk != f]).predict_proba(X[fk == f])[:, 1]
        pr = {n: oof[pg == n] for n in A}; po = po or pr
        acc += [np.mean([page_f1(R[n], pr[n], t, nt) for n in A]) for t, nt in grid]
    acc /= 5; i = int(np.argmax(acc)); a_oof, (t, nt) = float(acc[i]), grid[i]
    base = {p: float(np.mean([f1([b for b, s in zip(R[n][1], R[n][0][:, 0]) if s >= 0.5], R[n][2]) for n in sp[p]])) for p in ("A_train", "B_test")}
    clf = mk().fit(X, y)
    pb = {n: clf.predict_proba(R[n][0])[:, 1] if len(R[n][0]) else np.zeros(0) for n in Bt}
    b = float(np.mean([page_f1(R[n], pb[n], t, nt) for n in Bt]))
    print(f"{mn}{suf} 二次打分: 基线 A {base['A_train']:.4f} B {base['B_test']:.4f} | A 交叉验证 {a_oof:.4f} (t {t}, nms {nt}) -> B {b:.4f}", flush=True)
    per = {n: (round(page_f1(R[n], pb[n], t, nt), 3), round(f1([bb for bb, s in zip(R[n][1], R[n][0][:, 0]) if s >= 0.5], R[n][2]), 3)) for n in Bt}
    for n, v in per.items(): print("  ", v, n[7:60])
    import pickle
    pickle.dump({"R": {n: (R[n][1], R[n][2]) for n in R}, "pB": pb, "pA_oof": po, "t": t, "nt": nt, "clf": clf}, open(f"{CACHE}/{mn}_rescore{suf}.pkl", "wb"))
    json.dump({"base": base, "A_oof": a_oof, "t": t, "nt": nt, "B": b, "per": per}, open(f"{HERE}/results/rescore_{mn}{suf}.json", "w"), indent=1)
