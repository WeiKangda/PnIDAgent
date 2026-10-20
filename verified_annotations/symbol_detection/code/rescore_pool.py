"""Multi-view candidate pool + page-context re-scoring.
Candidate pool: detections (conf>=0.05) from five views (original + 1.25x + 0.8x + horizontal flip + vertical flip) are merged,
       near-coincident boxes with IoU>=0.8 become one candidate (box = highest-scoring member), recording its score in each view.
Features: per-view scores / number of views detecting it + rescore.py context features (size, text ratio, duplicates, within-page same-class consistency, page stats).
A half: thresholds chosen by document-grouped 5-fold cross-validation (repeated 5 times), train on all of A, B is only reported.
  python rescore_pool.py <cache name> [one2one] [views=a,b] [cnn|cnnp] [seed2] [prop] [deep]     (requires <name>.json and <name>_tta.json)"""
import glob, json, os, sys, re, pickle
import cv2, numpy as np
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from selfsim import f1, gt_of, sp, SRC, CACHE
from textfilter import text_mask, ink_in_text
from rescore import desc
VIEWS = ["base", "s1.25", "s0.8", "flipx", "flipy"]
FEATS = ["s", "v_base", "v_s1.25", "v_s0.8", "v_flipx", "v_flipy", "nview", "vmean", "lsz", "lasp", "horiz", "dens", "txt",
         "dup_iou", "dup_s", "covered", "n_in", "s_in", "n70", "n80", "n90", "smax80", "smax90", "stop3", "simhi",
         "page_n50", "page_q90", "page_rank", "ports", "port_sides", "thru", "ntext", "tchars", "tmax"]
RAW = "/home/suxinqi666/code/PnIDAgent_project/ft_verified/text_ocr/raw"


def line_mask(ink, sc):
    """Long straight lines (pipes/buses): ink with horizontal or vertical length >= 40px (scaled to 3300 width)."""
    L = max(15, int(40 / sc))
    return cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((1, L), np.uint8)) | cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((L, 1), np.uint8))


def port_feats(b, lm, sc):
    """Number of long line segments in a narrow band around the box (port count), how many sides have ports, and whether a line crosses the whole box."""
    H, W = lm.shape; m = max(2, int(4 / sc)); x0, y0, x1, y1 = [int(round(v)) for v in b]
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    if x1 - x0 < 2 or y1 - y0 < 2: return [0, 0, 0]
    bands = [lm[max(0, y0 - m):y0, x0:x1].any(0), lm[y1:min(H, y1 + m), x0:x1].any(0), lm[y0:y1, max(0, x0 - m):x0].any(1), lm[y0:y1, x1:min(W, x1 + m)].any(1)]
    segs = [int(np.sum(np.diff(np.concatenate([[0], v.astype(int), [0]])) == 1)) if v.size else 0 for v in bands]
    inner = lm[y0:y1, x0:x1]
    thru = float(inner.all(1).any() or inner.all(0).any()) if inner.size else 0.0
    return [sum(segs), sum(1 for s in segs if s), thru]


def text_feats(b, texts):
    """Number of OCR text items overlapping the box, total character count, and character count of the longest one."""
    hit = [t for t in texts if min(b[2], t[2]) - max(b[0], t[0]) > 0 and min(b[3], t[3]) - max(b[1], t[1]) > 0 and
           (min(b[2], t[2]) - max(b[0], t[0])) * (min(b[3], t[3]) - max(b[1], t[1])) >= 0.5 * (t[2] - t[0]) * (t[3] - t[1])]
    return [len(hit), sum(t[4] for t in hit), max([t[4] for t in hit], default=0)]


def iou_mat(a, b):
    ix = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = ix * iy; aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + ab[None] - inter, 1e-6), inter / np.maximum(aa[:, None], 1e-6)   # iou, fraction of a covered by b


def build_pool(views):
    allb = [(d[:4], d[4], vi) for vi, v in enumerate(VIEWS) for d in views[v] if d[2] - d[0] >= 2 and d[3] - d[1] >= 2]
    allb.sort(key=lambda t: -t[1])
    if not allb: return np.zeros((0, 4)), np.zeros((0, len(VIEWS)))
    X = np.array([t[0] for t in allb], float); io, _ = iou_mat(X, X)
    cid = -np.ones(len(allb), int); boxes, vs = [], []
    for i in range(len(allb)):
        if cid[i] >= 0: continue
        c = len(boxes); boxes.append(X[i]); v = np.zeros(len(VIEWS))
        mem = np.where((io[i] >= 0.8) & (cid < 0))[0]; cid[mem] = c
        for j in mem: v[allb[j][2]] = max(v[allb[j][2]], allb[j][1])
        vs.append(v)
    return np.array(boxes), np.array(vs)


STRUCT = "struct" in sys.argv[2:]; PIPE = "pipe" in sys.argv[2:]


def page_feats(args):
    n, views = args
    g = cv2.imread(glob.glob(f"{SRC}/{glob.escape(n)}.*")[0], 0); H, W = g.shape; sc = 3300 / max(W, H)
    ink = (g < 160).astype(np.uint8); tm = text_mask(n, g.shape); G = gt_of(n, W, H); lm = line_mask(ink, sc)
    texts = [t["box"] + [sum(ch.isalnum() for ch in t["text"])] for t in json.load(open(f"{RAW}/{n}.json")) if t["score"] >= 0.5]
    B, V = build_pool(views); N = len(B)
    if N == 0: return n, np.zeros((0, len(FEATS) + (9 if STRUCT else 0)), np.float32), [], G
    S = V.max(1); o = np.argsort(-S); B, V, S = B[o], V[o], S[o]
    io, cov = iou_mat(B, B); np.fill_diagonal(io, 0); np.fill_diagonal(cov, 0)
    DS = np.stack([desc(ink, b) for b in B]); sim = np.max(np.einsum("ik,jfk->ijf", DS[:, 0], DS), axis=2)
    w, h = B[:, 2] - B[:, 0], B[:, 3] - B[:, 1]; L, Sh = np.maximum(w, h), np.maximum(np.minimum(w, h), 1)
    szok = (np.abs(np.log(L[:, None] / L[None])) < 0.25) & (np.abs(np.log(Sh[:, None] / Sh[None])) < 0.3) & (io < 0.3)
    np.fill_diagonal(szok, False); sim = np.where(szok, sim, -1)
    hi = np.tril(np.ones((N, N), bool), -1)                      # j < i: higher-scoring candidates
    dup_iou = np.where(hi, io, 0).max(1); dup_s = np.where(hi & (io > 0.3), S[None], 0).max(1); covered = np.where(hi, cov, 0).max(1)
    ins = (cov.T > 0.8) & (io < 0.5)                              # ins[i,j]: j is contained in i
    n_in = ins.sum(1); s_in = np.where(ins, S[None], 0).max(1)
    def nsim(t): return (sim >= t).sum(1)
    def smax(t): return np.where(sim >= t, S[None], 0).max(1)
    top3 = np.argsort(-sim, 1)[:, :3]; t3v = np.take_along_axis(sim, top3, 1); t3s = S[top3]
    ok = t3v > 0.5; stop3 = np.where(ok, t3s, 0).sum(1) / np.maximum(ok.sum(1), 1)
    simhi = np.where((S >= 0.5)[None], sim, -1).max(1)
    dens = np.array([ink[int(b[1]):int(b[3]), int(b[0]):int(b[2])].mean() if (b[3] - b[1]) >= 1 and (b[2] - b[0]) >= 1 else 0 for b in B])
    txt = np.array([ink_in_text(b, ink, tm) for b in B])
    F = np.column_stack([S, V, (V > 0).sum(1), V.mean(1), np.log(L * sc), np.log(L / Sh), (w >= h).astype(float), dens, txt,
                         dup_iou, dup_s, covered, n_in, s_in, nsim(0.7), nsim(0.8), nsim(0.9), smax(0.8), smax(0.9), stop3, simhi,
                         np.full(N, (S >= 0.5).sum()), np.full(N, np.quantile(S, 0.9)), (S[None] > S[:, None]).mean(1),
                         np.array([port_feats(b, lm, sc) + text_feats(b, texts) for b in B])])
    if STRUCT:   # pipe structure features (structure.py, image only): endpoints facing box edges / flanked by collinear segments / line through box
        from structure import segments, struct_feats
        pm = cv2.imread(f"{HERE}/results/pipe_seg/{n}.png", 0) >= 128 if PIPE else None     # pipe: use learned pipe segmentation (cross-fitted) instead of rules
        Hs, Vs, _ = segments(g, sc, ink=pm); F = np.column_stack([F, struct_feats(B, Hs, Vs, sc)])
    return n, F.astype(np.float32), [list(b) for b in B], G


def nms(boxes, p, thr, ct=1.01):
    """IoU >= thr, or one box contained in another by >= ct, counts as a duplicate; keep the higher probability."""
    if not len(boxes): return []
    order = np.argsort(-p); B = np.array(boxes)[order]; io, cov = iou_mat(B, B); keep = []
    for i in range(len(order)):
        if all(io[i, k] < thr and cov[i, k] < ct and cov[k, i] < ct for k in keep): keep.append(i)
    return [order[i] for i in keep]


def select(B, probs, t, nt, ct=1.01):
    """t is a number: global threshold. t == "ef1": per-page expected-F1-optimal cutoff (no labels used):
    after NMS, sort by probability and take k maximizing 2*sum(p_1..k) / (k + sum(p_all)) (expected hits / expected F1 denominator)."""
    if t == "ef1":
        idx = np.where(probs >= 0.02)[0]; k = nms([B[i] for i in idx], probs[idx], nt, ct)
        p = np.array([probs[idx[i]] for i in k]); o = np.argsort(-p)
        if not len(p): return []
        ef = 2 * np.cumsum(p[o]) / (np.arange(1, len(p) + 1) + p.sum()); kk = int(np.argmax(ef)) + 1
        return [B[idx[k[i]]] for i in o[:kk]]
    idx = np.where(probs >= t)[0]; k = nms([B[i] for i in idx], probs[idx], nt, ct)
    return [B[idx[i]] for i in k]


def page_f1(R, probs, t, nt, ct=1.01):
    F, B, G = R
    return f1(select(B, probs, t, nt, ct), G)


if __name__ == "__main__":
    from sklearn.ensemble import HistGradientBoostingClassifier
    mn = sys.argv[1]; dets = json.load(open(f"{CACHE}/{mn}.json")); tta = json.load(open(f"{CACHE}/{mn}_tta.json"))
    # extra views: views=name1,name2 (selfsim_cache/<name>.json; A pages must come from a model not trained on that page, i.e. cross-fitted results)
    EXTRA = [a.split("=", 1)[1].split(",") for a in sys.argv[2:] if a.startswith("views=")]; EXTRA = EXTRA[0] if EXTRA else []
    ex = {v: json.load(open(f"{CACHE}/{v}.json")) for v in EXTRA}; VIEWS.extend(EXTRA)
    FEATS[FEATS.index("nview"):FEATS.index("nview")] = [f"v_{v}" for v in EXTRA]
    CV = ([a[3:] for a in sys.argv[2:] if a.startswith("cnnp")] or [""])[0]      # cnnp / cnnp2 ...: use the CNN self-trained with B pseudo labels (round 1/2..)
    USE_CNN = "cnn" in sys.argv[2:] or bool(CV); PROP = "prop" in sys.argv[2:]
    SEEDS = [""] + [f"s{a[4:]}" for a in sys.argv[2:] if a.startswith("seed")]     # seedN: also average in CNNs with a different random seed
    TAGC = (f"r{os.environ['CNN_S']}" if os.environ.get("CNN_S", "64") != "64" else "") + os.environ.get("CNN_TAGX", "")     # CNN_S=96: CNN with 96-pixel input; CNN_TAGX: other variants
    fsuf = TAGC and f"_{TAGC}"; fsuf += "".join(f"_{v}" for v in EXTRA) + ("_cnn" if "cnn" in sys.argv[2:] else "") + (f"_cnn{CV}" if CV else "") + "".join(f"_s{x}" for x in SEEDS if x) + ("_prop" if PROP else "") + ("_cnn4" if "cnn4" in sys.argv[2:] else "") + ("_struct" if STRUCT else "") + ("_pipe" if PIPE else "")
    fcache = f"{CACHE}/feat_{mn}{fsuf}.pkl"
    if os.path.exists(fcache):
        R = pickle.load(open(fcache, "rb")); print("feature cache", fcache, flush=True)
    else:
        with Pool(24) as pool: R = {n: (F, B, G) for n, F, B, G in pool.map(page_feats, [(n, {"base": dets[n], **tta[n], **{v: ex[v][n] for v in EXTRA}}) for n in sp["A_train"] + sp["B_test"]])}
        if USE_CNN:
            # crop CNN probability as a feature (cross-fitted: A1 pages use the CNN trained only on A2, and vice versa; B averages both fold models)
            import cv2 as _cv2
            from crop_cnn import load as cnn_load, score_embed
            if "cnn4" in sys.argv[2:]:   # 4-fold CNN: pages in fold k use the F{k} model that never saw them, B averages all 4
                f4 = json.load(open(f"{HERE}/folds4.json")); who = {**{n: [k] for k, v in f4.items() for n in v}, **{n: list(f4) for n in sp["B_test"]}}
                nets = {k: [cnn_load(k + CV + sd) for sd in SEEDS] for k in f4}
            else:
                folds = json.load(open(f"{HERE}/tile_folds.json")); who = {**{n: ["A2"] for n in folds["A1"]}, **{n: ["A1"] for n in folds["A2"]}, **{n: ["A1", "A2"] for n in sp["B_test"]}}
                nets = {k: [cnn_load(k + CV + TAGC + sd) for sd in SEEDS] for k in ("A1", "A2")}
            for n in R:
                F, B, G = R[n]
                if not len(B): R[n] = (np.zeros((0, F.shape[1] + (6 if PROP else 1)), np.float32), B, G); continue
                g = _cv2.imread(glob.glob(f"{SRC}/{glob.escape(n)}.*")[0], 0); S = F[:, 0]
                Bn = np.array(B); io = iou_mat(Bn, Bn)[0]; np.fill_diagonal(io, 1)
                cols = []
                for m in [m for k in who[n] for m in nets[k]]:
                    p, e = score_embed(m, g, B); c = [p]
                    if PROP:   # within-page propagation: candidates with similar appearance (CNN features) should be the same class
                        e = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-6); sim = np.where(io > 0.3, -1, e @ e.T)
                        strong = (p >= 0.7)[None]; top = np.argsort(-sim, 1)[:, :5]
                        c += [np.where(strong, sim, -1).max(1), p[top].mean(1), (sim >= 0.9).sum(1), S[top].mean(1),
                              (np.where(sim >= 0.85, p[None], 0).sum(1) + p) / ((sim >= 0.85).sum(1) + 1)]
                    cols.append(np.column_stack(c))
                R[n] = (np.column_stack([F, np.mean(cols, 0)]).astype(np.float32), B, G)
            FEATS.append("cnn"); print("CNN features added" + (" (+within-page propagation)" if PROP else ""), flush=True)
        pickle.dump(R, open(fcache, "wb"))
    ONE = "one2one" in sys.argv[2:]
    def lab(n):
        F, B, G = R[n]
        if not len(B) or not G: return np.zeros(len(B), int)
        io = iou_mat(np.array(B), np.array(G))[0]
        if not ONE: return (io.max(1) >= 0.5).astype(int)
        y = np.zeros(len(B), int)                       # one-to-one: each annotation matches only its highest-IoU candidate
        for gi in np.argsort(-io.max(0)):
            j = int(np.argmax(np.where(y == 1, -1, io[:, gi])))
            if io[j, gi] >= 0.5: y[j] = 1
        return y
    doc = lambda n: re.sub(r"_p\d+$", "", n).replace("SCS2_", "SCS_")
    A, Bt = sp["A_train"], sp["B_test"]
    X = np.concatenate([R[n][0] for n in A]); y = np.concatenate([lab(n) for n in A])
    grp = np.concatenate([[doc(n)] * len(R[n][0]) for n in A]); pg = np.concatenate([[n] * len(R[n][0]) for n in A])
    HP = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40, l2_regularization=1.0)
    if "deep" in sys.argv[2:]: HP = dict(max_iter=600, learning_rate=0.03, max_leaf_nodes=31, min_samples_leaf=20, l2_regularization=1.0)
    mk = lambda: HistGradientBoostingClassifier(random_state=0, **HP)
    grid = [(t, nt, ct) for t in (0.04, 0.06, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5) for nt in (0.1, 0.15, 0.2, 0.3, 0.4, 0.5) for ct in (0.5, 0.6, 0.7, 0.8, 1.01)]
    grid += [("ef1", nt, ct) for nt in (0.1, 0.15, 0.2, 0.3, 0.4, 0.5) for ct in (0.5, 0.6, 0.7, 0.8, 1.01)]
    docs = sorted(set(grp)); acc = np.zeros(len(grid)); po = {}
    for rep in range(5):
        perm = np.random.RandomState(rep).permutation(len(docs)); fold = {d: perm[i] % 5 for i, d in enumerate(docs)}
        fk = np.array([fold[d] for d in grp]); oof = np.zeros(len(y))
        for f in range(5): oof[fk == f] = mk().fit(X[fk != f], y[fk != f]).predict_proba(X[fk == f])[:, 1]
        pr = {n: oof[pg == n] for n in A}; po = po or pr
        acc += [np.mean([page_f1(R[n], pr[n], *g) for n in A]) for g in grid]
    acc /= 5; i = int(np.argmax(acc)); a_oof, (t, nt, ct) = float(acc[i]), grid[i]
    ig = int(np.argmax([a if g[0] != "ef1" else -1 for a, g in zip(acc, grid)])); ie = int(np.argmax([a if g[0] == "ef1" else -1 for a, g in zip(acc, grid)]))
    print(f"  A cross-validation: best global threshold {acc[ig]:.4f} {grid[ig]} | best per-page expected F1 {acc[ie]:.4f} {grid[ie]}", flush=True)
    clf = mk().fit(X, y); pb = {n: clf.predict_proba(R[n][0])[:, 1] if len(R[n][0]) else np.zeros(0) for n in Bt}
    b = float(np.mean([page_f1(R[n], pb[n], t, nt, ct) for n in Bt]))
    suf = ("_one2one" if ONE else "") + fsuf + ("_deep" if "deep" in sys.argv[2:] else "")
    print(f"{mn}{suf} multi-view pool + re-scoring: A cross-validation {a_oof:.4f} (t {t}, nms {nt}, containment {ct}) -> B {b:.4f}", flush=True)
    per = {n: round(page_f1(R[n], pb[n], t, nt, ct), 3) for n in Bt}
    pickle.dump({"R": {n: (R[n][1], R[n][2]) for n in R}, "pB": pb, "pA_oof": po, "t": t, "nt": nt, "ct": ct}, open(f"{CACHE}/{mn}_rescore_pool{suf}.pkl", "wb"))
    json.dump({"A_oof": a_oof, "t": str(t), "nt": nt, "ct": ct, "B": b, "per": per, "acc": acc.tolist(), "grid": [list(map(str, g)) for g in grid]}, open(f"{HERE}/results/rescore_pool_{mn}{suf}.json", "w"), indent=1)
