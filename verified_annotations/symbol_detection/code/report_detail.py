"""详细报告: 65 页逐页 (soup5 原模型 vs 级联关系模型; A 页用 5 折折外预测, B 页用 5 个折模型平均) + real4 逐张 (soup5 标准口径 vs 用全部 65 页训练的最终模型)."""
import glob, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
os.environ.setdefault("REL_DATA", "rel_data_pipe")
import rel_model as V1
from rescore_pool import select, iou_mat
from selfsim import sp, CACHE


def stats(pred, G):
    """贪心一一匹配 (IoU>=0.5): 返回 TP, FP, FN, P, R, F1."""
    pred = [list(p) for p in pred]; used = set(); tp = 0
    for g in G:
        best = max([(float(iou_mat(np.array([g]), np.array([p]))[0][0, 0]), j) for j, p in enumerate(pred) if j not in used], default=(0, -1))
        if best[0] >= 0.5: used.add(best[1]); tp += 1
    fp, fn = len(pred) - tp, len(G) - tp
    return tp, fp, fn, tp / max(1, tp + fp), tp / max(1, tp + fn), 2 * tp / max(1, 2 * tp + fp + fn)


def load_preds(tag):
    R = {}; Bm = {}
    for f in sorted(glob.glob(f"{V1.OUTD}/{tag}_fold*.npy")):
        for n, v in np.load(f, allow_pickle=True).item().items():
            (Bm.setdefault(n, []).append(v) if n in sp["B_test"] or n.startswith("real4_") else R.__setitem__(n, v))
    R.update({n: (np.mean([v[0] for v in Bm[n]], 0), np.mean([v[1] for v in Bm[n]], 0)) for n in Bm}); return R


if __name__ == "__main__":
    cfg = json.load(open(f"{V1.OUTD}/prt2cv_se_avg_first_eval.json"))["cfg"]
    base = json.load(open(f"{CACHE}/soup5.json")); P = load_preds("prt2cv_se_avg_first"); rows = []
    for part in ("A_train", "B_test"):
        for n in sp[part]:
            G = list(V1.load(n)["G"])
            s = stats([d[:4] for d in base[n] if d[4] >= 0.5], G); a = stats(select(list(P[n][1]), P[n][0], *cfg), G)
            rows.append({"page": n.replace("pseudo_", ""), "half": part[0], "gt": len(G), "soup5": s, "ours": a})
    json.dump(rows, open(f"{HERE}/results/report_65.json", "w"), indent=1)
    for half in ("A", "B", None):
        rr = [r for r in rows if half is None or r["half"] == half]
        agg = lambda k: np.array([r[k][:3] for r in rr]).sum(0)
        for k in ("soup5", "ours"):
            tp, fp, fn = agg(k); print(f"{half or '全部'} {k}: 宏F1 {np.mean([r[k][5] for r in rr]):.4f} | 宏精度 {np.mean([r[k][3] for r in rr]):.4f} 宏召回 {np.mean([r[k][4] for r in rr]):.4f} | 合计 命中 {tp} 误检 {fp} 漏检 {fn} (标注 {tp + fn})")
