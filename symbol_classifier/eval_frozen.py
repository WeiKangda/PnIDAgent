"""Baselines on frozen image features: logistic regression and nearest neighbour, leave-one-family-out.

  python eval_frozen.py --cache <dir>       # needs build_cache.py and extract_frozen.py; writes <dir>/frozen_results.json

Each of the 7 document families is held out in turn; the model is fit on the other 6. No setting is tuned on the
held-out families: logistic regression with C = 1 on standardised image features, context features unscaled (counts 0-3
and indicators), next to them.
"""
import argparse, json, os

import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from common import family_folds, load_cache, scores


def fit_predict(Xtr, ytr, Xte, n_img):
    sc = StandardScaler().fit(Xtr[:, :n_img])
    f = lambda X: np.hstack([sc.transform(X[:, :n_img]), X[:, n_img:]])
    m = LogisticRegression(C=1.0, max_iter=3000).fit(f(Xtr), ytr)
    return m.classes_[m.predict_proba(f(Xte)).argmax(1)]


def run_lr(X, y, folds, mask, n_img):
    """Leave-one-family-out predictions for the boxes in mask (the others get -1)."""
    pred = np.full(len(y), -1)
    jobs = []
    for _, tr, te in folds:
        tr = tr[mask[tr]]; te = te[mask[te]]
        if len(np.unique(y[tr])) == 1:
            pred[te] = y[tr][0]; continue
        jobs.append((te, delayed(fit_predict)(X[tr], y[tr], X[te], n_img)))
    res = Parallel(n_jobs=7)(j for _, j in jobs)
    for (te, _), p in zip(jobs, res): pred[te] = p
    return pred


def run_knn(X, y, folds):
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    pred = np.empty(len(y), int)
    for _, tr, te in folds:
        pred[te] = y[tr][(Xn[te] @ Xn[tr].T).argmax(1)]
    return pred


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cache", required=True); a = ap.parse_args()
    d, meta = load_cache(a.cache); folds = family_folds(d)
    feats = {m: np.load(os.path.join(a.cache, f"feat_{m}.npz")) for m in ("clip", "resnet18", "effnetv2s")}
    allm = np.ones(len(d["cls"]), bool); act_m = d["act"] != meta["actuators"].index("-"); pos_m = d["pos"] >= 0
    ok = d["review"] == ""                                       # boxes not on the review sheet
    results = {}

    def report(name, pc, pa=None, pp=None):
        r = {"all": scores(d, meta, pc, pa, pp), "not_in_review": scores(d, meta, pc, pa, pp, ok),
             "per_family": {f: scores(d, meta, pc, pa, pp, d["family"] == f)["class_acc"] for f, _, _ in folds}}
        results[name] = r
        s = r["all"]
        print(f"{name:42s} acc {s['class_acc']:.3f}  macroF1 {s['class_macro_f1']:.3f}"
              + (f"  act {s['act_acc']:.3f}  cls+act {s['class_act_acc']:.3f}" if pa is not None else "")
              + (f"  pos {s['pos_acc']:.3f}" if pp is not None else ""), flush=True)

    # majority class of the training families
    pc = np.empty(len(d["cls"]), int)
    for _, tr, te in folds: pc[te] = np.bincount(d["cls"][tr]).argmax()
    report("majority class", pc)

    for m in ("clip", "resnet18", "effnetv2s"):
        for o in ("id", "d4"):
            X = feats[m][o]
            report(f"LR  {m} ({o})", run_lr(X, d["cls"], folds, allm, X.shape[1]))
        report(f"1-NN {m} (d4)", run_knn(feats[m]["d4"], d["cls"], folds))

    best = max(("clip", "resnet18", "effnetv2s"), key=lambda m: results[f"LR  {m} (d4)"]["all"]["class_acc"])
    Xi = feats[best]["d4"]; n = Xi.shape[1]
    for ctx_name, ctx in (("text", d["ctx_text"]), ("text+ports", d["ctx_topo"])):
        X = np.hstack([Xi, ctx])
        pc = run_lr(X, d["cls"], folds, allm, n)
        pa = run_lr(X, d["act"], folds, act_m, n); pa[~act_m] = meta["actuators"].index("-")
        pp = run_lr(X, d["pos"], folds, pos_m, n)
        report(f"LR  {best} (d4) + {ctx_name}", pc, pa, pp)
    pc = run_lr(Xi, d["cls"], folds, allm, n)
    pa = run_lr(Xi, d["act"], folds, act_m, n); pa[~act_m] = meta["actuators"].index("-")
    pp = run_lr(Xi, d["pos"], folds, pos_m, n)
    report(f"LR  {best} (d4), all heads", pc, pa, pp)
    json.dump(results, open(os.path.join(a.cache, "frozen_results.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
