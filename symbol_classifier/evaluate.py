"""Scores of all classifiers from their held-out predictions -> results/results.json, results/results.md and
results/confusions.png (examples of the most frequent confusions of the main model).

  python evaluate.py --cache <dir> [--out results]

Reads <dir>/frozen_results.json (eval_frozen.py) and <dir>/cnn_*.npz (train_cnn.py).
"""
import argparse, collections, json, os

import cv2
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from common import load_cache, scores

RUNS = [("cnn_family", "new family", "ResNet-18, boxes only"), ("cnn_family_legend", "new family", "ResNet-18 + legend symbols"),
        ("cnn_page_legend", "new page", "ResNet-18 + legend symbols")]
FROZEN = {"majority class": ("most frequent class", "-"),
          "LR  clip (d4)": ("CLIP ViT-B/32 features + logistic regression", "image"),
          "LR  effnetv2s (d4)": ("EfficientNetV2-S features + logistic regression", "image"),
          "LR  effnetv2s (d4) + text": ("EfficientNetV2-S features + logistic regression", "image + text")}


def decide(P, mode, meta, act_classes):
    """Labels from the softmax outputs: the most likely class; an actuator only for the classes that have one (the most
    likely real actuator), else '-'; the more likely of open / closed."""
    pc = P[f"cls_{mode}"].argmax(1); conf = P[f"cls_{mode}"].max(1)
    dash = meta["actuators"].index("-")
    a = P[f"act_{mode}"].copy(); a[:, dash] = -1
    pa = np.where(np.isin(pc, act_classes), a.argmax(1), dash)
    return pc, pa, P[f"pos_{mode}"].argmax(1), conf


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cache", required=True)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")); a = ap.parse_args()
    d, meta = load_cache(a.cache); C = meta["classes"]
    fams = sorted(set(d["family"].tolist())); ok = d["review"] == ""
    frozen = json.load(open(os.path.join(a.cache, "frozen_results.json")))
    res = {"frozen": {k: dict(frozen[k], model=FROZEN[k][0], input=FROZEN[k][1], protocol="new family") for k in FROZEN}}
    main_pred = None
    for tag, desc, name in RUNS:
        p = os.path.join(a.cache, f"{tag}.npz")
        if not os.path.exists(p): continue
        P = dict(np.load(p))
        for mode, mdesc in (("img", "image only"), ("ctx", "image + text")):
            pc, pa, pp, conf = decide(P, mode, meta, P["act_classes"])
            r = {"model": name, "protocol": desc, "input": mdesc, "all": scores(d, meta, pc, pa, pp), "not_in_review": scores(d, meta, pc, pa, pp, ok),
                 "per_family": {f: scores(d, meta, pc, pa, pp, d["family"] == f)["class_acc"] for f in fams},
                 "coverage": {}}
            for t in (0.5, 0.7, 0.9):
                m = conf >= t
                r["coverage"][str(t)] = {"share": float(m.mean()), "class_acc": float((pc[m] == d["cls"][m]).mean())}
            res[f"{tag} / {mode}"] = r
            if tag == "cnn_family_legend" and mode == "ctx": main_pred = (pc, pa, pp, conf)
    # per class and confusions of the main model (new family, legend symbols, image + text)
    if main_pred:
        pc = main_pred[0]; y = d["cls"]; cnt = np.bincount(y, minlength=len(C))
        labels = [c for c in np.argsort(-cnt) if cnt[c] >= 5]
        pr, rc, f1, n = precision_recall_fscore_support(y, pc, labels=labels, zero_division=0)
        res["per_class"] = [{"class": C[c], "n": int(k), "precision": float(p_), "recall": float(r_), "f1": float(f_)}
                            for c, p_, r_, f_, k in zip(labels, pr, rc, f1, n)]
        conf_pairs = collections.Counter((C[t], C[p_]) for t, p_ in zip(y, pc) if t != p_)
        res["confusions"] = [{"true": t, "predicted": p_, "n": k} for (t, p_), k in conf_pairs.most_common(20)]
    os.makedirs(a.out, exist_ok=True)
    if main_pred: confusion_sheet(d, C, main_pred[0], os.path.join(a.out, "confusions.png"))
    json.dump(res, open(os.path.join(a.out, "results.json"), "w"), indent=1)
    write_md(res, fams, os.path.join(a.out, "results.md"))
    print(open(os.path.join(a.out, "results.md")).read())


def confusion_sheet(d, C, pc, path, n_pairs=10, per=8):
    """One row per frequent confusion: the true class, the predicted class and up to `per` of those held-out crops."""
    y = d["cls"]
    pairs = collections.Counter((t, p) for t, p in zip(y, pc) if t != p).most_common(n_pairs)
    rows = []
    for (t, p), k in pairs:
        lab = np.full((136, 300), 255, np.uint8)
        for j, txt in enumerate((f"true: {C[t]}", f"predicted: {C[p]}", f"{k} boxes")):
            cv2.putText(lab, txt, (6, 40 + 30 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1, cv2.LINE_AA)
        tiles = [d["crops"][i] for i in np.where((y == t) & (pc == p))[0][:per]]
        tiles += [np.full((128, 128), 255, np.uint8)] * (per - len(tiles))
        rows.append(np.hstack([lab] + [cv2.copyMakeBorder(c, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=255) for c in tiles]))
        rows.append(np.full((2, rows[-1].shape[1]), 200, np.uint8))
    cv2.imwrite(path, np.vstack(rows))


def pct(v): return f"{100 * v:.1f}"


def write_md(res, fams, path):
    rows = list(res["frozen"].values()) + [r for k, r in res.items() if k.startswith("cnn")]
    def cell(s, k): return pct(s[k]) if k in s else ""
    L = ["# Symbol classifier: held-out results", "",
         "Every number is on boxes the model did not see in training: 3,483 boxes on 65 pages, each predicted once.",
         "", "- new family: each of the 7 document families is held out in turn (the model has never seen that drawing style).",
         "- new page: 5 folds of whole pages; the other pages of the same family are in training. GE-Hitachi ABWR and NuScale "
         "have one page each, so they are a new family in both protocols.", "",
         "## Overall", "",
         "| model | test | input | class | class macro-F1 | actuator | class + actuator | fault-tree type | open/closed |",
         "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        s = r["all"]
        L.append(f"| {r['model']} | {r['protocol']} | {r['input']} | {cell(s, 'class_acc')} | {cell(s, 'class_macro_f1')} | "
                 f"{cell(s, 'act_acc')} | {cell(s, 'class_act_acc')} | {cell(s, 'ft_acc')} | {cell(s, 'pos_acc')} |")
    L += ["", "Accuracy in %. Macro-F1 is over the 45 classes with at least 5 boxes. Actuator accuracy is on the 2,347 boxes "
          "that have an actuator (valves, dampers, pumps). The fault-tree type (MOV, AOV, XVM, CKV, CRB, ...) follows from "
          "class + actuator (`ft_component_types.csv`); its accuracy is on the 2,963 boxes that have one. Open/closed "
          "accuracy is on the 1,626 boxes with a normal position.", "",
          "## Boxes not on the review sheet", "",
          "The 479 boxes on review.xlsx have a class or position that is still to be confirmed.", "",
          "| model | test | input | boxes | class | fault-tree type | open/closed |", "|---|---|---|---:|---:|---:|---:|"]
    for r in rows[len(res["frozen"]):]:
        s = r["not_in_review"]
        L.append(f"| {r['model']} | {r['protocol']} | {r['input']} | {s['n']} | {cell(s, 'class_acc')} | {cell(s, 'ft_acc')} | {cell(s, 'pos_acc')} |")
    L += ["", "## Class accuracy per family", "",
          "| model | test | input | " + " | ".join(fams) + " |", "|---|---|---|" + "---:|" * len(fams)]
    for r in rows:
        L.append(f"| {r['model']} | {r['protocol']} | {r['input']} | " + " | ".join(pct(r["per_family"][f]) for f in fams) + " |")
    L += ["", "## Confidence", "", "Share of boxes whose most likely class has at least this probability, and the class "
          "accuracy on them. `predict.py` marks the boxes below 0.7 with `needs_review`.", "",
          "| model | test | input | p >= 0.5 | p >= 0.7 | p >= 0.9 |", "|---|---|---|---|---|---|"]
    for r in rows[len(res["frozen"]):]:
        c = r["coverage"]
        L.append(f"| {r['model']} | {r['protocol']} | {r['input']} | " + " | ".join(
            f"{pct(c[t]['share'])}% of boxes, {pct(c[t]['class_acc'])}% right" for t in ("0.5", "0.7", "0.9")) + " |")
    if "per_class" in res:
        L += ["", "## Per class (ResNet-18 + legend symbols, new family, image + text)", "",
              "| class | boxes | precision | recall | F1 |", "|---|---:|---:|---:|---:|"]
        for r in res["per_class"]:
            L.append(f"| {r['class']} | {r['n']} | {pct(r['precision'])} | {pct(r['recall'])} | {pct(r['f1'])} |")
        L += ["", "## Most frequent confusions (same model)", "", "Examples of each: `confusions.png`.", "",
              "| true class | predicted | boxes |", "|---|---|---:|"]
        for r in res["confusions"]:
            L.append(f"| {r['true']} | {r['predicted']} | {r['n']} |")
    open(path, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
