"""Fine-tune the network of model.py on the symbol crops.

  python train_cnn.py --cache <dir> --protocol family [--legend]   # hold out each document family in turn
  python train_cnn.py --cache <dir> --protocol page [--legend]     # 5 folds of whole pages (all families seen)
      -> <dir>/cnn_<protocol>[_legend].npz: held-out softmax of every box, with and without the context
  python train_cnn.py --cache <dir> --final --legend --save model/symbol_classifier_v3.pt   # train on all 65 pages

--legend adds the legend symbols of legend_library (one crop each, class only) to every training set.
Recipe (fixed, not tuned on the held-out boxes): 30 epochs, batch 64, AdamW 5e-4 with warm-up and cosine decay,
weight decay 1e-4; class loss with label smoothing 0.1 and class weights (n / n_max) ** -0.25; actuator and position
losses (weight 0.5) only on the boxes that have them.
"""
import argparse, json, os, time

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
import torch.nn.functional as F

from common import CROP, CTX_DIM, HASH_DIM, family_folds, ft_types, load_cache, load_classes
from model import DEV, Net, augment, cosine_lr, predict_tta, to_input

EPOCHS, BS, LR = 30, 64, 5e-4


def page_folds(d, k=5):
    """k folds of whole pages with about the same number of boxes (largest pages first, each to the smallest fold)."""
    pages = sorted(set(d["page"].tolist()))
    cnt = {p: int((d["page"] == p).sum()) for p in pages}
    folds, load = [[] for _ in range(k)], [0] * k
    for p in sorted(pages, key=lambda p: (-cnt[p], p)):
        i = int(np.argmin(load)); folds[i].append(p); load[i] += cnt[p]
    out = []
    for i, ps in enumerate(folds):
        te = np.isin(d["page"], ps)
        out.append((f"page fold {i + 1}", np.where(~te)[0], np.where(te)[0]))
    return out


def head_classes(d, meta):
    """Classes whose boxes (mostly) have an actuator, and classes with a normal position (open / closed)."""
    dash = meta["actuators"].index("-")
    act = [c for c in range(len(meta["classes"])) if (d["cls"] == c).any() and (d["act"][d["cls"] == c] != dash).mean() > 0.5]
    pos = [c for c in range(len(meta["classes"])) if (d["pos"][d["cls"] == c] >= 0).any()]
    return act, pos


def train(ink, ctx, yc, ya, yp, n_cls, n_act, epochs=EPOCHS, seed=0, log=print):
    """ink (N, 1, H, W) float in [0, 1], 1 = black; ctx (N, CTX_DIM); ya / yp = -1 where the box has no actuator / position."""
    torch.manual_seed(seed); np.random.seed(seed)
    net = Net(n_cls, n_act, ctx.shape[1]).to(DEV)
    cnt = np.bincount(yc, minlength=n_cls).astype(float)
    w = torch.tensor(np.where(cnt > 0, (np.maximum(cnt, 1) / cnt.max()) ** -0.25, 0.0), dtype=torch.float32, device=DEV)
    X, C = ink.to(DEV), ctx.to(DEV)
    Yc, Ya, Yp = (torch.as_tensor(v, dtype=torch.long, device=DEV) for v in (yc, ya, yp))
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-4)
    n = len(yc); total = epochs * (n // BS); step = 0; t0 = time.time()
    for ep in range(epochs):
        net.train(); perm = torch.randperm(n, device=DEV); run = 0.0
        for s in range(0, n - BS + 1, BS):
            b = perm[s:s + BS]
            cosine_lr(opt, LR, step, total); step += 1
            keep = (torch.rand(len(b), 1, device=DEV) >= 0.3).float()          # drop the whole context of 30% of boxes
            lc, la, lp = net(to_input(augment(X[b])), C[b] * keep)
            loss = F.cross_entropy(lc, Yc[b], weight=w, label_smoothing=0.1)
            ma, mp = Ya[b] >= 0, Yp[b] >= 0
            if ma.any(): loss = loss + 0.5 * F.cross_entropy(la[ma], Ya[b][ma], label_smoothing=0.05)
            if mp.any(): loss = loss + 0.5 * F.cross_entropy(lp[mp], Yp[b][mp], label_smoothing=0.05)
            opt.zero_grad(); loss.backward(); opt.step(); run += loss.item()
        if (ep + 1) % 5 == 0 or ep == 0:
            log(f"    epoch {ep + 1:2d}/{epochs}  loss {run / max(1, n // BS):.3f}  {time.time() - t0:.0f}s")
    return net


def tensors(d, meta, legend=None):
    """Network inputs and targets of the boxes, and of the legend symbols when given (no context, no actuator / position)."""
    dash = meta["actuators"].index("-")
    ink = torch.from_numpy(255 - d["crops"]).float().div(255).unsqueeze(1)
    ctx = torch.from_numpy(d["ctx_text"])
    ya = np.where(d["act"] == dash, -1, d["act"]); yp = d["pos"]
    if legend is None: return ink, ctx, d["cls"], ya, yp
    lin = torch.from_numpy(255 - legend["crops"]).float().div(255).unsqueeze(1)
    m = len(lin)
    return lin, torch.zeros(m, ctx.shape[1]), legend["cls"], -np.ones(m, int), -np.ones(m, int)


def cat(*parts):
    return [torch.cat(x) if isinstance(x[0], torch.Tensor) else np.concatenate(x) for x in zip(*parts)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True); ap.add_argument("--protocol", choices=["family", "page"])
    ap.add_argument("--legend", action="store_true"); ap.add_argument("--final", action="store_true")
    ap.add_argument("--save"); ap.add_argument("--epochs", type=int, default=EPOCHS); ap.add_argument("--folds", type=int)
    a = ap.parse_args()
    d, meta = load_cache(a.cache)
    legend = dict(np.load(os.path.join(a.cache, "legend.npz"))) if a.legend else None
    n_cls, n_act = len(meta["classes"]), len(meta["actuators"])
    act_classes, pos_classes = head_classes(d, meta)
    boxes = tensors(d, meta); extra = tensors(d, meta, legend) if a.legend else None

    if a.final:
        parts = [boxes] + ([extra] if extra else [])
        ink, ctx, yc, ya, yp = cat(*parts)
        print(f"training on {len(yc)} crops ({len(d['cls'])} boxes{f' + {len(extra[2])} legend symbols' if extra else ''})")
        net = train(ink, ctx, yc, ya, yp, n_cls, n_act, a.epochs)
        os.makedirs(os.path.dirname(os.path.abspath(a.save)), exist_ok=True)
        torch.save({"state_dict": {k: (v.half() if v.is_floating_point() else v) for k, v in net.cpu().state_dict().items()},
                    "classes": meta["classes"], "actuators": meta["actuators"], "positions": meta["positions"],
                    "act_classes": [meta["classes"][c] for c in act_classes],
                    "pos_classes": [meta["classes"][c] for c in pos_classes],
                    "ft_types": ft_types(), "pra_modeled": {c: r["pra_modeled"] for c, r in load_classes()[1].items()},
                    "crop": CROP, "hash_dim": HASH_DIM, "ctx_dim": CTX_DIM, "epochs": a.epochs,
                    "trained_on": f"verified_annotations_v3, {len(d['cls'])} boxes on {len(set(d['page'].tolist()))} pages"
                                  + (f" + {len(extra[2])} legend symbols" if extra else "")}, a.save)
        print("saved", a.save, f"{os.path.getsize(a.save) / 1e6:.1f} MB")
        return

    folds = family_folds(d) if a.protocol == "family" else page_folds(d)
    if a.folds: folds = folds[:a.folds]
    N = len(d["cls"]); out = {f"{h}_{m}": np.zeros((N, k), np.float32) for m in ("ctx", "img")
                              for h, k in (("cls", n_cls), ("act", n_act), ("pos", 2))}
    fold_of = -np.ones(N, int)
    for fi, (name, tr, te) in enumerate(folds):
        t0 = time.time()
        parts = [[x[tr] for x in boxes]] + ([extra] if extra else [])
        net = train(*cat(*parts), n_cls, n_act, a.epochs, seed=fi, log=print)
        for mode, c in (("ctx", boxes[1][te]), ("img", torch.zeros_like(boxes[1][te]))):
            pc, pa, pp = predict_tta(net, boxes[0][te], c)
            out[f"cls_{mode}"][te], out[f"act_{mode}"][te], out[f"pos_{mode}"][te] = pc, pa, pp
        fold_of[te] = fi
        acc = (out["cls_ctx"][te].argmax(1) == d["cls"][te]).mean(); acc0 = (out["cls_img"][te].argmax(1) == d["cls"][te]).mean()
        print(f"{name:32s} {len(te):5d} boxes  class acc {acc:.3f} (image only {acc0:.3f})  {time.time() - t0:.0f}s", flush=True)
        del net
        if DEV == "mps": torch.mps.empty_cache()
    tag = f"cnn_{a.protocol}" + ("_legend" if a.legend else "")
    np.savez_compressed(os.path.join(a.cache, f"{tag}.npz"), fold=fold_of, fold_names=np.array([f[0] for f in folds]),
                        act_classes=np.array(act_classes), pos_classes=np.array(pos_classes), **out)
    print("wrote", tag)


if __name__ == "__main__":
    main()
