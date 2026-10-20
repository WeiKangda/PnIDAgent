"""Page Relational Transformer (PRT): end-to-end replacement for the hand-crafted features + gradient boosting + de-duplication + box regression of re-scoring.
One page = a set of tokens: symbol candidates (64px crop -> ResNet18 encoder, trained end-to-end with the whole model; + per-detector scores + geometry),
                  pipe segments (auto-extracted, geometry), OCR text boxes (geometry + character count).
Geometric relation attention: relative position / relative size / overlap / containment / gap of any two tokens -> one bias per attention head (like Relation Network).
Output: whether a candidate is a "correctly boxed symbol" (one-to-one target, so the model learns de-duplication itself) + four-side box refinement; auxiliary: whether a segment is a pipe (human pipe supervision on A pages).
  python rel_model.py fold <k> <gpu> <tag> [noline] [notext] [nocnn] [det] [mt] [pre|wpre] [pu=<previous-round tag>]    # fold k (A split into 5 folds by document), saves out-of-fold predictions + B predictions
  python rel_model.py eval <tag>                                         # pick thresholds on A out-of-fold predictions, report B"""
import glob, json, os, re, sys, time, random
import numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn, torchvision
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from selfsim import sp, f1
DATA = f"{HERE}/results/" + os.environ.get("REL_DATA", "rel_data"); OUTD = f"{HERE}/results/rel_runs"; os.makedirs(OUTD, exist_ok=True)
doc = lambda n: re.sub(r"_p\d+$", "", n).replace("SCS2_", "SCS_")


def folds5():
    docs = sorted({doc(n) for n in sp["A_train"]}); perm = np.random.RandomState(0).permutation(len(docs))
    f = {d: perm[i] % 5 for i, d in enumerate(docs)}; return [[n for n in sp["A_train"] if f[doc(n)] == k] for k in range(5)]


def load(n):
    z = np.load(f"{DATA}/{n}.npz"); d = {k: z[k] for k in z.files}
    p = f"{HERE}/results/rel_data_det/{n}.npy"
    if os.path.exists(p): d["det"] = np.load(p).astype(np.float32)      # soup5 neck ROI features (rel_detfeat.py)
    return d


# ---------------- Model ----------------
def pair_feats(bx, sc):
    """bx: N x 4 (original image pixels), sc: scale to the 3300 reference. Returns N x N x 8 relation features (i = query, j = key)."""
    cx, cy = (bx[:, 0] + bx[:, 2]) / 2, (bx[:, 1] + bx[:, 3]) / 2; w = (bx[:, 2] - bx[:, 0]).clamp(min=1); h = (bx[:, 3] - bx[:, 1]).clamp(min=1)
    dx = (cx[None] - cx[:, None]) * sc / 100; dy = (cy[None] - cy[:, None]) * sc / 100
    ix = (torch.minimum(bx[None, :, 2], bx[:, None, 2]) - torch.maximum(bx[None, :, 0], bx[:, None, 0])).clamp(min=0)
    iy = (torch.minimum(bx[None, :, 3], bx[:, None, 3]) - torch.maximum(bx[None, :, 1], bx[:, None, 1])).clamp(min=0)
    inter = ix * iy; a = w * h; iou = inter / (a[:, None] + a[None] - inter)
    gx = (torch.maximum(bx[None, :, 0], bx[:, None, 0]) - torch.minimum(bx[None, :, 2], bx[:, None, 2])).clamp(min=0)
    gy = (torch.maximum(bx[None, :, 1], bx[:, None, 1]) - torch.minimum(bx[None, :, 3], bx[:, None, 3])).clamp(min=0)
    gap = torch.sqrt(gx ** 2 + gy ** 2) * sc / 10
    return torch.stack([torch.sign(dx) * torch.log1p(dx.abs()), torch.sign(dy) * torch.log1p(dy.abs()), torch.log(w[None] / w[:, None]),
                        torch.log(h[None] / h[:, None]), iou, inter / a[:, None], inter / a[None], torch.log1p(gap)], -1)


class Layer(nn.Module):
    def __init__(s, d, H):
        super().__init__(); s.H = H; s.n1 = nn.LayerNorm(d); s.qkv = nn.Linear(d, 3 * d); s.o = nn.Linear(d, d)
        s.n2 = nn.LayerNorm(d); s.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d)); s.drop = nn.Dropout(float(os.environ.get("REL_DROP", 0.1)))
    def forward(s, x, bias):                                   # x: N x d, bias: H x N x N
        N, d = x.shape; q, k, v = s.qkv(s.n1(x)).view(N, 3, s.H, d // s.H).unbind(1)
        att = torch.einsum("ihc,jhc->hij", q, k) / (d // s.H) ** 0.5 + bias
        x = x + s.drop(s.o(torch.einsum("hij,jhc->ihc", att.softmax(-1), v).reshape(N, d)))
        return x + s.drop(s.ff(s.n2(x)))


class PRT(nn.Module):
    def __init__(s, nview=7, d=256, H=8, L=4, use_line=True, use_text=True, use_cnn=True, use_det=False):
        super().__init__(); s.use_line, s.use_text, s.use_cnn, s.use_det = use_line, use_text, use_cnn, use_det
        if use_det: s.det_in = nn.Sequential(nn.LayerNorm(3840), nn.Dropout(0.2), nn.Linear(3840, d), nn.GELU())
        if use_cnn:
            r = torchvision.models.resnet18(weights="IMAGENET1K_V1"); r.fc = nn.Identity(); s.cnn = r
        s.cand = nn.Sequential(nn.Linear((512 if use_cnn else 0) + (d if use_det else 0) + nview + 5, d), nn.GELU(), nn.Linear(d, d))
        s.line = nn.Sequential(nn.Linear(5, d), nn.GELU(), nn.Linear(d, d)); s.text = nn.Sequential(nn.Linear(7, d), nn.GELU(), nn.Linear(d, d))
        s.typ = nn.Embedding(3, d); s.rel = nn.Sequential(nn.Linear(8, 32), nn.GELU(), nn.Linear(32, H * L)); s.H, s.L = H, L
        s.layers = nn.ModuleList([Layer(d, H) for _ in range(L)])
        s.out_c = nn.Linear(d, 1); s.out_r = nn.Linear(d, 4); s.out_l = nn.Linear(d, 1)

    def forward(s, P):
        B, sc, S = P["B"], P["sc"], P["S"]; w = (B[:, 2] - B[:, 0]).clamp(min=1); h = (B[:, 3] - B[:, 1]).clamp(min=1)
        geo = torch.stack([(B[:, 0] + B[:, 2]) / 2 / S, (B[:, 1] + B[:, 3]) / 2 / S, torch.log(w * sc), torch.log(h * sc), torch.log(w / h)], 1)
        f = [P["V"], geo]
        if s.use_cnn: f.insert(0, s.cnn(P["img"]))
        if s.use_det: f.insert(0, s.det_in(P["det"]))
        toks = [s.cand(torch.cat(f, 1)) + s.typ.weight[0] + (s.stage_emb.weight[P["stage"]] if "stage" in P and hasattr(s, "stage_emb") else 0)]; boxes = [B]
        if s.use_line and len(P["L"]):
            Lb = P["L"][:, :4]; lw = (Lb[:, 2] - Lb[:, 0]).clamp(min=1); lh = (Lb[:, 3] - Lb[:, 1]).clamp(min=1)
            toks.append(s.line(torch.stack([(Lb[:, 0] + Lb[:, 2]) / 2 / S, (Lb[:, 1] + Lb[:, 3]) / 2 / S, torch.log(lw * sc), torch.log(lh * sc), P["L"][:, 4]], 1)) + s.typ.weight[1]); boxes.append(Lb)
        if s.use_text and len(P["TX"]):
            Tb = P["TX"][:, :4]; tw = (Tb[:, 2] - Tb[:, 0]).clamp(min=1); th = (Tb[:, 3] - Tb[:, 1]).clamp(min=1)
            toks.append(s.text(torch.stack([(Tb[:, 0] + Tb[:, 2]) / 2 / S, (Tb[:, 1] + Tb[:, 3]) / 2 / S, torch.log(tw * sc), torch.log(th * sc), torch.log(tw / th),
                                            torch.log1p(P["TX"][:, 4]), P["TX"][:, 5] / P["TX"][:, 4].clamp(min=1)], 1)) + s.typ.weight[2]); boxes.append(Tb)
        x = torch.cat(toks); bx = torch.cat(boxes); nc, nl = len(B), (len(P["L"]) if s.use_line else 0)
        bias = s.rel(pair_feats(bx, sc)).permute(2, 0, 1)        # (H*L) x N x N
        for i, layer in enumerate(s.layers): x = layer(x, bias[i * s.H:(i + 1) * s.H])
        xc = x[:nc]; xl = x[nc:nc + nl]
        return s.out_c(xc).squeeze(1), s.out_r(xc), (s.out_l(xl).squeeze(1) if nl else None)


# ---------------- Data -> tensors (with augmentation) ----------------
def make_mask(B):
    w, h = B[:, 2] - B[:, 0], B[:, 3] - B[:, 1]; L = 1.5 * np.maximum(w, h) + 8; k = 64 / L
    xs = np.arange(64) + 0.5; mx = (np.abs(xs[None] - 32) <= (w * k / 2)[:, None]); my = (np.abs(xs[None] - 32) <= (h * k / 2)[:, None])
    return (my[:, :, None] & mx[:, None, :]).astype(np.float32)


def to_tensors(z, dev, aug=False):
    W, H = z["WH"]; B, crops, L, TX = z["B"].copy(), z["crops"].astype(np.float32) / 255, z["L"].copy(), z["TX"].copy()
    keep = np.ones(len(B), bool); lk = np.ones(len(L), bool); tk = np.ones(len(TX), bool); fx = fy = False
    if aug:
        fx, fy = random.random() < 0.5, random.random() < 0.5
        if random.random() < 0.5:     # random sub-window
            ww, hh = W * random.uniform(0.55, 1), H * random.uniform(0.55, 1); x0, y0 = random.uniform(0, W - ww), random.uniform(0, H - hh)
            inw = lambda b: ((b[:, 0] + b[:, 2]) / 2 >= x0) & ((b[:, 0] + b[:, 2]) / 2 <= x0 + ww) & ((b[:, 1] + b[:, 3]) / 2 >= y0) & ((b[:, 1] + b[:, 3]) / 2 <= y0 + hh)
            keep, lk, tk = inw(B), inw(L) if len(L) else lk, inw(TX) if len(TX) else tk
            if keep.sum() < 5: keep[:] = True; lk[:] = True; tk[:] = True
    def flip(b, axis, size):
        b = b.copy(); i0, i1 = (0, 2) if axis == 0 else (1, 3); b[:, [i0, i1]] = size - b[:, [i1, i0]]; return b
    if fx: B = flip(B, 0, W); L[:, :4] = flip(L[:, :4], 0, W) if len(L) else L[:, :4]; TX[:, :4] = flip(TX[:, :4], 0, W) if len(TX) else TX[:, :4]; crops = crops[:, :, ::-1]
    if fy: B = flip(B, 1, H); L[:, :4] = flip(L[:, :4], 1, H) if len(L) else L[:, :4]; TX[:, :4] = flip(TX[:, :4], 1, H) if len(TX) else TX[:, :4]; crops = crops[:, ::-1, :]
    img = np.stack([crops, make_mask(B), crops], 1)[keep]
    t = lambda a: torch.tensor(np.ascontiguousarray(a), dtype=torch.float32, device=dev)
    tgt = z["tgt"].copy()
    if fx: tgt = flip(tgt, 0, W)
    if fy: tgt = flip(tgt, 1, H)
    return {"det": t(z["det"][keep]) if "det" in z else None, "B": t(B[keep]), "V": t(z["V"][keep]), "img": t(img), "L": t(L[lk]), "TX": t(TX[tk]), "sc": 3300 / max(W, H), "S": float(max(W, H)),
            "y": t(z["y"][keep]), "tgt": t(tgt[keep]), "has": t(z["has"][keep]), "ly": t(z["ly"][lk]), "keep": keep}


def deltas(B, T):
    w, h = (B[:, 2] - B[:, 0]).clamp(min=1), (B[:, 3] - B[:, 1]).clamp(min=1)
    return torch.stack([(T[:, 0] - B[:, 0]) / w, (T[:, 1] - B[:, 1]) / h, (T[:, 2] - B[:, 2]) / w, (T[:, 3] - B[:, 3]) / h], 1).clamp(-1, 1)


def apply_deltas(B, d):
    w, h = B[:, 2] - B[:, 0], B[:, 3] - B[:, 1]
    return np.column_stack([B[:, 0] + d[:, 0] * w, B[:, 1] + d[:, 1] * h, B[:, 2] + d[:, 2] * w, B[:, 3] + d[:, 3] * h])


@torch.no_grad()
def predict(model, z, dev):
    model.eval(); P = to_tensors(z, dev); outs = []
    for fx in (False, True):        # test-time horizontal-flip averaging
        Q = to_tensors(z, dev)
        if fx:
            W = float(z["WH"][0]); Q["B"] = torch.stack([W - Q["B"][:, 2], Q["B"][:, 1], W - Q["B"][:, 0], Q["B"][:, 3]], 1); Q["img"] = Q["img"].flip(3)
            for k in ("L", "TX"):
                if len(Q[k]): Q[k] = torch.cat([torch.stack([W - Q[k][:, 2], Q[k][:, 1], W - Q[k][:, 0], Q[k][:, 3]], 1), Q[k][:, 4:]], 1)
        c, r, _ = model(Q); r = r.clone()
        if fx: r = torch.stack([-r[:, 2], r[:, 1], -r[:, 0], r[:, 3]], 1)
        outs.append((torch.sigmoid(c).cpu().numpy(), r.cpu().numpy()))
    return np.mean([o[0] for o in outs], 0), np.mean([o[1] for o in outs], 0)


def train_fold(k, dev, tag, flags):
    torch.manual_seed(k); random.seed(k); np.random.seed(k)
    F5 = folds5(); tr = [n for i, f in enumerate(F5) if i != k for n in f]; te = F5[k]
    Z = {n: load(n) for n in tr + te + sp["B_test"]}
    model = PRT(d=int(os.environ.get("REL_D", 256)), L=int(os.environ.get("REL_L", 4)), use_line="noline" not in flags, use_text="notext" not in flags, use_cnn="nocnn" not in flags, use_det="det" in flags).to(dev)
    if "wpre" in flags:  # init crop encoder from the appearance network pretrained with large-scale weak supervision on P&IDs (pretrain_cnn.py, no human annotations, same for all folds)
        sd = torch.load(f"{HERE}/results/{os.environ.get('REL_PRE', 'cnn_weakpre')}.pt", map_location=dev); sd = {kk: v for kk, v in sd.items() if not kk.startswith("fc.")}
        print("Weakly supervised pretrained encoder:", model.cnn.load_state_dict(sd, strict=False), flush=True)
    elif "pre" in flags:   # init crop encoder from this fold's pretrained crop CNN (crop_cnn.py train K<k>)
        sd = torch.load(f"{HERE}/results/cnn_K{k}.pt", map_location=dev); sd = {kk: v for kk, v in sd.items() if not kk.startswith("fc.")}
        print("Pretrained encoder:", model.cnn.load_state_dict(sd, strict=False), flush=True)
    cnn_p = [p for n_, p in model.named_parameters() if n_.startswith("cnn.")]; rest = [p for n_, p in model.named_parameters() if not n_.startswith("cnn.")]
    EP = int(os.environ.get("REL_EP", 60)); steps = EP * len(tr)
    opt = torch.optim.AdamW([{"params": cnn_p, "lr": 2e-4}, {"params": rest, "lr": 4e-4}], weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, [2e-4, 4e-4], total_steps=steps, pct_start=0.08)
    PU = [f.split("=", 1)[1] for f in flags if f.startswith("pu=")]
    if PU:   # nnPU adaptation: on target pages (this fold's held-out A pages + B pages), candidates with previous-round p>=0.6 are positives, all others "unlabeled" (not negatives)
        srcp = np.load(f"{OUTD}/{PU[0]}_fold{k}.npy", allow_pickle=True).item()
        tgt_pages = te + sp["B_test"]; posm = {n: srcp[n][0] >= 0.6 for n in tgt_pages}
        prior = {n: float(np.clip(srcp[n][0].mean(), 0.02, 0.6)) for n in tgt_pages}     # per-page positive ratio = previous round's expected hits / number of candidates
    MT = "mt" in flags; unl = te + sp["B_test"]        # Mean Teacher: unlabeled pages = this fold's held-out A pages + B pages (same conditions for both, so out-of-fold evaluation is fair)
    if MT:
        import copy; teacher = copy.deepcopy(model).eval()
        for p in teacher.parameters(): p.requires_grad_(False)
    t0 = time.time(); step = 0
    for ep in range(EP):
        model.train(); tot = 0
        for n in random.sample(tr, len(tr)):
            P = to_tensors(Z[n], dev, aug=True); c, r, l = model(P)
            loss = Fn.binary_cross_entropy_with_logits(c, P["y"], pos_weight=torch.tensor(2.0, device=dev))
            m = P["has"] > 0
            if m.any(): loss = loss + 2 * Fn.smooth_l1_loss(r[m], deltas(P["B"][m], P["tgt"][m]), beta=0.05)
            if l is not None:
                lm = P["ly"] >= 0
                if lm.any(): loss = loss + 0.3 * Fn.binary_cross_entropy_with_logits(l[lm], P["ly"][lm])
            if PU:
                u = random.choice(tgt_pages); Pu = to_tensors(Z[u], dev, aug=True); cu, _, _ = model(Pu)
                pm = torch.tensor(posm[u][Pu["keep"]], device=dev); pi = prior[u]; ramp = min(1.0, step / (0.3 * steps))
                if pm.any():
                    rp_pos = Fn.softplus(-cu[pm]).mean(); rp_neg = Fn.softplus(cu[pm]).mean(); ru_neg = Fn.softplus(cu).mean()
                    loss = loss + ramp * (pi * rp_pos + torch.clamp(ru_neg - pi * rp_neg, min=0))
            if MT:   # consistency: teacher sees the original page, student the same page augmented; student probabilities should match the teacher's soft probabilities
                u = random.choice(unl); Ps = to_tensors(Z[u], dev, aug=True)
                with torch.no_grad(): pt = torch.sigmoid(teacher(to_tensors(Z[u], dev))[0])[torch.tensor(Ps["keep"], device=dev)]
                cs, _, _ = model(Ps); ramp = min(1.0, step / (0.3 * steps))
                loss = loss + ramp * Fn.binary_cross_entropy_with_logits(cs, pt)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step(); tot += loss.item(); step += 1
            if MT:
                with torch.no_grad():
                    for pt_, ps_ in zip(teacher.parameters(), model.parameters()): pt_.mul_(0.995).add_(ps_.detach(), alpha=0.005)
                    for bt, bs in zip(teacher.buffers(), model.buffers()): bt.copy_(bs)
        if ep % 10 == 9 or ep == EP - 1: print(f"[fold {k}] ep {ep + 1}/{EP} loss {tot / len(tr):.4f} {time.time() - t0:.0f}s", flush=True)
    res = {n: predict(teacher if MT else model, Z[n], dev) for n in te + sp["B_test"]}
    np.save(f"{OUTD}/{tag}_fold{k}.npy", {n: (p, r) for n, (p, r) in res.items()}, allow_pickle=True)
    print(f"FOLD_DONE {k}", flush=True)


def evaluate(tag):
    sys.path.insert(0, HERE); from rescore_pool import select
    R = {}; B5 = {}
    for k in range(5):
        d = np.load(f"{OUTD}/{tag}_fold{k}.npy", allow_pickle=True).item()
        for n, v in d.items():
            if n in sp["B_test"]: B5.setdefault(n, []).append(v)
            else: R[n] = v
    pb = {n: (np.mean([v[0] for v in B5[n]], 0), np.mean([v[1] for v in B5[n]], 0)) for n in B5}
    Z = {n: load(n) for n in list(R) + list(pb)}
    def run(n, pr, reg, g, use_reg):
        B = apply_deltas(Z[n]["B"], reg) if use_reg else Z[n]["B"]
        return f1(select(list(B), pr, *g), list(Z[n]["G"]))
    grid = [(t, nt, ct) for t in (0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6) for nt in (0.3, 0.5, 0.7, 1.01) for ct in (0.6, 0.8, 1.01)]
    best = None
    for ur in (False, True):
        acc = [np.mean([run(n, *R[n], g, ur) for n in sp["A_train"]]) for g in grid]; i = int(np.argmax(acc))
        b = np.mean([run(n, *pb[n], grid[i], ur) for n in sp["B_test"]])
        print(f"{tag} {'+box refine' if ur else 'no box refine'}: A out-of-fold {acc[i]:.4f} {grid[i]} -> B {b:.4f}", flush=True)
        if best is None or acc[i] > best[0]: best = (acc[i], ur, grid[i], b)
    print(f"{tag} selected on A: A {best[0]:.4f} -> B {best[3]:.4f}  (box refine {best[1]}, {best[2]})")
    json.dump({"A": best[0], "B": best[3], "reg": best[1], "cfg": best[2]}, open(f"{OUTD}/{tag}_eval.json", "w"))


if __name__ == "__main__":
    if sys.argv[1] == "fold": train_fold(int(sys.argv[2]), f"cuda:{sys.argv[3]}", sys.argv[4], sys.argv[5:])
    else: evaluate(sys.argv[2])
