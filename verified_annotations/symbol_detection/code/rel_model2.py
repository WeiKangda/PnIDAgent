"""页面关系模型 v2: 级联框修正 + 实时 ROI 图块.
与 v1 (rel_model.py) 的区别:
  1. 图块不再预存: 整页灰度图放显卡, 按框用 ROIAlign 实时裁 64x64 (正方形, 边长 1.5x 框长边 + 8, 与 crop_cnn.crop 同规则) -> 任意位置的框都能裁.
  2. 级联: 第 1 遍输出 (是不是符号, 框修正) -> 在修正后的框上重新裁图块, 同一网络再过第 2 遍 -> 最终概率 + 再修正.
     一一对应标签 / 回归目标按每一遍的框实时重算 (第 2 遍的框更准, 判断"框准不准"的标准更严).
  3. 训练时候选框随机抖动 (+-6%), 标签随之重算, 让定位学得更稳.
其余 (三类 token, 几何关系注意力, 文字/管线, 预训练编码器, 5 折, 评估) 同 v1.
  python rel_model2.py fold <k> <gpu> <tag> [wpre] [stages=2]
  python rel_model2.py eval <tag>"""
import glob, json, os, re, sys, time, random
import cv2, numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn
from torchvision.ops import roi_align
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from selfsim import sp, f1, SRC
import rel_model as V1
from rel_prep import one2one
OUTD = V1.OUTD; JIT = float(os.environ.get("REL_JIT", 0))


def page_img(n):
    if n.startswith("real4_"): return cv2.imread(f"{HERE}/real4/{n}.png", 0)
    return cv2.imread(glob.glob(f"{SRC}/{glob.escape(n)}.*")[0], 0)


def roi_crops(img, B):
    """img: 1x1xHxW (墨迹 0-1), B: Nx4 -> Nx3x64x64 [墨迹, 框掩码, 墨迹]."""
    w, h = (B[:, 2] - B[:, 0]).clamp(min=1), (B[:, 3] - B[:, 1]).clamp(min=1); L = 1.5 * torch.maximum(w, h) + 8
    cx, cy = (B[:, 0] + B[:, 2]) / 2, (B[:, 1] + B[:, 3]) / 2
    rois = torch.stack([torch.zeros_like(cx), cx - L / 2, cy - L / 2, cx + L / 2, cy + L / 2], 1)
    a = roi_align(img, rois, (64, 64), spatial_scale=1.0, sampling_ratio=0, aligned=True)[:, 0]   # 0 = 自适应采样 (约等于面积平均, 大框缩小不混叠)
    k = 64 / L; xs = torch.arange(64, device=B.device) + 0.5
    mx = (xs[None] - 32).abs() <= (w * k / 2)[:, None]; my = (xs[None] - 32).abs() <= (h * k / 2)[:, None]
    m = (my[:, :, None] & mx[:, None, :]).float()
    return torch.stack([a, m, a], 1)


STORED = os.environ.get("REL_STORED") == "1"     # 诊断: 第 1 遍用预存图块 (和 v1 完全相同的输入)
CVCROP = os.environ.get("REL_CVCROP") == "1"     # 诊断: 实时裁图改用 OpenCV (和预存图块的生成方法完全相同)


def cv_crops(g, B):
    from crop_cnn import crop as _crop
    Bn = B.detach().cpu().numpy(); a = np.array([_crop(g, b)[0] for b in Bn]) if len(Bn) else np.zeros((0, 64, 64), np.float32)
    a = torch.tensor((a * 255).astype(np.uint8).astype(np.float32) / 255, device=B.device)
    return torch.stack([a, torch.tensor(V1.make_mask(Bn), device=B.device), a], 1)


STAGE_EMB = os.environ.get("REL_STAGEEMB") == "1"     # 级联: 阶段嵌入 + 第 2 遍修正幅度减半 (Cascade R-CNN 式)


class PRT2(V1.PRT):
    def __init__(s, *a, **kw):
        super().__init__(*a, **kw)
        if STAGE_EMB: s.stage_emb = nn.Embedding(3, 256)

    def tokens_forward(s, P, B, img, first=False, stage=0):
        P = dict(P); P["B"] = B
        if STAGE_EMB:
            P["stage"] = stage; c, r, l = s._fwd(P, B, img, first); return c, r * (1.0 if stage == 0 else 0.5), l
        return s._fwd(P, B, img, first)

    def _fwd(s, P, B, img, first):
        P["img"] = P["img0"] if (STORED and first) else (cv_crops(P["g_np"], B) if CVCROP else roi_crops(img, B)); return V1.PRT.forward(s, P)


def build(z, img_np, dev, aug=False):
    """返回 (P 字典, 页面墨迹张量, 标注框 G)."""
    W, H = z["WH"]; B, L, TX, G = z["B"].copy(), z["L"].copy(), z["TX"].copy(), z["G"].copy()
    keep = np.ones(len(B), bool); lk = np.ones(len(L), bool); tk = np.ones(len(TX), bool); fx = fy = False; ink = 1 - img_np.astype(np.float32) / 255
    if aug:
        fx, fy = random.random() < 0.5, random.random() < 0.5
        if random.random() < 0.5:
            ww, hh = W * random.uniform(0.55, 1), H * random.uniform(0.55, 1); x0, y0 = random.uniform(0, W - ww), random.uniform(0, H - hh)
            inw = lambda b: ((b[:, 0] + b[:, 2]) / 2 >= x0) & ((b[:, 0] + b[:, 2]) / 2 <= x0 + ww) & ((b[:, 1] + b[:, 3]) / 2 >= y0) & ((b[:, 1] + b[:, 3]) / 2 <= y0 + hh)
            keep = inw(B); lk = inw(L) if len(L) else lk; tk = inw(TX) if len(TX) else tk
            if keep.sum() < 5: keep[:] = True; lk[:] = True; tk[:] = True
        if JIT:   # 候选框抖动 (可选)
            bw, bh = B[:, 2] - B[:, 0], B[:, 3] - B[:, 1]; B = B + np.random.uniform(-JIT, JIT, B.shape) * np.stack([bw, bh, bw, bh], 1)
    def flip(b, axis, size):
        b = b.copy(); i0, i1 = (0, 2) if axis == 0 else (1, 3); b[:, [i0, i1]] = size - b[:, [i1, i0]]; return b
    for f, ax, sz in ((fx, 0, W), (fy, 1, H)):
        if not f: continue
        B = flip(B, ax, sz); G = flip(G, ax, sz) if len(G) else G
        if len(L): L[:, :4] = flip(L[:, :4], ax, sz)
        if len(TX): TX[:, :4] = flip(TX[:, :4], ax, sz)
        ink = ink[:, ::-1] if ax == 0 else ink[::-1, :]
    t = lambda a: torch.tensor(np.ascontiguousarray(a), dtype=torch.float32, device=dev)
    cr = z["crops"].astype(np.float32) / 255
    if fx: cr = cr[:, :, ::-1]
    if fy: cr = cr[:, ::-1, :]
    img0 = np.stack([cr, V1.make_mask(B), cr], 1)[keep]
    P = {"g_np": np.ascontiguousarray(((1 - ink) * 255).astype(np.uint8)), "img0": t(img0), "V": t(z["V"][keep]), "L": t(L[lk]), "TX": t(TX[tk]), "sc": 3300 / max(W, H), "S": float(max(W, H)), "ly": t(z["ly"][lk]),
         "det": t(z["det"][keep]) if "det" in z else None, "keep": keep}
    return P, t(B[keep]), t(ink)[None, None], G


def stage_loss(c, r, B, G, dev):
    y, tgt, has = one2one(B.detach().cpu().numpy(), G)
    y, tgt, has = (torch.tensor(a, device=dev) for a in (y, tgt, has))
    loss = Fn.binary_cross_entropy_with_logits(c, y, pos_weight=torch.tensor(2.0, device=dev)); m = has > 0
    if m.any(): loss = loss + 2 * Fn.smooth_l1_loss(r[m], V1.deltas(B[m], tgt[m]), beta=0.05)
    return loss


def refine(B, r):
    """按修正量移动四边; 保证框有效 (宽高 >= 原来的 30%, 且不翻转)."""
    w, h = (B[:, 2] - B[:, 0]).clamp(min=1), (B[:, 3] - B[:, 1]).clamp(min=1)
    x0, y0, x1, y1 = B[:, 0] + r[:, 0] * w, B[:, 1] + r[:, 1] * h, B[:, 2] + r[:, 2] * w, B[:, 3] + r[:, 3] * h
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2; nw = torch.maximum(x1 - x0, 0.3 * w); nh = torch.maximum(y1 - y0, 0.3 * h)
    return torch.stack([cx - nw / 2, cy - nh / 2, cx + nw / 2, cy + nh / 2], 1)


@torch.no_grad()
def predict(model, z, img_np, dev, stages):
    model.eval(); outs = []
    for fx in (False, True):
        P, B, img, _ = build(z, img_np, dev)
        W = float(z["WH"][0])
        if fx:   # 翻转那一遍: 框/整页墨迹/OpenCV 用的整页灰度/预存图块 全部一起翻
            B = torch.stack([W - B[:, 2], B[:, 1], W - B[:, 0], B[:, 3]], 1); img = img.flip(3)
            P = dict(P); P["g_np"] = np.ascontiguousarray(P["g_np"][:, ::-1]); P["img0"] = P["img0"].flip(3)
            for k in ("L", "TX"):
                if len(P[k]): P[k] = torch.cat([torch.stack([W - P[k][:, 2], P[k][:, 1], W - P[k][:, 0], P[k][:, 3]], 1), P[k][:, 4:]], 1)
        Bc = B; ps, bs = [], []
        for st in range(stages):
            c, r, _ = model.tokens_forward(P, Bc, img, first=(st == 0), stage=st); Bc = refine(Bc, r.clamp(-1, 1)); ps.append(torch.sigmoid(c)); bs.append(Bc)
        # 推理组合 (Cascade R-CNN 式): CASC = last (最后一遍分数+框) / avg_last (各遍分数平均+最后框) / avg_first (平均分数+第一遍框) / last_first (最后分数+第一遍框)
        mode = os.environ.get("REL_CASC", "last")
        p = ps[-1] if mode in ("last", "last_first") else torch.stack(ps).mean(0); Bo = bs[0] if mode in ("avg_first", "last_first") else bs[-1]
        if fx: Bo = torch.stack([W - Bo[:, 2], Bo[:, 1], W - Bo[:, 0], Bo[:, 3]], 1)
        outs.append((p.cpu().numpy(), Bo.cpu().numpy()))
    return np.mean([o[0] for o in outs], 0), np.mean([o[1] for o in outs], 0)


def train_fold(k, dev, tag, flags):
    """k = "all": 用全部 65 页 (A+B) 训练, 预测 real4 (最终模型)."""
    sd0 = (99 if k == "all" else k) + 100 * int(os.environ.get("REL_SEED", 0)); torch.manual_seed(sd0); random.seed(sd0); np.random.seed(sd0)   # REL_SEED: 换随机种子重跑
    ST = int(([f.split("=")[1] for f in flags if f.startswith("stages=")] or [2])[0])
    if k == "all":
        tr = sp["A_train"] + sp["B_test"]; te = []; test_pages = sorted(os.path.basename(f)[:-4] for f in glob.glob(f"{HERE}/real4/real4_*.png"))
    else:
        F5 = V1.folds5(); tr = [n for i, f in enumerate(F5) if i != k for n in f]; te = F5[k]; test_pages = sp["B_test"]
    Z = {n: V1.load(n) for n in tr + te + test_pages}; IM = {n: page_img(n) for n in Z}
    model = PRT2(use_det="det" in flags).to(dev)
    if "wpre" in flags:
        sd = torch.load(f"{HERE}/results/{os.environ.get('REL_PRE', 'cnn_weakpre')}.pt", map_location=dev)
        model.cnn.load_state_dict({kk: v for kk, v in sd.items() if not kk.startswith("fc.")}, strict=False)
    cnn_p = [p for n_, p in model.named_parameters() if n_.startswith("cnn.")]; rest = [p for n_, p in model.named_parameters() if not n_.startswith("cnn.")]
    EP = int(os.environ.get("REL_EP", 60)); steps = EP * len(tr)
    opt = torch.optim.AdamW([{"params": cnn_p, "lr": 2e-4}, {"params": rest, "lr": 4e-4}], weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, [2e-4, 4e-4], total_steps=steps, pct_start=0.08); t0 = time.time()
    for ep in range(EP):
        model.train(); tot = 0
        for n in random.sample(tr, len(tr)):
            P, B, img, G = build(Z[n], IM[n], dev, aug=True); loss = 0; Bc = B
            for st in range(ST):
                c, r, l = model.tokens_forward(P, Bc, img, first=(st == 0), stage=st); loss = loss + stage_loss(c, r, Bc, G, dev)
                if st == 0 and l is not None:
                    lm = P["ly"] >= 0
                    if lm.any(): loss = loss + 0.3 * Fn.binary_cross_entropy_with_logits(l[lm], P["ly"][lm])
                Bc = refine(Bc, r.clamp(-1, 1)).detach()
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step(); tot += loss.item()
        if ep % 10 == 9 or ep == EP - 1: print(f"[fold {k}] ep {ep + 1}/{EP} loss {tot / len(tr):.4f} {time.time() - t0:.0f}s", flush=True)
    torch.save(model.state_dict(), f"{OUTD}/{tag}_fold{k}.pt")
    res = {n: predict(model, Z[n], IM[n], dev, ST) for n in te + test_pages}
    np.save(f"{OUTD}/{tag}_fold{k}.npy", res, allow_pickle=True); print(f"FOLD_DONE {k}", flush=True)


def evaluate(tag):
    from rescore_pool import select
    R = {}; B5 = {}
    for k in range(5):
        for n, v in np.load(f"{OUTD}/{tag}_fold{k}.npy", allow_pickle=True).item().items():
            (B5.setdefault(n, []).append(v) if n in sp["B_test"] else R.__setitem__(n, v))
    pb = {n: (np.mean([v[0] for v in B5[n]], 0), np.mean([v[1] for v in B5[n]], 0)) for n in B5}
    G = {n: list(V1.load(n)["G"]) for n in list(R) + list(pb)}
    grid = [(t, nt, ct) for t in (0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6) for nt in (0.3, 0.5, 0.7, 1.01) for ct in (0.6, 0.8, 1.01)]
    acc = [np.mean([f1(select(list(R[n][1]), R[n][0], *g), G[n]) for n in sp["A_train"]]) for g in grid]; i = int(np.argmax(acc))
    b = np.mean([f1(select(list(pb[n][1]), pb[n][0], *grid[i]), G[n]) for n in sp["B_test"]])
    print(f"{tag} 按 A 选: A {acc[i]:.4f} -> B {b:.4f}  {grid[i]}")
    json.dump({"A": float(acc[i]), "B": float(b), "cfg": grid[i]}, open(f"{OUTD}/{tag}_eval.json", "w"))


OUTTAG = None


def repredict(tag, flags):
    """用已保存的各折模型重新预测 (修预测 bug 时不必重训)."""
    ST = int(([f.split("=")[1] for f in flags if f.startswith("stages=")] or [2])[0]); dev = "cuda:0"; F5 = V1.folds5()
    for k in range(5):
        model = PRT2(use_det="det" in flags).to(dev); model.load_state_dict(torch.load(f"{OUTD}/{tag}_fold{k}.pt", map_location=dev))
        pages = F5[k] + sp["B_test"]; res = {n: predict(model, V1.load(n), page_img(n), dev, ST) for n in pages}
        np.save(f"{OUTD}/{OUTTAG}_fold{k}.npy", res, allow_pickle=True); print("REPRED", k, flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "repredict":
        OUTTAG = sys.argv[2] + os.environ.get("REL_OUTSUF", ""); repredict(sys.argv[2], sys.argv[3:]); evaluate(OUTTAG); sys.exit()
    if sys.argv[1] == "fold": train_fold(sys.argv[2] if sys.argv[2] == "all" else int(sys.argv[2]), f"cuda:{sys.argv[3]}", sys.argv[4], sys.argv[5:])
    else: evaluate(sys.argv[2])
