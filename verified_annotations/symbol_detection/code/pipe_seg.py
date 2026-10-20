"""方向 2: 学习式管线分割. 目标 = A 页人工核验管线 (lines_verified, 实线) 经过的墨迹像素; 只在墨迹像素上算损失 (空白像素天然不是管线).
网络: ResNet18 编码器 + U-Net 解码器, 原分辨率 512 块训练, 整页滑窗推理.
交叉拟合: A1 模型预测 A2 页, A2 模型预测 A1 页, B 用两者平均.  输出 results/pipe_seg/<page>.png (管线概率 x255, 只在墨迹上非零)
  python pipe_seg.py train <A1|A2> <gpu>
  python pipe_seg.py predict <gpu>
  python pipe_seg.py check            # 和人工管线比 (留出半), 对照规则提取"""
import glob, json, os, sys, random
import cv2, numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn, torchvision
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from selfsim import SRC, sp
LV = "/home/suxinqi666/code/PnIDAgent_project/line_results/lines_verified"; OUT = f"{HERE}/results/pipe_seg"; os.makedirs(OUT, exist_ok=True)
FOLDS = json.load(open(f"{HERE}/tile_folds.json"))


class UNet(nn.Module):
    def __init__(s):
        super().__init__(); r = torchvision.models.resnet18(weights="IMAGENET1K_V1")
        s.l0 = nn.Sequential(r.conv1, r.bn1, r.relu); s.l1 = nn.Sequential(r.maxpool, r.layer1); s.l2, s.l3, s.l4 = r.layer2, r.layer3, r.layer4
        def up(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True), nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))
        s.d3, s.d2, s.d1, s.d0 = up(512 + 256, 256), up(256 + 128, 128), up(128 + 64, 64), up(64 + 64, 32)
        s.head = nn.Sequential(nn.Conv2d(32 + 1, 16, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(16, 1, 1))
    def forward(s, x):
        x0 = s.l0(x); x1 = s.l1(x0); x2 = s.l2(x1); x3 = s.l3(x2); x4 = s.l4(x3)
        u = lambda a, b: torch.cat([Fn.interpolate(a, size=b.shape[-2:], mode="bilinear", align_corners=False), b], 1)
        y = s.d3(u(x4, x3)); y = s.d2(u(y, x2)); y = s.d1(u(y, x1)); y = s.d0(u(y, x0))
        y = Fn.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return s.head(torch.cat([y, x[:, :1]], 1)).squeeze(1)


def page(n):
    g = cv2.imread(glob.glob(f"{SRC}/{glob.escape(n)}.*")[0], 0); return g, (g < 160).astype(np.uint8)


def target(n, g, ink):
    sc = 3300 / max(g.shape); m = np.zeros_like(g, np.uint8)
    for x0, y0, x1, y1 in json.load(open(f"{LV}/{n}.json"))["solid"]: cv2.line(m, (int(x0), int(y0)), (int(x1), int(y1)), 1, max(5, int(9 / sc)))
    return m & ink


def to_in(g):   # 灰度 -> 3 通道 [0,1] 张量
    a = 1 - g.astype(np.float32) / 255; return np.stack([a, a, a])


def train(name, dev):
    random.seed(0); torch.manual_seed(0)
    pages = FOLDS[name]; D = []
    for n in pages:
        g, ink = page(n); D.append((g, ink, target(n, g, ink)))
    m = UNet().to(dev); opt = torch.optim.AdamW(m.parameters(), 3e-4, weight_decay=1e-4); IT = 4000; bs = 8; T = 512
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=IT)
    for it in range(IT):
        xs, ys, ws = [], [], []
        for _ in range(bs):
            g, ink, t = random.choice(D); H, W = g.shape
            for _ in range(20):          # 偏向有墨迹的块
                y0, x0 = random.randint(0, H - T), random.randint(0, W - T)
                if ink[y0:y0 + T, x0:x0 + T].mean() > 0.01: break
            a, b, c = g[y0:y0 + T, x0:x0 + T], ink[y0:y0 + T, x0:x0 + T], t[y0:y0 + T, x0:x0 + T]
            k = random.randint(0, 3); a, b, c = np.rot90(a, k), np.rot90(b, k), np.rot90(c, k)
            if random.random() < 0.5: a, b, c = a[:, ::-1], b[:, ::-1], c[:, ::-1]
            xs.append(to_in(np.ascontiguousarray(a))); ys.append(c.astype(np.float32)); ws.append(b.astype(np.float32))
        x = torch.tensor(np.array(xs), device=dev); y = torch.tensor(np.array(ys), device=dev); w = torch.tensor(np.array(ws), device=dev)
        z = m(x); loss = (Fn.binary_cross_entropy_with_logits(z, y, reduction="none") * w).sum() / w.sum().clamp(min=1)
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        if it % 500 == 499: print(f"[{name}] it {it + 1}/{IT} loss {loss.item():.4f}", flush=True)
    torch.save(m.state_dict(), f"{OUT}/unet_{name}.pt"); print("PIPE_TRAIN_DONE", name, flush=True)


@torch.no_grad()
def infer(m, g, dev, T=1024, S=896):
    H, W = g.shape; acc = np.zeros((H, W), np.float32); cnt = np.zeros((H, W), np.float32)
    ys = list(range(0, max(1, H - T), S)) + [max(0, H - T)]; xs = list(range(0, max(1, W - T), S)) + [max(0, W - T)]
    for y0 in sorted(set(ys)):
        for x0 in sorted(set(xs)):
            a = g[y0:y0 + T, x0:x0 + T]; h, w = a.shape
            p = torch.sigmoid(m(torch.tensor(to_in(a)[None], device=dev)))[0].cpu().numpy()
            acc[y0:y0 + h, x0:x0 + w] += p; cnt[y0:y0 + h, x0:x0 + w] += 1
    return acc / np.maximum(cnt, 1)


if __name__ == "__main__":
    if sys.argv[1] == "train": train(sys.argv[2], f"cuda:{sys.argv[3]}")
    elif sys.argv[1] == "predict":
        dev = f"cuda:{sys.argv[2]}"; M = {}
        for k in ("A1", "A2"):
            M[k] = UNet().to(dev).eval(); M[k].load_state_dict(torch.load(f"{OUT}/unet_{k}.pt", map_location=dev))
        for n in sp["A_train"] + sp["B_test"]:
            g, ink = page(n)
            use = ["A2"] if n in FOLDS["A1"] else (["A1"] if n in FOLDS["A2"] else ["A1", "A2"])
            p = np.mean([infer(M[k], g, dev) for k in use], 0) * ink
            cv2.imwrite(f"{OUT}/{n}.png", (p * 255).astype(np.uint8))
        print("PIPE_PRED_DONE", flush=True)
    else:
        from structure import segments
        res = {"学习式": [], "规则": []}
        for n in sp["A_train"]:
            g, ink = page(n); t = target(n, g, ink); pr = cv2.imread(f"{OUT}/{n}.png", 0) >= 128; sc = 3300 / max(g.shape)
            H_, V_, _ = segments(g, sc); rule = np.zeros_like(ink)
            for x0, x1, y, th in H_: rule[int(y - th / 2):int(y + th / 2) + 1, int(x0):int(x1) + 1] = 1
            for y0, y1, x, th in V_: rule[int(y0):int(y1) + 1, int(x - th / 2):int(x + th / 2) + 1] = 1
            rule &= ink
            for k, p in (("学习式", pr), ("规则", rule.astype(bool))):
                tp = (p & (t > 0)).sum(); res[k].append((tp / max(1, p.sum()), tp / max(1, (t > 0).sum())))
        for k, v in res.items():
            P, R = np.mean([a for a, b in v]), np.mean([b for a, b in v]); print(f"{k}: 管线像素 精度 {P:.3f} 召回 {R:.3f} F1 {2 * P * R / max(1e-6, P + R):.3f}")
