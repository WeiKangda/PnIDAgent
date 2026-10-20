"""Large-scale weakly supervised pretraining of the appearance network (direction 1, stage 1):
Data = the 5 datasets used to train soup5 (yolo_quality / apr26 / real31 / notation34 / dpid_nuke, 1280px tiles, automatic pseudo labels, no human-verified annotations).
Samples: positive = symbol box + IoU>=0.7 jitter; negative = offset boxes with IoU 0.1-0.45 (learn box accuracy) + random boxes far from symbols (text/lines/borders/blank, learn symbol vs. not).
Crop format is identical to crop_cnn.crop (64px, [ink, box mask, ink]); the pretrained network directly initializes crop_cnn.
  python pretrain_cnn.py <gpu>   -> results/cnn_weakpre.pt"""
import glob, os, sys, random
import cv2, numpy as np, torch
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from crop_cnn import crop, jitter, model
from rescore_pool import iou_mat
DS = "/home/suxinqi666/code/PnIDAgent_project/datasets"; SETS = ["yolo_quality", "yolo_apr26", "yolo_real31", "yolo_notation34", "yolo_dpid_nuke"]


def work(args):
    ip, seed = args; rnd = random.Random(seed)
    g = cv2.imread(ip, 0); H, W = g.shape; lp = ip.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
    if not os.path.exists(lp): return [], []
    G = np.array([[(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H] for cx, cy, w, h in
                  (list(map(float, l.split()[1:5])) for l in open(lp) if l.strip())]).reshape(-1, 4)
    if not len(G): return [], []
    X, Y = [], []
    for b in G[rnd.sample(range(len(G)), min(25, len(G)))]:
        X.append(crop(g, b)); Y.append(1)
        for lo, hi, y in ((0.7, 1.01, 1), (0.1, 0.45, 0)):
            nb = jitter(b, rnd, lo, hi)
            if nb is not None: X.append(crop(g, nb)); Y.append(y)
        w, h = b[2] - b[0], b[3] - b[1]                          # same-size random box, far from all symbols and containing ink
        for _ in range(10):
            x0, y0 = rnd.uniform(0, W - w), rnd.uniform(0, H - h); nb = [x0, y0, x0 + w, y0 + h]
            if iou_mat(np.array([nb]), G)[0].max() < 0.1 and (g[int(y0):int(y0 + h), int(x0):int(x0 + w)] < 160).mean() > 0.02:
                X.append(crop(g, nb)); Y.append(0); break
    return [(x[:2] * 255).astype(np.uint8) for x in X], Y


if __name__ == "__main__":
    dev = f"cuda:{sys.argv[1]}"; random.seed(0)
    ims = []
    for s in SETS:
        f = sorted(glob.glob(f"{DS}/{s}/images/train/*.png")); ims += random.Random(1).sample(f, min(int(os.environ.get("PRE_N", 650)), len(f)))
    with Pool(24) as p: res = p.map(work, [(ip, i) for i, ip in enumerate(ims)])
    X = np.array([x for r in res for x in r[0]]); Y = np.array([y for r in res for y in r[1]], np.float32)
    print("Pretraining samples", len(X), "positive", int(Y.sum()), flush=True)
    EP = int(os.environ.get("PRE_EP", 6)); bs = 256; ARCH = os.environ.get("PRE_ARCH", "resnet18")
    if ARCH == "resnet18": m = model()
    else:
        import torchvision; m = getattr(torchvision.models, ARCH)(weights="IMAGENET1K_V1"); m.fc = torch.nn.Linear(m.fc.in_features, 1)
    m = m.to(dev); opt = torch.optim.AdamW(m.parameters(), 1e-3, weight_decay=1e-4)
    steps = EP * ((len(X) + bs - 1) // bs); sch = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=steps); lossf = torch.nn.BCEWithLogitsLoss()
    Xt = torch.tensor(X); Yt = torch.tensor(Y)
    for ep in range(EP):
        m.train(); perm = torch.randperm(len(X)); tot = 0
        for i in range(0, len(X), bs):
            idx = perm[i:i + bs]; a = Xt[idx].float().to(dev) / 255
            x = torch.stack([a[:, 0], a[:, 1], a[:, 0]], 1)
            if random.random() < 0.5: x = x.flip(3)
            if random.random() < 0.5: x = x.flip(2)
            k = random.randint(0, 3); x = torch.rot90(x, k, (2, 3)) if k else x
            l = lossf(m(x).squeeze(1), Yt[idx].to(dev)); opt.zero_grad(); l.backward(); opt.step(); sch.step(); tot += l.item() * len(idx)
        print(f"  epoch {ep + 1}/{EP} loss {tot / len(X):.4f}", flush=True)
    torch.save(m.cpu().state_dict(), f"{HERE}/results/{os.environ.get('PRE_OUT', 'cnn_weakpre')}.pt"); print("PRETRAIN_DONE", flush=True)
