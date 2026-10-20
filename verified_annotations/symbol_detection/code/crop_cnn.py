"""Candidate-box crop classifier CNN (stage 2): decides "is this box a correctly labeled symbol box".
Input: square crop centered on the box (side = 1.5 x box long side + 8, with context), resized to 64x64; 3 channels = [gray, box-position mask, gray].
Training samples (A pages only): boxes in the candidate pool (positive if IoU>=0.5 with a label), label boxes themselves + small jitters with IoU>=0.7 (positive),
  shifted/scaled boxes with IoU 0.1-0.45 (hard negatives, teach it whether a box is accurate). Augmentation: flips / 90-degree rotation.
Model: ResNet18 (ImageNet pretrained), trained on CPU.
Cross-fitting: cnn_A1 (trained on A1 pages only) scores A2 pages, cnn_A2 scores A1 pages, cnn_A (all of A) scores B.
  python crop_cnn.py train <A1|A2|A>"""
import glob, json, os, sys, pickle, random
import cv2, numpy as np, torch, torchvision
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from selfsim import SRC, CACHE, sp, gt_of
from rescore_pool import iou_mat
torch.set_num_threads(16); S = int(os.environ.get("CNN_S", 64)); EPOCHS = int(os.environ.get("CNN_EP", 12))
TAG = (f"r{S}" if S != 64 else "") + os.environ.get("CNN_TAGX", "")    # CNN_TAGX: extra name suffix (e.g. q = pseudo labels from the best pipeline)
DEV = os.environ.get("CNN_DEV", "cpu"); SEED = int(os.environ.get("SEED", 0)); SUF = f"s{SEED}" if SEED else ""
torch.manual_seed(SEED); random.seed(SEED)
NEGAGREE = os.environ.get("NEGAGREE") == "1"; NEGAGREE_T = 0.3 if NEGAGREE else 9


def crop(g, b):
    w, h = b[2] - b[0], b[3] - b[1]; L = 1.5 * max(w, h) + 8; cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    x0, y0 = cx - L / 2, cy - L / 2; k = S / L
    M = np.float32([[k, 0, -x0 * k], [0, k, -y0 * k]])
    im = cv2.warpAffine(g, M, (S, S), flags=cv2.INTER_AREA, borderValue=255)
    m = np.zeros((S, S), np.float32); cv2.rectangle(m, (int((b[0] - x0) * k), int((b[1] - y0) * k)), (int((b[2] - x0) * k), int((b[3] - y0) * k)), 1.0, -1)
    a = 1 - im.astype(np.float32) / 255
    return np.stack([a, m, a])


def jitter(b, rnd, lo, hi, tries=30):
    w, h = b[2] - b[0], b[3] - b[1]
    for _ in range(tries):
        s = rnd.uniform(-0.6, 0.6)
        nb = [b[0] + rnd.uniform(-0.5, 0.5) * w, b[1] + rnd.uniform(-0.5, 0.5) * h, 0, 0]
        nb[2] = nb[0] + w * np.exp(rnd.uniform(-0.6, 0.6)); nb[3] = nb[1] + h * np.exp(rnd.uniform(-0.6, 0.6))
        v = iou_mat(np.array([nb]), np.array([b]))[0][0, 0]
        if lo <= v < hi: return nb
    return None


def build(pages, R):
    rnd = random.Random(0); X, Y = [], []
    for n in pages:
        g = cv2.imread(glob.glob(f"{SRC}/{glob.escape(n)}.*")[0], 0); B, G = R[n]
        lab = iou_mat(np.array(B), np.array(G))[0].max(1) >= 0.5 if len(B) and len(G) else np.zeros(len(B), bool)
        for b, l in zip(B, lab): X.append(crop(g, b)); Y.append(float(l))
        for b in G:
            X.append(crop(g, b)); Y.append(1.0)
            for lo, hi, y in ((0.7, 1.01, 1.0), (0.1, 0.45, 0.0), (0.1, 0.45, 0.0)):
                nb = jitter(b, rnd, lo, hi)
                if nb is not None: X.append(crop(g, nb)); Y.append(y)
    return torch.tensor(np.array(X)), torch.tensor(Y)


def model():
    m = torchvision.models.resnet18(weights="IMAGENET1K_V1"); m.fc = torch.nn.Linear(512, 1)
    if os.environ.get("CNN_INIT"): m.load_state_dict(torch.load(os.environ["CNN_INIT"], map_location="cpu"))   # weakly supervised pretrained weights (pretrain_cnn.py)
    return m


def train(X, Y, epochs=None, bs=128):
    epochs = epochs or EPOCHS
    m = model().to(DEV); X, Y = X.to(DEV), Y.to(DEV); opt = torch.optim.AdamW(m.parameters(), 1e-3, weight_decay=1e-4)
    steps = epochs * ((len(X) + bs - 1) // bs); sch = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=steps)
    lossf = torch.nn.BCEWithLogitsLoss()
    for ep in range(epochs):
        m.train(); perm = torch.randperm(len(X)); tot = 0
        for i in range(0, len(X), bs):
            idx = perm[i:i + bs].to(DEV); x = X[idx]
            if random.random() < 0.5: x = x.flip(3)
            if random.random() < 0.5: x = x.flip(2)
            k = random.randint(0, 3); x = torch.rot90(x, k, (2, 3)) if k else x
            l = lossf(m(x).squeeze(1), Y[idx]); opt.zero_grad(); l.backward(); opt.step(); sch.step(); tot += l.item() * len(idx)
        print(f"  epoch {ep + 1}/{epochs} loss {tot / len(X):.4f}", flush=True)
    return m.cpu().eval()


@torch.no_grad()
def score(m, g, boxes, bs=256):
    if not len(boxes): return np.zeros(0)
    X = torch.tensor(np.array([crop(g, b) for b in boxes])); out = []
    for i in range(0, len(X), bs):
        x = X[i:i + bs]   # average over 4 orientations (rotation invariant)
        out.append(torch.stack([torch.sigmoid(m(torch.rot90(x, k, (2, 3))).squeeze(1)) for k in range(4)]).mean(0))
    return torch.cat(out).numpy()


@torch.no_grad()
def score_embed(m, g, boxes, bs=256):
    """Probability + penultimate-layer features (512-d, averaged over 4 orientations), for within-page propagation."""
    if not len(boxes): return np.zeros(0), np.zeros((0, 512))
    feat = {}
    h = m.fc.register_forward_hook(lambda mod, i, o: feat.__setitem__("e", i[0]))
    X = torch.tensor(np.array([crop(g, b) for b in boxes])); P, E = [], []
    for i in range(0, len(X), bs):
        x = X[i:i + bs]; ps, es = [], []
        for k in range(4):
            ps.append(torch.sigmoid(m(torch.rot90(x, k, (2, 3))).squeeze(1))); es.append(feat["e"])
        P.append(torch.stack(ps).mean(0)); E.append(torch.stack(es).mean(0))
    h.remove(); return torch.cat(P).numpy(), torch.cat(E).numpy()


def load(name):
    m = model(); m.load_state_dict(torch.load(f"{HERE}/results/cnn_{name}.pt")); return m.eval()


def det_score(n, B):
    """Detector score of candidate boxes: highest score among the five soup5 views with IoU>=0.8 to it (same as the candidate pool merge rule)."""
    from rescore_pool import iou_mat as _iou
    d = json.load(open(f"{CACHE}/soup5.json"))[n]; t = json.load(open(f"{CACHE}/soup5_tta.json"))[n]
    D = np.array([x for v in [d] + list(t.values()) for x in v]).reshape(-1, 5)
    if not len(D) or not len(B): return np.zeros(len(B))
    io = _iou(np.array(B), D[:, :4])[0]
    return np.where(io >= 0.8, D[None, :, 4], 0).max(1)


def build_pseudo(pages, R, pk, hi=0.6, lo=0.05):
    """Pseudo labels for B pages (transductive self-training, no B labels): boxes kept by the current pipeline with prob>=hi are positive (plus IoU>=0.7 jitter),
    candidates with prob<=lo that overlap no pseudo-positive box (IoU<0.3) are negative."""
    from rescore_pool import select
    rnd = random.Random(1); X, Y = [], []
    for n in pages:
        B = R[n][0]; p = pk["pB"][n]
        if not len(B): continue
        g = cv2.imread(glob.glob(f"{SRC}/{glob.escape(n)}.*")[0], 0)
        pos = [b for b in select(B, p, hi, pk["nt"], pk.get("ct", 1.01))]
        for b in pos:
            X.append(crop(g, b)); Y.append(1.0); nb = jitter(b, rnd, 0.7, 1.01)
            if nb is not None: X.append(crop(g, nb)); Y.append(1.0)
        io = iou_mat(np.array(B), np.array(pos))[0].max(1) if pos else np.zeros(len(B))
        # NEGAGREE: pseudo negatives need agreement (low pipeline prob AND low detector score); boxes the detector finds symbol-like are not used as negatives (keeps self-training from suppressing unseen true symbols)
        ds = det_score(n, B) if NEGAGREE else np.zeros(len(B))
        for b, pp, o, d in zip(B, p, io, ds):
            if pp <= lo and o < 0.3 and d < NEGAGREE_T: X.append(crop(g, b)); Y.append(0.0)
    return torch.tensor(np.array(X)), torch.tensor(Y)


if __name__ == "__main__":
    name = sys.argv[2]; folds = json.load(open(f"{HERE}/tile_folds.json"))
    if name.startswith("K"):     # K0..K4: crop-encoder pretraining for fold k of the relation model (only true labels of that fold's training pages, with jittered hard negatives)
        from rel_model import folds5
        k = int(name[1:]); pages = [n for n in sp["A_train"] if n not in folds5()[k]]
        R = pickle.load(open(f"{CACHE}/soup5_rescore_pool_one2one.pkl", "rb"))["R"]
        X, Y = build(pages, R); print(name, "samples", len(X), "pos", int(Y.sum()), flush=True)
        m = train(X, Y); torch.save(m.state_dict(), f"{HERE}/results/cnn_{name}{TAG}{SUF}.pt"); print("CNN_DONE", name + TAG + SUF, flush=True); sys.exit()
    if "p" in name:     # A1p / A2p (round 1), A1p2 (round 2)...: this fold's true labels + B pseudo labels (from the previous round's pipeline results)
        base, rd = name.split("p")[0], int(name.split("p")[1] or 1)
        f4 = json.load(open(f"{HERE}/folds4.json"))   # F0..F3: train = A minus fold k (about 30 pages)
        pages = {"A1": folds["A1"], "A2": folds["A2"], **{k: [n for n in sp["A_train"] if n not in v] for k, v in f4.items()}}[base]
        R = pickle.load(open(f"{CACHE}/soup5_rescore_pool_one2one.pkl", "rb"))["R"]
        src = "_cnn" if rd == 1 else ("_cnnp" if rd == 2 else f"_cnnp{rd - 1}")
        src = os.environ.get("PSEUDO_SRC", src)     # pseudo labels for the pretrained variant come from its own round-0 pipeline
        pk = pickle.load(open(f"{CACHE}/soup5_rescore_pool_one2one{src}.pkl", "rb")); print("pseudo label source", src, flush=True)
        X, Y = build(pages, R); Xp, Yp = build_pseudo(sp["B_test"], pk["R"], pk)
        print(name, "true-label samples", len(X), "pos", int(Y.sum()), "| B pseudo-label samples", len(Xp), "pos", int(Yp.sum()), flush=True)
        m = train(torch.cat([X, Xp]), torch.cat([Y, Yp])); torch.save(m.state_dict(), f"{HERE}/results/cnn_{name}{TAG}{SUF}.pt"); print("CNN_DONE", name + TAG + SUF, flush=True)
        sys.exit()
    pages = {"A1": folds["A1"], "A2": folds["A2"], "A": sp["A_train"]}[name]
    R = pickle.load(open(f"{CACHE}/soup5_rescore_pool_one2one.pkl", "rb"))["R"]
    X, Y = build(pages, R); print(name, "samples", len(X), "pos", int(Y.sum()), flush=True)
    m = train(X, Y); torch.save(m.state_dict(), f"{HERE}/results/cnn_{name}{TAG}{SUF}.pt"); print("CNN_DONE", name + TAG + SUF, flush=True)
