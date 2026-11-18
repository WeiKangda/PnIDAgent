"""The network: ResNet-18 (ImageNet weights) on the grey symbol crop, with three heads (class, actuator, normal position).

Each head adds a linear term of the context vector (common.context_features) to its image logits. During training the
whole context vector of a box is dropped (set to 0) for 30% of the boxes, so the network also works without context
(predict.py without a text file).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm

DEV = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class Net(nn.Module):
    def __init__(self, n_cls, n_act, ctx_dim, pretrained=True):
        super().__init__()
        m = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        m.fc = nn.Identity()
        self.backbone, self.drop = m, nn.Dropout(0.2)
        self.cls, self.act, self.pos = nn.Linear(512, n_cls), nn.Linear(512, n_act), nn.Linear(512, 2)
        self.ctx_cls, self.ctx_act, self.ctx_pos = nn.Linear(ctx_dim, n_cls), nn.Linear(ctx_dim, n_act), nn.Linear(ctx_dim, 2)
        self.ctx_drop = nn.Dropout(0.1)

    def forward(self, x, ctx):
        f = self.drop(self.backbone(x)); c = self.ctx_drop(ctx)
        return self.cls(f) + self.ctx_cls(c), self.act(f) + self.ctx_act(c), self.pos(f) + self.ctx_pos(c)


def to_input(ink):
    """(N, 1, H, W) ink in [0, 1] (1 = black) -> the normalised 3-channel image the backbone expects."""
    x = (1 - ink).repeat(1, 3, 1, 1)
    return (x - MEAN.to(x.device)) / STD.to(x.device)


def d4(x, k, flip):
    x = x.flip(-1) if flip else x
    return torch.rot90(x, k, (-2, -1))


def augment(ink):
    """Random rotation by 90 degrees / mirror image, small affine jitter of the box (scale, shift, a few degrees), thicker
    or blurred lines, fainter ink and noise. ink: (N, 1, H, W) on the device."""
    N = ink.shape[0]; dev = ink.device
    out = torch.empty_like(ink)
    k = torch.randint(0, 4, (N,)); fl = torch.rand(N) < 0.5
    for kk in range(4):
        for ff in (False, True):
            sel = ((k == kk) & (fl == ff)).nonzero().flatten().to(dev)
            if len(sel): out[sel] = d4(ink[sel], kk, ff)
    s = torch.empty(N, device=dev).uniform_(0.87, 1.15)
    a = torch.empty(N, device=dev).uniform_(-0.05, 0.05)
    t = torch.empty(N, 2, device=dev).uniform_(-0.06, 0.06)
    theta = torch.stack([torch.stack([torch.cos(a) / s, -torch.sin(a) / s, t[:, 0]], 1),
                         torch.stack([torch.sin(a) / s, torch.cos(a) / s, t[:, 1]], 1)], 1)
    grid = F.affine_grid(theta, list(out.shape), align_corners=False)
    out = F.grid_sample(out, grid, padding_mode="zeros", align_corners=False)          # outside = white
    r = torch.rand(N, device=dev)
    thick = (r < 0.15).view(N, 1, 1, 1); blur = ((r >= 0.15) & (r < 0.3)).view(N, 1, 1, 1)
    out = torch.where(thick, F.max_pool2d(out, 3, 1, 1), out)
    out = torch.where(blur, F.avg_pool2d(out, 3, 1, 1), out)
    out = out * torch.empty(N, 1, 1, 1, device=dev).uniform_(0.6, 1.0)
    noise = (torch.rand(N, 1, 1, 1, device=dev) < 0.3).float() * 0.04 * torch.randn_like(out)
    return (out + noise).clamp(0, 1)


@torch.no_grad()
def predict_tta(net, ink, ctx, bs=256):
    """Softmax of the three heads, averaged over the 8 rotations / mirror images. ink: (N, 1, H, W) CPU float."""
    net.eval(); P = [[], [], []]
    for s in range(0, len(ink), bs):
        x = ink[s:s + bs].to(DEV); c = ctx[s:s + bs].to(DEV)
        acc = [0, 0, 0]
        for kk in range(4):
            for ff in (False, True):
                outs = net(to_input(d4(x, kk, ff)), c)
                for i in range(3): acc[i] = acc[i] + outs[i].softmax(1)
        for i in range(3): P[i].append((acc[i] / 8).cpu())
    return [torch.cat(p).numpy() for p in P]


def cosine_lr(opt, base, step, total, warm=100):
    lr = base * min(1.0, (step + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, step / total)))
    for g in opt.param_groups: g["lr"] = lr
