"""Image features of the symbol crops from pretrained networks that are not trained further (the baselines).

  python extract_frozen.py --cache <dir>        # reads <dir>/dataset.npz, writes <dir>/feat_<model>.npz

For each model two features per box: of the crop as it is ('id') and the mean over its 8 rotations and mirror images
('d4'). A symbol means the same in any orientation, so the mean is the orientation-free description.
Models: CLIP ViT-B/32 (Hugging Face openai/clip-vit-base-patch32), ResNet-18 and EfficientNetV2-S (ImageNet weights of
torchvision). They are loaded from the local caches; nothing is downloaded.
"""
import argparse, os

os.environ.setdefault("USE_TF", "0")            # transformers must not import TensorFlow (protobuf clash)
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch
import torch.nn.functional as F

DEV = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
IMAGENET = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
CLIP_NORM = ([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])


def d4(x):
    """The 8 rotations / mirror images of a batch (N, C, H, W)."""
    out = []
    for f in (x, x.flip(-1)):
        for k in range(4): out.append(torch.rot90(f, k, (-2, -1)))
    return out


def load(name):
    if name == "clip":
        from transformers import CLIPModel
        m = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True).eval().to(DEV)
        return (lambda x: m.get_image_features(pixel_values=x)), CLIP_NORM
    import torchvision.models as tvm
    if name == "resnet18":
        m = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1); m.fc = torch.nn.Identity()
    elif name == "effnetv2s":
        m = tvm.efficientnet_v2_s(weights=tvm.EfficientNet_V2_S_Weights.IMAGENET1K_V1); m.classifier = torch.nn.Identity()
    m = m.eval().to(DEV)
    return m, IMAGENET


@torch.no_grad()
def features(crops, name, bs=128):
    f, (mean, std) = load(name)
    mean = torch.tensor(mean, device=DEV).view(1, 3, 1, 1); std = torch.tensor(std, device=DEV).view(1, 3, 1, 1)
    ident, avg = [], []
    for s in range(0, len(crops), bs):
        x = torch.from_numpy(crops[s:s + bs]).float().div(255).unsqueeze(1).to(DEV)
        x = F.interpolate(x, size=224, mode="bilinear", align_corners=False).repeat(1, 3, 1, 1)
        x = (x - mean) / std
        outs = [f(t).float() for t in d4(x)]
        ident.append(outs[0].cpu().numpy()); avg.append(torch.stack(outs).mean(0).cpu().numpy())
    return np.concatenate(ident), np.concatenate(avg)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cache", required=True)
    ap.add_argument("--models", default="clip,resnet18,effnetv2s"); a = ap.parse_args()
    crops = np.load(os.path.join(a.cache, "dataset.npz"))["crops"]
    for name in a.models.split(","):
        i, m = features(crops, name)
        np.savez(os.path.join(a.cache, f"feat_{name}.npz"), id=i, d4=m)
        print(name, i.shape)


if __name__ == "__main__":
    main()
