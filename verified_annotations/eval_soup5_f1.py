"""Per-page symbol F1 of a YOLO model on the verified pages (the numbers in README.md).

    python eval_soup5_f1.py --weights best.pt --images <dir with the page images> \
        --labels labels --out f1.json [--device 0]

Protocol: adaptive imgsz (longest side scaled to the training scale, clamped to 1280-2048),
conf 0.10, then keep score >= 0.50, greedy matching at IoU 0.5, macro-F1 over pages.
"""
import argparse, glob, json, os
from PIL import Image
from ultralytics import YOLO

Image.MAX_IMAGE_PIXELS = None
ap = argparse.ArgumentParser()
ap.add_argument("--weights", required=True)
ap.add_argument("--images", required=True)
ap.add_argument("--labels", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "labels"))
ap.add_argument("--out", default="f1.json")
ap.add_argument("--device", default="cpu")
args = ap.parse_args()


def adaptive(longest):
    return max(1280, min(2048, int(round(1280 * longest / 3300 / 32)) * 32))


def iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    i = (x1 - x0) * (y1 - y0)
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)


model = YOLO(args.weights)
res = {}
for f in sorted(glob.glob(os.path.join(args.labels, "*.txt"))):
    name = os.path.basename(f)[:-4]
    img = glob.glob(os.path.join(args.images, glob.escape(name) + ".*"))[0]
    W, H = Image.open(img).size
    gt = []
    for line in open(f):
        p = line.split()
        if len(p) == 5:
            _, cx, cy, w, h = map(float, p)
            gt.append([(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H])
    r = model.predict(img, imgsz=adaptive(max(W, H)), conf=0.10, iou=0.5, max_det=1500, verbose=False, device=args.device)[0]
    pred = [b.tolist() for b, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()) if c >= 0.50]
    used, tp = set(), 0
    for g in gt:
        best = max([(iou(g, p), j) for j, p in enumerate(pred) if j not in used], default=(0, -1))
        if best[0] >= 0.5:
            used.add(best[1]); tp += 1
    fp, fn = len(pred) - tp, len(gt) - tp
    res[name] = {"gt": len(gt), "tp": tp, "fp": fp, "fn": fn,
                 "P": tp / max(1, tp + fp), "R": tp / max(1, tp + fn), "F1": 2 * tp / max(1, 2 * tp + fp + fn)}
    print(f"{res[name]['F1']:.3f}  {name}")
json.dump(res, open(args.out, "w"), indent=1)
print(f"macro-F1 {sum(v['F1'] for v in res.values()) / len(res):.4f} over {len(res)} pages")
