"""Classify the symbol boxes of a P&ID page with the trained network.

  python predict.py --image page.png --boxes <labels.txt | detections.json> [--text text.json]
                    [--model model/symbol_classifier_v3.pt] [--out classification.json] [--overlay overlay.png]

--boxes  a YOLO label file (class cx cy w h, normalised; box id = line number), or a JSON with 'symbols':
         [{'id', 'bbox': [x1, y1, x2, y2]}, ...] (the classification JSON of the pipeline), or a plain list of such boxes.
--text   optional: the text detections of the pipeline (a list of {'bbox', 'text'}) or a text layer of the
         annotations ({'items': [{'box', 'text'}]}). The words in and around a box help the class and the actuator.

The output is the classification JSON the rest of the pipeline reads (digitize_pnid.py): 'symbols' with 'id', 'bbox',
'category' (the class) and 'confidence', and in addition 'actuator', 'normal_position' (open / closed for the classes that
have one, else n/a), their probabilities, the fault-tree component type that follows from class + actuator, whether the
class is usually modeled in a PRA, the three most likely classes and 'needs_review' (class probability < 0.7).
"""
import argparse, collections, json, os

import cv2
import numpy as np
import torch

from common import CROP, context_features, crop, ft_type
from model import DEV, Net, predict_tta

HERE = os.path.dirname(os.path.abspath(__file__))
REVIEW_BELOW = 0.7


def load_boxes(path, W, H):
    if path.endswith(".txt"):
        out = []
        for i, line in enumerate(open(path)):
            p = line.split()
            if len(p) < 5: continue
            cx, cy, w, h = (float(v) for v in p[1:5])
            out.append({"id": i, "bbox": [(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H]})
        return out
    j = json.load(open(path))
    syms = j.get("symbols", []) if isinstance(j, dict) else j
    return [{"id": s.get("id", i), "bbox": [float(v) for v in s["bbox"][:4]], "mask_id": s.get("mask_id")}
            for i, s in enumerate(syms)]


def load_text(path):
    j = json.load(open(path))
    items = j["items"] if isinstance(j, dict) else j
    out = []
    for it in items:
        b = it.get("box") or it.get("bbox")
        if not b or not it.get("text"): continue
        if isinstance(b[0], (list, tuple)):                          # polygon -> its bounding box
            xs, ys = [p[0] for p in b], [p[1] for p in b]; b = [min(xs), min(ys), max(xs), max(ys)]
        out.append({"box": [float(v) for v in b[:4]], "text": it["text"]})
    return out


def load_model(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = Net(len(ck["classes"]), len(ck["actuators"]), ck["ctx_dim"], pretrained=False)
    net.load_state_dict({k: (v.float() if v.is_floating_point() else v) for k, v in ck["state_dict"].items()})
    return net.to(DEV).eval(), ck


def classify(image, boxes, texts=None, model=os.path.join(HERE, "model", "symbol_classifier_v3.pt")):
    net, ck = load_model(model)
    gray = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    if gray is None: raise SystemExit(f"cannot read {image}")
    assert ck["crop"] == CROP
    ink = torch.from_numpy(255 - np.stack([crop(gray, b["bbox"]) for b in boxes])).float().div(255).unsqueeze(1)
    rows = [{"bbox": b["bbox"], "box_id": b["id"]} for b in boxes]
    ctx = context_features(rows, texts) if texts else np.zeros((len(boxes), ck["ctx_dim"]), np.float32)
    pc, pa, pp = predict_tta(net, ink, torch.from_numpy(ctx))
    C, A = ck["classes"], ck["actuators"]; dash = A.index("-")
    symbols = []
    for i, b in enumerate(boxes):
        c = int(pc[i].argmax()); cls = C[c]
        a = pa[i].copy(); a[dash] = -1; ai = int(a.argmax())
        x0, y0, x1, y1 = b["bbox"]
        act = A[ai] if cls in ck["act_classes"] else "-"
        s = {"id": b["id"], "mask_id": b["mask_id"] if b.get("mask_id") is not None else b["id"],
             "bbox": [round(v, 1) for v in b["bbox"]], "size": [round(x1 - x0, 1), round(y1 - y0, 1)],
             "area": round((x1 - x0) * (y1 - y0), 1), "cluster_id": c, "category": cls, "confidence": round(float(pc[i, c]), 4),
             "actuator": act,
             "actuator_confidence": round(float(pa[i, ai] / max(1e-9, 1 - pa[i, dash])), 4) if cls in ck["act_classes"] else None,
             "normal_position": ck["positions"][int(pp[i].argmax())] if cls in ck["pos_classes"] else "n/a",
             "position_confidence": round(float(pp[i].max()), 4) if cls in ck["pos_classes"] else None,
             "ft_component_type": ft_type(cls, act, ck["ft_types"]), "pra_modeled": ck["pra_modeled"].get(cls, ""),
             "top3": [[C[k], round(float(pc[i, k]), 4)] for k in np.argsort(-pc[i])[:3]],
             "needs_review": bool(pc[i, c] < REVIEW_BELOW)}
        symbols.append(s)
    cats = collections.Counter(s["category"] for s in symbols)
    return {"image_path": str(image), "classifier": os.path.basename(model), "num_symbols_detected": len(symbols),
            "num_clusters": len(cats), "categories": dict(cats.most_common()), "discarded_categories": {},
            "symbols": symbols, "discarded_symbols": [], "num_symbols_valid": len(symbols), "num_symbols_discarded": 0}


def overlay(image, result, path):
    im = cv2.imread(image)
    for s in result["symbols"]:
        x0, y0, x1, y1 = (int(v) for v in s["bbox"])
        col = (0, 0, 230) if s["needs_review"] else (0, 150, 0)
        cv2.rectangle(im, (x0, y0), (x1, y1), col, 2)
        lab = s["category"] + ("" if s["actuator"] in ("-", "NONE") else f"/{s['actuator']}") + \
              ("" if s["normal_position"] in ("n/a", "open") else " NC")
        cv2.putText(im, lab, (x0, max(10, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
    cv2.imwrite(path, im)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True); ap.add_argument("--boxes", required=True); ap.add_argument("--text")
    ap.add_argument("--model", default=os.path.join(HERE, "model", "symbol_classifier_v3.pt"))
    ap.add_argument("--out"); ap.add_argument("--overlay"); a = ap.parse_args()
    g = cv2.imread(a.image, cv2.IMREAD_GRAYSCALE); H, W = g.shape
    res = classify(a.image, load_boxes(a.boxes, W, H), load_text(a.text) if a.text else None, a.model)
    out = a.out or os.path.splitext(a.image)[0] + "_classification.json"
    json.dump(res, open(out, "w"), indent=1)
    if a.overlay: overlay(a.image, res, a.overlay)
    n_rev = sum(s["needs_review"] for s in res["symbols"])
    print(f"{res['num_symbols_detected']} symbols -> {out}  ({n_rev} with class probability < {REVIEW_BELOW})")
    print(res["categories"])


if __name__ == "__main__":
    main()
