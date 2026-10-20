"""PP-OCRv5 (server det + server rec) text pre-annotation for the 65 verified pages.
- Detect at native resolution; run once per page orientation 0/90cw/90ccw (vertical text), map back to original coordinates
- Of overlapping results across the three orientations, keep only the highest recognition score
- Output raw/<page>.json: [{"poly": [[x,y]x4] original pixels, "box": [x0,y0,x1,y1] original pixels, "text", "score", "rot"}]
  python ocr5_pages.py <k> <K> <device>     (shard k of K)
"""
import glob, json, os, sys
os.environ.setdefault("FLAGS_allocator_strategy", "auto_growth")
import cv2, numpy as np
from paddleocr import PaddleOCR
R = "/home/suxinqi666/code/PnIDAgent_project"; HERE = os.path.dirname(os.path.abspath(__file__))
OUT = f"{HERE}/raw"; CAND = f"{HERE}/cand"; os.makedirs(OUT, exist_ok=True); os.makedirs(CAND, exist_ok=True)
k, K, dev = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
IMG_DIR = os.environ.get("IMG_DIR")    # run OCR on other images (e.g. real4): the pngs under IMG_DIR
pages = sorted(os.path.basename(f)[:-4] for f in glob.glob(f"{IMG_DIR}/*.png")) if IMG_DIR else sorted(os.path.basename(f)[:-4] for f in glob.glob(f"{R}/line_results/labels_verified/*.txt"))
def make(limit, box, thr):
    return PaddleOCR(text_detection_model_name="PP-OCRv5_server_det", text_recognition_model_name="PP-OCRv5_server_rec",
                     use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=True,
                     text_det_limit_type="max", text_det_limit_side_len=limit, text_det_box_thresh=box, text_det_thresh=thr,
                     text_rec_score_thresh=0.0, device=dev, **({"cpu_threads": int(os.environ.get("THREADS", 6)), "enable_mkldnn": True} if dev == "cpu" else {}))


# detection is scale sensitive: one pass at native resolution with low thresholds + one resized to 2000 px, merged (a single scale misses whole label lines)
OCRS = {"hi": make(6000, 0.3, 0.2), "lo": make(2000, 0.5, 0.3)}


def back(pts, rot, W, H):
    """Points on the rotated image -> original coordinates. cw: orig (x,y)->(H-1-y, x); ccw: orig (x,y)->(y, W-1-x)."""
    p = np.asarray(pts, float)
    if rot == "cw": return np.stack([p[:, 1], H - 1 - p[:, 0]], 1)
    if rot == "ccw": return np.stack([W - 1 - p[:, 1], p[:, 0]], 1)
    return p


def overlap(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    if ix == 0: return 0.0
    return ix / min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))


def quality(c):
    """Pick the most credible of the orientation readings of the same text: rec score x valid-char count^0.5; down-weight boxes whose shape contradicts the reading direction."""
    import re
    w, h = c["box"][2] - c["box"][0], c["box"][3] - c["box"][1]; n = len(re.sub(r"[^A-Za-z0-9]", "", c["text"]))
    horiz = c["rot"] in ("0", "180")
    bad = (horiz and h > 1.5 * w and n > 1) or (not horiz and w > 1.5 * h and n > 1)
    return c["score"] * max(1, n) ** 0.5 * (0.2 if bad else 1.0)


def dedup(cand):
    keep = []
    for c in sorted(cand, key=lambda c: -quality(c)):
        if all(overlap(c["box"], d["box"]) < 0.5 for d in keep): keep.append(c)
    return keep


for i, n in enumerate(pages):
    if i % K != k or os.path.exists(f"{CAND}/{n}.json") or (os.environ.get("ONLY") and os.environ["ONLY"] not in n): continue
    img = cv2.imread(f"{IMG_DIR}/{n}.png" if IMG_DIR else glob.glob(f"{R}/datasets/yolo_pseudo_v2/images/train/{glob.escape(n)}.*")[0]); H, W = img.shape[:2]
    cand = []
    for rot, code in (("0", None), ("cw", cv2.ROTATE_90_CLOCKWISE), ("ccw", cv2.ROTATE_90_COUNTERCLOCKWISE)):
        im2 = img if code is None else cv2.rotate(img, code)
        for pas, ocr in OCRS.items():
            r = ocr.predict(im2)[0]
            ang = list(r.get("textline_orientation_angles", []))
            if len(ang) != len(r["rec_polys"]): ang = [0] * len(r["rec_polys"])   # do not correct if they do not match
            for poly, t, s, a in zip(r["rec_polys"], r["rec_texts"], r["rec_scores"], ang):
                p = back(poly, rot, W, H)
                # the textline orientation model flips upside-down text before recognition: true orientation = this pass's orientation + 180
                true_rot = {"0": "180", "cw": "ccw", "ccw": "cw"}[rot] if int(a) == 1 else rot
                cand.append({"poly": p.round(1).tolist(), "box": [float(p[:, 0].min()), float(p[:, 1].min()), float(p[:, 0].max()), float(p[:, 1].max())],
                             "text": t, "score": float(s), "rot": true_rot, "pass_rot": rot, "pass": pas})
    json.dump(cand, open(f"{CAND}/{n}.json", "w"), ensure_ascii=False)
    json.dump(dedup(cand), open(f"{OUT}/{n}.json", "w"), ensure_ascii=False)
    print(f"{n[:50]} cand {len(cand)}", flush=True)
print("OCR5_DONE", k)
