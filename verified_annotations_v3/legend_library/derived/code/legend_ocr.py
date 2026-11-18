"""PP-OCRv5 on the legend PDF pages in reading orientation (the orientation the legend images were matched in, legend_loc.json).
Runs on the 600 dpi page shrunk to 300 dpi; boxes are returned in 600 dpi rotated-page pixels. Output legend_ocr/<src>_r<rot>.json."""
import json, os, sys
os.environ.setdefault("FLAGS_allocator_strategy", "auto_growth")
import cv2
from paddleocr import PaddleOCR
HERE = "/home/suxinqi666/code/PnIDAgent_project/symbol_class"; LP = f"{HERE}/legend_pages"; OUT = f"{HERE}/legend_ocr"; os.makedirs(OUT, exist_ok=True)
G = json.load(open(f"{HERE}/legend.json")); LOC = json.load(open(f"{HERE}/legend_loc.json"))
ROT = {0: None, 1: cv2.ROTATE_90_CLOCKWISE, 2: cv2.ROTATE_90_COUNTERCLOCKWISE}
need = sorted({(G[int(j)]["src"], v[0]) for j, v in LOC.items()})
ocr = PaddleOCR(text_detection_model_name="PP-OCRv5_server_det", text_recognition_model_name="PP-OCRv5_server_rec",
                use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False,
                text_det_limit_type="max", text_det_limit_side_len=4000, text_det_box_thresh=0.4, text_det_thresh=0.25,
                text_rec_score_thresh=0.0, device=sys.argv[1] if len(sys.argv) > 1 else "gpu:0")
for src, rot in need:
    f = f"{OUT}/{src}_r{rot}.json"
    if os.path.exists(f): continue
    P = cv2.imread(f"{LP}/{src}.png"); P = P if ROT[rot] is None else cv2.rotate(P, ROT[rot]); small = cv2.resize(P, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    r = ocr.predict(small)[0]; out = []
    for poly, t, s in zip(r["dt_polys"], r["rec_texts"], r["rec_scores"]):
        xs = [p[0] * 2 for p in poly]; ys = [p[1] * 2 for p in poly]
        out.append({"box": [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))], "text": t, "score": round(float(s), 3)})
    json.dump(out, open(f, "w")); print(src, rot, len(out), flush=True)
