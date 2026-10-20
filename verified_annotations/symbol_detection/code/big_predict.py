"""大设备检测器整页推理 (imgsz 1280, conf 0.1), 写 selfsim_cache/<name>.json (已有就合并).
  python big_predict.py <gpu> <name> <weights> <A1|A2|A|B>"""
import glob, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from selfsim import SRC, CACHE, sp
from ultralytics import YOLO
dev, name, wp, which = int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
folds = {**json.load(open(f"{HERE}/tile_folds.json")), **json.load(open(f"{HERE}/folds4.json"))}; pages = {**{k: v for k, v in folds.items()}, "A": sp["A_train"], "B": sp["B_test"]}[which]
m = YOLO(wp); p = f"{CACHE}/{name}.json"; out = json.load(open(p)) if os.path.exists(p) else {}
for n in pages:
    r = m.predict(glob.glob(f"{SRC}/{glob.escape(n)}.*")[0], imgsz=1280, conf=0.1, iou=0.5, verbose=False, device=dev)[0]
    out[n] = [b.tolist() + [float(c)] for b, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy())]
json.dump(out, open(p, "w")); print("BIGPRED_DONE", name, which, flush=True)
