"""文字误检过滤 (测试时只用 OCR 原始结果, 不用人工标注):
预测框里的墨迹, 如果绝大部分都落在 OCR 文字框里 (框里几乎只有字, 没有圆圈/阀门等图形), 就判为文字误检删掉.
在 A 半上选阈值, B 半上报告.  python textfilter.py <cache name> [selfsim参数json]"""
import glob, json, os, sys
import cv2, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from selfsim import f1, gt_of, sp, SRC, CACHE, propagate, ink_img
R = "/home/suxinqi666/code/PnIDAgent_project"; RAW = f"{R}/ft_verified/text_ocr/raw"


def text_mask(n, shape):
    m = np.zeros(shape, np.uint8)
    for t in json.load(open(f"{RAW}/{n}.json")):
        if t["score"] < 0.5 or sum(ch.isalnum() for ch in t["text"]) < 2: continue
        x0, y0, x1, y1 = [int(round(v)) for v in t["box"]]; cv2.rectangle(m, (x0 - 1, y0 - 1), (x1 + 1, y1 + 1), 1, -1)
    return m


def ink_in_text(box, ink, tm):
    x0, y0, x1, y1 = [int(round(v)) for v in box[:4]]
    a = ink[y0:y1, x0:x1]; s = a.sum()
    return float((a * tm[y0:y1, x0:x1]).sum() / s) if s > 0 else 0.0


if __name__ == "__main__":
    mn = sys.argv[1]; dets = json.load(open(f"{CACHE}/{mn}.json"))
    pre = {}
    for n in sp["A_train"] + sp["B_test"]:
        g = cv2.imread(glob.glob(f"{SRC}/{glob.escape(n)}.*")[0], 0); H, W = g.shape
        ink = (g < 160).astype(np.uint8); tm = text_mask(n, g.shape)
        boxes = [d for d in dets[n] if d[4] >= 0.5]
        pre[n] = ([(d, ink_in_text(d, ink, tm)) for d in boxes], gt_of(n, W, H))
    def score(part, t):
        return float(np.mean([f1([d[:4] for d, r in pre[n][0] if r < t], pre[n][1]) for n in sp[part]]))
    base = {p: score(p, 9) for p in ("A_train", "B_test")}
    rows = [(score("A_train", t), t) for t in (0.6, 0.7, 0.8, 0.85, 0.9, 0.95)]
    for a, t in rows: print(f"  t {t}: A {a:.4f}")
    a, t = max(rows)
    print(f"{mn} 文字过滤: 基线 A {base['A_train']:.4f} B {base['B_test']:.4f} | 按A选 t={t}: A {a:.4f} B {score('B_test', t):.4f}")
