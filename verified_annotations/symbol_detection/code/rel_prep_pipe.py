"""关系模型数据的管线 token 换成学习式管线分割 (pipe_seg.py, 交叉拟合) 提取的线段; 其余 token 不变.
输出 results/rel_data_pipe/<page>.npz  (rel_model.py 用环境变量 REL_DATA=rel_data_pipe 选用)"""
import glob, json, os, sys
import cv2, numpy as np
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from selfsim import SRC, sp
from structure import segments
IN, OUT = f"{HERE}/results/rel_data", f"{HERE}/results/rel_data_pipe"; os.makedirs(OUT, exist_ok=True)
LV = "/home/suxinqi666/code/PnIDAgent_project/line_results/lines_verified"


def work(n):
    z = dict(np.load(f"{IN}/{n}.npz")); g = cv2.imread(glob.glob(f"{SRC}/{glob.escape(n)}.*")[0], 0); sc = 3300 / max(g.shape)
    pm = cv2.imread(f"{HERE}/results/pipe_seg/{n}.png", 0) >= 128
    Hs, Vs, _ = segments(g, sc, ink=pm)
    L = np.concatenate([np.column_stack([Hs[:, 0], Hs[:, 2] - Hs[:, 3] / 2, Hs[:, 1], Hs[:, 2] + Hs[:, 3] / 2, np.zeros(len(Hs))]),
                        np.column_stack([Vs[:, 2] - Vs[:, 3] / 2, Vs[:, 0], Vs[:, 2] + Vs[:, 3] / 2, Vs[:, 1], np.ones(len(Vs))])]) if len(Hs) + len(Vs) else np.zeros((0, 5))
    ly = np.full(len(L), -1, np.float32)
    if n in sp["A_train"] and os.path.exists(f"{LV}/{n}.json"):
        mg = np.zeros_like(g, np.uint8)
        for x0, y0, x1, y1 in json.load(open(f"{LV}/{n}.json"))["solid"]: cv2.line(mg, (int(x0), int(y0)), (int(x1), int(y1)), 1, max(5, int(8 / sc)))
        ly = np.array([mg[int(max(0, (b + d) / 2 - 1)):int((b + d) / 2 + 2), int(a):int(c) + 1].mean() if t == 0 else
                       mg[int(b):int(d) + 1, int(max(0, (a + c) / 2 - 1)):int((a + c) / 2 + 2)].mean() for a, b, c, d, t in L], np.float32)
        ly = (ly >= 0.5).astype(np.float32)
    z["L"], z["ly"] = L, ly; np.savez_compressed(f"{OUT}/{n}.npz", **z)
    return len(L), int((ly == 1).sum())


if __name__ == "__main__":
    with Pool(12) as p: r = np.array(p.map(work, sp["A_train"] + sp["B_test"]))
    print("线段", r[:, 0].sum(), "其中 A 页对得上人工管线的", r[:, 1].sum())
