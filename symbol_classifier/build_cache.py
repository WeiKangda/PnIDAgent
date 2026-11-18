"""Crops, context features and labels of all symbol boxes, cached in one .npz for the experiments, and the crops of the
legend symbols (legend_library) that stand for a class.

  python build_cache.py --out <dir>      # writes <dir>/dataset.npz, <dir>/legend.npz, <dir>/meta.json
"""
import argparse, json, os

import numpy as np

from common import CROP, by_page, crop, legend_crops, load_classes, load_dataset, page_context, page_image


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = load_dataset(); classes, cinfo = load_classes()
    crops = np.zeros((len(rows), CROP, CROP), np.uint8)
    ctx_topo, ctx_text = None, None
    for page, ids in by_page(rows).items():
        g = page_image(page)
        for i in ids: crops[i] = crop(g, rows[i]["bbox"])
        pr = [rows[i] for i in ids]
        c1 = page_context(page, pr, use_topology=True); c2 = page_context(page, pr, use_topology=False)
        if ctx_topo is None:
            ctx_topo = np.zeros((len(rows), c1.shape[1]), np.float32); ctx_text = np.zeros_like(ctx_topo)
        ctx_topo[ids] = c1; ctx_text[ids] = c2
    acts = sorted({r["actuator"] for r in rows})
    pos = ["open", "closed"]
    np.savez_compressed(
        os.path.join(a.out, "dataset.npz"), crops=crops, ctx_topo=ctx_topo, ctx_text=ctx_text,
        cls=np.array([classes.index(r["class"]) for r in rows]),
        act=np.array([acts.index(r["actuator"]) for r in rows]),
        pos=np.array([pos.index(r["normal_position"]) if r["normal_position"] in pos else -1 for r in rows]),
        page=np.array([r["page"] for r in rows]), family=np.array([r["family"] for r in rows]),
        box_id=np.array([r["box_id"] for r in rows]),
        review=np.array([r["review"] for r in rows]), uncertain=np.array([r["uncertain"] for r in rows]),
        ft=np.array([r["ft_component_type"] for r in rows]))
    lc, lcls, lsrc, lid = legend_crops(classes)
    np.savez_compressed(os.path.join(a.out, "legend.npz"), crops=lc, cls=lcls, source=lsrc, legend_id=lid)
    print(len(lc), "legend symbols of", len(set(lcls.tolist())), "classes")
    json.dump({"classes": classes, "actuators": acts, "positions": pos,
               "groups": {c: cinfo[c]["group"] for c in classes}}, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    print(len(rows), "boxes;", len(classes), "classes;", acts)


if __name__ == "__main__":
    main()
