"""Check images of the symbol classes and the page index:

  ../overlay/<page>.jpg   the drawing with every box, its id and its class tag (colour = class group); a closed valve or
                          damper is marked NC, a locked one LO / LC
  ../index.md             one row per page: symbols, link to its overview

  python render_overlays.py
"""
import collections, csv, os
from urllib.parse import quote
import cv2, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.dirname(HERE); V = os.path.dirname(OUT)
GROUP_BGR = {"valve": (40, 40, 220), "equipment": (200, 110, 30), "instrument": (40, 160, 40), "electrical": (170, 40, 170),
             "other": (0, 140, 230)}


def main():
    groups = {r["class"]: r["group"] for r in csv.DictReader(open(f"{OUT}/classes.csv", encoding="utf-8"))}
    rows = list(csv.DictReader(open(f"{OUT}/symbol_classes.csv", encoding="utf-8")))
    by_page = collections.defaultdict(list)
    for r in rows: by_page[r["page"]].append(r)
    os.makedirs(f"{OUT}/overlay", exist_ok=True)
    index = ["# Symbol class check images (65 pages)", "",
             f"One overview per page for the {len(rows)} symbol boxes: the whole drawing, every box with its id and class tag "
             "(red = valve or damper, blue = equipment, green = instrument, purple = electrical, orange = other). "
             "NC = normally closed, LO / LC = locked open / closed. Use it to find a box id on the drawing.", "",
             "| No. | Page | Symbols | Overview |", "|---|---|---|---|"]
    for k, page in enumerate(sorted(by_page), 1):
        g = cv2.imread(f"{V}/topology_verified/images/{page}.png"); H, W = g.shape[:2]; sc = max(W, H) / 3300
        im = (g.astype(np.float32) * 0.6 + 255 * 0.4).astype(np.uint8); lw = max(2, int(3 * sc))
        for r in by_page[page]:
            b = [int(float(r[q])) for q in ("x0", "y0", "x1", "y1")]; c = GROUP_BGR.get(groups.get(r["class"]), (0, 0, 0))
            cv2.rectangle(im, tuple(b[:2]), tuple(b[2:]), c, lw)
            tag = f"{r['box_id']} {r['class']}" + (f"/{r['actuator']}" if r["actuator"] not in ("-", "", "NONE") else "")
            tag += " NC" if r["normal_position"] == "closed" else ""
            tag += f" {r['locked']}" if r["locked"] else ""
            cv2.putText(im, tag, (b[0], b[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55 * sc, c, max(1, lw - 1))
        pad = int(60 * sc); out = cv2.copyMakeBorder(im, pad, 0, 25, 25, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        cv2.putText(out, f"{page[7:]}  -  {len(by_page[page])} symbols", (25, int(40 * sc)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8 * sc, (0, 0, 0), max(1, lw - 1))
        cv2.imwrite(f"{OUT}/overlay/{page}.jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 82])
        index.append(f"| {k} | {page[7:]} | {len(by_page[page])} | [overview](overlay/{quote(page)}.jpg) |")
    open(f"{OUT}/index.md", "w", encoding="utf-8").write("\n".join(index) + "\n")
    print("rendered", len(by_page), "pages")


if __name__ == "__main__":
    main()
