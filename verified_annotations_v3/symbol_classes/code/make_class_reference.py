"""Class reference images in ../class_reference/:

  <CLASS>.png              the legend symbols the class stands for (AP1000 / APR1400 P&ID legends), each with its PDF page,
                           entry number and printed description, and the number of boxes on the 65 pages tied to it; the
                           symbols tied to boxes come first. A class with no legend symbol shows drawn examples instead.
  actuator_<ACTUATOR>.png  the operator symbols tied to the boxes with this actuator (drawn examples where there is none)
  overview.png             the most-cited legend symbol of every class that has one, on one sheet

The symbol images are those of ../../legend_library (derived/snippets/<id>_sym.png, else the extracted 600 dpi image).

  python make_class_reference.py
"""
import collections, csv, json, os
import cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.dirname(HERE); V = os.path.dirname(OUT)
LEG = f"{V}/legend_library"; REF = f"{OUT}/class_reference"
FONT = next((f for f in ("/System/Library/Fonts/Helvetica.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf") if os.path.exists(f)), None)
GROUP_RGB = {"valve": (200, 40, 40), "equipment": (40, 90, 200), "instrument": (30, 140, 70), "electrical": (130, 50, 170),
             "other": (220, 120, 20), "actuator": (70, 70, 70)}
TINT = {"AP1000": (255, 240, 225), "APR1400": (230, 240, 252), "drawing": (238, 238, 238)}
TW, TH, IMG_H, PAD, PER_ROW, MAX_TILES = 300, 210, 140, 10, 4, 12


def font(n): return ImageFont.truetype(FONT, n) if FONT else ImageFont.load_default()


def fit(im, w, h):
    im = im.convert("RGB"); f = min(w / im.width, h / im.height, 3.0)
    return im.resize((max(1, int(im.width * f)), max(1, int(im.height * f))), Image.LANCZOS)


def legend_image(e):
    sym = f"{LEG}/derived/snippets/{e['legend_id']}_sym.png"
    p = sym if os.path.exists(sym) else os.path.normpath(os.path.join(f"{LEG}/derived", e["image"]))
    return Image.open(p) if os.path.exists(p) else None


def drawn_example(r):
    g = cv2.imread(f"{V}/topology_verified/images/{r['page']}.png")
    x0, y0, x1, y1 = (int(float(r[k])) for k in ("x0", "y0", "x1", "y1")); H, W = g.shape[:2]; pad = int(max(15, 0.25 * max(x1 - x0, y1 - y0)))
    c = g[max(0, y0 - pad):min(H, y1 + pad), max(0, x0 - pad):min(W, x1 + pad)].copy()
    cv2.rectangle(c, (x0 - max(0, x0 - pad), y0 - max(0, y0 - pad)), (x1 - max(0, x0 - pad), y1 - max(0, y0 - pad)), (0, 0, 220), 2)
    return Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))


def tile(img, line1, line2, line3, tint):
    t = Image.new("RGB", (TW, TH), "white"); d = ImageDraw.Draw(t)
    d.rectangle((0, 0, TW - 1, IMG_H + 9), fill=tint)
    box = Image.new("RGB", (TW - 20, IMG_H), "white")
    if img is not None:
        f = fit(img, TW - 30, IMG_H - 10); box.paste(f, ((box.width - f.width) // 2, (box.height - f.height) // 2))
    t.paste(box, (10, 5)); d.rectangle((10, 5, TW - 11, IMG_H + 4), outline=(170, 170, 170))
    d.rectangle((0, IMG_H + 10, TW - 1, IMG_H + 52), fill=tint)
    d.text((6, IMG_H + 12), line1, fill=(40, 40, 40), font=font(15))
    d.text((6, IMG_H + 31), line2[:38] + ("..." if len(line2) > 38 else ""), fill=(40, 40, 40), font=font(14))
    d.text((6, IMG_H + 56), line3[:46] + ("..." if len(line3) > 46 else ""), fill=(90, 90, 90), font=font(12))
    return t


def sheet(title, sub, color, tiles, footer=""):
    rows = max(1, (len(tiles) + PER_ROW - 1) // PER_ROW)
    W = PAD + PER_ROW * (TW + PAD); H = 46 + PAD + rows * (TH + PAD) + (24 if footer else 0)
    im = Image.new("RGB", (W, H), "white"); d = ImageDraw.Draw(im)
    d.rectangle((0, 0, W, 38), fill=color); d.text((12, 7), title, fill="white", font=font(22))
    d.text((24 + d.textlength(title, font=font(22)), 12), sub, fill="white", font=font(15))
    for k, t in enumerate(tiles):
        im.paste(t, (PAD + (k % PER_ROW) * (TW + PAD), 46 + PAD + (k // PER_ROW) * (TH + PAD)))
    if not tiles: d.text((14, 60), "no symbol", fill=(90, 90, 90), font=font(15))
    if footer: d.text((12, H - 22), footer, fill=(90, 90, 90), font=font(13))
    return im


def legend_tiles(entries, cites, examples, cap=MAX_TILES):
    order = sorted(entries, key=lambda e: (-cites.get(e["legend_id"], 0), e["legend_id"]))
    tiles = []
    for e in order[:cap]:
        n = cites.get(e["legend_id"], 0); src = "AP1000" if e["pdf"].startswith("AP1000") else "APR1400"
        ex = examples.get(e["legend_id"])
        l3 = f"cited by {n} boxes" + (f"  (e.g. {ex})" if ex else "") if n else "not drawn on the 65 pages"
        var = f" ({e['variant']})" if e.get("variant") else ""
        tiles.append(tile(legend_image(e), f"{src} p.{e['pdf_page']} #{e['entry']}{var}", f"\"{e['description']}\"", l3, TINT[src]))
    more = len(order) - cap
    return tiles, (f"+ {more} more legend symbols of this class, not drawn on the 65 pages" if more > 0 else "")


def main():
    os.makedirs(REF, exist_ok=True)
    classes = list(csv.DictReader(open(f"{OUT}/classes.csv", encoding="utf-8")))
    acts = list(csv.DictReader(open(f"{OUT}/actuators.csv", encoding="utf-8")))
    rows = list(csv.DictReader(open(f"{OUT}/symbol_classes.csv", encoding="utf-8")))
    idx = json.load(open(f"{LEG}/derived/legend_index.json")); byid = {e["legend_id"]: e for e in idx}
    links = json.load(open(f"{LEG}/derived/symbol_legend_links.json"))
    row = {(r["page"], int(r["box_id"])): r for r in rows}
    cite = collections.defaultdict(collections.Counter); ex = collections.defaultdict(dict)
    opc = collections.defaultdict(collections.Counter); opex = collections.defaultdict(dict)
    for p, es in links.items():
        for e in es:
            if e.get("legend_symbol") is not None:
                cite[e["class"]][e["legend_symbol"]] += 1; ex[e["class"]].setdefault(e["legend_symbol"], f"{p[7:27]} #{e['box_id']}")
            if e.get("legend_operator") is not None and e["actuator"] not in ("-", "NONE"):
                opc[e["actuator"]][e["legend_operator"]] += 1; opex[e["actuator"]].setdefault(e["legend_operator"], f"{p[7:27]} #{e['box_id']}")
    by_class = collections.defaultdict(list)
    for e in idx:
        if e.get("class") and not e.get("not_applicable_variant"): by_class[e["class"]].append(e)
    overview = []
    for c in classes:
        name, n = c["class"], int(c["count"])
        unc = sum(1 for r in rows if r["class"] == name and r["uncertain"].strip())
        sub = f"{c['definition']}   |   group: {c['group']}   |   {n} boxes" + (f", {unc} uncertain" if unc else "") + \
              ("   |   not modeled in PRA" if c["pra_modeled"] == "no" else "")
        entries = by_class.get(name, [])
        if entries:
            tiles, footer = legend_tiles(entries, cite[name], ex[name])
            top = max(entries, key=lambda e: cite[name].get(e["legend_id"], 0))
            overview.append((name, c["group"], legend_image(top)))
        elif name == "DAMPER_MOTOR":
            parts = [e for e in idx if e["legend_id"] in (257, 267)]
            tiles = [tile(legend_image(e), f"AP1000 p.{e['pdf_page']} #{e['entry']}", f"\"{e['description']}\"",
                          "drawn together: damper + motor operator", TINT["AP1000"]) for e in parts]
            footer = "No legend symbol of its own: a damper symbol with the motor operator drawn on it."
        else:
            sel = [r for r in rows if r["class"] == name][:8]
            tiles = [tile(drawn_example(r), f"drawing: {r['page'][7:30]}", f"box {r['box_id']}", r["note"], TINT["drawing"]) for r in sel]
            footer = "No legend symbol: examples as drawn on the pages (red frame = the box); the class rests on the drawing text."
        sheet(name, sub, GROUP_RGB[c["group"]], tiles, footer).save(f"{REF}/{name}.png", optimize=True)
    for a in acts:
        name = a["actuator"]
        if name in ("-", "NONE"): continue
        n = sum(1 for r in rows if r["actuator"] == name)
        sub = f"{a['definition']}   |   {n} boxes"
        if opc[name]:
            tiles, footer = legend_tiles([byid[i] for i in opc[name]], opc[name], opex[name])
        else:
            sel = [r for r in rows if r["actuator"] == name][:8]
            tiles = [tile(drawn_example(r), f"drawing: {r['page'][7:30]}", f"box {r['box_id']}  {r['class']}", r["note"], TINT["drawing"]) for r in sel]
            footer = "No operator symbol is tied to these boxes: examples as drawn on the pages (red frame = the box)."
        sheet(f"{name} (actuator)", sub, GROUP_RGB["actuator"], tiles, footer).save(f"{REF}/actuator_{name}.png", optimize=True)
    # overview: one cell per class with a legend symbol
    cw, ch, per = 190, 150, 8
    rows_n = (len(overview) + per - 1) // per
    im = Image.new("RGB", (PAD + per * (cw + PAD), 50 + rows_n * (ch + PAD)), "white"); d = ImageDraw.Draw(im)
    d.text((12, 12), f"The most-cited legend symbol of each of the {len(overview)} classes that have one (frame colour = group)", fill=(30, 30, 30), font=font(18))
    for k, (name, g, img) in enumerate(overview):
        x, y = PAD + (k % per) * (cw + PAD), 50 + (k // per) * (ch + PAD)
        d.rectangle((x, y, x + cw, y + ch), outline=GROUP_RGB[g], width=3)
        if img is not None:
            f = fit(img, cw - 16, ch - 40); im.paste(f, (x + (cw - f.width) // 2, y + 8 + (ch - 40 - f.height) // 2))
        d.text((x + 6, y + ch - 26), name, fill=GROUP_RGB[g], font=font(14))
    im.save(f"{REF}/overview.png", optimize=True)
    print("wrote", len(os.listdir(REF)), "images")


if __name__ == "__main__":
    main()
