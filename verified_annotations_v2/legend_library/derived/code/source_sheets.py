"""Evidence sheets: one image per page (split into parts of 48), one tile per symbol box. Each tile shows the symbol as drawn,
the legend PDF symbol it was matched to, the legend PDF operator (for powered valves), the label, the evidence level and the
evidence itself (page legend, drawing text, source document, note). English only.
Output: <out>/<page>_<part>.jpg"""
import json, os, re, sys, collections, textwrap
from PIL import Image, ImageDraw, ImageFont
R = "/home/suxinqi666/code/PnIDAgent_project"; HERE = f"{R}/symbol_class"
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
S = json.load(open(f"{HERE}/symbols.json")); L = json.load(open(f"{HERE}/labels_current.json"))
SR = json.load(open(f"{HERE}/sources.json")); SN = json.load(open(f"{HERE}/legend_snip.json")); LK = json.load(open(f"{HERE}/legend_link.json")); LG = json.load(open(f"{HERE}/legend.json"))
FT = {}
import csv
for r in csv.DictReader(open(f"{R}/PnIDAgent/verified_annotations_v2/symbol_classes/symbol_classes.csv")): FT[(r["page"], int(r["box_id"]))] = r["ft_component_type"]
DJ = "/usr/share/fonts/truetype/dejavu/"
f_b = ImageFont.truetype(DJ + "DejaVuSans-Bold.ttf", 15); f_t = ImageFont.truetype(DJ + "DejaVuSans.ttf", 13); f_c = ImageFont.truetype(DJ + "DejaVuSans.ttf", 12)
f_h = ImageFont.truetype(DJ + "DejaVuSans-Bold.ttf", 26); f_hs = ImageFont.truetype(DJ + "DejaVuSans.ttf", 15)
LEV = {"A": ((46, 125, 50), (232, 245, 233), "A  confirmed by the drawing / a document"),
       "B": ((40, 90, 160), (232, 240, 250), "B  legend symbol shape / visual identification only"),
       "C": ((214, 100, 0), (255, 236, 220), "C  uncertain, check first")}
TW, TH, COLS, PER = 680, 262, 4, 48
SW, SH = 182, 112                      # image slot

def ascii_(t): return re.sub(r"[^\x20-\x7e]", "-", t.replace("—", "-").replace("–", "-").replace("″", '"').replace("′", "'"))
def fit(im, w, h, up=4.0):
    k = min(w / im.width, h / im.height, up); return im.resize((max(1, int(im.width * k)), max(1, int(im.height * k))), Image.LANCZOS if k < 1 else Image.BICUBIC)
def legref(j):
    e = LG[j]; fam = "AP1000" if e["src"].startswith("ap1000") else "APR1400"
    v = os.path.basename(e["file"])[:-4]; var = " (closed)" if v.endswith("_b") else ""
    return f"{fam} legend p.{int(e['src'][-1]) + 1} #{e['n']}{var}", ascii_(re.sub(r"\s+", " ", e.get("description") or ""))
def slot(d, img, x, y, im, cap1, cap2, col=(90, 90, 90), SW=182):
    d.rectangle([x, y, x + SW, y + SH], outline=(170, 170, 170))
    if im is not None:
        t = fit(im, SW - 8, SH - 8); img.paste(t, (x + (SW - t.width) // 2, y + (SH - t.height) // 2))
    else:
        for k, ln in enumerate(cap1.split("\n")): d.text((x + SW // 2, y + SH // 2 - 8 + 17 * k), ln, font=f_t, fill=col, anchor="mm")
        return
    d.text((x + 2, y + SH + 2), cap1, font=f_c, fill=col)
    if cap2: d.text((x + 2, y + SH + 17), cap2, font=f_c, fill=(60, 60, 60))
def short(x):
    x = ascii_(x)
    x = x.replace("drawing text next to the box: ", "text next to it: ").replace("drawing text inside: ", "text in box: ")
    if x.startswith("no operator drawn"): return "no operator drawn (legend: hand actuator not shown)"
    if x.startswith("legend PDF"): return None
    return x
def tile(i):
    s = S[i]; c, a = L[str(i)][:2]; lev = SR[str(i)]["level"]; dark, light, _ = LEV[lev]; lk = LK[str(i)]
    img = Image.new("RGB", (TW - 10, TH - 10), "white"); d = ImageDraw.Draw(img)
    d.rectangle([0, 0, TW - 11, TH - 11], fill=light, outline=dark, width=2); d.rectangle([0, 0, TW - 11, 24], fill=dark)
    ft = FT.get((s["page"], s["id"]), ""); act = "" if a in ("-",) else f" / {a}"
    d.text((8, 3), f"#{s['id']}   {c}{act}" + (f"   -> {ft}" if ft else ""), font=f_b, fill="white")
    d.text((TW - 34, 3), lev, font=f_b, fill="white")
    y = 32; x = 8
    W1, W2, W3 = 170, 230, 230
    slot(d, img, x, y, Image.open(f"{HERE}/crops/{s['page']}/{s['id']}.png").convert("RGB"), "this drawing", f"box {s['box'][0]:.0f},{s['box'][1]:.0f}", SW=W1)
    snip = lambda j: Image.open(f"{HERE}/legend_snip/{j}_sym.png" if LG[j]["src"] == "ap1000_p0" else LG[j]["file"]).convert("RGB")
    cut = lambda j: "(description cut off on the PDF)" if SN[str(j)]["mode"] == "symbol only" else '"' + textwrap.shorten(ascii_(re.sub(r"\s+", " ", LG[j]["description"])), 36, placeholder="...") + '"'
    if lk["body"] is not None:
        c1, c2 = legref(lk["body"]); slot(d, img, x + W1 + 8, y, snip(lk["body"]), c1.replace(" legend", ""), cut(lk["body"]), (20, 70, 150), SW=W2)
    else:
        slot(d, img, x + W1 + 8, y, None, "no legend PDF entry\n(class from drawing\ntext / note below)", "", (150, 60, 0), SW=W2)
    if lk["op"] is not None:
        c1, c2 = legref(lk["op"]); slot(d, img, x + W1 + W2 + 16, y, snip(lk["op"]), c1.replace(" legend", ""), cut(lk["op"]), (20, 70, 150), SW=W3)
    elif a not in ("-",) and c.startswith("VLV"):
        slot(d, img, x + W1 + W2 + 16, y, None, {"NONE": "no operator drawn", "UNSPEC": "operator drawn,\ntype not shown"}.get(a, a.lower()), "", SW=W3)
    lines = []
    for it in SR[str(i)]["items"]:
        t = short(it)
        if t: lines += textwrap.wrap(t, 80, subsequent_indent="   ")
    ty = y + SH + 34
    for k, ln in enumerate(lines[:5]):
        if k == 4 and len(lines) > 5: ln = ln[:76] + " ..."
        bold = ln.startswith(("page legend", "document", "uncertain"))
        d.text((8, ty + 16 * k), ln, font=f_t, fill=(150, 60, 0) if ln.startswith("uncertain") else (0, 0, 0) if bold else (40, 40, 40))
    return img

pages = sorted(set(s["page"] for s in S)); made = []
for pno, n in enumerate(pages, 1):
    idx = sorted([i for i, s in enumerate(S) if s["page"] == n], key=lambda i: S[i]["id"]); parts = (len(idx) + PER - 1) // PER
    cnt = collections.Counter(SR[str(i)]["level"] for i in idx)
    own = any(x.startswith("page legend") for i in idx for x in SR[str(i)]["items"])
    for p in range(parts):
        sub = idx[p * PER:(p + 1) * PER]; rows = (len(sub) + COLS - 1) // COLS
        W = COLS * TW + 20; H = 128 + rows * TH + 10
        sheet = Image.new("RGB", (W, H), "white"); d = ImageDraw.Draw(sheet)
        d.text((14, 10), ascii_(f"{pno}. {n.replace('pseudo_', '')}") + (f"   (part {p + 1}/{parts})" if parts > 1 else ""), font=f_h, fill="black")
        d.text((14, 46), f"{len(idx)} symbols: A {cnt['A']}, B {cnt['B']}, C {cnt['C']}.   Box ids as in ../overlay/{ascii_(n)}.jpg.   "
               "Each tile: the symbol on this drawing | the legend PDF symbol and its printed description | the legend PDF operator.", font=f_hs, fill=(40, 40, 40))
        d.text((14, 68), "Legend PDFs: 'AP1000 (PID Legend).pdf' and 'APR1400 (PID Legend).pdf'. " +
               ("This drawing has its own legend: cited as 'page legend'." if own else "This drawing has no legend of its own."), font=f_hs, fill=(40, 40, 40))
        x = 14
        for k in "ABC":
            dark, light, txt = LEV[k]; d.rectangle([x, 94, x + 18, 112], fill=dark); d.text((x + 24, 94), txt, font=f_hs, fill=dark); x += 24 + int(d.textlength(txt, font=f_hs)) + 30
        for k, i in enumerate(sub):
            sheet.paste(tile(i), (10 + (k % COLS) * TW, 124 + (k // COLS) * TH))
        fn = f"{n}_{p + 1}.jpg" if parts > 1 else f"{n}.jpg"
        sheet.save(f"{OUT}/{fn}", quality=82, optimize=True); made.append(fn)
    print(pno, n, len(idx), parts, flush=True)
json.dump(made, open(f"{OUT}/_files.json", "w"))
print(len(made), "sheets")
