"""Redraw the illustration images of the pages changed by the review fixes (fixes.json), from the fixed annotations:

  ../overlay/<page>.jpg                                  symbols (red) and pipes (green; dashed flow pipes in teal)
  ../topology_verified/overlays/<page>_symbols_lines.jpg  the same on a faded drawing, with symbol ids, full resolution
  ../topology_verified/<page>.jpg                        connections on the drawing, one colour per pipe segment
  ../topology_verified/<page>_explained.png              the drawing with pipes coloured by connected network, next to
                                                         the extracted network (symbols and branch points at their
                                                         positions, one straight line per pipe segment)
  ../symbol_classes/overlay/<page>.jpg                   symbol boxes coloured by class group, with the class tag

  python render_images.py               # every page of fixes.json
  python render_images.py SCS_APR1400   # pages whose name contains the argument
"""
import collections, colorsys, csv, json, os, re, sys
import cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont
HERE = os.path.dirname(os.path.abspath(__file__)); V = os.path.dirname(HERE); T = f"{V}/topology_verified"
FONT = next((f for f in ("/System/Library/Fonts/Helvetica.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf") if os.path.exists(f)), None)
GROUP_BGR = {"valve": (40, 40, 220), "equipment": (200, 110, 30), "instrument": (40, 160, 40), "electrical": (170, 40, 170), "other": (0, 150, 200)}


def font(size):
    return ImageFont.truetype(FONT, size) if FONT else ImageFont.load_default()


def palette(n):
    return [tuple(int(255 * c) for c in colorsys.hsv_to_rgb((i * 0.618) % 1, 0.85, 0.85))[::-1] for i in range(max(n, 1))]


def faded(g, k=0.45):
    return (g.astype(np.float32) * k + 255 * (1 - k)).astype(np.uint8)


def titles():
    out = {}
    txt = open(f"{T}/README.md", encoding="utf-8").read()
    for m in re.finditer(r"^### \d+\. (.+?)\n\n.*?\]\((pseudo_[^)]*?)_explained\.png\)", txt, re.M | re.S):
        name = m.group(2).replace("%28", "(").replace("%29", ")").replace("%20", " ")
        t = m.group(1); doc = re.search(r"\(([^()]*(?:\([^()]*\))?[^()]*)\)\s*$", t)
        out[name] = (t[:doc.start()].strip() if doc else t, doc.group(1) if doc else "")
    return out


def load(page):
    g = cv2.imread(f"{T}/images/{page}.png")
    L = json.load(open(f"{V}/lines/{page}.json"))
    topo = json.load(open(f"{T}/{page}.json"))
    rows = [r for r in csv.DictReader(open(f"{V}/symbol_classes/symbol_classes.csv", encoding="utf-8")) if r["page"] == page]
    return g, L, topo, rows


def networks(topo):
    adj = collections.defaultdict(set)
    for l in topo["links"]: adj[str(l["source"])].add(str(l["target"])); adj[str(l["target"])].add(str(l["source"]))
    comp, n = {}, 0
    for s in [str(s["id"]) for s in topo["symbols"]] + [j["id"] for j in topo["junctions"]]:
        if s in comp: continue
        stack = [s]; comp[s] = n
        while stack:
            for v in adj[stack.pop()]:
                if v not in comp: comp[v] = n; stack.append(v)
        n += 1
    sizes = collections.Counter(comp.values())
    real = sorted((c for c in sizes if sizes[c] > 1), key=lambda c: -sizes[c])
    return comp, {c: i for i, c in enumerate(real)}


def draw_lines(img, L, w, solid=(0, 170, 0), dashed=(170, 150, 0)):
    for s in L["solid"]: cv2.line(img, (int(s[0]), int(s[1])), (int(s[2]), int(s[3])), solid, w)
    for s in L.get("dashed", []): cv2.line(img, (int(s[0]), int(s[1])), (int(s[2]), int(s[3])), dashed, w)


def render(page, ttl):
    g, L, topo, rows = load(page); H, W = g.shape[:2]; sc = max(W, H) / 3300
    lw = max(2, int(3 * sc)); fs = 0.7 * sc
    # 1. overlay/<page>.jpg
    im = g.copy(); draw_lines(im, L, lw)
    for s in topo["symbols"]: cv2.rectangle(im, tuple(s["bbox"][:2]), tuple(s["bbox"][2:]), (0, 0, 230), lw)
    f = 2000 / W; cv2.imwrite(f"{V}/overlay/{page}.jpg", cv2.resize(im, None, fx=f, fy=f, interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 88])
    # 2. topology_verified/overlays/<page>_symbols_lines.jpg
    im = faded(g); draw_lines(im, L, lw)
    for s in topo["symbols"]:
        cv2.rectangle(im, tuple(s["bbox"][:2]), tuple(s["bbox"][2:]), (0, 0, 230), lw)
        cv2.putText(im, str(s["id"]), (s["bbox"][0], s["bbox"][1] - 4), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 230), max(1, lw - 1))
    cv2.imwrite(f"{T}/overlays/{page}_symbols_lines.jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 88])
    # 3. topology_verified/<page>.jpg: one colour per pipe segment (link)
    im = faded(g, 0.35); pal = palette(len(topo["links"]))
    for k, l in enumerate(topo["links"]):
        for p in l["path"]: cv2.line(im, (int(p[0]), int(p[1])), (int(p[2]), int(p[3])), pal[k], max(3, int(5 * sc)))
    for s in topo["symbols"]:
        cv2.rectangle(im, tuple(s["bbox"][:2]), tuple(s["bbox"][2:]), (0, 0, 200), lw)
        cv2.putText(im, str(s["id"]), (s["bbox"][0], s["bbox"][1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 1.1 * sc, (0, 0, 200), max(2, lw))
    for j in topo["junctions"]: cv2.circle(im, (int(j["xy"][0]), int(j["xy"][1])), max(4, int(6 * sc)), (0, 0, 0), -1)
    f = 1600 / W; cv2.imwrite(f"{T}/{page}.jpg", cv2.resize(im, None, fx=f, fy=f, interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 88])
    # 4. topology_verified/<page>_explained.png
    comp, rank = networks(topo); npal = palette(len(rank))
    col = lambda node: npal[rank[comp[str(node)]]] if comp.get(str(node)) in rank else (150, 150, 150)
    left = faded(g, 0.3)
    for l in topo["links"]:
        for p in l["path"]: cv2.line(left, (int(p[0]), int(p[1])), (int(p[2]), int(p[3])), col(l["source"]), max(3, int(6 * sc)))
    for s in topo["symbols"]:
        cv2.rectangle(left, tuple(s["bbox"][:2]), tuple(s["bbox"][2:]), (60, 60, 60), lw)
        cv2.putText(left, str(s["id"]), (s["bbox"][0], s["bbox"][1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 1.3 * sc, (20, 20, 20), max(2, lw))
    right = np.full_like(g, 255)
    pos = {str(s["id"]): ((s["bbox"][0] + s["bbox"][2]) / 2, (s["bbox"][1] + s["bbox"][3]) / 2) for s in topo["symbols"]}
    pos.update({j["id"]: tuple(j["xy"]) for j in topo["junctions"]})
    for l in topo["links"]:
        a, b = pos[str(l["source"])], pos[str(l["target"])]
        cv2.line(right, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), col(l["source"]), max(3, int(5 * sc)))
    for j in topo["junctions"]: cv2.circle(right, (int(j["xy"][0]), int(j["xy"][1])), max(5, int(7 * sc)), (0, 0, 0), -1)
    for s in topo["symbols"]:
        c = pos[str(s["id"])]; r = max(14, int(20 * sc))
        cv2.circle(right, (int(c[0]), int(c[1])), r, col(s["id"]), -1)
        t = str(s["id"]); (tw, th), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.9 * sc, max(2, lw))
        cv2.putText(right, t, (int(c[0] - tw / 2), int(c[1] + th / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.9 * sc, (255, 255, 255), max(2, lw))
    canvas = Image.new("RGB", (1920, 1080), "white"); d = ImageDraw.Draw(canvas)
    name, doc = ttl
    deg = collections.Counter(x for l in topo["links"] for x in (l["source"], l["target"]))
    tj = sum(1 for j in topo["junctions"] if deg[j["id"]] >= 3); oe = sum(1 for j in topo["junctions"] if deg[j["id"]] == 1)
    d.text((40, 28), name, fill=(20, 20, 20), font=font(40)); d.text((40, 80), doc, fill=(110, 110, 110), font=font(22))
    d.text((1700, 34), "Verified", fill=(40, 140, 60), font=font(30))
    stats = f"{len(topo['symbols'])} symbols  ·  {len(topo['links'])} pipe segments  ·  {tj + oe} branch points  ·  {len(rank)} connected network{'s' if len(rank) != 1 else ''}"
    d.text((1880 - d.textlength(stats, font=font(22)), 80), stats, fill=(60, 60, 60), font=font(22))
    d.text((40, 128), "Left: the drawing, pipes coloured by connected network (same colour = connected)", fill=(110, 110, 110), font=font(19))
    d.text((980, 128), "Right: extracted network (coloured dot = symbol, black dot = branch point, line = pipe segment)", fill=(110, 110, 110), font=font(19))
    xs = [v for s in topo["symbols"] for v in (s["bbox"][0], s["bbox"][2])] + [v for l in topo["links"] for q in l["path"] for v in (q[0], q[2])]
    ys = [v for s in topo["symbols"] for v in (s["bbox"][1], s["bbox"][3])] + [v for l in topo["links"] for q in l["path"] for v in (q[1], q[3])]
    m = int(0.05 * max(W, H)); cx0, cy0 = max(0, int(min(xs)) - m), max(0, int(min(ys)) - m); cx1, cy1 = min(W, int(max(xs)) + m), min(H, int(max(ys)) + m)
    for k, panel in enumerate((left, right)):
        panel = panel[cy0:cy1, cx0:cx1]; pw, ph = 900, 830; f = min(pw / panel.shape[1], ph / panel.shape[0])
        p = cv2.cvtColor(cv2.resize(panel, None, fx=f, fy=f, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        x0 = 30 + k * 950 + (pw - p.shape[1]) // 2; y0 = 170 + (ph - p.shape[0]) // 2
        canvas.paste(Image.fromarray(p), (x0, y0)); d.rectangle((30 + k * 950, 170, 30 + k * 950 + pw, 170 + ph), outline=(210, 210, 210))
    d.text((40, 1030), "Numbers match on both sides. The right side is the drawing simplified: pipes straightened and text removed, with exactly the same connections.",
           fill=(110, 110, 110), font=font(19))
    canvas.save(f"{T}/{page}_explained.png")
    # 5. symbol_classes/overlay/<page>.jpg
    groups = {r["class"]: r["group"] for r in csv.DictReader(open(f"{V}/symbol_classes/classes.csv", encoding="utf-8"))}
    im = faded(g, 0.6)
    for r in rows:
        b = [int(float(r[k])) for k in ("x0", "y0", "x1", "y1")]; c = GROUP_BGR.get(groups.get(r["class"], "other"), (0, 0, 0))
        cv2.rectangle(im, tuple(b[:2]), tuple(b[2:]), c, lw)
        tag = f"{r['box_id']} {r['class']}" + (f"/{r['actuator']}" if r["actuator"] not in ("-", "") else "")
        cv2.putText(im, tag, (b[0], b[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55 * sc, c, max(1, lw - 1))
    pad = int(60 * sc); out = cv2.copyMakeBorder(im, pad, 0, 25, 25, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    cv2.putText(out, f"{page[7:]}  -  {len(rows)} symbols (class pre-labels; red valve, blue equipment, green instrument, purple electrical, orange other)",
                (25, int(40 * sc)), cv2.FONT_HERSHEY_SIMPLEX, 0.8 * sc, (0, 0, 0), max(1, lw - 1))
    cv2.imwrite(f"{V}/symbol_classes/overlay/{page}.jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 85])


if __name__ == "__main__":
    fixes = [p for p in json.load(open(f"{HERE}/fixes.json", encoding="utf-8")) if not p.startswith("_")]
    args = sys.argv[1:]; tt = titles()
    for p in fixes:
        if args and not any(a in p for a in args): continue
        render(p, tt.get(p, (p[7:], ""))); print("rendered", p)
