#!/usr/bin/env python3
"""Build the class reference images in class_reference/.

For every class in classes.csv, <CLASS>.png shows the AP1000 / APR1400 legend PDF entries
that the boxes of this class were tied to, with the number of boxes citing each entry.
For every actuator value, actuator_<ACTUATOR>.png shows the operator legend entries in the
same way. overview.png shows the most-cited legend entry of every class on one sheet.

The legend PDFs themselves are not in the repository. The entry images are cut from the
evidence tiles (evidence/<page>[_<part>].png): the second panel of a tile is the legend
entry of the box, the third panel the legend entry of its operator, each with the PDF
page, entry number and printed description underneath. Panels are grouped by a coarse
image hash, so the same entry cited from several pages appears once.

Run from this folder:  python3 make_class_reference.py     (needs Pillow and numpy)
"""
import csv, os
from collections import defaultdict
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
EV_DIR = os.path.join(HERE, 'evidence')
OUT_DIR = os.path.join(HERE, 'class_reference')
os.makedirs(OUT_DIR, exist_ok=True)

# Evidence tile grid: 4 x 12 tiles of 680 x 262 px per image, 24 px header.
TILE_W, TILE_H, X0, Y0, HDR = 680, 262, 10, 124, 24
# Panels relative to the tile origin: legend entry image and the two caption lines under it.
CLASS_PANEL = (187, HDR + 9, 416, HDR + 120)
CLASS_CAPTION = (187, HDR + 122, 416, HDR + 152)
ACT_PANEL = (425, HDR + 9, 654, HDR + 120)
ACT_CAPTION = (425, HDR + 122, 654, HDR + 152)
NO_LEGEND = {'BLOCK', 'ENGINE', 'OFFPAGE', 'OTHER', 'PRESSURIZER', 'SUMP'}

SCALE, PER_ROW, MAX_ENTRIES, PAD = 1.4, 4, 8, 10
GROUP_COLOR = {'valve': (200, 40, 40), 'equipment': (40, 90, 200), 'instrument': (30, 140, 70),
               'electrical': (130, 50, 170), 'other': (220, 120, 20), 'actuator': (70, 70, 70)}


def font(size, bold=False):
    for f in ['/System/Library/Fonts/Helvetica.ttc', '/System/Library/Fonts/Supplemental/Arial.ttf',
              '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf' if bold else
              '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']:
        if os.path.exists(f):
            try:
                return ImageFont.truetype(f, size, index=1 if (bold and f.endswith('.ttc')) else 0)
            except Exception:
                return ImageFont.truetype(f, size)
    return ImageFont.load_default()


F_H, F_B, F_S = font(20, True), font(15), font(14)


def evidence_parts(page):
    if os.path.exists(os.path.join(EV_DIR, page + '.png')):
        return [page + '.png']
    parts, k = [], 1
    while os.path.exists(os.path.join(EV_DIR, f'{page}_{k}.png')):
        parts.append(f'{page}_{k}.png'); k += 1
    return parts


_cache = {}
def tile_origin(page, box_id):
    f = evidence_parts(page)[box_id // 48]
    if f not in _cache:
        _cache.clear()
        _cache[f] = Image.open(os.path.join(EV_DIR, f)).convert('RGB')
    i = box_id % 48
    return _cache[f], X0 + (i % 4) * TILE_W, Y0 + (i // 4) * TILE_H


def crop(im, tx, ty, box):
    return im.crop((tx + box[0], ty + box[1], tx + box[2], ty + box[3]))


def signature(im):
    """Non-white pixels of the caption at half resolution: robust to the anti-aliasing and palette
    differences between evidence images, while a different entry number or '(closed)' still differs."""
    g = np.array(im.convert('L').resize((im.width // 2, im.height // 2), Image.BOX), dtype=float)
    return (g < np.median(g) - 25).astype(np.uint8).ravel()   # median = the tile's tinted background


def same_entry(a, b):
    """Same legend entry: the signatures differ by at most 18 % of the larger one (plus 6 px)."""
    return np.count_nonzero(a != b) <= 0.18 * max(a.sum(), b.sum(), 1) + 6


def fit(d, s, f, width):
    if d.textlength(s, font=f) <= width:
        return s
    while d.textlength(s + '…', font=f) > width and len(s) > 4:
        s = s[:-1]
    return s + '…'


def collect(rows, panel, caption):
    """Distinct legend panels cited by these rows: [(image, caption image, count, first row)]."""
    reps = []
    for r in sorted(rows, key=lambda r: (r['page'], r['box_id'])):
        im, tx, ty = tile_origin(r['page'], r['box_id'])
        p, c = crop(im, tx, ty, panel), crop(im, tx, ty, caption)
        # identity = the caption (PDF page, entry number, description); the entry image itself is
        # placed slightly differently from tile to tile, so it is not compared
        s = signature(c)
        for rep in reps:
            if same_entry(rep[0], s):
                rep[3] += 1
                break
        else:
            reps.append([s, p, c, 1, r])
    reps.sort(key=lambda x: -x[3])
    return [(p, c, n, r) for s, p, c, n, r in reps]


def has_legend_image(panel):
    """A legend entry panel has a white image inside; a 'no legend PDF entry' panel is tinted text only."""
    a = np.array(panel.convert('L'))
    return (a >= 250).mean() > 0.05


def short_page(p):
    import re
    p = p.replace('pseudo_', '')
    p = re.sub(r'_NUREG-CR-4550.*$', ' (Surry NUREG)', p)
    p = re.sub(r'_\(Surry\) FSAR$', ' (Surry FSAR)', p)
    p = re.sub(r'_0520_R325C.*$', ' (CE training)', p)
    p = re.sub(r'_\(DAVIS-BESSE\).*_p(\d+)$', r' (Davis-Besse p\1)', p)
    return p.strip('"')


def sheet(title, subtitle, color, entries, note=None):
    pw, ph = int((CLASS_PANEL[2] - CLASS_PANEL[0]) * SCALE), int((CLASS_PANEL[3] - CLASS_PANEL[1]) * SCALE)
    ch = int((CLASS_CAPTION[3] - CLASS_CAPTION[1]) * SCALE)
    cell_w, cell_h = pw + PAD, ph + ch + 22 + PAD
    n = min(len(entries), MAX_ENTRIES)
    cols = min(PER_ROW, max(n, 1))
    rows_n = max(1, (n + PER_ROW - 1) // PER_ROW)
    W = PAD + PER_ROW * cell_w + PAD
    H = 44 + rows_n * cell_h + PAD + (20 if note else 0)
    im = Image.new('RGB', (W, H), 'white')
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, W, 36), fill=color)
    d.text((PAD, 8), title, font=F_H, fill='white')
    d.text((PAD + d.textlength(title, font=F_H) + 14, 11), subtitle, font=F_B, fill='white')
    if not entries:
        d.text((PAD, 60), note or 'no instance in the 65 pages', font=F_B, fill=(150, 150, 150))
        return im
    for k, (p, c, cnt, r) in enumerate(entries[:MAX_ENTRIES]):
        x = PAD + (k % PER_ROW) * cell_w
        y = 44 + (k // PER_ROW) * cell_h
        im.paste(p.resize((pw, ph), Image.LANCZOS), (x, y))
        d.rectangle((x, y, x + pw, y + ph), outline=(170, 170, 170))
        im.paste(c.resize((pw, ch), Image.LANCZOS), (x, y + ph + 2))
        d.text((x, y + ph + ch + 4), fit(d, f'cited by {cnt} box{"es" if cnt != 1 else ""}'
               f'  (e.g. {short_page(r["page"])} #{r["box_id"]})', F_S, pw), font=F_S, fill=(110, 110, 110))
    if len(entries) > MAX_ENTRIES:
        d.text((PAD, H - 22), f'... and {len(entries) - MAX_ENTRIES} more entries cited by '
               f'{sum(e[2] for e in entries[MAX_ENTRIES:])} boxes', font=F_S, fill=(110, 110, 110))
    elif note:
        d.text((PAD, H - 22), note, font=F_S, fill=(110, 110, 110))
    return im


def main():
    classes = list(csv.DictReader(open(os.path.join(HERE, 'classes.csv'), encoding='utf-8')))
    actuators = list(csv.DictReader(open(os.path.join(HERE, 'actuators.csv'), encoding='utf-8')))
    rows = list(csv.DictReader(open(os.path.join(HERE, 'symbol_classes.csv'), encoding='utf-8')))
    for r in rows:
        r['box_id'] = int(r['box_id'])
    by_cls, by_act = defaultdict(list), defaultdict(list)
    for r in rows:
        by_cls[r['class']].append(r)
        if r['actuator'] not in ('-', 'NONE'):
            by_act[r['actuator']].append(r)
    overview = []
    for c in classes:
        cls = c['class']; cls_rows = by_cls.get(cls, [])
        entries = collect(cls_rows, CLASS_PANEL, CLASS_CAPTION) if cls_rows else []
        unc = sum(r['uncertain'] == 'yes' for r in cls_rows)
        sub = f"{c['definition']}   |   group: {c['group']}   |   {len(cls_rows)} boxes" + (f', {unc} uncertain' if unc else '')
        note = ('no legend PDF entry: the panels show the drawing text the class rests on'
                if cls in NO_LEGEND else None)
        im = sheet(cls, sub, GROUP_COLOR[c['group']], entries, note)
        im.save(os.path.join(OUT_DIR, cls + '.png'), optimize=True)
        print(f'{cls:14s} {len(cls_rows):5d} boxes -> {len(entries):2d} legend entries')
        real = [e for e in entries if has_legend_image(e[0])]
        if real:
            overview.append((cls, c['group'], real[0][0]))
    for a in actuators:
        act = a['actuator']; act_rows = by_act.get(act, [])
        if act in ('-', 'NONE'):
            continue
        entries = collect(act_rows, ACT_PANEL, ACT_CAPTION) if act_rows else []
        im = sheet(act, f"{a['definition']}   |   {len(act_rows)} boxes", GROUP_COLOR['actuator'], entries)
        im.save(os.path.join(OUT_DIR, f'actuator_{act}.png'), optimize=True)
        print(f'actuator {act:10s} {len(act_rows):5d} boxes -> {len(entries):2d} operator entries')
    # overview: most-cited legend entry per class
    cols, pw, ph = 6, 229, 111
    cw, chh = pw + 12, ph + 30
    rows_n = (len(overview) + cols - 1) // cols
    ov = Image.new('RGB', (PAD * 2 + cols * cw, PAD * 2 + rows_n * chh), 'white')
    d = ImageDraw.Draw(ov)
    for i, (cls, grp, p) in enumerate(overview):
        x, y = PAD + (i % cols) * cw, PAD + (i // cols) * chh
        ov.paste(p, (x, y))
        d.rectangle((x, y, x + pw, y + ph), outline=GROUP_COLOR[grp], width=2)
        d.text((x, y + ph + 4), cls, font=F_B, fill=GROUP_COLOR[grp])
    ov.save(os.path.join(OUT_DIR, 'overview.png'), optimize=True)
    print('overview', ov.size, len(overview), 'classes with a legend entry')


if __name__ == '__main__':
    main()
