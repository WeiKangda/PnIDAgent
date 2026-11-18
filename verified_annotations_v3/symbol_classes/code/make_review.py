"""Review workbook for the symbol classes: one sheet per kind of decision that needs a person, every row with a crop
of the symbol on its drawing (red frame = the box), the proposal and empty columns for the decision.

  python make_review.py             # writes ../review.xlsx (run derive_classes.py first)
"""
import csv, io, os
import cv2
from openpyxl import Workbook
from openpyxl.drawing.image import Image as XLImage
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.dirname(HERE); V = os.path.dirname(OUT)
IMG = f"{V}/topology_verified/images"
SHEETS = [
    ("fan_or_pump", "Fans and pumps can share the same circular symbol on HVAC drawings: confirm each box is a fan (or say which pump class it is).",
     "FAN / PUMP_CENTRIFUGAL / PUMP_VERTICAL / other"),
    ("normal_position", "Normally closed valves and dampers (and the ones whose position could not be read): confirm the normal position. "
                        "Evidence = where the proposal comes from (the labelers' note, or the automatic match to the legend's open / closed symbol).",
     "closed / open / not shown"),
    ("auto_split", "The finer class of these boxes rests only on the automatic match to a legend entry: confirm the class.", "a class of the sheet classes"),
    ("other", "Boxes of class OTHER: give a class from the sheet classes where one fits, or keep OTHER.", "a class of the sheet classes, or OTHER"),
    ("uncertain", "Boxes the labelers marked uncertain: confirm or correct the class and the actuator.", "a class of the sheet classes"),
]
COLS = ["crop", "page", "box_id", "proposed class", "actuator", "proposed normal position", "locked", "class evidence",
        "position evidence", "legend entry (automatic match)", "note", "decision (ok / change)", "corrected value", "comment"]
WIDTH = {"crop": 30, "page": 26, "box_id": 7, "proposed class": 20, "actuator": 10, "proposed normal position": 13,
         "locked": 7, "class evidence": 18, "position evidence": 20, "legend entry (automatic match)": 28, "note": 36,
         "decision (ok / change)": 12, "corrected value": 18, "comment": 30}
REVIEW_FILL = PatternFill("solid", fgColor="FFF2CC"); HEAD_FILL = PatternFill("solid", fgColor="D9D9D9")
GREEN = PatternFill("solid", fgColor="E2EFDA"); RED_FONT = Font(color="FF0000")


def crop_png(img, r, h=150, w=210):
    x0, y0, x1, y1 = (float(r[k]) for k in ("x0", "y0", "x1", "y1"))
    H, W = img.shape[:2]; pad = int(max(30, 0.6 * max(x1 - x0, y1 - y0)))
    a, b, c, d = max(0, int(x0) - pad), max(0, int(y0) - pad), min(W, int(x1) + pad), min(H, int(y1) + pad)
    cr = img[b:d, a:c].copy()
    t = max(1, int(round(max(cr.shape[:2]) / 160)))
    cv2.rectangle(cr, (int(x0) - a, int(y0) - b), (int(x1) - a, int(y1) - b), (0, 0, 230), t)
    f = min(h / cr.shape[0], w / cr.shape[1])
    cr = cv2.resize(cr, None, fx=f, fy=f, interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_CUBIC)
    ok, buf = cv2.imencode(".png", cr); return io.BytesIO(buf.tobytes())


def main():
    rows = list(csv.DictReader(open(f"{OUT}/symbol_classes.csv", encoding="utf-8")))
    classes = list(csv.DictReader(open(f"{OUT}/classes.csv", encoding="utf-8")))
    wb = Workbook(); ws = wb.active; ws.title = "README"
    lines = ["Symbol classes: review", "",
             "Each sheet lists the symbol boxes that need a person's decision, with a crop of the symbol on its drawing (red frame = the box).",
             "Fill the yellow columns: 'decision' = ok (the proposal is right) or change; 'corrected value' = the right class or position when it is change; 'comment' = anything else.",
             "Classes: see the sheet classes (green = typically not modeled in a PRA; red = no symbol in the AP1000 / APR1400 legends).", ""]
    for name, what, value in SHEETS:
        n = sum(1 for r in rows if name in r["review"].split(";"))
        lines.append(f"{name} ({n} boxes): {what} Corrected value: {value}.")
    for i, l in enumerate(lines, 1):
        ws.cell(i, 1, l).font = Font(bold=(i == 1), size=14 if i == 1 else 11)
    ws.column_dimensions["A"].width = 160
    cs = wb.create_sheet("classes")
    for j, h in enumerate(classes[0].keys(), 1):
        cs.cell(1, j, h).font = Font(bold=True); cs.cell(1, j).fill = HEAD_FILL
    for i, c in enumerate(classes, 2):
        for j, (k, v) in enumerate(c.items(), 1):
            cell = cs.cell(i, j, int(v) if k == "count" else v)
            if c["pra_modeled"] == "no": cell.fill = GREEN
            if k == "class" and c["legend_png"] == "no": cell.font = RED_FONT
    for col, w in zip("ABCDEF", (22, 70, 12, 11, 12, 8)): cs.column_dimensions[col].width = w
    cache = {}
    for name, what, value in SHEETS:
        sel = [r for r in rows if name in r["review"].split(";")]
        sh = wb.create_sheet(name)
        sh.cell(1, 1, what).font = Font(italic=True); sh.merge_cells(start_row=1, start_column=1, end_row=1, end_column=len(COLS))
        for j, h in enumerate(COLS, 1):
            c = sh.cell(2, j, h); c.font = Font(bold=True); c.fill = REVIEW_FILL if j >= 12 else HEAD_FILL
            c.alignment = Alignment(wrap_text=True, vertical="center")
            sh.column_dimensions[c.column_letter].width = WIDTH[h]
        dv = DataValidation(type="list", formula1='"ok,change"', allow_blank=True); sh.add_data_validation(dv)
        for i, r in enumerate(sorted(sel, key=lambda r: (r["page"], int(r["box_id"]))), 3):
            if r["page"] not in cache:
                cache.clear(); cache[r["page"]] = cv2.imread(f"{IMG}/{r['page']}.png")
            img = XLImage(crop_png(cache[r["page"]], r)); sh.add_image(img, f"A{i}")
            sh.row_dimensions[i].height = 118
            vals = ["", r["page"][7:], int(r["box_id"]), r["class"], r["actuator"], r["normal_position"], r["locked"],
                    r["class_evidence"], r["position_evidence"], r["legend_entry"], r["note"], "", "", ""]
            for j, v in enumerate(vals, 1):
                c = sh.cell(i, j, v); c.alignment = Alignment(wrap_text=True, vertical="top")
                if j >= 12: c.fill = REVIEW_FILL
            dv.add(f"L{i}")
        sh.freeze_panes = "B3"
        print(name, len(sel))
    wb.save(f"{OUT}/review.xlsx"); print("wrote", f"{OUT}/review.xlsx")


if __name__ == "__main__":
    main()
