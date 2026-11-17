"""Apply the review fixes of fixes.json to the second-release annotations.

Every run starts from the files of the base commit (BASE below), so the script can be run again after fixes.json
changes.  For each page listed in fixes.json it rewrites, consistently:

  labels/<page>.txt, topology_verified/symbols/<page>.{txt,json}      symbol boxes (removed boxes dropped, the
                                                                       remaining ones renumbered, added boxes appended)
  lines/<page>.json, topology_verified/lines/<page>.json              pipe lines (removed, moved, added; dashed flow
                                                                       pipes go to "dashed")
  symbol_classes/symbol_classes.{csv,json}, symbol_classes/labels/<page>.txt     class and actuator per box
  legend_library/derived/symbol_legend_links.json                     box ids, class and actuator
  topology_verified/manual_edits.json, topology_verified/shaft_links.json        symbol ids, added lines
  topology_verified/<page>.json, topology_verified/rebuilt/<page>.json           connectivity, rebuilt
  stats.json                                                          per-page counts

  python apply_fixes.py               # all pages of fixes.json
  python apply_fixes.py SCS_APR1400   # pages whose name contains the argument (the others keep the base files)
"""
import collections, csv, io, json, os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
V = os.path.dirname(HERE)                       # verified_annotations_v2
REPO = os.path.dirname(V)
T = f"{V}/topology_verified"
BASE = "1d42d1d"
sys.path.insert(0, f"{T}/code")
import build_connectivity as bc  # noqa: E402

CSV_FIELDS = ["page", "box_id", "x0", "y0", "x1", "y1", "class", "actuator", "ft_component_type", "uncertain", "note"]


def base(path):
    """Content of a file of the verified_annotations_v2 folder at the base commit."""
    rel = os.path.relpath(path, REPO)
    return subprocess.run(["git", "-C", REPO, "show", f"{BASE}:{rel}"], capture_output=True, check=True).stdout.decode("utf-8")


def base_json(path):
    return json.loads(base(path))


def dump(obj, path, indent=None):
    """Written as the base files are: no trailing newline, the indent of the file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False)


def same(a, b, tol=2.0):
    return max(abs(x - y) for x, y in zip(a, b)) <= tol or max(abs(x - y) for x, y in zip(a, b[2:] + b[:2])) <= tol


def ft_type(table, cls, act):
    return table.get(f"{cls}|{act}", table.get(f"{cls}|*", ""))


def main():
    fixes = json.load(open(f"{HERE}/fixes.json", encoding="utf-8"))
    fixes = {k: v for k, v in fixes.items() if not k.startswith("_")}
    args = sys.argv[1:]
    pages = [p for p in fixes if not args or any(a in p for a in args)]
    classes = [l.strip() for l in open(f"{V}/symbol_classes/classes.txt") if l.strip()]
    ft = {r["class|actuator"]: r["ft_component_type"] for r in csv.DictReader(open(f"{V}/symbol_classes/ft_component_types.csv"))}

    rows_all = list(csv.DictReader(io.StringIO(base(f"{V}/symbol_classes/symbol_classes.csv"))))
    cls_json = base_json(f"{V}/symbol_classes/symbol_classes.json")
    links = base_json(f"{V}/legend_library/derived/symbol_legend_links.json")
    edits = base_json(f"{T}/manual_edits.json")
    shafts = base_json(f"{T}/shaft_links.json")
    stats = base_json(f"{V}/stats.json")
    # pages not selected this run keep their base rows; selected pages are rebuilt below
    rows_out = {p: [r for r in rows_all if r["page"] == p] for p in dict.fromkeys(r["page"] for r in rows_all)}

    for page in pages:
        fx = fixes[page]
        sym = base_json(f"{T}/symbols/{page}.json"); W, H = sym["width"], sym["height"]
        yolo = [l for l in base(f"{V}/labels/{page}.txt").splitlines() if l.strip()]
        rows = rows_out[page]
        assert len(rows) == len(yolo), page
        # --- symbols: remove, renumber, relabel, add
        remove = {}
        for r in fx.get("remove_boxes", []):
            b = [float(rows[r["id"]][k]) for k in ("x0", "y0", "x1", "y1")]
            assert same(b, r["bbox"], 3), (page, r["id"], b, r["bbox"])
            remove[r["id"]] = r
        keep = [i for i in range(len(yolo)) if i not in remove]
        idmap = {old: new for new, old in enumerate(keep)}
        new_yolo = [yolo[i] for i in keep]
        new_rows = []
        for i in keep:
            r = dict(rows[i]); r["box_id"] = str(idmap[i]); new_rows.append(r)
        for r in fx.get("relabel", []):
            b = [float(rows[r["id"]][k]) for k in ("x0", "y0", "x1", "y1")]
            assert same(b, r["bbox"], 3), (page, r["id"], b, r["bbox"])
            row = new_rows[idmap[r["id"]]]
            if "box" in r:                                    # a corrected box for the same symbol
                x0, y0, x1, y1 = r["box"]
                new_yolo[idmap[r["id"]]] = f"0 {(x0 + x1) / 2 / W:.6f} {(y0 + y1) / 2 / H:.6f} {(x1 - x0) / W:.6f} {(y1 - y0) / H:.6f}"
                for k, v in zip(("x0", "y0", "x1", "y1"), r["box"]): row[k] = f"{float(v):.1f}"
            row["class"] = r.get("class", row["class"]); row["actuator"] = r.get("actuator", row["actuator"])
            row["ft_component_type"] = ft_type(ft, row["class"], row["actuator"])
            if "note" in r: row["note"] = r["note"]
            if "uncertain" in r: row["uncertain"] = r["uncertain"]
        for a in fx.get("add_boxes", []):
            x0, y0, x1, y1 = a["bbox"]
            new_yolo.append(f"0 {(x0 + x1) / 2 / W:.6f} {(y0 + y1) / 2 / H:.6f} {(x1 - x0) / W:.6f} {(y1 - y0) / H:.6f}")
            new_rows.append({"page": page, "box_id": str(len(new_rows)), "x0": f"{x0:.1f}", "y0": f"{y0:.1f}", "x1": f"{x1:.1f}",
                             "y1": f"{y1:.1f}", "class": a["class"], "actuator": a.get("actuator", "-"),
                             "ft_component_type": ft_type(ft, a["class"], a.get("actuator", "-")), "uncertain": "", "note": a.get("note", "")})
        rows_out[page] = new_rows
        txt = "\n".join(new_yolo) + "\n"
        for p in (f"{V}/labels/{page}.txt", f"{T}/symbols/{page}.txt"):
            open(p, "w").write(txt)
        boxes = []
        for l in new_yolo:
            _, cx, cy, w, h = map(float, l.split())
            boxes.append([int((cx - w / 2) * W), int((cy - h / 2) * H), int((cx + w / 2) * W), int((cy + h / 2) * H)])
        dump({"name": page, "width": W, "height": H, "symbols": [{"id": i, "bbox": b} for i, b in enumerate(boxes)]}, f"{T}/symbols/{page}.json")
        with open(f"{V}/symbol_classes/labels/{page}.txt", "w") as f:
            for l, r in zip(new_yolo, new_rows):
                f.write(f"{classes.index(r['class'])} " + " ".join(l.split()[1:]) + "\n")
        cls_json[page] = [{k: (int(r[k]) if k == "box_id" else float(r[k]) if k in ("x0", "y0", "x1", "y1") else r[k])
                           for k in CSV_FIELDS if k != "page"} for r in new_rows]
        # --- legend links
        old = {e["box_id"]: e for e in links.get(page, [])}
        by_kind = {}
        for pg, es in links.items():
            for e in es:
                by_kind.setdefault((e["class"], e["actuator"]), []).append((pg == page, e))
        new_links = []
        for r in new_rows:
            bid = int(r["box_id"]); src = keep[bid] if bid < len(keep) else None
            e = dict(old[src]) if src is not None and src in old else {"legend_symbol": None, "legend_operator": None}
            if e.get("class") != r["class"] or e.get("actuator") != r["actuator"]:
                # the legend entry of the symbol follows the class, the entry of the operator follows the actuator
                same_page_first = lambda lst: sorted(lst, key=lambda t: not t[0])
                if e.get("class") != r["class"] or e.get("legend_symbol") is None:
                    by_class = same_page_first([t for (c, _), lst in by_kind.items() if c == r["class"] for t in lst if t[1]["legend_symbol"] is not None])
                    exact = same_page_first([t for t in by_kind.get((r["class"], r["actuator"]), []) if t[1]["legend_symbol"] is not None])
                    e["legend_symbol"] = (exact or by_class)[0][1]["legend_symbol"] if (exact or by_class) else None
                if r["actuator"] in ("NONE", "-", "UNSPEC"):
                    e["legend_operator"] = None
                elif e.get("actuator") != r["actuator"]:
                    ops = same_page_first([t for (_, a), lst in by_kind.items() if a == r["actuator"] for t in lst if t[1]["legend_operator"] is not None])
                    e["legend_operator"] = ops[0][1]["legend_operator"] if ops else None
            new_links.append({"box_id": bid, "class": r["class"], "actuator": r["actuator"],
                              "legend_symbol": e["legend_symbol"], "legend_operator": e["legend_operator"]})
        links[page] = new_links
        # --- lines
        L = base_json(f"{V}/lines/{page}.json")
        solid, dashed = list(L["solid"]), list(L.get("dashed", []))
        for r in fx.get("remove_lines", []):
            hit = [s for s in solid if same(s, r["seg"])]
            assert len(hit) == 1, (page, "remove_lines", r["seg"], len(hit))
            solid.remove(hit[0])
        for r in fx.get("move_lines", []):
            hit = [i for i, s in enumerate(solid) if same(s, r["seg"])]
            assert len(hit) == 1, (page, "move_lines", r["seg"], len(hit))
            solid[hit[0]] = [round(float(v), 1) for v in r["new"]]
        solid += [[round(float(v), 1) for v in r["seg"]] for r in fx.get("add_solid", [])]
        dashed += [[round(float(v), 1) for v in r["seg"]] for r in fx.get("add_dashed", [])]
        # --- manual edits and shaft links: renumber symbol ids, drop edits of removed boxes
        e = edits[page]
        # --- line annotations = the verified pipe set: lines that are not pipes leave the annotations (and the
        #     removed_lines of the connectivity review), drawn pipe pieces of the review join them (and leave added_lines)
        sync = fx.get("line_sync", {})
        for r in sync.get("drop_removed_lines", []):
            hit = [s for s in solid if same(s, r["seg"])]
            assert hit, (page, "drop_removed_lines", r["seg"])
            for h in hit: solid.remove(h)
            hm = [d for d in e.get("removed_lines", []) if same(d, r["seg"])]
            assert len(hm) == 1, (page, "removed_lines", r["seg"], len(hm))
            e["removed_lines"].remove(hm[0])
        for r in sync.get("adopt_added_lines", []):
            hm = [d for d in e.get("added_lines", []) if same(d, r["seg"])]
            assert len(hm) == 1, (page, "added_lines", r["seg"], len(hm))
            e["added_lines"].remove(hm[0])
            (dashed if r.get("dashed") else solid).append([round(float(v), 1) for v in hm[0]])
        for p in (f"{V}/lines/{page}.json", f"{T}/lines/{page}.json"):
            dump({"solid": solid, "dashed": dashed}, p)
        if e.get("unlinks"): e["unlinks"] = [[idmap[s], x, y] for s, x, y in e["unlinks"] if s in idmap]
        if e.get("pass_through"): e["pass_through"] = [idmap[s] for s in e["pass_through"] if s in idmap]
        if e.get("no_join"): e["no_join"] = [idmap[s] for s in e["no_join"] if s in idmap]
        for k, v in fx.get("manual_edits", {}).items():
            if k == "remove_added_lines":
                for seg in v:
                    hit = [a for a in e.get("added_lines", []) if same(a, seg)]
                    assert len(hit) == 1, (page, "remove_added_lines", seg)
                    e["added_lines"].remove(hit[0])
            elif k.startswith("add_"):
                e.setdefault(k[4:], []).extend(v)
            else:
                e[k] = v
        if page in shafts:
            shafts[page] = [[idmap[a], idmap[b]] for a, b in shafts[page] if a in idmap and b in idmap]
        # --- connectivity
        pub = base_json(f"{T}/{page}.json")
        r = bc.build(page, e)
        out = {"name": page, "width": W, "height": H, "status": pub["status"],
               "status_note": pub["status_note"] + " " + fx.get("status_note", ""),
               "symbols": r["symbols"], "junctions": r["junctions"], "links": r["links"], "neighbors": r["neighbors"],
               "crossings": r["crossings"], "notes": fx.get("notes", pub["notes"])}
        out["status_note"] = out["status_note"].strip()
        dump(out, f"{T}/{page}.json")
        dump(r, f"{T}/rebuilt/{page}.json")
        # --- stats
        st = stats[page]
        st["symbols_v2"] = len(new_yolo)
        st["symbol_area_v2"] = round(sum(float(l.split()[3]) * W * float(l.split()[4]) * H for l in new_yolo))
        st["lines_v2"] = len(solid) + len(dashed)     # lines_into_symbols_v2 is kept: added lines stop at the box edges
        print(f"{page}: {len(yolo)} -> {len(new_yolo)} symbols, {len(L['solid']) + len(L.get('dashed', []))} -> {len(solid) + len(dashed)} lines "
              f"({len(dashed)} dashed), {len(r['links'])} links")

    with open(f"{V}/symbol_classes/symbol_classes.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, lineterminator="\r\n"); w.writeheader()
        for p in rows_out:
            for r in rows_out[p]: w.writerow(r)
    dump(cls_json, f"{V}/symbol_classes/symbol_classes.json", indent=0)
    # class counts in classes.csv (4th column)
    counts = collections.Counter(r["class"] for p in rows_out for r in rows_out[p])
    raw = base(f"{V}/symbol_classes/classes.csv"); nl = "\r\n" if "\r\n" in raw else "\n"
    out = []
    for i, line in enumerate(raw.split(nl)):
        if i == 0 or not line: out.append(line); continue
        cells = next(csv.reader([line])); cells[-1] = str(counts.get(cells[0], 0))
        sio = io.StringIO(); csv.writer(sio, lineterminator="").writerow(cells); out.append(sio.getvalue())
    open(f"{V}/symbol_classes/classes.csv", "w", encoding="utf-8", newline="").write(nl.join(out))
    dump(links, f"{V}/legend_library/derived/symbol_legend_links.json", indent=1)
    dump(edits, f"{T}/manual_edits.json", indent=1)
    dump(shafts, f"{T}/shaft_links.json", indent=1)
    dump(stats, f"{V}/stats.json", indent=1)


if __name__ == "__main__":
    main()
