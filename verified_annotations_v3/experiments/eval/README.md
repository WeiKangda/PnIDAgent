# Scoring

`score.py` scores the outputs of the experiments on the 65 verified pages of this release with the metrics of the
paper. It needs Python 3.9 or later, `numpy` and `opencv-python`.

```
python score.py symbols --pred FILE [--classes]
python score.py text    --pred DIR
python score.py lines   --pred DIR
python score.py graph   --symbols FILE|gt --lines DIR|gt [--no-dot]
python score.py pairs   --pred FILE [--match iou|center]
python score.py export-lines --dir DIR [--pages all|test|train]
```

Options of every subcommand:

- `--pages test` (default): the 25 `B_test` pages of [`../split.json`](../split.json); `train`: the 40 `A_train` pages;
  `all`: the 65 pages.
- `--v3 DIR`: the dataset folder (default: two levels up from this folder, `verified_annotations_v3/`).
- `--out FILE`: write a JSON file with the scores of every page (`per_page`), their means (`mean`) and the summed
  counts (`totals`).

Each subcommand prints a short table. Scores are means over pages (macro averages) unless stated. Coordinates are pixels
of `topology_verified/images/<page>.png`. Page names are the file stems, for example `pseudo_AFW_APR1400`; in prediction
files and folders the leading `pseudo_` may be left out. A page without a prediction is scored as an empty prediction,
with a warning.

## symbols

Input: `{"pages": {"<page>": [[x0, y0, x1, y1, score], ...]}}`, the format of `test_predictions_*.json` in
`verified_annotations/symbol_detection/results/` on branch `verified-annotations`. A box can also be
`{"bbox": [x0, y0, x1, y1], "score": s, "class": c, "actuator": a}`, and a page can be given as `{"symbols": [...]}`
(the format of `pairs`).

The predictions, in order of decreasing score, each take the unmatched reference box with the highest IoU, if that IoU
is at least 0.5 (the matching of the paper). Output: precision, recall and F1 per page and their means; TP, FP and FN
summed over the pages. The reference boxes are `labels/<page>.txt` in pixels. The symbol id is the line number, as in
`topology_verified/<page>.json` (the script warns if the two disagree).

`--classes` adds two rows. The class is the 6th element of a box list (an optional 7th is the actuator) or the keys
`class` and `actuator`.

- `class`: a match must also have the class of the reference box in `symbol_classes/symbol_classes.csv`.
- `ft_type`: a match must also have the fault-tree component type of the reference box. The type of a prediction is the
  one of its (class, actuator) in `symbol_classes/ft_component_types.csv`. If that pair is not in the table, the class
  gives the type when all its rows have the same type; otherwise the box has no type and matches nothing. Classes
  without a fault-tree type (empty in the table) count as one type.

## text

Input: a folder with `<page>.json` in the format of `text/<page>.json`:
`{"items": [{"box": [x0, y0, x1, y1], "text": "...", "rot": "0"}, ...]}`, with `rot` one of `0`, `cw`, `ccw`, `180`
(default `0`). An item can carry a `score` (default 1), which orders the matching.

- Detection: precision, recall and F1 at IoU >= 0.5, with the matching of `symbols`.
- Recognition, over the matched pairs, after removing whitespace and converting to upper case: exact-match rate;
  character error rate (CER), the Levenshtein distance divided by the reference length, both summed over the pairs of a
  page; reading-direction accuracy (the same `rot`).
- Read rate: the share of reference lines whose text (normalized the same way) equals the text of some prediction that
  covers at least half of the reference box (intersection divided by the area of the reference box). There is no
  one-to-one matching here.

The table has three rows: all lines, horizontal lines (reference `rot` 0 or 180) and vertical lines (`cw` or `ccw`). In
the horizontal and vertical rows a matched prediction counts with its reference line and an unmatched one with its own
`rot`. A page enters the mean of a row only if it has reference lines of that orientation (column `pages`), and the
recognition means only if it also has matched pairs.

Reference lines with the text `###` (handwriting that cannot be read: 6 lines on the training pages p2, p3, p5, p7 and
p9 of the Davis-Besse reliability analysis) are excluded from all text scores: they count neither as found nor as
missed, and a prediction matched to one is not counted.

## lines

Input: a folder with `<page>.json` in the format of `lines/<page>.json`: `{"solid": [[x1, y1, x2, y2], ...],
"dashed": [...]}`. Solid and dashed segments are scored together, rounded to whole pixels.

The count F1 of the paper: a reference segment is found when predicted segments within 8 px and 10 degrees cover at
least half of its length; a predicted segment is correct when reference segments cover at least half of its length in
the same way. Precision is the share of correct predicted segments, recall the share of found reference segments.
Columns `ref` and `pred`: the numbers of segments. The reference is the line layer described under "Ground-truth pipe
lines".

## graph

`--symbols gt` takes the reference symbol boxes, otherwise a file in the format of `symbols` (every box is used, ids
in file order). `--lines gt` takes the ground-truth line layer, otherwise a folder in the format of `lines`. All
segments are pipes. As in the paper, the graph is built with `build` of
[`../../topology_verified/code/topo.py`](../../topology_verified/code/topo.py) (filled-dot test at crossings, without the
manual edits of the connectivity review), or with `--no-dot` by `build_topology` of `pnid_graph_upstream.py` in the same
folder (12 px tolerance, every crossing joined).

- `edges`, the graph metric of the paper: predicted symbols are matched one to one with reference symbols in order of
  decreasing IoU (at least 0.5), predicted junctions with reference junctions within 20 px. An edge is correct when both
  ends are matched and the matched pair is a link of `topology_verified/<page>.json`. Precision, recall, F1 and
  sheet-exact: 1 for a page where every reference edge is found and no other edge is predicted, so its mean is the share
  of such pages.
- `symbol pairs`: the pairs of symbols that the built graph connects (along links and through junctions, without
  passing another symbol, as `neighbors` in `build_connectivity.py`), mapped with the same symbol matching and scored as
  in `pairs`.
- `symbols`: precision, recall and F1 of that symbol matching. They can differ a little from `symbols`, which matches in
  order of score on unrounded boxes.

On the 25 test pages `graph` takes about 25 s with `topo.build` and 9 s with `--no-dot`.

## pairs

For graphs without junctions, for example the answer of a language model. Input:
`{"pages": {"<page>": {"symbols": [{"id": 1, "bbox": [x0, y0, x1, y1]}, ...], "connections": [[1, 2], ...]}}}`; ids
can be numbers or strings. Without `symbols`, the ids in `connections` are taken as reference symbol ids (for symbols
marked on the page with their reference ids).

Predicted symbols are matched one to one with reference symbols: with `--match iou` (default) in order of decreasing
IoU, at least 0.5, as in `graph`; with `--match center` when the center of the predicted box lies inside the reference
box, nearest centers first. The predicted pairs are mapped to reference ids and compared with the reference pairs, the
`neighbors` of `topology_verified/<page>.json`. Each unordered pair counts once, and a pair with an unmatched symbol is a
false positive. Output: precision, recall and F1 of the pairs and of the symbols under the matching; TP, FP and FN
summed over the pages.

## export-lines

Writes the ground-truth line layer (next section) of the chosen pages (default: all 65) as `<page>.json` in the format of
`lines/`, with the solid or dashed type of every segment kept; added pieces and vector-region lines are solid.
[`../gt_lines/`](../gt_lines/) holds its output for the 65 pages (6,464 segments). This is the layer to train a pipe
model on and to score predicted lines against; `lines --pred ../gt_lines --pages all` gives 1.000, and
`graph --symbols gt --lines ../gt_lines` gives the same scores as `--lines gt`.

## Ground-truth pipe lines

`lines/<page>.json` holds the pipe lines as annotated (`topology_verified/lines/` is an identical copy). The connectivity
review recorded corrections in `topology_verified/manual_edits.json`, and `lines/` does not contain them: all 55
`removed_lines` are still in `lines/`, none of the 69 `added_lines` is (none is even half covered within 8 px by the
lines that remain), and on `pseudo_AC Power_APR1400` the 99 labeled lines that run into the two `vector_regions` are
still there beside the 139 `vector_lines` that replace them inside the regions.
`topology_verified/code/build_connectivity.py` applies these corrections when it builds the connectivity; built this
way, all 65 connectivity files are identical to the published ones.

The ground truth of `lines` and of `graph --lines gt` is therefore the segment list of `build_connectivity.build`:
`lines/` (solid and dashed) without the `removed_lines`, with the labeled lines inside `vector_regions` replaced by the
`vector_lines`, plus the `added_lines`, each correction applied once. The 50 dashed segments (flow pipes drawn dashed,
on the training pages `pseudo_Battery system_(Surry) FSAR`, `pseudo_CSS (APR1400)`, `pseudo_SCS_APR1400` and
`pseudo_SCS2_APR1400`) are pipes of the connectivity and part of the ground truth.

## Expected outputs

To check a setup: with the published test predictions of `verified_annotations/symbol_detection/results/` (branch
`verified-annotations`) and with the reference itself, `score.py` gives on the test pages:

| Command | Result |
|---|---|
| `symbols --pred test_predictions_new_method.json` | P 0.919, R 0.796, F1 0.849; TP 1375, FP 115, FN 393 |
| `symbols --pred test_predictions_soup5.json` | P 0.742, R 0.708, F1 0.704; TP 1183, FP 414, FN 585 |
| `graph --symbols gt --lines gt` | edge F1 0.924, sheet-exact 6 of 25; symbol-pair F1 0.907 |
| `graph --symbols gt --lines gt --pages all` | edge F1 0.941, sheet-exact 24 of 65 |
| `graph --symbols gt --lines gt --no-dot` | edge F1 0.820, sheet-exact 1 of 25 |
| `graph --symbols test_predictions_new_method.json --lines gt` | edge F1 0.687 |
| `lines --pred ../gt_lines --pages all` | 1.000 |
