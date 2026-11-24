# Experiments on this release

What the experiments of the NucP&ID paper need, run on this folder: the data, the split, the tasks, the files each
task returns and the scripts that score them.

Every experiment uses this folder at the git tag **`v3-freeze-2026-11-24`**. At that tag the symbol boxes, pipe lines,
text and connectivity are final. The symbol classes may still change through the review of
[`symbol_classes/review.xlsx`](../symbol_classes/review.xlsx) (task 8); only the classification experiments depend on
them, and they are run after that review.

## Data

| Layer | Where (in `verified_annotations_v3/`) | Format |
|---|---|---|
| Drawings (65 pages) | `topology_verified/images/<page>.png` | grayscale PNG, page pixels |
| Symbol boxes | `labels/<page>.txt` | YOLO, one class; line number = symbol id |
| Symbol classes | `symbol_classes/symbol_classes.csv`; `symbol_classes/labels/<page>.txt` | one row per box: class (63), actuator, normal position, fault-tree type, ...; YOLO with the class index |
| Pipe lines, as annotated | `lines/<page>.json` | `{"solid": [[x1,y1,x2,y2], ...], "dashed": [...]}` (dashed = flow pipes drawn dashed) |
| Pipe lines of the connectivity | [`gt_lines/<page>.json`](gt_lines/) | the same format: `lines/` with the corrections of the connectivity review (`topology_verified/manual_edits.json`) applied: lines that are not pipes removed, missing pipe pieces added. The ground truth of the line and graph scores, and the layer to train a pipe model on |
| Text | `text/<page>.json` | `{"items": [{"id", "box": [x0,y0,x1,y1], "text", "rot"}]}`, rot `0` / `cw` / `ccw` / `180`; `###` = unreadable |
| Connectivity | `topology_verified/<page>.json` | symbols, junctions, links, `neighbors` (symbol pairs), crossings |
| Legend symbols | `legend_library/` | AP1000 / APR1400 legend entries and the link of every box to its entry |
| Split | [`split.json`](split.json) | `A_train`: 40 training pages, `B_test`: 25 test pages, split by drawing |

Totals: 65 pages, 3,493 symbols, 6,389 annotated pipe line segments (6,464 in the line layer of the connectivity),
6,061 text lines, 7,237 connected symbol pairs.

## Code that exists

| What | Where |
|---|---|
| Symbol detection: base detector candidates, tile and large-equipment detectors, PP-OCRv5 text, pipe U-Net, crop encoder pretraining, relational re-scorer | branch `verified-annotations`, `verified_annotations/symbol_detection/code/` (file table in its `README.md`, "Code"); the published test predictions and their format: `verified_annotations/symbol_detection/results/test_predictions_*.json` |
| Topology assembler (graph from symbols and lines, dot test at crossings) | `topology_verified/code/topo.py` (`build`); the variant without the dot test: `topology_verified/code/pnid_graph_upstream.py` (`build_topology`) |
| Symbol classifier (class, actuator, normal position) | `symbol_classifier/` at the root of this branch |
| Scoring with the metrics of the paper | [`eval/`](eval/) |

## Tasks

The paper's tables need the results of tasks 1 to 6. Tasks 7 to 9 are facts and decisions for the text. Task 10 is a
baseline with an open-weight vision-language model.

### 1. Retrain the symbol detector on this release

Retrain every component that uses the labels of the training pages, on the 40 `A_train` pages of this folder:

- the pipe U-Net (`pipe_seg.py`), two folds by drawing, trained on the pipe lines of [`gt_lines/`](gt_lines/); its output
  is the pipe context of the re-scorer;
- the tile detector and the large-equipment detector, four folds by drawing plus one model on all training pages;
- the relational re-scorer, five folds by drawing, with the pipe context of the retrained U-Net and the text of task 3;
- the crop encoder pretraining needs no labels and stays as it is.

Return, in `experiments/results/symbol_detection/`:

| File | Content |
|---|---|
| `test_predictions_rescorer.json`, `test_predictions_base.json` | the 25 test pages, format of `test_predictions_*.json` (`{"format", "method", "selection", "pages": {"<page without pseudo_>": [[x0,y0,x1,y1,score], ...]}}`); the base detector alone at score >= 0.50 |
| `oof_predictions_rescorer.json`, `oof_predictions_base.json` | the 40 training pages, each predicted by a model that did not see its labels (same format) |
| `candidates/<page>.json` | the merged candidate pool of every page (65): `{"views": [the 7 view names], "boxes": [[x0,y0,x1,y1, s1..s7], ...]}` |
| `coverage.json` | on the test pages, the share of reference boxes that a candidate matches at IoU >= 0.5: base detector alone, and all 7 views |
| `README.md` | the commands that produced the files, the seeds, the training times |

Return in `experiments/folds/` the fold lists as JSON (`pipe_unet_2folds.json`, `detectors_4folds.json`,
`rescorer_5folds.json`: fold name -> list of pages).

Upload as assets of a GitHub release of this repository (files above 100 MB cannot go into git):

- `pipe_probability_maps.zip`: the U-Net pipe probability of every page as an 8-bit PNG of the page size (training pages
  out of fold, test pages the mean of the two fold models);
- all model weights (task 6).

### 2. Base detector fine-tuned on the training pages

The base detector fine-tuned on the 40 training pages (the baseline that spends the annotated pages on retraining the
detector instead of on a re-scorer). Return `test_predictions_finetuned.json` and, from fold models,
`oof_predictions_finetuned.json` in `experiments/results/symbol_detection/`.

### 3. PP-OCRv5 text of the 65 pages

`ocr5_pages.py` on the 65 pages. Return the raw output in `experiments/results/text_ocr/raw/<page>.json` and the
deduplicated readings in the format of `text/<page>.json` (with a `score` per item) in
`experiments/results/text_ocr/text/<page>.json`.

### 4. Ablations of the re-scorer

Each run removes one component from the final model of task 1 and returns its test predictions as
`experiments/results/symbol_detection/ablations/<name>.json`:

| name | change |
|---|---|
| `no_attention` | each candidate scored alone from the same inputs |
| `no_text` | without text tokens |
| `no_pipe` | without pipe tokens |
| `imagenet_encoder` | crop encoder initialized from ImageNet, without the pretraining |
| `one_pass` | one pass instead of two |
| `base_candidates` | candidates from the base detector only |
| `seed1` | the full model with a second random seed |

### 5. The whole pipeline on the test pages

Run text, lines, symbols and graph assembly on the 25 test pages with the models of tasks 1 and 3 and keep every
intermediate output, in `experiments/results/pipeline/<page>/`: `text.json`, `lines.json` (pipe segments, format of
`lines/<page>.json`), `symbols.json` (`[[x0,y0,x1,y1,score], ...]`), `graph.json` (the graph of `topo.build`, format of
`topology_verified/<page>.json`) and `graph_llm.json` (node ids, categories, captions; links as pairs). These outputs give
the end-to-end row of the graph table, the stage figure and the JSON example of the paper.

### 6. Release: weights, code and one entry point

- All final weights as release assets: base detector, tile and large-equipment detectors (each fold and the full model),
  crop encoder, pipe U-Nets, re-scorers (each fold and seed).
- The code without absolute paths: the paths go into one config file or command-line arguments.
- One entry point that runs the whole pipeline on a page image and writes the files of task 5.
- A `README.md` with the environment (Python, CUDA, package versions) and the command that reproduces each result of
  tasks 1 to 5.

### 7. Two facts for the paper

- The teacher detectors that pseudo-labeled the training data of the base detector, and its initial checkpoint (one or
  two sentences).
- Who did which part of the annotation work (the independent connectivity audit, the second review of the classes, the
  legend library), with what background, and which AI tools were used and how (the journal asks for this disclosure).

### 8. Review of the symbol classes

[`symbol_classes/review.xlsx`](../symbol_classes/review.xlsx): one sheet per kind of decision (fan or pump, normal
position, finer class from the legend match alone, class OTHER, uncertain), each row with a crop of the symbol on its
drawing, the proposal and empty columns for the decision. Its `README` sheet explains the columns. Return the filled
workbook as `experiments/review/review_decisions.xlsx`; the decisions are then applied to `symbol_classes/`.

### 9. Seven plain crossings

Seven crossings on the pages of the Davis-Besse reliability analysis are drawn without a dot. This drawing marks
non-connected crossings with a hop, so the connectivity treats these seven as connected (a decision from the function of
the system). Images in [`review/crossings/`](review/crossings/), one per crossing, the crossing circled:

| Page | Crossing | Here |
|---|---|---|
| p1 | (1190,1378) | connected |
| p3 | (1259,1431) | connected |
| p5 | (1272,882), (1270,1104) | connected |
| p8 | (1184,1374) | connected |
| p9 | (2257,1258), (2484,1260) | connected |

Confirm each from the system it belongs to, or say which are not connected and why, in
`experiments/review/crossings.md`.

### 10. Baseline with an open-weight vision-language model

The baselines with vision-language models are described in [`llm_baselines.md`](llm_baselines.md). The closed models run
through their APIs; this task is the same baselines with an open-weight model on a local GPU.

## Scoring

[`eval/`](eval/) scores the outputs of these tasks with the metrics of the paper: symbols (IoU >= 0.5, macro over pages),
text (detection, exact match, character error rate, read rate), pipe lines (many-to-many count F1 within 8 px and
10 degrees, against `gt_lines/`), graph (edge F1 and sheet-exact rate, and the F1 of connected symbol pairs). Its `README.md` gives the
commands.
