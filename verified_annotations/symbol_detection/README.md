# Symbol detection: original method (soup5) vs new method

This folder compares two ways of detecting P&ID symbols on the 65 human-verified pages:

- **Original method: soup5.** A YOLO11x detector (weight average of 5 models) applied to the whole page. It is unchanged here.
- **New method: cascade relational re-scoring.** Detectors propose candidate boxes generously. A relational model then decides, box by box, whether each one matches the annotation conventions of the verified labels, and corrects its position.

Real4 (the 4 standard test drawings) is not included in this folder.

## Results

All numbers use the same metric. A prediction counts as correct when IoU ≥ 0.5 with a verified box, and each verified box can be matched once. F1 is computed per page and then averaged over pages (macro F1).

| Data | Method | Macro F1 | TP | FP | FN |
|---|---|---|---|---|---|
| **B half, 25 test pages** (never used for training) | soup5 (original) | 0.728 | 1253 | 344 | 510 |
| | soup5 fine-tuned on the A half | 0.742 | – | – | – |
| | **new method** | **0.863** | 1393 | 97 | 370 |
| A half, 40 pages (5-fold out-of-fold) | soup5 (original) | 0.646 | 1319 | 711 | 440 |
| | **new method** | **0.826** | 1382 | 223 | 377 |
| All 65 pages | soup5 (original) | 0.678 | 2572 | 1055 | 950 |
| | **new method** | **0.840** | 2775 | 320 | 747 |

- On all 65 pages, false positives fall by 70% (1055 → 320) and misses by 21% (950 → 747).
- 58 of the 65 pages improve and 6 get worse. The pages that get worse are mostly APR1400 simplified block diagrams.
- Per-page numbers are in [`results/per_page_65.csv`](results/per_page_65.csv), and totals in [`results/summary.json`](results/summary.json).
- A second training run with a different random seed gives 0.861 on the B half.

### Test set (B half)

- [`results/split.json`](results/split.json): which pages are in A (training) and which are in B (test). The split is by document, so all pages of one document are on the same side.
- [`results/test_predictions_soup5.json`](results/test_predictions_soup5.json) and [`results/test_predictions_new_method.json`](results/test_predictions_new_method.json): predicted boxes for the 25 test pages, as pixel coordinates `[x0, y0, x1, y1, score]`. Scoring these files against `../labels/` reproduces 0.7275 and 0.8629 exactly.
- [`test_set_overlays/`](test_set_overlays/): one image per test page, with soup5 on the left and the new method on the right. Green = correct, red = false positive, dashed orange = missed.

## How leakage is prevented

- **B half:** never used for training. Every model applied to B was trained on A-half labels only.
- **A half:** split by document into 5 folds. Each A page is predicted by the model trained on the other 4 folds, so no page is predicted by a model that saw its labels.
- **Detectors used as inputs:** their predictions on A pages also come from cross-fitted models (4 folds), so no model's input carries information from the labels of the page it is scoring.
- **Threshold:** the selection threshold was chosen on A only.

Caveats:
- soup5 itself was originally trained with machine-generated pseudo labels that include these 65 pages (not the human labels). This applies equally to both methods.
- The B results were looked at several times during development, so the cleanest number is the A-half out-of-fold result (0.646 → 0.826).

## How the new method works

1. **Candidates.** Boxes are proposed from 7 views and merged when nearly identical:
   - soup5 on the original page, scaled ×1.25, scaled ×0.8, flipped left-right, and flipped up-down;
   - a high-resolution tile detector, which looks at 640×640 tiles at native resolution;
   - a large-equipment detector.

   A page usually has 200–1000 candidates. Together they cover 91.7% of the B-half symbols, against 81% for soup5 alone.
2. **Relational re-scoring.** Every element on the page becomes a token:
   - candidate boxes: a 64×64 crop through a ResNet18, the detector scores and the box geometry;
   - pipe segments, taken from a learned pipe segmentation;
   - OCR text boxes.

   A 4-layer Transformer (8 heads) adds a learned geometric bias to the attention between every pair of elements. The bias is computed from relative position, size ratio, IoU, mutual coverage and gap. For each candidate the model outputs a probability (is this a correctly boxed symbol under the verified conventions?) and a box correction.
3. **Training target.** Each verified box marks only its best-matching candidate as positive; all overlapping duplicates are negative. The model therefore learns de-duplication itself. Box corrections use Smooth-L1, and an auxiliary task classifies each segment as pipe or not, supervised by the verified pipe lines.
4. **Two-pass cascade.** The second pass crops again at the corrected boxes and receives a stage embedding. Its correction is damped by half. The final score is the mean of the two passes, the box is the first-pass correction, and the whole procedure is also run on a left-right flipped page and averaged.
5. **Appearance pretraining.** Before learning the conventions, the crop encoder is pretrained on about 229,000 P&ID crops with automatic labels. These come from the datasets soup5 was trained on, and no human labels are used. This step gives the largest gain: on the B half, 0.777 → 0.839.

### Which parts use which labels

| Component | Labels |
|---|---|
| soup5 | pseudo labels (original training, not retrained here) |
| Appearance pretraining | automatic pseudo labels |
| Relational model | human-verified symbols, A half only |
| Tile detector, large-equipment detector | human-verified symbols, A half only. The tile detector is initialized from soup5. |
| Pipe segmentation | human-verified pipe lines, A half only |

## Code

Everything is in [`code/`](code/). This is research code and contains absolute paths from the original server, so adjust the paths before running it.

| Step | Files |
|---|---|
| Split and common utilities | `split.py`, `selfsim.py` (soup5 candidates, F1), `textfilter.py`, `rescore.py`, `rescore_pool.py` (candidate pool, selection) |
| Multi-view soup5 candidates | `selfsim.py cache`, `tta_cache.py` |
| Tile detector | `tiles.py`, `folds4.py`, `train_tile.py`, `tile_predict.py` |
| Large-equipment detector | `train_big.py`, `big_predict.py`, `merge_cache.py` |
| OCR text | `ocr5_pages.py` (PP-OCRv5) |
| Pipe segmentation | `pipe_seg.py`, `structure.py` |
| Appearance pretraining | `pretrain_cnn.py`, `crop_cnn.py` |
| Relational model | `rel_prep.py`, `rel_prep_pipe.py`, `rel_model.py`, `rel_model2.py` |
| Evaluation | `report_detail.py` |

The final model is trained with:

```
REL_STAGEEMB=1 REL_CVCROP=1 REL_DATA=rel_data_pipe REL_CASC=avg_first \
  python rel_model2.py fold <k> <gpu> <tag> wpre stages=2      # k = 0..4
python rel_model2.py eval <tag>
```

Model weights are not in this repository, because the tile detector alone exceeds GitHub's 100 MB file limit. On the original server, relative to the project root:

- soup5: `runs/detect/unsupervised_symbol_recognition/runs/soup5/weights/best.pt`
- tile detectors: `runs/exp65/tile_{F0..F3,A}/weights/last.pt`
- large-equipment detectors: `runs/detect/runs/exp65/big_{F0..F3,A}/weights/`
- appearance pretraining: `ft_verified/exp65/results/cnn_weakpre.pt`
- pipe segmentation: `ft_verified/exp65/results/pipe_seg/unet_{A1,A2}.pt`
- relational models: `ft_verified/exp65/results/rel_runs/prt2cv_se_fold{0..4}.pt` (seed 0) and `prt2cv_se_r1_fold{0..4}.pt` (seed 1)

## 中文摘要

在没参与训练的 25 页测试集（B 半）上，宏 F1 从 soup5 的 **0.728** 提到新方法的 **0.863**（+13.5 点）；全部 65 页从 0.678 提到 0.840，误检减少 70%。

新方法 = 检测器宁多勿漏地找候选 + 关系模型按人工标注习惯逐个审核每个框（同时看框里的图块、管线、文字和它们之间的位置关系），再在修正后的位置复查一遍。

测试集上两种方法的预测框和逐页对比图都在本文件夹，real4 不在其中。
