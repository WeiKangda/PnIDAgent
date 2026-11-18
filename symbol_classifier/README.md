# Symbol classifier

A classifier for the symbol boxes of a P&ID page. For each box it predicts three things:

- the class, one of the 63 classes of [`verified_annotations_v3/symbol_classes`](../verified_annotations_v3/symbol_classes/README.md)
  (the reference image of every class is there);
- the actuator of a valve, damper or pump (`NONE`, `MOTOR`, `AIR`, `EH`, `SOLENOID`, `TURBINE`, `DIESEL`, `HYDRAULIC`,
  `UNSPEC`);
- the normal position, open or closed, of the 8 classes that have one: gate, globe, butterfly, angle, 3-way and
  stop-check valves, and pneumatic and electro-hydraulic dampers.

It takes the boxes of the symbol detector (or of the annotations) and writes the classification JSON that
`digitize_pnid.py` reads, so it can stand in for the clustering step (stage 3) of the pipeline.

## Results

Every number is on boxes the model did not see in training: the 3,483 boxes, each predicted once. Two tests:

- new family: each of the 7 document families (APR1400 DCD, AFW of other plants, Surry FSAR, Davis-Besse report,
  NUREG/CR-4550, GE-Hitachi ABWR, NuScale) is held out in turn. This is what to expect on drawings in a style the
  model has never seen.
- new page: 5 folds of whole pages, so the other pages of the same family are in training. This is what to expect on
  more pages of the documents it was trained on.

| accuracy (%), ResNet-18 + legend symbols, with `--text` | new family | new page |
|---|---:|---:|
| class | 71.8 | 84.4 |
| class, macro-F1 over the 45 classes with at least 5 boxes | 33.1 | 50.1 |
| actuator (2,347 valves, dampers and pumps) | 80.2 | 93.7 |
| class and actuator both right | 64.9 | 82.5 |
| fault-tree type that follows from class + actuator (2,963 boxes with one) | 74.9 | 88.2 |
| open / closed (1,626 boxes) | 94.5 | 95.3 |

Baselines on the new-family test, class accuracy: the most frequent class (`VLV_GATE`) 37.3; frozen EfficientNetV2-S or
CLIP features with logistic regression 57.6 and 57.4; the same ResNet-18 trained on the boxes only, without the legend
symbols, 61.6. All tables (per family, per class, confidence, confusions) are in [`results/results.md`](results/results.md),
examples of the confusions in [`results/confusions.png`](results/confusions.png).

What the numbers say:

- The legend symbols matter most on a new family. 316 of the 1,270 APR1400 boxes (25%) are of classes that no other
  family has (`VLV_GLOBE` 132, `COIL` 40, dampers 72, `PUMP_VERTICAL` 13), so without the legend the model cannot learn
  them when APR1400 is held out. APR1400 class accuracy is 35.6 without the legend symbols, 61.1 with them, and 85.7 when
  the other APR1400 pages are in training.
- The words around a box do not help: with and without `--text` the overall scores differ by at most half a point.
- Open / closed is reliable. Most of its errors (77 of 90 on a new family, 42 of 76 on new pages) are on boxes whose
  normal position is still on the review sheet.
- On a new family the errors are mostly symbols drawn in a style that no training family uses: APR1400 globe valves
  (a bowtie with a dot) read as gate valves (94 boxes), APR1400 butterfly valves read as switches or dampers (64),
  stop-check valves read as check valves (28, 21 of them in the Surry FSAR), the circuit breakers of NUREG/CR-4550 read
  as check valves (26).
- Confidence: on a new family 57% of the boxes get `needs_review` (class probability below 0.7) and the others are
  94.5% right; on new pages 30% get it and the others are 97.9% right.

What would help next: for a new document family, its legend symbols and one or two of its labelled pages (on APR1400
the other pages raised class accuracy from 61.1 to 85.7). The weakest families are the ones with one page, GE-Hitachi
ABWR and NuScale (64.3 each), and NUREG/CR-4550 (71.8).

Limits of these numbers:

- Each fold was trained once (seed = fold number). Run-to-run variation was not measured, so differences of about one
  point between rows should not be read as real.
- The text features come from the verified text layer; OCR text can only be worse.
- The 479 boxes on `review.xlsx` may still change class or position.
- Tried after the first results and not kept: a separate text model combined with the image model by Bayes' rule (class
  accuracy 71.7 to 62.9 on a new family, 84.3 to 80.2 on new pages), and averaging the predictions of near-identical
  symbols on a page (0.1 to 0.8 points lower).

## Use

```bash
python predict.py --image page.png --boxes page_classification.json --text page_step3_text.json --overlay check.png
```

- `--boxes`: the boxes, either as the pipeline's classification JSON (`symbols` with `id` and `bbox` = `[x1, y1, x2, y2]`),
  a plain list of such boxes, or a YOLO label file (`class cx cy w h`, normalised; the box id is the line number).
- `--text` (optional): the text detections of the pipeline (a list of `{bbox, text}`) or a text layer of the annotations
  (`{items: [{box, text}]}`), for the words in and around each box. In the tests it changed the overall scores by at
  most half a point (see Results); without it the classifier uses the image alone.
- `--model`: default `model/symbol_classifier_v3.pt`. Model files are not kept in git (see `.gitignore`); train one with
  `train_cnn.py --final` (below) or get it from the person who trained it.
- `--overlay`: draws the boxes with their class; red boxes are the ones to check (`needs_review`).

Every symbol in the output has the keys the pipeline uses (`id`, `mask_id`, `bbox`, `size`, `area`, `cluster_id`,
`category`, `confidence`) and in addition:

| key | meaning |
|---|---|
| `actuator` | for the classes with an actuator, the most likely one; `-` for the others |
| `normal_position` | `open` or `closed` for the 8 classes with a normal position; `n/a` for the others |
| `actuator_confidence`, `position_confidence` | their probabilities |
| `ft_component_type` | the fault-tree component type that follows from class + actuator (`MOV`, `AOV`, `XVM`, `CKV`, `CRB`, ...; empty when there is none), as in `ft_component_types.csv` |
| `pra_modeled` | `no` for the classes that are usually not modeled in a PRA (instruments, orifices, steam generators, ...), as in `classes.csv` |
| `top3` | the three most likely classes with their probabilities |
| `needs_review` | `true` when the class probability is below 0.7 |

## How it works

- Crop: the box with a margin of 8% of its longer side (+3 px), padded to a square with white, 128 x 128 grey.
- Network: ResNet-18 with ImageNet weights, trained further on the crops, with three heads (class, actuator, normal
  position). Each head adds a linear term of the box's context to its image logits.
- Context: the words inside the box and within one box size around it, as hashed tokens with the digits collapsed
  (`MOV-1018` gives `MOV` and `#`; `FT345D` gives `FT#D` and `FT#`), so the operator letters (`M`, `S`, `E/H`), the
  position marks (`NC`, `LO`, `LC`) and the tag prefixes (`FT`, `PP`, `TK`, ...) count; and the box size relative to the
  median symbol of the page, and its aspect ratio. During training the whole context of 30% of the boxes is dropped, so
  the network also works without text.
- Training data: the 3,483 boxes of the 65 pages, and 264 legend symbols of 57 classes from the AP1000 and APR1400 P&ID
  legends of `verified_annotations_v3/legend_library` (class only).
- Augmentation: rotations by 90 degrees and mirror images (a symbol is drawn in any orientation), small shifts, scales and
  turns of the box, thicker or blurred lines, fainter ink and noise. At test time the softmax is averaged over the 8
  rotations and mirror images.
- Recipe (fixed beforehand, not tuned on held-out data): 30 epochs, batch 64, AdamW 5e-4 with warm-up and cosine decay,
  weight decay 1e-4. Class loss with label smoothing 0.1 and class weights (n / n_max) ^ -0.25, so rare classes count a
  little more; actuator and position losses (weight 0.5) only on the boxes that have them.
- The actuator is reported only for the classes whose boxes have one, the position only for the 8 classes with a
  position. They follow from the predicted class.

## Reproduce

```bash
python build_cache.py --out /tmp/clf                       # crops, context, labels, legend crops (10 s)
python extract_frozen.py --cache /tmp/clf                  # baseline features: CLIP, ResNet-18, EfficientNetV2-S (5 min)
python eval_frozen.py --cache /tmp/clf                     # baselines (1 min)
python train_cnn.py --cache /tmp/clf --protocol family --legend
python train_cnn.py --cache /tmp/clf --protocol page --legend
python train_cnn.py --cache /tmp/clf --protocol family
python evaluate.py --cache /tmp/clf                        # results/results.md, results.json, confusions.png
python train_cnn.py --cache /tmp/clf --final --legend --save model/symbol_classifier_v3.pt
```

Times on an Apple M1 Max (PyTorch on MPS): the cache 10 s, the frozen features 5 min, each cross-validation 15 to 25 min
(2 to 3.5 min per fold), the final model 3 min, `predict.py` a few seconds per page.

Requirements: torch, torchvision, opencv-python, numpy, scikit-learn; transformers only for the CLIP baseline. The
pretrained weights are those of torchvision and Hugging Face.

## Files

| file | what it does |
|---|---|
| `common.py` | labels, document families, crops, context features, scores |
| `model.py` | the network, augmentation, test-time averaging |
| `build_cache.py` | crops and context of all boxes and legend symbols into one cache |
| `extract_frozen.py`, `eval_frozen.py` | baselines on frozen pretrained features |
| `train_cnn.py` | cross-validation and the final model |
| `evaluate.py` | the scores of `results/` |
| `predict.py` | classify the boxes of a page |
| `results/` | `results.md` (all tables), `results.json`, `confusions.png` |
