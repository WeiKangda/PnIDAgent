# Zero-annotation symbol detector

A single-class P&ID symbol detector trained with **no human-labeled bounding boxes**. It is
a pseudo-label + synthesis self-training pipeline: teacher models auto-label real pages,
symbol crops are pasted onto real backgrounds (paste locations give exact labels), and each
model is trained from the previous one — a bootstrap where every generation both labels data
for and initializes the next.

The best single checkpoint reaches macro-F1 0.883 on the four held-out real drawings, versus
0.821 for a supervised baseline.

The four held-out real drawings and their human-annotated labels (274 symbols, YOLO format)
are published in [`../nuclear_pid_real4_dataset/`](../nuclear_pid_real4_dataset/) — the test
set the numbers on this page are measured against.

**On reproducibility (measured).** Because the pipeline generates its own training data with
its own models, a full rebuild from scratch does not land exactly on 0.883. A clean-room run
of the whole chain measured **macro-F1 ≈ 0.853** (per-drawing 0.921 / 0.931 / 0.776 / 0.784)
— still well above the 0.821 supervised baseline. The early stages reproduce closely (the
step-1 model came back at 0.866, matching the original); the gap is nondeterminism compounding
over five sequential fine-tunes and landing on CVCS_Surry, the smallest/most sensitive drawing
(37 symbols). The exact 0.883 depends on the original intermediate checkpoints. So: expect
~0.85 from a from-scratch rebuild, 0.883 only if you start from the original checkpoints.

There are three training steps, interleaved with three data-prep steps (some prep needs a
checkpoint an earlier training step produces). Run everything from the project root.

**To reproduce, you need the asset bundle from the authors** (data + base model checkpoints,
~1 GB — not in this repo). See **[ASSETS.md](ASSETS.md)** for the exact list, where each file
goes, and the full command sequence in the correct order.

## Where the data goes

The scripts read and write paths relative to the directory you run them from. That
directory must contain:

```
<project root>/
├── datasets/                         # datasets live here (created/read by the scripts)
│   ├── yolo_quality/                 # created by step 1
│   ├── yolo_quality_x2/              # a copy of yolo_quality (you make this, see step 2)
│   ├── yolo_real31/                  # created by step 3
│   ├── yolo_apr26/                   # built by datasets_prep/ (APR-family synthesis)
│   ├── yolo_dpid_nuke/               # built by datasets_prep/ (programmatic pages)
│   └── yolo_pseudo_v3_union/         # built by datasets_prep/ (real-page pseudo-labels)
│
├── runs/                             # three teacher/base checkpoints go here (see ASSETS.md)
│   ├── pid_combined/pseudo_v2_clean/weights/best.pt            # teacher 1
│   ├── detect/.../synth_det_v2/weights/best.pt                 # teacher 2
│   └── pid_combined/y11x_heavyaug_final/weights/best.pt        # student base
│                                      # (v16_union, the step-1 base, is GENERATED, not provided)
│
├── PID_merged/Symbol - Legend/extracted/
│   ├── *.json                        # legend entries
│   └── sym600/                       # high-res symbol crops (the symbol bank)
│
├── unsupervised_symbol_recognition/
│   └── realcrop_pool/*.png           # symbol crops taken from real pages
│
├── All PID Diagram/*.pdf             # background drawings (300 dpi rendered)
├── PID_merged_organized/png/*/*.png  # more background drawings
│
└── gpt_detect/llm_baseline/
    ├── config.py                     # defines the 4 evaluation drawings + their labels
    └── eval.py                       # F1 metric
```

The symbol bank, background drawings and base model are not in this repo (they are several
GB). The three `yolo_apr26 / yolo_dpid_nuke / yolo_pseudo_v3_union` datasets aren't external
either — you build them from the symbol bank and backgrounds with the scripts in
`datasets_prep/` (see `datasets_prep/README.md`). Cloning the repo alone will not run end to
end — you need those assets in place first. The four evaluation drawings are excluded from all
synthesis, so nothing you train on overlaps with what you test on.

## Requirements

- Python 3.9, a CUDA GPU (training uses ~15 GB at imgsz 1280 / batch 4).
- `pip install ultralytics torch opencv-python Pillow numpy PyMuPDF`

## How to run

The order matters and is interleaved — some data-prep steps need a checkpoint that an
earlier training step produces (`make_yolo_apr26` needs the step-1 model). Full command
sequence is in [ASSETS.md](ASSETS.md); the summary:

```
dpid_nuke → pseudo_v3_union → step 1 → apr26 → (copy x2) → step 2 → step 3 → evaluate
```

### Prep A — programmatic pages + pseudo-labels

```bash
python PnIDAgent/synth_symbol_v31/datasets_prep/make_yolo_dpid_nuke.py
python PnIDAgent/synth_symbol_v31/datasets_prep/make_yolo_pseudo_v3_union.py
```

`make_yolo_dpid_nuke` draws whole synthetic pages (labels exact by construction).
`make_yolo_pseudo_v3_union` pseudo-labels 65 real pages with two teachers **and trains the
`v16_union` base that step 1 starts from** (so it must run before step 1). See
`datasets_prep/README.md`.

### Step 1 — synthesize and train

```bash
CUDA_VISIBLE_DEVICES=0 python PnIDAgent/synth_symbol_v31/step1_synthesize.py
```

Builds a symbol pool from the legend crops and real crops, detects the pipes on each
background with the base model, pseudo-labels any symbols already on the background, pastes
new symbols onto the pipes (white out the center, leave a small interface, blend it in),
adds tag numbers, degrades the page. Each page's labels = pseudo-labels (existing symbols) +
exact labels (pasted symbols). Writes 1600 pages to `datasets/yolo_quality/`, then fine-tunes
for 12 epochs. Produces the v25 model.

### Prep B — APR-family set (needs the step-1 model)

```bash
CUDA_VISIBLE_DEVICES=0 python PnIDAgent/synth_symbol_v31/datasets_prep/make_yolo_apr26.py
```

### Step 2 — rebalance and train

First make the 2x oversample copy the script expects:

```bash
cp -r datasets/yolo_quality datasets/yolo_quality_x2
```

Then:

```bash
CUDA_VISIBLE_DEVICES=0 python PnIDAgent/synth_symbol_v31/step2_rebalance.py
```

No new synthesis here. It trains on all the datasets together (with `yolo_quality`
counted twice, to give that family more weight) and fine-tunes from step 1's checkpoint
for 12 epochs.

### Step 3 — synthesize and train (final)

```bash
CUDA_VISIBLE_DEVICES=0 python PnIDAgent/synth_symbol_v31/step3_synthesize.py
```

Same idea as step 1 but with more careful compositing: symbols are matched to the
background line width, the box is filled with the page's paper color instead of white,
symbols are blended (not pasted over), short stubs reconnect them to the pipe, and the
whole page is degraded together after pasting. Writes 1600 pages to `datasets/yolo_real31/`
and fine-tunes from step 2's checkpoint. This is the final model (0.883 for the original
checkpoint; ~0.85 for a from-scratch rebuild — see the reproducibility note at the top).

If the data is already synthesized and you only want to retrain (this avoids a GPU
out-of-memory issue when synthesis and training share a card):

```bash
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python PnIDAgent/synth_symbol_v31/step3_train_only.py
```

### Evaluate

The protocol is fixed — follow it or the numbers aren't comparable: predict at
`imgsz=1280, conf=0.10, iou=0.5, max_det=400`, keep boxes with `score >= 0.65`, then
average per-drawing F1 at IoU 0.5 across the four drawings.

```bash
python PnIDAgent/synth_symbol_v31/evaluate.py \
  --weights runs/detect/unsupervised_symbol_recognition/runs/v31_realism/weights/best.pt \
  --baseline-dir gpt_detect/llm_baseline
```

Expected: 0.883 (per-drawing 0.915 / 0.931 / 0.873 / 0.813) from the original checkpoint;
~0.853 (0.921 / 0.931 / 0.776 / 0.784) from a full from-scratch rebuild.

## Training settings (same at every step)

```
start from previous checkpoint | SGD | lr0=0.0004 | cos_lr | epochs=12
imgsz=1280 | batch=4 | single_cls=True | mosaic=0.3 close_mosaic=3
scale=0.25 degrees=2 | patience=5-6
```

Low learning rate and light augmentation are deliberate: these are gentle fine-tunes of a
model that already works, and P&IDs are clean line drawings where heavy augmentation hurts.
