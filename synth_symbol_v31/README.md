# Zero-annotation symbol detector

A single-class P&ID symbol detector trained without any hand-labeled data. Symbol crops
are pasted onto real drawing backgrounds so they look real, the paste locations become
free bounding boxes, and each training round starts from the previous model's weights.

On the four held-out real drawings it reaches macro-F1 0.883, versus 0.821 for a
supervised baseline.

There are three training steps. Each one synthesizes pages, then fine-tunes from the
previous checkpoint. Run everything from the project root.

## Where the data goes

The scripts read and write paths relative to the directory you run them from. That
directory must contain:

```
<project root>/
├── datasets/                         # datasets live here (created/read by the scripts)
│   ├── yolo_quality/                 # created by step 1
│   ├── yolo_quality_x2/              # a copy of yolo_quality (you make this, see step 2)
│   ├── yolo_real31/                  # created by step 3
│   ├── yolo_apr26/                   # provide these three (APR-family, synthetic, real)
│   ├── yolo_dpid_nuke/
│   └── yolo_pseudo_v3_union/
│
├── runs/detect/unsupervised_symbol_recognition/runs/
│   └── <base>/weights/best.pt        # the base model you start from (provide this)
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

The symbol bank, background drawings, base model and the three provided datasets are not
in this repo (they are several GB). Cloning the repo alone will not run end to end — you
need those assets in place first. The four evaluation drawings are excluded from all
synthesis, so nothing you train on overlaps with what you test on.

## Requirements

- Python 3.9, a CUDA GPU (training uses ~15 GB at imgsz 1280 / batch 4).
- `pip install ultralytics torch opencv-python Pillow numpy PyMuPDF`

## How to run

### Step 1 — synthesize and train

```bash
CUDA_VISIBLE_DEVICES=0 python PnIDAgent/synth_symbol_v31/step1_synthesize.py
```

Builds a symbol pool from the legend crops and real crops, detects the pipes on each
background with the base model, pastes symbols onto those pipes (white out the center,
leave a small interface, blend it in), adds tag numbers, degrades the page. Writes 1600
pages to `datasets/yolo_quality/`, then fine-tunes for 12 epochs.

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
and fine-tunes from step 2's checkpoint. This is the 0.883 model.

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

Expected: 0.883, per-drawing 0.915 / 0.931 / 0.873 / 0.813.

## Training settings (same at every step)

```
start from previous checkpoint | SGD | lr0=0.0004 | cos_lr | epochs=12
imgsz=1280 | batch=4 | single_cls=True | mosaic=0.3 close_mosaic=3
scale=0.25 degrees=2 | patience=5-6
```

Low learning rate and light augmentation are deliberate: these are gentle fine-tunes of a
model that already works, and P&IDs are clean line drawings where heavy augmentation hurts.
