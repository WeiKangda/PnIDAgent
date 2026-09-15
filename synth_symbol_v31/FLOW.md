# The full chain, step by step

This is the runnable version of the pipeline: the exact order of scripts that takes you
from a weak base model to the 0.883 single checkpoint, with what each step produces and
the score it lands on. Run everything from the project root (`PnIDAgent_project`).

The chain is `v16 -> v25 -> v28b -> v31`. Each model starts from the previous one's
weights and fine-tunes at a low learning rate — nothing is trained from scratch, which is
how the accumulated skill carries forward. All three synthesis steps exclude the four
evaluation drawings via the `BAN` list, so there is no leakage.

```
v16_union  (base, ~0.855)
   │  v25_quality_synth.py     synth 1600 "quality" pages (Surry family), fine-tune
   ▼
v25_quality  (~0.866, Surry strong)                 -> dataset: yolo_quality
   │  v28b_rebalance.py        re-mix datasets, Surry oversampled 2x, fine-tune
   ▼
v28b_rebalance  (Surry recovered)
   │  synth_and_train.py       synth 1600 "realism" pages, fine-tune
   ▼
v31_realism  (0.883, best single checkpoint)        -> dataset: yolo_real31
```

## Step 0 — prerequisites

You need the base model `v16_union/weights/best.pt`, the mined symbol bank (`sym600/`,
`realcrop_pool/`), the background drawings (`All PID Diagram/*.pdf`,
`PID_merged_organized/png/`), and — for the v28b step — the sibling datasets `yolo_apr26`,
`yolo_dpid_nuke`, `yolo_pseudo_v3_union`. See the main README for where these come from.
None of them are in the repo; they are several GB.

## Step 1 — v25 (quality synthesis, from v16)

```bash
CUDA_VISIBLE_DEVICES=0 python PnIDAgent/synth_symbol_v31/v25_quality_synth.py
```

What it does: builds a symbol pool from the legend crops plus real crops, detects the real
pipes on each background with the v16 teacher, embeds symbols onto those pipes (white out
the center, leave a 4px interface, min-blend), scatters tag numbers, and degrades the page.
1600 pages land in `datasets/yolo_quality/`. Then it fine-tunes v16 for 12 epochs.

Produces: `runs/.../v25_quality/weights/best.pt` and the `yolo_quality` dataset. Around 0.866,
with the Surry family strong.

## Step 2 — v28b (rebalance, from v25)

```bash
CUDA_VISIBLE_DEVICES=0 python PnIDAgent/synth_symbol_v31/v28b_rebalance.py
```

What it does: no new synthesis. It writes a joint `data.yaml` that lists `yolo_quality`
twice (via a copy `yolo_quality_x2`, a plain 2x oversample of the Surry data) alongside
`yolo_apr26`, `yolo_dpid_nuke` and `yolo_pseudo_v3_union`, then fine-tunes from v25 for 12
epochs. The whole point is to fix CVCS_Surry recall, which earlier joint training had let
slip — oversampling the Surry data brings it back without losing APR.

Note: the script reads `datasets/yolo_quality_x2`. It's just a copy of `yolo_quality`; make
one before running:

```bash
cp -r datasets/yolo_quality datasets/yolo_quality_x2
```

Produces: `runs/.../v28b_rebalance/weights/best.pt`. This is what v31 fine-tunes from.

## Step 3 — v31 (realism synthesis, from v28b)

```bash
CUDA_VISIBLE_DEVICES=0 python PnIDAgent/synth_symbol_v31/synth_and_train.py
```

Same idea as v25 but with the full realism treatment: stroke-weight matching to the
background line width, paper-color fill instead of white, min-blend, pipe-stub
reconnection at both ends, orientation alignment, and whole-page degradation after pasting.
1600 pages in `datasets/yolo_real31/`, joined with a few sibling datasets, fine-tuned from
v28b for 12 epochs.

If you already synthesized the data and only want to (re)train, use the OOM-safe variant on
a clean GPU:

```bash
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python PnIDAgent/synth_symbol_v31/train_from_synth.py
```

Produces: `runs/.../v31_realism/weights/best.pt`. This is the 0.883 checkpoint.

## Step 4 — evaluate

Same fixed protocol for every step (predict at conf 0.10, keep score ≥ 0.65, macro-F1 over
the four drawings):

```bash
python PnIDAgent/synth_symbol_v31/eval_standard.py \
  --weights runs/detect/unsupervised_symbol_recognition/runs/v31_realism/weights/best.pt \
  --baseline-dir gpt_detect/llm_baseline
```

Expected per step: v25 ~0.866, v31 0.883.

## Beyond v31

v31 is the best single checkpoint. To reach the reported 0.888 you average five
checkpoints (v25, v28b, v31, v34, v36) into one network — see PIPELINE.md, section 6. The
0.892 and 0.901 numbers add a VLM arbiter and an OCR text-symbol path at inference time and
are a system, not a single model.

## The training recipe (identical at every step)

```
start from previous ckpt | SGD | lr0=0.0004 | cos_lr | epochs=12 | imgsz=1280 | batch=4
single_cls=True | mosaic=0.3 close_mosaic=3 | scale=0.25 degrees=2 | patience=5-6
```
