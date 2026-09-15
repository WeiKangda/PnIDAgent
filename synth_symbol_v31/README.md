# v31 — zero-annotation symbol detector

A single-class P&ID symbol detector trained without any hand-labeled data. The idea
is to paste symbol crops onto real drawing backgrounds in a way that actually looks
real, take the paste locations as free bounding boxes, and fine-tune from the previous
checkpoint at a low learning rate.

On the four held-out real drawings it scores macro-F1 0.883, which beats the supervised
baseline (0.821) by about 6 points:

```
                       NorthANA  AFW_Surry  CVCS_Surry  APR1400   macro-F1
supervised baseline      0.899     0.917       0.678      0.789     0.821
v31 (best single ckpt)   0.915     0.931       0.873      0.813     0.883
v31 re-run (this code)   0.915     0.931       0.873      0.813     0.883
soup5 (weight-avg of 5)  0.922     0.932       0.873      0.824     0.888
```

The re-run row is a full retrain from scratch with these scripts — it lands on the
exact same per-drawing numbers, so 0.883 isn't luck.

## Why the synthesis is the way it is

Naively pasting a symbol PNG onto a drawing leaves it looking obviously fake — wrong
line weight, a white patch behind it, a hard edge — and the model just learns those
artifacts instead of the symbol. v31 fixes that with a few tricks (all in
`synth_and_train.py`):

- Match stroke weight: binarize the symbol and dilate it to the background line width
  so it's as thick as the surrounding lines (`render_symbol`).
- Orient along the pipe: rotate the symbol 90° on vertical runs so its long axis
  follows the line.
- Fill with paper color: before pasting, fill the box with the page's own paper color
  (90th-percentile pixel) instead of white.
- Blend, don't overwrite: paste with `np.minimum(background, symbol)` so it darkens
  onto the drawing rather than covering it.
- Reconnect the pipe: draw short stubs at both ends at the background line width so the
  symbol sits on the line instead of floating.
- Degrade together: after everything is pasted, blur and add noise to the whole page at
  once, so the symbol goes through the same scan degradation as the drawing.

A couple of things that matter:

- Paste locations come from the previous model (v28b) running on the real drawing — it
  finds the pipes, so symbols end up on pipes. This is the bootstrap step.
- Where you paste is the label, so annotation is free.
- The four evaluation drawings are excluded everywhere via the `BAN` list — they never
  touch training data.
- The symbol bank is filtered by `enhance()` to drop text, fragments and solid blobs.

## Requirements

- Python 3.9, a CUDA GPU. Training at imgsz 1280 / batch 4 uses ~15 GB.
- Packages: `ultralytics`, `torch`, `opencv-python`, `Pillow`, `numpy`, `PyMuPDF`.
- For evaluation: `config.py` and `eval.py` from `gpt_detect/llm_baseline` in the main
  repo (they define the four eval drawings and the F1 metric).

This is the important caveat: the scripts read data assets that are **not** in this repo
and are several GB. You need all of them to reproduce 0.883:

- Symbol bank: `sym600/` (legend crops) and `realcrop_pool/`.
- Backgrounds: `All PID Diagram/*.pdf` and `PID_merged_organized/png/` (with the four
  eval drawings excluded).
- Base model: the previous checkpoint `v28b_rebalance/weights/best.pt`.
- Joint datasets to keep old skills: `yolo_quality`, `yolo_apr26`, `yolo_dpid_nuke`,
  `yolo_pseudo_v3_union`.

Cloning this branch alone will not run end to end without those.

## Running it

Run everything from the project root (`PnIDAgent_project`).

Full pipeline — synthesize 1600 pages, fine-tune 12 epochs from v28b, then evaluate:

```bash
CUDA_VISIBLE_DEVICES=0 python PnIDAgent/synth_symbol_v31/synth_and_train.py
```

One gotcha: the v28b model used during synthesis holds GPU memory, and if it isn't freed
before training starts you can OOM. The safer path is to synthesize once, then train in a
separate process on a clean GPU:

```bash
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python PnIDAgent/synth_symbol_v31/train_from_synth.py
```

Training settings (deliberately gentle, since we're fine-tuning a converged model and
don't want it to drift):

```
start from v28b | SGD | lr0=0.0004 | cos_lr | epochs=12 | imgsz=1280 | batch=4
single_cls=True | mosaic=0.3 close_mosaic=3 | scale=0.25 degrees=2 | patience=5
```

The light augmentation is on purpose — P&IDs are clean line drawings and heavy augmentation
hurts.

Evaluation uses a fixed protocol; you have to follow it or the numbers aren't comparable:
predict at `imgsz=1280, conf=0.10, iou=0.5, max_det=400`, keep boxes with `score >= 0.65`,
then average per-drawing F1@IoU0.5 across the four drawings.

```bash
python PnIDAgent/synth_symbol_v31/eval_standard.py \
  --weights runs/detect/unsupervised_symbol_recognition/runs/v31_realism/weights/best.pt \
  --baseline-dir gpt_detect/llm_baseline
```

Expect 0.883.

## Files

- `synth_and_train.py` — synthesis (1600 pages) plus fine-tuning from v28b, with a
  self-eval at the end.
- `train_from_synth.py` — training only, reusing already-synthesized data on a clean GPU.
- `eval_standard.py` — standalone evaluation using the standard protocol.
