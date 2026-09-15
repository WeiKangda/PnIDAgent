# How this was built, start to finish

This is the whole story behind the 0.883 number, not just the v31 fine-tune step. v31
starts from a checkpoint called v28b, which itself came from a long chain of earlier
models. If you only look at `synth_and_train.py` you see the last link; this file
explains the rest.

The short version: we never labeled a single training image by hand. We mined symbols
out of the drawings, learned to paste them back onto real pages convincingly, and let
each model teach the next one. The score climbed from 0.356 (the original off-the-shelf
model) to 0.883 for the best single checkpoint, and 0.901 once a couple of inference-time
helpers are added.

## 1. Starting point

Real P&IDs, no annotations. The original author's detector scored F1 0.356 on the four
held-out drawings — basically unusable on high-res scans, mostly because it ran inference
at 640px and small symbols disappeared. Bumping inference to 1280px and retraining a plain
baseline already got to ~0.70. Everything after that is about generating training data
without a human labeler.

## 2. Mining a symbol bank

We need symbol images to paste, and we don't have a labeled legend. Two ways to get them:

- Unsupervised mining: strip the lines off a page, take connected components as symbol
  candidates, embed them with a self-supervised model, and greedily cluster. The cluster
  centers act as a de-facto legend.
- Direct crops: `sym600` is a set of high-res crops from the legend pages, and
  `realcrop_pool` is symbols cut out of 65 real pages using an earlier model as the
  teacher (keeping only confident boxes, score ≥ 0.6).

Both feed a bank of a few thousand clean symbol crops. A structural filter (`enhance()`
in the v31 script) throws out text strings, single strokes and solid blobs.

## 3. Learning to synthesize convincingly

Early synthesis just pasted crops onto backgrounds and the models overfit to paste
artifacts. Each version fixed one more source of "fakeness":

- size alignment — paste at 22–52px so symbols match the real size distribution at 1280px;
- real-crop compositing — paste real symbol crops onto real backgrounds, embed them into
  the lines, add tag numbers;
- programmatic whole-page synthesis (dpid-style) — generate entire synthetic sheets with
  orthogonal pipe networks, junctions, dashed spare lines, instrument bubbles and title
  blocks, so labels are exact by construction;
- family-targeted synthesis — separate recipes for the Surry-family drawings and the
  APR1400-family drawings, because they look different.

v31 is the version that pulled all the realism tricks together (stroke-weight matching,
paper-color fill, min-blend, pipe-stub reconnection, whole-page degradation). Those are
described in the main README.

## 4. Fighting catastrophic forgetting

A recurring problem: training hard on one drawing family wrecks performance on another.
The fix was joint training — always train on a mix of datasets at once rather than one at
a time:

- `yolo_quality` (Surry family), `yolo_apr26` (APR family), `yolo_dpid_nuke` (programmatic
  synthetic), plus a small `yolo_pseudo_v3_union` (65 real pages, ~5000 boxes) to anchor
  the real distribution.

v28b is the "rebalanced" joint model that oversamples the Surry data 2x. It's the base v31
fine-tunes from. Every model in the chain starts from the previous one's weights at a low
learning rate, not from scratch — that's what keeps the accumulated skill.

## 5. The checkpoints that matter

By the end, five checkpoints trained with different recipes were each strong on different
drawings:

- v25 (quality / Surry-focused)
- v28b (rebalanced joint)
- v31 (realistic compositing) — best single checkpoint at 0.883
- v34 (notation-heavy)
- v36 (APR-final)

## 6. Model soup

Averaging those five checkpoints' weights (0.16·v25 + 0.20·v28b + 0.26·v31 + 0.20·v34 +
0.18·v36) gives one network, `soup5`, at macro-F1 0.888. It's still a single model, single
forward pass, fixed threshold — the averaging just cancels each checkpoint's overfitting
and is a bit steadier, mostly helping the hardest drawing (APR1400: 0.813 → 0.824). A greedy
search over mixing ratios couldn't beat 0.888, so that's the single-model ceiling here.

## 7. System on top (optional, not single-model)

Two inference-time additions push past 0.888:

- VLM arbitration — for low-confidence boxes on big equipment, ask a vision-language model
  to confirm; recovers large symbols the detector is unsure about → 0.892.
- Text-symbol path — an OCR route that recovers notation symbols by intersecting part-name
  text with bounding boxes on pipes → 0.901.

These aren't part of the single-model number; they're a system.

## The numbers in one place

```
original author model              0.356
baseline retrained at 1280px      ~0.70
iterative pseudo-label             0.82  (deploy) / 0.84 (val+TTA)
v31, best single checkpoint        0.883
soup5, weight-average of 5         0.888   <- single-model ceiling
+ VLM arbitration                  0.892   (system)
+ OCR text-symbol path             0.901   (system)
supervised baseline, for reference 0.821
```

## What's in this folder vs. what isn't

This folder contains the v31 slice — the synthesis+train script, a train-only variant, a
standalone evaluator, and this write-up. The earlier checkpoints (v25…v28b), the mined
symbol bank, the background drawings, and the joint datasets live outside the repo and are
several GB. You can read the whole method here, but reproducing the full chain from zero
needs those assets. Reproducing just the v31 step needs v28b plus the four joint datasets;
that's the part we re-ran and it landed back on 0.883 exactly.
