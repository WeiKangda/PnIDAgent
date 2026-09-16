# Required assets (get these from the authors)

The code is in this repo; the data and the base model checkpoints are not (they're private
and several GB). Ask the authors for the asset bundle. Once every item below is in place at
the exact path shown — all paths relative to the project root you run from — the pipeline
runs end to end. A full from-scratch rebuild measured **macro-F1 ≈ 0.853** (above the 0.821
supervised baseline); the original best checkpoint is 0.883. See the reproducibility note in
the main README for why a rebuild doesn't land exactly on 0.883.

Total bundle is roughly 1 GB.

## 1. Raw data

```
PID_merged/Symbol - Legend/extracted/          # symbol bank                (~11 MB)
    *.json                                     #   legend entries
    sym600/                                    #   high-res symbol crops
unsupervised_symbol_recognition/realcrop_pool/*.png   # symbol crops from real pages (~14 MB)
All PID Diagram/*.pdf                           # background drawings         (~44 MB)
PID_merged_organized/png/*/*.png               # more backgrounds            (~63 MB)
datasets/yolo_pseudo_v2/                        # 65 real pages, source for the pseudo set (~58 MB)
```

## 2. Base model checkpoints

These three checkpoints come from earlier work that predates this repo, so they must be
shipped in the bundle — you cannot rebuild them from what's here.

```
runs/pid_combined/pseudo_v2_clean/weights/best.pt                               # teacher 1
runs/detect/unsupervised_symbol_recognition/runs/synth_det_v2/weights/best.pt   # teacher 2
runs/pid_combined/y11x_heavyaug_final/weights/best.pt                           # student base
```

Note: the pipeline's own base model, `v16_union`, is **not** shipped — it is produced by
`datasets_prep/make_yolo_pseudo_v3_union.py` (it trains the student base on the pseudo set
for 50 epochs and saves it as `v16_union`). Step 1 then starts from that. So run the pseudo
step before step 1.

## 3. Evaluation data + metric code

```
yolo_traindata/                                 # the 4 held-out eval drawings + GT (~112 MB)
gpt_detect/llm_baseline/config.py               # defines the 4 drawings and their labels
gpt_detect/llm_baseline/eval.py                 # F1 metric
```

Important: `config.py` has a hardcoded absolute path at the top:

```python
ROOT = Path("/home/suxinqi666/code/PnIDAgent_project")
```

Change it to your own project root before running anything. `DATA_ROOT = ROOT/"yolo_traindata"`
is derived from it and points at the eval drawings.

## 4. System font

The synthesis scripts draw tag numbers with DejaVu Sans:

```
/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf
```

Present by default on most Linux distros (`apt install fonts-dejavu-core` if missing).

## What gets generated (don't ship these — the scripts create them)

```
datasets/yolo_apr26/            # datasets_prep/make_yolo_apr26.py
datasets/yolo_dpid_nuke/        # datasets_prep/make_yolo_dpid_nuke.py
datasets/yolo_pseudo_v3_union/  # datasets_prep/make_yolo_pseudo_v3_union.py
datasets/yolo_quality/          # step1_synthesize.py
datasets/yolo_quality_x2/       # a copy of yolo_quality you make by hand (see main README)
datasets/yolo_real31/           # step3_synthesize.py
runs/.../{v25_quality,v28b_rebalance,v31_realism}/weights/best.pt   # trained by each step
```

## Once everything is in place

The order matters: some data-prep steps depend on checkpoints that earlier training steps
produce (e.g. `make_yolo_apr26` needs the step-1 model), so prep and training interleave.

```bash
# from the project root, in this exact order:

# 1. programmatic pages (needs only the symbol bank)
python PnIDAgent/synth_symbol_v31/datasets_prep/make_yolo_dpid_nuke.py

# 2. pseudo-label set — also produces the v16_union base that step 1 needs
python PnIDAgent/synth_symbol_v31/datasets_prep/make_yolo_pseudo_v3_union.py

# 3. step 1: quality synthesis + train (needs v16_union) -> produces the v25 model
python PnIDAgent/synth_symbol_v31/step1_synthesize.py

# 4. APR-family set (needs the v25 model from step 1)
python PnIDAgent/synth_symbol_v31/datasets_prep/make_yolo_apr26.py

# 5. the 2x oversample copy
cp -r datasets/yolo_quality datasets/yolo_quality_x2

# 6. step 2: rebalance + train
python PnIDAgent/synth_symbol_v31/step2_rebalance.py

# 7. step 3: realism synthesis + train -> the 0.883 model
python PnIDAgent/synth_symbol_v31/step3_synthesize.py

# 8. evaluate
python PnIDAgent/synth_symbol_v31/evaluate.py \
  --weights runs/detect/unsupervised_symbol_recognition/runs/v31_realism/weights/best.pt \
  --baseline-dir gpt_detect/llm_baseline
# from-scratch rebuild: ~0.853  |  original checkpoint: 0.883
```
