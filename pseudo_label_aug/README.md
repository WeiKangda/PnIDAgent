# Pseudo-Label Augmentation (yolo-aug)

Research scripts used to improve YOLO symbol detection on real P&IDs via
pseudo-label augmentation. **These are research scripts with hardcoded local
paths (`datasets/`, `runs/`) — adapt paths before running.**

| Script | Purpose |
|---|---|
| `make_copypaste.py` | Copy-Paste augmentation: crop real symbols, paste onto real backgrounds (keeps black lines, transparent white bg) to expand training data. |
| `train_pseudo_v2_clean.py` | Iterative pseudo-label training — re-label real images with a stronger model, then re-train. Produced the best single model (real4 F1 ~0.82 deploy / 0.84 val+TTA). |
| `make_prelabels.py` | Weighted-Box-Fusion ensemble to generate pre-labels for human correction. |

## Result (real4: 4 real P&IDs, 274 GT, held-out)
- Original author model: F1 0.356
- Reproduced baseline: F1 0.70
- This work (iterative pseudo-label): **F1 0.84 (val+TTA) / 0.82 (deploy)**

Inference must use imgsz=1280 (see `finetune_yolo_symbols.py`); the default 640
collapses on small symbols in high-res scans.
