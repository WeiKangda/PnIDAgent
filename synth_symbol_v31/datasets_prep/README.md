# Dataset preparation

The three datasets that steps 2 and 3 mix in (`yolo_apr26`, `yolo_dpid_nuke`,
`yolo_pseudo_v3_union`) are not external — they're produced by these scripts from the
same symbol bank and backgrounds. Run them before step 2. Run from the project root.

Output goes under `datasets/` (each script creates its own folder).

## make_yolo_apr26.py  →  datasets/yolo_apr26/  (~1834 pages)

APR1400-family targeted synthesis. Same pasting approach as the main steps, but it
upweights the (clean, vector) APR1400 legend symbols and uses style-matched degradation
(light for vector backgrounds, heavy for scans) to fix the APR1400 blind spots.

```bash
CUDA_VISIBLE_DEVICES=0 python PnIDAgent/synth_symbol_v31/datasets_prep/make_yolo_apr26.py
```

## make_yolo_dpid_nuke.py  →  datasets/yolo_dpid_nuke/  (~1375 pages)

Programmatic whole-page synthesis (CPU only). Instead of pasting onto a real background,
it draws entire synthetic sheets from a P&ID layout grammar — orthogonal pipe networks,
jump arcs, junction dots, dashed spare lines, embedded symbols, tag numbers, instrument
bubbles, title blocks, gray-background noise. Symbols come from the same bank. Because the
whole page is drawn, the labels are exact by construction. (Layout follows Dataset-P&ID,
Paliwal et al., arXiv:2109.03794.)

```bash
python PnIDAgent/synth_symbol_v31/datasets_prep/make_yolo_dpid_nuke.py
```

## make_yolo_pseudo_v3_union.py  →  datasets/yolo_pseudo_v3_union/  (65 real pages)

The only dataset with real pages. It pseudo-labels 65 real drawings using two teachers and
takes the union of their boxes (a general-symbol detector at conf 0.5 / 1280, plus a
legend-synthesis detector at high resolution / conf 0.45). This anchors training to the
real symbol distribution. Needs the two teacher checkpoints referenced inside the script.

```bash
CUDA_VISIBLE_DEVICES=0 python PnIDAgent/synth_symbol_v31/datasets_prep/make_yolo_pseudo_v3_union.py
```

These are research scripts with hardcoded paths to the symbol bank, backgrounds and
teacher models — check the paths at the top of each before running.
