# Evaluation tools (Dataset-P&ID + real4)

Copied from the working-tree `tools/` directory. Path assumptions: this repo
directory is named `PnIDAgent` and the dataset sits beside it:

```
<root>/PnIDAgent/            # this repo (eval_tools/ inside)
<root>/dataset/{image_2,mask,ann}   # Dataset-P&ID
<root>/results/split.json    # fixed 450/50 split (seed 42)
```

| file | purpose |
|---|---|
| dpid.py | loader, annotation repairs, derive_graph / derive_edges (connectivity GT) |
| fetch_ann.py | resumable Drive fetch of the 3500 annotation files, with cool-down |
| eval_lines.py, line_variants.py, diag_line_misses.py | line metrics, config A/B, per-stage recall |
| line_seg.py, lineseg_sweep.py | learned centreline segmenter (train / predict) and post-processing sweep |
| eval_ocr.py, ocr_post.py | stratified OCR metrics; offline scoring of tag_grammar.correct |
| eval_symbols.py, eval_real4.py | sheet-level mAP (synthetic); real4 symbol scoring |
| graph_assembly.py, eval_graph.py | adapters to pnid_graph / digitize_pnid; connectivity evaluator + ablation ladder |

See `weekly_update/2026-09-21_FINDINGS.md` for every measured number and the commands.
