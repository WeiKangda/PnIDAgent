# Human-verified annotations

Symbol boxes and pipe lines on real nuclear P&ID pages, pre-labeled by the detectors
(symbols: soup5; lines: lineseg U-Net) and then manually verified/corrected in a browser
editor.

- `labels/<page>.txt` — symbols, YOLO format: `0 cx cy w h` (normalized, single class).
- `lines/<page>.json` — pipes: `{"solid": [[x1,y1,x2,y2],...], "dashed": [...]}` in image pixels.
- `overlay/<page>.jpg` — the verified result drawn on the page (red = symbol, green = solid
  pipe, orange = dashed), **for quick visual review**.

Pages are the real sheets from `datasets/yolo_pseudo_v2`. In-progress set (**30 pages** so far).

## Best symbol model measured against these pages

Model: **soup5** (best single model, weight-average of 5 checkpoints). Inference: adaptive
imgsz (longest side scaled to the model's training scale), conf 0.10 then keep score >= 0.50,
evaluated at IoU 0.5 against the verified symbol GT.

**Overall macro-F1 = 0.7043** across the 30 pages (vs 0.888 on the clean real4 set).

| Page | GT | F1 | P | R |
|---|---|---|---|---|
| AFW (Davis -Besse) | 57 | 0.957 | 0.933 | 0.982 |
| AFW (Vogtle) | 76 | 0.953 | 0.973 | 0.934 |
| Charging pump cooling system NUREG-CR-4550 Vol.3 Part1 Analysis of Core Damage F | 91 | 0.929 | 0.924 | 0.934 |
| AFW (Farley) | 73 | 0.909 | 0.864 | 0.959 |
| CCWS NUREG (Surry) | 41 | 0.902 | 0.902 | 0.902 |
| AFW (ST. LUCIE) | 55 | 0.885 | 0.806 | 0.982 |
| AFW (SALEM) | 64 | 0.881 | 0.843 | 0.922 |
| AFW (South Texas) | 60 | 0.876 | 0.869 | 0.883 |
| Boron recovery system (Surry) p3 | 30 | 0.871 | 0.844 | 0.900 |
| AFW (Turkey Point) | 174 | 0.861 | 0.885 | 0.839 |
| SCS2 (APR1400) | 104 | 0.845 | 0.802 | 0.894 |
| AFW (Haddam Neck) | 51 | 0.800 | 0.719 | 0.902 |
| SCS (APR1400) | 21 | 0.791 | 0.773 | 0.810 |
| Control room HVAC (APR1400) | 97 | 0.788 | 0.792 | 0.784 |
| AFW (Brain Wood) | 67 | 0.780 | 0.857 | 0.716 |
| Instrument Air (APR1400) | 68 | 0.752 | 0.769 | 0.735 |
| Boron recovery system (Surry) p1 | 25 | 0.750 | 0.783 | 0.720 |
| CCWS (Surry) p2 | 9 | 0.737 | 0.700 | 0.778 |
| ECWS (APR1400) | 62 | 0.733 | 0.949 | 0.597 |
| CSS NUREG (Surry) | 24 | 0.706 | 0.667 | 0.750 |
| AFW (DAVIS-BESSE) Reliability p1 | 52 | 0.677 | 0.564 | 0.846 |
| CCWS (Surry) p1 | 19 | 0.667 | 0.514 | 0.947 |
| AFW (DAVIS-BESSE) Reliability p8 | 54 | 0.647 | 0.544 | 0.796 |
| Boron recovery system (Surry) p2 | 9 | 0.632 | 0.600 | 0.667 |
| SI (APR1400) | 93 | 0.609 | 0.577 | 0.645 |
| CCWS (Surry) p5 | 11 | 0.516 | 0.400 | 0.727 |
| EPS NUREG (Surry) p2 | 12 | 0.333 | 0.333 | 0.333 |
| CCWS (Surry) p3 | 7 | 0.273 | 0.200 | 0.429 |
| Battery system (Surry) | 22 | 0.069 | 0.143 | 0.045 |
| CCWS (Surry) p4 | 6 | 0.000 | 0.000 | 0.000 |
| **macro (30)** | | **0.704** | 0.684 | 0.745 |

### Reading this

The clean real4 number (0.888) overstates generalization. On this broader, messier set
(different plants, resolutions, scan quality) soup5 is ~0.70.

The near-zero pages are **not a soup5 malfunction** — they are a category mismatch. soup5 is a
small-symbol detector (valves, instruments, ~30-80 px). The pages it scores near 0 on
(CCWS p4 = 0.00, Battery system = 0.07, EPS p2 = 0.33) consist almost entirely of **large
equipment** — tanks, heat exchangers, coolers at 150-500 px — which soup5 is not built to
detect (big-equipment detection is a known open weakness of this pipeline). On pages dominated
by normal small symbols it scores 0.80-0.96.

Caveat: this GT was seeded from soup5 prelabels and then human-corrected, so it is somewhat
anchored to what soup5 already found. The takeaway: small-symbol detection on real pages is
usable (~0.8+), large-equipment detection is the gap, and real labeled data (this set) is the
lever for closing it.
