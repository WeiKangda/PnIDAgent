# Human-verified annotations

Symbol boxes and pipe lines on real nuclear P&ID pages, pre-labeled by the detectors
(symbols: soup5; lines: lineseg U-Net) and then manually verified/corrected in a browser
editor.

- `labels/<page>.txt`: symbols, YOLO format `0 cx cy w h` (normalized, single class).
- `lines/<page>.json`: pipes, `{"solid": [[x1,y1,x2,y2],...], "dashed": [...]}` in image pixels.
- `overlay/<page>.jpg`: the verified result drawn on the page (red = symbol, green = solid
  pipe, orange = dashed), **for quick visual review**.

Pages are the real sheets from `datasets/yolo_pseudo_v2`. **65 pages** (all 65 sheets of that set, 3522 symbols).

## Labeling rules

- One box per symbol. A valve together with its actuator (M, solenoid) and LO/LC mark is one box.
  Tag circles and instrument circles get their own box.
- Large equipment is boxed: pumps, heat exchangers, tanks, steam generators, turbines, condensers,
  heaters, off-page connectors.
- Not boxed: plain text, dashed signal boxes, segment-boundary marks, flow arrows, legend, frame/title block.
- Lines follow pipe centerlines. Dashed/dash-dot lines (signal lines, area and seismic boundaries) are not labeled.
- Lines stop at the edge of a symbol box instead of running through it (a box that holds several other
  symbols, e.g. a tank with spargers inside, is not cut).

The 35 pages marked † below were labeled with the last two rules in place. The first 30 pages were
verified before those rules, so some of their lines still run through symbols (and one page has a dashed line).

## Best symbol model measured against these pages

Model: **soup5** (best single model, weight-average of 5 checkpoints). Inference: adaptive
imgsz (longest side scaled to the model's training scale), conf 0.10 then keep score >= 0.50,
evaluated at IoU 0.5 against the verified symbol GT. Script: `eval_soup5_f1.py` in this folder
(it reproduces the earlier 30-page number exactly, 0.7043).

**Overall macro-F1 = 0.678** across the 65 pages (first 30 pages: 0.704, new 35 pages: 0.655;
for reference 0.888 on the clean real4 set).

| Page | GT | F1 | P | R |
|---|---|---|---|---|
| AFW (Davis -Besse) | 57 | 0.957 | 0.933 | 0.982 |
| AFW (Vogtle) | 76 | 0.953 | 0.973 | 0.934 |
| SI NUREG (Surry) † | 44 | 0.935 | 0.896 | 0.977 |
| Charging pump cooling system NUREG-CR-4550 Vol.3 Part1 Analysis of Core Damage F | 91 | 0.929 | 0.924 | 0.934 |
| LPI-LPR NUREG (Surry) † | 42 | 0.920 | 0.889 | 0.952 |
| AFW (APR1400) † | 75 | 0.909 | 0.956 | 0.867 |
| AFW (Farley) | 73 | 0.909 | 0.864 | 0.959 |
| CCWS NUREG (Surry) | 41 | 0.902 | 0.902 | 0.902 |
| AFW (Seabrook) † | 47 | 0.889 | 0.846 | 0.936 |
| AFW (ST. LUCIE) | 55 | 0.885 | 0.806 | 0.982 |
| AFW (SALEM) | 64 | 0.881 | 0.843 | 0.922 |
| AFW (South Texas) | 60 | 0.876 | 0.869 | 0.883 |
| Boron recovery system (Surry) p3 | 30 | 0.871 | 0.844 | 0.900 |
| SWS (Surry) † | 65 | 0.871 | 0.915 | 0.831 |
| AFW (Turkey Point) | 174 | 0.861 | 0.885 | 0.839 |
| Condensate and Feedwater (APR1400) † | 42 | 0.860 | 0.841 | 0.881 |
| SCS2 (APR1400) | 104 | 0.845 | 0.802 | 0.894 |
| AFW (Cristal River) p2 † | 58 | 0.845 | 0.845 | 0.845 |
| AFW CE Cross Training R325C ch.11.5 † | 76 | 0.843 | 0.922 | 0.776 |
| HVAC (APR1400) p1 † | 19 | 0.811 | 0.833 | 0.789 |
| AFW (Haddam Neck) | 51 | 0.800 | 0.719 | 0.902 |
| ESW (APR1400) † | 48 | 0.791 | 0.895 | 0.708 |
| SCS (APR1400) | 21 | 0.791 | 0.773 | 0.810 |
| Control room HVAC (APR1400) | 97 | 0.788 | 0.792 | 0.784 |
| AFW (San Onofre) Design Review † | 60 | 0.784 | 0.754 | 0.817 |
| AFW (Brain Wood) | 67 | 0.780 | 0.857 | 0.716 |
| AFW (Cristal River) p1 † | 36 | 0.775 | 0.705 | 0.861 |
| Feedwater system (Surry) † | 90 | 0.765 | 0.861 | 0.689 |
| Instrument Air (APR1400) | 68 | 0.752 | 0.769 | 0.735 |
| Boron recovery system (Surry) p1 | 25 | 0.750 | 0.783 | 0.720 |
| Steam and Power Conversion (GE-Hitachi ABWR) † | 140 | 0.749 | 0.722 | 0.779 |
| CCWS (Surry) p2 | 9 | 0.737 | 0.700 | 0.778 |
| ECWS (APR1400) | 62 | 0.733 | 0.949 | 0.597 |
| AFW INL System Study † | 49 | 0.729 | 0.861 | 0.633 |
| RCS-POSRV (APR1400) † | 135 | 0.716 | 0.754 | 0.681 |
| CSS NUREG (Surry) | 24 | 0.706 | 0.667 | 0.750 |
| AFW (DAVIS-BESSE) Reliability p1 | 52 | 0.677 | 0.564 | 0.846 |
| CCWS (APR1400) † | 85 | 0.667 | 0.716 | 0.624 |
| CCWS (Surry) p1 | 19 | 0.667 | 0.514 | 0.947 |
| AFW (DAVIS-BESSE) Reliability p3 † | 54 | 0.652 | 0.551 | 0.796 |
| HVAC (APR1400) p2 † | 23 | 0.651 | 0.700 | 0.609 |
| AFW (DAVIS-BESSE) Reliability p8 | 54 | 0.647 | 0.544 | 0.796 |
| AFW (DAVIS-BESSE) Reliability p9 † | 37 | 0.633 | 0.508 | 0.838 |
| Boron recovery system (Surry) p2 | 9 | 0.632 | 0.600 | 0.667 |
| AFW (DAVIS-BESSE) Reliability p5 † | 36 | 0.624 | 0.509 | 0.806 |
| SI (APR1400) | 93 | 0.609 | 0.577 | 0.645 |
| Main Steam (APR1400) † | 66 | 0.595 | 0.655 | 0.545 |
| Boron recovery system (Surry) p4 † | 17 | 0.579 | 0.524 | 0.647 |
| EPS NUREG (Surry) p1 † | 71 | 0.559 | 0.775 | 0.437 |
| HVAC (APR1400) p3 † | 115 | 0.531 | 0.662 | 0.443 |
| CCWS (Surry) p5 | 11 | 0.516 | 0.400 | 0.727 |
| Power conversion system (NuScale) † | 45 | 0.515 | 0.385 | 0.778 |
| AFW (DAVIS-BESSE) Reliability p6 † | 27 | 0.513 | 0.392 | 0.741 |
| RHR (Surry) † | 14 | 0.511 | 0.364 | 0.857 |
| AFW (DAVIS-BESSE) Reliability p7 † | 25 | 0.494 | 0.365 | 0.760 |
| Compressed Air system (Surry) † | 37 | 0.468 | 0.450 | 0.486 |
| AFW (DAVIS-BESSE) Reliability p2 † | 27 | 0.463 | 0.345 | 0.704 |
| CSS (APR1400) † | 72 | 0.398 | 0.360 | 0.444 |
| AFW (DAVIS-BESSE) Reliability p4 † | 26 | 0.386 | 0.281 | 0.615 |
| EPS NUREG (Surry) p2 | 12 | 0.333 | 0.333 | 0.333 |
| AC Power (APR1400) † | 177 | 0.274 | 0.714 | 0.169 |
| CCWS (Surry) p3 | 7 | 0.273 | 0.200 | 0.429 |
| Main Steam system (Surry) † | 8 | 0.214 | 0.150 | 0.375 |
| Battery system (Surry) | 22 | 0.069 | 0.143 | 0.045 |
| CCWS (Surry) p4 | 6 | 0.000 | 0.000 | 0.000 |
| **macro (65)** | | **0.678** | 0.668 | 0.724 |

† = added in this update (35 pages).

### Reading this

The clean real4 number (0.888) overstates generalization. On this broader, messier set
(different plants, resolutions, scan quality) soup5 is about 0.68.

The low pages fall into a few groups, and none of them is a soup5 malfunction:

- **Large equipment pages** (CCWS (Surry) p4 0.00, Battery system 0.07, EPS NUREG p2 0.33). soup5 is a
  small-symbol detector (valves, instruments, ~30-80 px); these pages are mostly tanks, heat
  exchangers and coolers at 150-500 px.
- **Electrical one-line diagrams** (AC Power (APR1400) 0.27). Most symbols are breakers, transformers
  and connectors, which soup5 never saw in training. Precision is fine (0.71), recall is 0.17.
- **Old scanned, hand-drawn sheets** (AFW (DAVIS-BESSE) Reliability p2-p9, 0.39-0.65). Recall is
  OK (0.6-0.8), precision is low: handwritten tags and scan noise produce false boxes.
- **Different box convention on one page** (CSS (APR1400) 0.40). On this page each valve and its tag
  circle are labeled as one box, while soup5 puts them in separate boxes, so many matches fall below IoU 0.5.
- **Very few symbols** (Main Steam system (Surry), 8 GT): a couple of misses swing F1 a lot.

On pages made of normal small symbols it scores 0.80-0.96.

Caveat: this GT was seeded from soup5 prelabels and then human-corrected, so it is somewhat
anchored to what soup5 already found. The takeaway: small-symbol detection on real pages is
usable (~0.8+), large equipment and electrical symbols are the gap, and real labeled data
(this set) is the way to close it.
