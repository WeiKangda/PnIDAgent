# Pipe connectivity: 5 verified pages

Each image below shows a drawing on the **left** (pipes drawn in colour, each symbol a numbered box) and the extracted pipe network on the **right** (coloured dot = symbol, small black dot = branch point, line = one pipe segment). The numbers match on both sides. The right side is the left side simplified: pipes straightened and text removed, with exactly the same connections.

## How to read

![How to read](how_to_read.png)

Start from a symbol and follow the pipe. At a branch point (black dot), follow every branch, and stop at the next symbol. The symbols you reach are the ones connected to it. Two pipes that only cross (no solid dot on the drawing) are **not** connected.

## How to check

For each page, compare the left and right sides:

1. **Nothing missing:** every pipe on the drawing appears on the right, between the same two symbols. A missing line means a lost connection.
2. **Nothing extra:** every line on the right corresponds to a pipe on the drawing. An extra line means a wrong connection.
3. **Crossings:** where two pipes cross on the drawing, a solid dot means they are connected (a branch point on the right). Without a dot they only cross, and the two lines on the right must not meet.
4. **No symbol missing:** every symbol has a dot on the right, and the numbers match.

To report an error, give the page and the symbol numbers, for example "RHR: 0 should not connect to 9".

All 5 pages were cross-checked automatically against the human-verified symbols and pipe lines, and inspected visually; no errors were found. See the [main README](../README.md#connectivity-verified-pages) for the method. The same material as slides: [topology_5pages_verified.pptx](topology_5pages_verified.pptx).

## The 5 pages

### 1. AFW · Davis-Besse reliability analysis, p7 (Third Train Configuration)

25 symbols · 32 pipe segments · 7 branch points · status: **verified**

![AFW · Davis-Besse reliability analysis, p7 (Third Train Configuration)](pseudo_AFW_%28DAVIS-BESSE%29%20AFW%20system%20Reliability%20Analysis_Final%20Report_p7_explained.png)

Files: [connectivity data (json)](pseudo_AFW_%28DAVIS-BESSE%29%20AFW%20system%20Reliability%20Analysis_Final%20Report_p7.json) · [connections drawn on the original drawing](pseudo_AFW_%28DAVIS-BESSE%29%20AFW%20system%20Reliability%20Analysis_Final%20Report_p7.jpg)

### 2. RHR · Residual heat removal system (Surry FSAR)

14 symbols · 40 pipe segments · 22 branch points · status: **verified**

![RHR · Residual heat removal system (Surry FSAR)](pseudo_RHR_%28Surry%29%20FSAR_explained.png)

Files: [connectivity data (json)](pseudo_RHR_%28Surry%29%20FSAR.json) · [connections drawn on the original drawing](pseudo_RHR_%28Surry%29%20FSAR.jpg)

### 3. AFW · Auxiliary feedwater system (Seabrook Station, NUREG/CR-5500)

47 symbols · 56 pipe segments · 8 branch points · status: **verified**

![AFW · Auxiliary feedwater system (Seabrook Station, NUREG/CR-5500)](pseudo_AFW%20%28Seabrook%29_explained.png)

Files: [connectivity data (json)](pseudo_AFW%20%28Seabrook%29.json) · [connections drawn on the original drawing](pseudo_AFW%20%28Seabrook%29.jpg)

### 4. AFW · Auxiliary feedwater system (San Onofre design review)

60 symbols · 98 pipe segments · 28 branch points · status: **verified**

![AFW · Auxiliary feedwater system (San Onofre design review)](pseudo_AFW_%28San%20Onofre%29%20AFW%20Design%20Review%20and%20Reliability%20Evaluation_explained.png)

Files: [connectivity data (json)](pseudo_AFW_%28San%20Onofre%29%20AFW%20Design%20Review%20and%20Reliability%20Evaluation.json) · [connections drawn on the original drawing](pseudo_AFW_%28San%20Onofre%29%20AFW%20Design%20Review%20and%20Reliability%20Evaluation.jpg)

### 5. CSS · Containment spray system (APR1400 design document)

72 symbols · 94 pipe segments · 15 branch points · status: **verified**

![CSS · Containment spray system (APR1400 design document)](pseudo_CSS%20%28APR1400%29_explained.png)

Files: [connectivity data (json)](pseudo_CSS%20%28APR1400%29.json) · [connections drawn on the original drawing](pseudo_CSS%20%28APR1400%29.jpg)
