# PnIDAgent - P&ID Digitization Pipeline

A comprehensive pipeline for digitizing Piping and Instrumentation Diagrams (P&IDs). This tool processes P&ID images through symbol detection, text recognition, and line detection to produce structured JSON outputs suitable for downstream LLM processing.

## Verified annotations v3 (this branch, symbol classes in review)

[`verified_annotations_v3/`](verified_annotations_v3/README.md) holds 65 real nuclear P&ID pages annotated by hand and
verified in five layers: symbol boxes, symbol classes, pipe lines (solid, and the flow pipes drawn dashed or dotted), text
lines and pipe connectivity, with the AP1000 / APR1400 P&ID legends they follow. It is the training and evaluation data
of the pipeline's symbol classifier.

| | |
|---|---:|
| Pages | 65 |
| Symbol boxes | 3483 |
| Symbol classes | 63 (56 occur on the pages) |
| Pipe line segments | 6381 (50 dashed) |
| Text lines | 6064 |
| Connections | 7223 |

The symbol classes ([`verified_annotations_v3/symbol_classes/`](verified_annotations_v3/symbol_classes/README.md)):

- **One class per component that has its own legend symbol**: valves by body (gate, globe, butterfly, check, stop-check,
  safety / relief, pilot-operated relief, vacuum breaker, ...), dampers by operator (motor, pneumatic, electro-hydraulic,
  none), pumps by type (centrifugal, turbine-driven, vertical), filters, strainers, carbon adsorbers, moisture and air
  separators, spray (nozzles, headers, spargers), and the other equipment, instruments and electrical components.
- **Actuator** of valves, dampers and pumps, so the fault-tree type (MOV, AOV, SOV, HOV, CKV, XVM, SRV, PORV, VBV, MDP,
  TDP, ...) follows from class + actuator.
- **Normal position** of valves and dampers (open 1269, closed 357) and the locked marks (LO 155, LC 20).
- **PRA flag**: components extracted but typically not modeled in a PRA (instruments, orifices, steam generators,
  off-page connectors, penetrations, ...; 417 boxes) are marked, to be screened out when the fault trees are built.
- **Review**: 479 boxes that need a person's decision (fan or pump, normally closed valves and dampers, `OTHER`, ...) are
  listed with crops of the drawings in `review.xlsx`.

## Symbol classifier (this branch)

[`symbol_classifier/`](symbol_classifier/README.md) gives each detected symbol box one of the 63 v3 classes, the actuator
of valves, dampers and pumps, and the normal position (open / closed) of valves and dampers, and writes the
classification JSON that `digitize_pnid.py` reads. It is a ResNet-18 trained on the crops of the 3,483 boxes and on the
legend symbols. On pages of a document family it was not trained on it gets 71.8% of the classes right, on new pages of
the families it was trained on 84.4%; open / closed is right for about 95% of the valves and dampers.

```bash
python symbol_classifier/predict.py --image page.png --boxes page_classification.json --overlay check.png
```

## Pipeline Overview

The pipeline consists of 8 stages:

1. **Symbol Detection** - Detect symbols using SAM2 or YOLO
2. **Symbol Mask Editing** - Interactive cleanup of detected masks
3. **Symbol Classification** - Cluster and label similar symbols (or classify them with the trained
   [symbol classifier](symbol_classifier/README.md))
4. **Text Detection** - Extract text using PaddleOCR
5. **Text Editing** - Interactive text correction
6. **Line Detection** - Detect solid and dashed lines
7. **Line Editing** - Interactive line cleanup
8. **Final Digitization** - Generate graph structure (nodes & links)

## Environment Setup

### Prerequisites

- Python 3.9 (recommended)
- CUDA-capable GPU (recommended for faster processing)
- tkinter (for interactive editors)

### Step 1: Create Virtual Environment

```bash
# Create and activate virtual environment
python -m venv pnid_env
source pnid_env/bin/activate  # Linux/macOS
# or
pnid_env\Scripts\activate  # Windows
```

### Step 2: Install Dependencies

```bash
cd PnIDAgent
pip install -r requirements.txt
```

All dependencies including PyTorch, transformers (for SAM2), PaddleOCR, etc. are included in requirements.txt.

### Step 3: Install tkinter (if not available)

```bash
# Ubuntu/Debian
sudo apt-get install python3-tk

# macOS - usually included with Python
```

### Optional: GPU Support for PaddleOCR

If you want GPU acceleration for text detection, uncomment and install:

```bash
pip install paddlepaddle-gpu==2.6.2
```

## Model Files

The pipeline supports two detection models:

| Model | File | Description |
|-------|------|-------------|
| YOLO | `best.pt` | Pre-trained YOLO model for symbol detection |
| SAM2 | `best_model.pth` | Fine-tuned SAM2 model for mask generation |

Both detection model files should be placed in the `PnIDAgent` directory. The symbol classifier reads
`symbol_classifier/model/symbol_classifier_v3.pt`; `python symbol_classifier/train_cnn.py --final` trains it (see its
README).

## Usage

### Basic Usage (YOLO Detection - Recommended)

```bash
python process_single_pnid.py \
    --image /path/to/your/pnid.jpg \
    --out /path/to/output/ \
    --yolo_model best.pt
```

### With Interactive Editing

```bash
python process_single_pnid.py \
    --image /path/to/your/pnid.jpg \
    --out /path/to/output/ \
    --yolo_model best.pt \
    --interactive
```

### Using SAM2 Detection

```bash
python process_single_pnid.py \
    --image /path/to/your/pnid.jpg \
    --out /path/to/output/ \
    --detector sam2 \
    --sam2_model best_model.pth \
    --interactive
```

### Skip Visualization

```bash
python process_single_pnid.py \
    --image /path/to/your/pnid.jpg \
    --out /path/to/output/ \
    --yolo_model best.pt \
    --no-visualization
```

## Command Line Arguments

### Required Arguments

| Argument | Description |
|----------|-------------|
| `--image` | Path to input P&ID image |
| `--out` | Output directory for results |

### Detection Model Selection

| Argument | Default | Description |
|----------|---------|-------------|
| `--detector` | `yolo` | Detection model: `sam2` or `yolo` |
| `--yolo_model` | - | Path to YOLO model (required if detector=yolo) |
| `--sam2_model` | - | Path to SAM2 model (required if detector=sam2) |
| `--confidence` | `0.5` | Confidence threshold for detection |

### Symbol Classification

| Argument | Default | Description |
|----------|---------|-------------|
| `--embedding_model` | `clip` | Model for embeddings: `clip`, `dinov2`, `vit` |
| `--clustering_method` | `hdbscan` | Clustering method: `hdbscan`, `kmeans` |
| `--sensitivity` | `high` | Clustering sensitivity: `low`, `medium`, `high`, `very_high` |

### Text Detection

| Argument | Default | Description |
|----------|---------|-------------|
| `--target-width` | `7168` | Resize image width for text detection |
| `--lang` | `en` | PaddleOCR language |
| `--gpu` | `False` | Use GPU for PaddleOCR |

### Line Detection

| Argument | Default | Description |
|----------|---------|-------------|
| `--min-len` | `22` | Minimum line segment length |
| `--line-mode` | `solid` | Line mode: `solid` or `solid_dashed` |
| `--suppress-text` | `True` | Remove text regions before line detection |

### Interactive Editing

| Argument | Description |
|----------|-------------|
| `--interactive` | Enable all interactive editors |
| `--skip-symbol-edit` | Skip symbol mask editing |
| `--skip-text-edit` | Skip text editing |
| `--skip-line-edit` | Skip line editing |

### Visualization

| Argument | Default | Description |
|----------|---------|-------------|
| `--no-visualization` | `False` | Skip visualization generation |
| `--no-labels` | `False` | Hide category labels on symbols |
| `--vis-line-width` | `2` | Width of connection lines |
| `--vis-bbox-width` | `3` | Width of bounding boxes |

### General

| Argument | Default | Description |
|----------|---------|-------------|
| `--device` | `cuda` | Device: `cuda` or `cpu` |

## Output Files

After processing, the output directory will contain:

| File | Description |
|------|-------------|
| `{name}_sam2_results.json` | Symbol detection results |
| `{name}_masks.npz` | Symbol masks in NumPy format |
| `{name}_classification.json` | Symbol classifications |
| `{name}_step3_text.json` | Detected text boxes |
| `{name}_step4_lines.json` | Detected lines |
| `{name}_digitized.json` | Full digitized output |
| `{name}_digitized_llm.json` | LLM-friendly JSON output |
| `{name}_digitized_visualization.png` | Visualization image |

## Example Workflow

```bash
# 1. Process a P&ID with interactive editing
python process_single_pnid.py \
    --image example_pnid.png \
    --out results/example/ \
    --yolo_model best.pt \
    --interactive

# 2. Review and edit symbols in the interactive mask editor
# 3. Classify and label symbols in the interactive classifier
# 4. Review and correct text in the text editor
# 5. Review and adjust lines in the line editor
# 6. Final JSON outputs are generated automatically
```

## Troubleshooting

### UMAP Import Error

```bash
pip uninstall umap umap-learn
pip install umap-learn
```

### Matplotlib Backend Error

```bash
# Set backend before running
export MPLBACKEND=TkAgg
```

### tkinter Not Found

```bash
# Ubuntu/Debian
sudo apt-get install python3-tk

# Fedora
sudo dnf install python3-tkinter
```

### CUDA Out of Memory

Try reducing the image size or use CPU:

```bash
python process_single_pnid.py \
    --image pnid.jpg \
    --out results/ \
    --yolo_model best.pt \
    --device cpu
```

## License

This project is for research purposes.
