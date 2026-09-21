#!/usr/bin/env python3
"""
YOLO Fine-tuning for P&ID Symbol Detection (tiled, 32-way)

Trains a YOLO model on Dataset-P&ID symbol boxes and runs tiled inference on
full-resolution sheets.  Detections carry a class id and are also exported as
center points for SAM2 prompts.

Why tiles: sheets are 7168x4561 and symbols are 33-155 px (median 69).  Fed
whole at imgsz 640 a symbol shrinks to ~5 px; at imgsz 1280 tiles cut from the
native-resolution sheet it keeps its true size.  Boxes are clipped at tile
borders (kept if >= --min-vis of the box survives) and inference merges the
overlapping tiles with NMS.

Why 32 classes: symbols.npy holds a class column (1..32).  The previous
converter never read it and emitted every box as class 0, so the model could
not classify.  --single-cls restores that behaviour for comparison.

Dataset layout (dataset root):
    image_2/<idx>.jpg
    ann/<idx>/<idx>_symbols.npy      rows [symbol_id, [x1,y1,x2,y2], class_id]
The old flat layout <root>/<idx>/<idx>_symbols.npy is still accepted.

Usage:
    # Convert (writes split.json once; later runs reuse it -- never tune on val)
    python finetune_yolo_symbols.py --mode convert --data_root ../dataset --output_dir ./yolo_tiles

    # Train on 8 GPUs at 1280
    python finetune_yolo_symbols.py --mode train --data_root ../dataset --output_dir ./yolo_tiles \
        --model_size yolo11m --imgsz 1280 --batch_size 64 --device 0,1,2,3,4,5,6,7 --epochs 60

    # Tiled inference on one sheet
    python finetune_yolo_symbols.py --mode inference --model runs/train/pid_symbols/weights/best.pt --image ../dataset/image_2/3.jpg

    # Predictions for a list of sheets -> JSON (for tools/eval_symbols.py / eval_graph.py)
    python finetune_yolo_symbols.py --mode predict_sheets --model best.pt --data_root ../dataset \
        --ids ./yolo_tiles/split.json:val --pred_out preds_val.json
"""

import os
import sys
import json
import argparse
import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import yaml
from tqdm import tqdm

try:
    from ultralytics import YOLO
except ImportError:
    print("Please install ultralytics: pip install ultralytics")
    sys.exit(1)

N_CLASSES = 32
CLASS_NAMES = {i: f"c{i + 1:02d}" for i in range(N_CLASSES)}   # dataset ids 1..32 -> 0..31


@dataclass
class SymbolDetection:
    """A detected symbol: center point (SAM2 prompt), confidence, box and class."""
    x_center: float
    y_center: float
    confidence: float
    bbox: Optional[Tuple[float, float, float, float]] = None   # (x1, y1, x2, y2)
    cls: int = 0                                                # 0..31
    cls_name: str = "symbol"

    def to_sam2_prompt(self) -> Tuple[float, float]:
        return (self.x_center, self.y_center)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["center"] = [self.x_center, self.y_center]
        d["bbox"] = list(self.bbox) if self.bbox else None
        return d


# ============================================================================
# Dataset conversion
# ============================================================================

def tile_grid(H: int, W: int, tile: int, overlap: int) -> List[Tuple[int, int]]:
    """Top-left corners of tiles covering HxW; the last row/col is border-aligned."""
    stride = max(1, tile - overlap)

    def axis(n):
        if n <= tile:
            return [0]
        pos = list(range(0, n - tile, stride))
        if pos[-1] != n - tile:
            pos.append(n - tile)
        return pos
    return [(x, y) for y in axis(H) for x in axis(W)]


def load_symbols_from_mask(mask_path: Path, min_area: int = 20) -> List[Tuple[int, int, int, int, int]]:
    """[(cls0, x1, y1, x2, y2)] from mask/<idx>_mask.png (0 = bg, 1..32 = class).

    Fallback when symbols.npy is missing.  Checked on 15 sheets against
    symbols.npy: 1638/1640 boxes matched at IoU >= 0.5, mean IoU 0.972, class
    agreement 100%.  Mask boxes are pixel-tight, so slightly smaller.
    """
    m = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    out = []
    for c in np.unique(m):
        if c == 0:
            continue
        n, _, st, _ = cv2.connectedComponentsWithStats((m == c).astype(np.uint8), connectivity=8)
        for i in range(1, n):
            x, y, w, h, a = (int(v) for v in st[i])
            if a >= min_area:
                out.append((int(c) - 1, x, y, x + w, y + h))
    return out


def load_symbols(symbols_path: Path) -> List[Tuple[int, int, int, int, int]]:
    """[(cls0, x1, y1, x2, y2)] from symbols.npy (or a mask png); class column may be str or int."""
    if str(symbols_path).endswith(".png"):
        return load_symbols_from_mask(symbols_path)
    out = []
    for row in np.load(symbols_path, allow_pickle=True):
        _sid, box, cls = row[0], row[1], row[2]
        x1, y1, x2, y2 = (int(v) for v in box)
        c = int(cls) - 1
        if not 0 <= c < N_CLASSES:
            raise ValueError(f"class id {cls} out of range in {symbols_path}")
        out.append((c, min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)))
    return out


def _convert_one(job) -> Tuple[int, int, int]:
    (img_path, sym_path, images_dir, labels_dir, tile, overlap, min_vis,
     neg_keep, single_cls, seed, quality) = job
    img = cv2.imread(str(img_path))
    if img is None:
        return 0, 0, 0
    H, W = img.shape[:2]
    boxes = load_symbols(sym_path)
    rng = random.Random(seed * 100003 + int(Path(img_path).stem))
    n_tiles = n_boxes = n_neg = 0
    stem = Path(img_path).stem
    for (x0, y0) in tile_grid(H, W, tile, overlap):
        th, tw = min(tile, H - y0), min(tile, W - x0)
        labels = []
        for c, x1, y1, x2, y2 in boxes:
            x2, y2 = min(x2, W), min(y2, H)          # mask png has one extra row
            ix1, iy1 = max(x1, x0), max(y1, y0)
            ix2, iy2 = min(x2, x0 + tw), min(y2, y0 + th)
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            area = (x2 - x1) * (y2 - y1)
            if area <= 0 or ((ix2 - ix1) * (iy2 - iy1)) / area < min_vis:
                continue
            cx = ((ix1 + ix2) / 2 - x0) / tw
            cy = ((iy1 + iy2) / 2 - y0) / th
            bw, bh = (ix2 - ix1) / tw, (iy2 - iy1) / th
            labels.append(f"{0 if single_cls else c} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        if not labels:
            if rng.random() >= neg_keep:
                continue
            n_neg += 1
        name = f"{stem}_{x0}_{y0}"
        cv2.imwrite(str(images_dir / f"{name}.jpg"), img[y0:y0 + th, x0:x0 + tw],
                    [cv2.IMWRITE_JPEG_QUALITY, quality])
        with open(labels_dir / f"{name}.txt", "w") as f:
            f.write("\n".join(labels))
        n_tiles += 1
        n_boxes += len(labels)
    return n_tiles, n_boxes, n_neg


class PIDDatasetConverter:
    """Dataset-P&ID -> tiled YOLO dataset with real class ids and a fixed split."""

    def __init__(self, data_root: str, output_dir: str, train_ratio: float = 0.9,
                 tile: int = 1280, overlap: int = 256, min_vis: float = 0.5,
                 neg_keep: float = 0.1, single_cls: bool = False, seed: int = 42,
                 workers: int = 32, jpeg_quality: int = 95):
        self.data_root = Path(data_root)
        self.output_dir = Path(output_dir)
        self.train_ratio = train_ratio
        self.tile, self.overlap, self.min_vis = tile, overlap, min_vis
        self.neg_keep, self.single_cls, self.seed = neg_keep, single_cls, seed
        self.workers, self.quality = workers, jpeg_quality
        self.image_dir = self.data_root / "image_2"

    def symbols_path(self, idx: int) -> Optional[Path]:
        for p in (self.data_root / "ann" / str(idx) / f"{idx}_symbols.npy",
                  self.data_root / str(idx) / f"{idx}_symbols.npy",
                  self.data_root / "mask" / f"{idx}_mask.png"):      # fallback
            if p.exists():
                return p
        return None

    def sheet_ids(self) -> List[int]:
        ids = []
        for p in sorted(self.image_dir.glob("*.jpg"), key=lambda q: int(q.stem)):
            if self.symbols_path(int(p.stem)) is not None:
                ids.append(int(p.stem))
        return ids

    def split(self) -> Dict[str, List[int]]:
        """Fixed held-out split, persisted so the val set never changes."""
        path = self.output_dir / "split.json"
        if path.exists():
            with open(path) as f:
                return json.load(f)
        ids = self.sheet_ids()
        rng = random.Random(self.seed)
        rng.shuffle(ids)
        k = int(len(ids) * self.train_ratio)
        sp = {"train": sorted(ids[:k]), "val": sorted(ids[k:]), "seed": self.seed}
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(sp, f, indent=1)
        return sp

    def convert(self) -> str:
        sp = self.split()
        print(f"Converting {len(sp['train'])} train / {len(sp['val'])} val sheets "
              f"into {self.tile}px tiles (overlap {self.overlap})")
        for name in ("train", "val"):
            images_dir = self.output_dir / "images" / name
            labels_dir = self.output_dir / "labels" / name
            images_dir.mkdir(parents=True, exist_ok=True)
            labels_dir.mkdir(parents=True, exist_ok=True)
            jobs = [(self.image_dir / f"{i}.jpg", self.symbols_path(i), images_dir,
                     labels_dir, self.tile, self.overlap, self.min_vis,
                     self.neg_keep if name == "train" else 1.0,   # keep all val tiles
                     self.single_cls, self.seed, self.quality) for i in sp[name]]
            tiles = boxes = neg = 0
            with ProcessPoolExecutor(self.workers) as ex:
                for t, b, n in tqdm(ex.map(_convert_one, jobs), total=len(jobs), desc=name):
                    tiles += t
                    boxes += b
                    neg += n
            print(f"  {name}: {tiles} tiles ({neg} empty), {boxes} boxes")

        names = {0: "symbol"} if self.single_cls else CLASS_NAMES
        data_yaml = self.output_dir / "data.yaml"
        with open(data_yaml, "w") as f:
            yaml.dump({"path": str(self.output_dir.absolute()),
                       "train": "images/train", "val": "images/val",
                       "names": names, "nc": len(names)}, f, default_flow_style=False)
        print(f"  data config: {data_yaml}")
        return str(data_yaml)


# ============================================================================
# Detector
# ============================================================================

def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> List[int]:
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thr]
    return keep


def merge_tile_detections(dets: List[Tuple[float, float, float, float, float, int]],
                          iou_thr: float = 0.6, agnostic: bool = False
                          ) -> List[Tuple[float, float, float, float, float, int]]:
    """NMS across tiles. dets: (x1, y1, x2, y2, conf, cls) in sheet coordinates."""
    if not dets:
        return []
    arr = np.array([d[:5] for d in dets], dtype=np.float64)
    cls = np.array([d[5] for d in dets])
    out = []
    groups = [np.arange(len(dets))] if agnostic else [np.where(cls == c)[0] for c in np.unique(cls)]
    for idx in groups:
        keep = _nms(arr[idx, :4], arr[idx, 4], iou_thr)
        out.extend(dets[int(idx[k])] for k in keep)
    return out


class YOLOSymbolDetector:
    """YOLO-based symbol detector with tiled inference and class output."""

    def __init__(self, model_path: Optional[str] = None, model_size: str = "yolo11m"):
        self.model_size = model_size
        if model_path and os.path.exists(model_path):
            print(f"Loading trained model from {model_path}")
            self.model = YOLO(model_path)
        else:
            print(f"Initializing {model_size} model")
            self.model = YOLO(f"{model_size}.pt")

    def train(self, data_yaml: str, epochs: int = 60, imgsz: int = 1280,
              batch_size: int = 64, device: str = "0", project: str = "./runs/train",
              name: str = "pid_symbols", patience: int = 20, lr0: float = 0.01,
              lrf: float = 0.01, resume: bool = False, augment: bool = True,
              single_cls: bool = False, pretrained: bool = True, workers: int = 16,
              cache: Union[bool, str] = False) -> str:
        print(f"\n{'=' * 60}\nYOLO training: {self.model_size} imgsz={imgsz} batch={batch_size} "
              f"device={device} epochs={epochs} single_cls={single_cls}\n{'=' * 60}")
        dev = [int(d) for d in str(device).split(",")] if "," in str(device) else device
        self.model.train(
            data=data_yaml, epochs=epochs, imgsz=imgsz, batch=batch_size, device=dev,
            project=project, name=name, patience=patience, lr0=lr0, lrf=lrf,
            resume=resume, augment=augment, pretrained=pretrained,
            single_cls=single_cls, exist_ok=True,
            save=True, save_period=-1, plots=True, verbose=True, seed=42,
            deterministic=False, cos_lr=True, close_mosaic=10, amp=True,
            workers=workers, cache=cache, optimizer="auto", warmup_epochs=3.0,
            # symbols are drawn at fixed scale and never flipped, so keep the
            # geometric augmentation mild; hue/saturation are meaningless on
            # black-on-grey drawings
            mosaic=1.0, mixup=0.0, degrees=0.0, translate=0.1, scale=0.2,
            shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.0,
            hsv_h=0.0, hsv_s=0.0, hsv_v=0.2, erasing=0.0,
            box=7.5, cls=0.5, dfl=1.5,
        )
        best = Path(project) / name / "weights" / "best.pt"
        print(f"\nTraining complete. Best weights: {best}")
        return str(best)

    def detect(self, image: Union[str, np.ndarray], conf_threshold: float = 0.25,
               iou_threshold: float = 0.6, imgsz: int = 1280, tile: int = 1280,
               overlap: int = 256, batch: int = 16, agnostic_nms: bool = False,
               max_det: int = 300, border: int = 2) -> List[SymbolDetection]:
        """Inference over a full sheet.

        tile > 0: tiled inference at native resolution, tiles merged with NMS
        (the Dataset-P&ID 32-class model is trained this way).  A box touching a
        tile edge that is not the sheet edge is a clipped view of a symbol the
        neighbouring tile sees whole (overlap 256 px > largest symbol 155 px); a
        half-visible clip has IoU ~0.5 with the full box and survives NMS, so
        such boxes are dropped (`border` px).

        tile = 0: whole image resized to `imgsz` -- the protocol of the private
        single-class real-drawing model (imgsz 1280, conf 0.10, then keep
        score >= 0.65; macro-F1 0.888 on real4).
        """
        img = cv2.imread(str(image)) if isinstance(image, (str, Path)) else image
        H, W = img.shape[:2]
        names = self.model.names if hasattr(self.model, "names") else CLASS_NAMES
        if not tile:
            r = self.model.predict(img, imgsz=imgsz, conf=conf_threshold, iou=iou_threshold,
                                   max_det=max_det, agnostic_nms=agnostic_nms, verbose=False)[0]
            out = []
            if r.boxes is not None:
                for (x1, y1, x2, y2), c, l in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(),
                                                  r.boxes.cls.cpu().numpy().astype(int)):
                    out.append(SymbolDetection(x_center=float(x1 + x2) / 2, y_center=float(y1 + y2) / 2,
                                               confidence=float(c), bbox=(float(x1), float(y1), float(x2), float(y2)),
                                               cls=int(l), cls_name=str(names.get(int(l), l)) if isinstance(names, dict) else str(l)))
            return out
        grid = tile_grid(H, W, tile, overlap)
        dets: List[Tuple[float, float, float, float, float, int]] = []
        for k in range(0, len(grid), batch):
            chunk = grid[k:k + batch]
            patches = [img[y0:y0 + tile, x0:x0 + tile] for x0, y0 in chunk]
            results = self.model.predict(patches, imgsz=imgsz, conf=conf_threshold,
                                         iou=iou_threshold, max_det=max_det,
                                         verbose=False, half=True)
            for (x0, y0), r in zip(chunk, results):
                if r.boxes is None or len(r.boxes) == 0:
                    continue
                xyxy = r.boxes.xyxy.cpu().numpy()
                cf = r.boxes.conf.cpu().numpy()
                cl = r.boxes.cls.cpu().numpy().astype(int)
                th, tw = min(tile, H - y0), min(tile, W - x0)
                for (x1, y1, x2, y2), c, l in zip(xyxy, cf, cl):
                    if ((x0 > 0 and x1 <= border) or (y0 > 0 and y1 <= border)
                            or (x0 + tw < W and x2 >= tw - border)
                            or (y0 + th < H and y2 >= th - border)):
                        continue
                    dets.append((float(x1 + x0), float(y1 + y0), float(x2 + x0),
                                 float(y2 + y0), float(c), int(l)))
        merged = merge_tile_detections(dets, iou_thr=iou_threshold, agnostic=agnostic_nms)
        return [SymbolDetection(x_center=(d[0] + d[2]) / 2, y_center=(d[1] + d[3]) / 2,
                                confidence=d[4], bbox=(d[0], d[1], d[2], d[3]), cls=d[5],
                                cls_name=str(names.get(d[5], d[5])) if isinstance(names, dict) else str(d[5]))
                for d in merged]

    def predict_sheets(self, data_root: str, ids: Sequence[int], out_path: str,
                       **kw) -> Dict[str, List[Dict]]:
        """{sheet_idx: [{"bbox", "cls", "conf"}]} for tools/eval_symbols.py."""
        out: Dict[str, List[Dict]] = {}
        for i in tqdm(ids, desc="predict"):
            dets = self.detect(str(Path(data_root) / "image_2" / f"{i}.jpg"), **kw)
            out[str(i)] = [{"bbox": [round(v, 1) for v in d.bbox], "cls": d.cls,
                            "conf": round(d.confidence, 4)} for d in dets]
        with open(out_path, "w") as f:
            json.dump(out, f)
        print(f"wrote {out_path} ({len(out)} sheets)")
        return out

    def get_sam2_prompts(self, image, conf_threshold: float = 0.25) -> List[Tuple[float, float]]:
        return [d.to_sam2_prompt() for d in self.detect(image, conf_threshold=conf_threshold)]

    def export_points_json(self, image, output_path: str, conf_threshold: float = 0.25) -> dict:
        dets = self.detect(image, conf_threshold=conf_threshold)
        result = {"image": str(image) if isinstance(image, (str, Path)) else "numpy_array",
                  "num_detections": len(dets),
                  "detections": [d.to_dict() for d in dets],
                  "sam2_prompts": {"points": [[d.x_center, d.y_center] for d in dets],
                                   "labels": [1] * len(dets)}}
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        return result

    def visualize(self, image, output_path: Optional[str] = None,
                  conf_threshold: float = 0.25) -> np.ndarray:
        img = cv2.imread(str(image)) if isinstance(image, (str, Path)) else image.copy()
        for d in self.detect(image, conf_threshold=conf_threshold):
            x1, y1, x2, y2 = [int(v) for v in d.bbox]
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(img, f"{d.cls_name} {d.confidence:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.circle(img, (int(d.x_center), int(d.y_center)), 5, (0, 255, 0), -1)
        if output_path:
            cv2.imwrite(output_path, img)
        return img

    def validate(self, data_yaml: str, imgsz: int = 1280, **kw) -> dict:
        """Tile-level validation via ultralytics (sheet-level: tools/eval_symbols.py)."""
        return self.model.val(data=data_yaml, imgsz=imgsz, **kw)

    def export(self, format: str = "onnx", imgsz: int = 1280, half: bool = False,
               dynamic: bool = False) -> str:
        return self.model.export(format=format, imgsz=imgsz, half=half, dynamic=dynamic)


def export_all_points(detector: YOLOSymbolDetector, data_root: str, output_dir: str,
                      conf_threshold: float = 0.25):
    data_root, output_dir = Path(data_root), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_results = {}
    for img_path in tqdm(sorted((data_root / "image_2").glob("*.jpg")), desc="Exporting points"):
        res = detector.export_points_json(str(img_path), str(output_dir / f"{img_path.stem}_points.json"),
                                          conf_threshold=conf_threshold)
        all_results[img_path.stem] = res
    with open(output_dir / "all_points.json", "w") as f:
        json.dump(all_results, f, indent=2)
    return all_results


def _parse_ids(spec: str) -> List[int]:
    """'1,2,3' | path to a text file | 'split.json:val'."""
    if ":" in spec and spec.rsplit(":", 1)[0].endswith(".json"):
        path, key = spec.rsplit(":", 1)
        with open(path) as f:
            return list(json.load(f)[key])
    if os.path.exists(spec):
        return [int(x) for x in open(spec).read().split()]
    return [int(x) for x in spec.split(",")]


def main():
    p = argparse.ArgumentParser(description="YOLO fine-tuning for P&ID symbol detection (tiled, 32-way)")
    p.add_argument("--mode", required=True,
                   choices=["convert", "train", "inference", "predict_sheets", "export_points",
                            "visualize", "validate", "export"])
    p.add_argument("--data_root", default="../dataset", help="dataset root with image_2/ and ann/")
    p.add_argument("--output_dir", default="./yolo_tiles", help="tiled YOLO dataset dir")
    p.add_argument("--model", default=None, help="trained weights")
    p.add_argument("--model_size", default="yolo11m")
    p.add_argument("--image", default=None)
    p.add_argument("--ids", default=None, help="sheet ids: '1,2' | file | split.json:val")
    p.add_argument("--pred_out", default="preds.json")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--tile", type=int, default=1280, help="tile size; 0 = whole image at --imgsz (private real-drawing model protocol)")
    p.add_argument("--overlap", type=int, default=256)
    p.add_argument("--min_vis", type=float, default=0.5, help="min visible box fraction at tile border")
    p.add_argument("--neg_keep", type=float, default=0.1, help="fraction of empty train tiles kept")
    p.add_argument("--single_cls", action="store_true", help="class-agnostic training (old behaviour)")
    p.add_argument("--device", default="0")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--agnostic_nms", action="store_true",
                   help="merge tiles class-agnostically: one box per symbol (class-aware NMS keeps "
                        "a second box of another class on 12%% of symbols)")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.6)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--project", default="./runs/train")
    p.add_argument("--name", default="pid_symbols")
    p.add_argument("--export_format", default="onnx")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--no_augment", action="store_true")
    p.add_argument("--cache", default="false", help="false | ram | disk")
    args = p.parse_args()

    conv = lambda: PIDDatasetConverter(args.data_root, args.output_dir, tile=args.tile,
                                       overlap=args.overlap, min_vis=args.min_vis,
                                       neg_keep=args.neg_keep, single_cls=args.single_cls,
                                       workers=args.workers)
    cache = {"false": False, "true": True}.get(args.cache.lower(), args.cache)

    if args.mode == "convert":
        print(f"\nData YAML: {conv().convert()}")

    elif args.mode == "train":
        data_yaml = Path(args.output_dir) / "data.yaml"
        if not data_yaml.exists():
            data_yaml = conv().convert()
        det = YOLOSymbolDetector(model_path=args.model, model_size=args.model_size)
        print(f"\nBest model: {det.train(str(data_yaml), epochs=args.epochs, imgsz=args.imgsz, batch_size=args.batch_size, device=args.device, project=args.project, name=args.name, patience=args.patience, lr0=args.lr, resume=args.resume, augment=not args.no_augment, single_cls=args.single_cls, workers=args.workers, cache=cache)}")

    elif args.mode in ("inference", "visualize", "export_points", "predict_sheets", "validate", "export"):
        if not args.model:
            sys.exit("--model is required")
        det = YOLOSymbolDetector(model_path=args.model)
        kw = dict(conf_threshold=args.conf, iou_threshold=args.iou, imgsz=args.imgsz,
                  tile=args.tile, overlap=args.overlap, agnostic_nms=args.agnostic_nms)
        if args.mode == "inference":
            dets = det.detect(args.image, **kw)
            print(f"\nDetected {len(dets)} symbols:")
            for i, d in enumerate(dets):
                print(f"  {i + 1}. {d.cls_name} conf {d.confidence:.3f} center ({d.x_center:.0f}, {d.y_center:.0f})")
        elif args.mode == "visualize":
            out = f"{Path(args.image).stem}_detected.jpg"
            det.visualize(args.image, output_path=out, conf_threshold=args.conf)
            print(f"saved {out}")
        elif args.mode == "export_points":
            export_all_points(det, args.data_root, str(Path(args.output_dir) / "sam2_prompts"), args.conf)
        elif args.mode == "predict_sheets":
            ids = _parse_ids(args.ids) if args.ids else conv().split()["val"]
            det.predict_sheets(args.data_root, ids, args.pred_out, **kw)
        elif args.mode == "validate":
            print(det.validate(str(Path(args.output_dir) / "data.yaml"), imgsz=args.imgsz))
        else:
            print(det.export(format=args.export_format, imgsz=args.imgsz))


if __name__ == "__main__":
    main()
