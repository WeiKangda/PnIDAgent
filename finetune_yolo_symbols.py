#!/usr/bin/env python3
"""
YOLO Fine-tuning for P&ID Symbol Detection

This script fine-tunes a YOLO model (v8/v10/v11) to detect a single class "symbol"
in P&ID diagrams. During inference, bounding box predictions are converted to
center points (x_center, y_center) for use as SAM2 point prompts.

Usage:
    # Training
    python finetune_yolo_symbols.py --mode train --data_root ./DigitizePID_Dataset

    # Inference (outputs center points)
    python finetune_yolo_symbols.py --mode inference --model ./runs/train/best.pt --image ./test.jpg

    # Export center points for SAM2
    python finetune_yolo_symbols.py --mode export_points --model ./runs/train/best.pt --data_root ./DigitizePID_Dataset
"""

import os
import sys
import json
import shutil
import argparse
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass
from tqdm import tqdm
import cv2
import yaml
import random

# Ultralytics YOLO import
try:
    from ultralytics import YOLO
    print("Ultralytics YOLO imported successfully")
except ImportError:
    print("Please install ultralytics: pip install ultralytics")
    sys.exit(1)


@dataclass
class SymbolDetection:
    """Represents a detected symbol with center point and confidence."""
    x_center: float
    y_center: float
    confidence: float
    bbox: Optional[Tuple[float, float, float, float]] = None  # (x1, y1, x2, y2)

    def to_sam2_prompt(self) -> Tuple[float, float]:
        """Convert to SAM2 point prompt format."""
        return (self.x_center, self.y_center)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'center': [self.x_center, self.y_center],
            'confidence': self.confidence,
            'bbox': list(self.bbox) if self.bbox else None
        }


class PIDDatasetConverter:
    """Converts DigitizePID_Dataset to YOLO format for training."""

    def __init__(self, data_root: str, output_dir: str, train_ratio: float = 0.9):
        """
        Args:
            data_root: Path to DigitizePID_Dataset
            output_dir: Output directory for YOLO-formatted dataset
            train_ratio: Ratio of training data (rest is validation)
        """
        self.data_root = Path(data_root)
        self.output_dir = Path(output_dir)
        self.train_ratio = train_ratio
        self.image_dir = self.data_root / 'image_2'

    def convert(self, random_seed: int = 42) -> str:
        """
        Convert the dataset to YOLO format.

        Returns:
            Path to the generated data.yaml file
        """
        print(f"Converting dataset from {self.data_root} to YOLO format...")

        # Create output directories
        train_images = self.output_dir / 'images' / 'train'
        train_labels = self.output_dir / 'labels' / 'train'
        val_images = self.output_dir / 'images' / 'val'
        val_labels = self.output_dir / 'labels' / 'val'

        for d in [train_images, train_labels, val_images, val_labels]:
            d.mkdir(parents=True, exist_ok=True)

        # Collect all P&ID samples
        pid_dirs = sorted([d for d in self.data_root.iterdir()
                          if d.is_dir() and d.name.isdigit()])

        print(f"Found {len(pid_dirs)} P&ID directories")

        # Shuffle and split
        random.seed(random_seed)
        random.shuffle(pid_dirs)
        split_idx = int(len(pid_dirs) * self.train_ratio)
        train_pids = pid_dirs[:split_idx]
        val_pids = pid_dirs[split_idx:]

        print(f"Train: {len(train_pids)} images, Val: {len(val_pids)} images")

        # Process training set
        train_count = self._process_split(train_pids, train_images, train_labels, "train")

        # Process validation set
        val_count = self._process_split(val_pids, val_images, val_labels, "val")

        # Create data.yaml
        data_yaml = self.output_dir / 'data.yaml'
        data_config = {
            'path': str(self.output_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'names': {
                0: 'symbol'
            },
            'nc': 1  # number of classes
        }

        with open(data_yaml, 'w') as f:
            yaml.dump(data_config, f, default_flow_style=False)

        print(f"\nDataset conversion complete!")
        print(f"  Train images: {train_count}")
        print(f"  Val images: {val_count}")
        print(f"  Data config: {data_yaml}")

        return str(data_yaml)

    def _process_split(self, pid_dirs: List[Path], images_dir: Path,
                       labels_dir: Path, split_name: str) -> int:
        """Process a data split (train or val)."""
        processed = 0
        total_symbols = 0

        for pid_dir in tqdm(pid_dirs, desc=f"Processing {split_name}"):
            pid_id = pid_dir.name
            image_path = self.image_dir / f"{pid_id}.jpg"
            symbols_path = pid_dir / f"{pid_id}_symbols.npy"

            if not image_path.exists() or not symbols_path.exists():
                continue

            try:
                # Load image to get dimensions
                img = cv2.imread(str(image_path))
                if img is None:
                    continue
                img_height, img_width = img.shape[:2]

                # Load symbols
                symbols_data = np.load(symbols_path, allow_pickle=True)

                # Extract bounding boxes and convert to YOLO format
                labels = []
                for row in symbols_data:
                    bbox = None
                    for item in row:
                        if isinstance(item, (list, np.ndarray)) and len(item) == 4:
                            bbox = item
                            break

                    if bbox is not None:
                        x1, y1, x2, y2 = bbox

                        # Convert to YOLO format (normalized center + width/height)
                        x_center = ((x1 + x2) / 2) / img_width
                        y_center = ((y1 + y2) / 2) / img_height
                        width = (x2 - x1) / img_width
                        height = (y2 - y1) / img_height

                        # Clamp values to [0, 1]
                        x_center = max(0, min(1, x_center))
                        y_center = max(0, min(1, y_center))
                        width = max(0, min(1, width))
                        height = max(0, min(1, height))

                        # Class 0 = symbol
                        labels.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
                        total_symbols += 1

                if len(labels) > 0:
                    # Copy image
                    dest_image = images_dir / f"{pid_id}.jpg"
                    shutil.copy2(image_path, dest_image)

                    # Write label file
                    label_file = labels_dir / f"{pid_id}.txt"
                    with open(label_file, 'w') as f:
                        f.write('\n'.join(labels))

                    processed += 1

            except Exception as e:
                print(f"Error processing P&ID {pid_id}: {e}")
                continue

        print(f"  {split_name}: {processed} images, {total_symbols} symbols")
        return processed


class YOLOSymbolDetector:
    """YOLO-based symbol detector with center point output for SAM2."""

    def __init__(self, model_path: Optional[str] = None,
                 model_size: str = 'yolov8n'):
        """
        Args:
            model_path: Path to trained model weights (for inference)
            model_size: YOLO model size for training (yolov8n, yolov8s, yolov8m, etc.)
        """
        self.model_size = model_size

        if model_path and os.path.exists(model_path):
            print(f"Loading trained model from {model_path}")
            self.model = YOLO(model_path)
        else:
            print(f"Initializing {model_size} model")
            self.model = YOLO(f"{model_size}.pt")

    def train(self, data_yaml: str, epochs: int = 100, imgsz: int = 640,
              batch_size: int = 16, device: str = 'auto',
              project: str = './runs/train', name: str = 'pid_symbols',
              patience: int = 20, lr0: float = 0.01, lrf: float = 0.01,
              resume: bool = False, augment: bool = True,
              mosaic: float = 1.0, mixup: float = 0.0,
              pretrained: bool = True) -> str:
        """
        Train the YOLO model on P&ID symbol detection.

        Args:
            data_yaml: Path to data.yaml file
            epochs: Number of training epochs
            imgsz: Image size for training
            batch_size: Batch size
            device: Device to use ('auto', 'cpu', '0', '0,1', etc.)
            project: Project directory for saving runs
            name: Experiment name
            patience: Early stopping patience
            lr0: Initial learning rate
            lrf: Final learning rate factor
            resume: Resume from last checkpoint
            augment: Enable data augmentation
            mosaic: Mosaic augmentation probability
            mixup: Mixup augmentation probability
            pretrained: Use pretrained weights

        Returns:
            Path to best model weights
        """
        print(f"\n{'='*60}")
        print("Starting YOLO Training for P&ID Symbol Detection")
        print(f"{'='*60}")
        print(f"Model: {self.model_size}")
        print(f"Data: {data_yaml}")
        print(f"Epochs: {epochs}")
        print(f"Image Size: {imgsz}")
        print(f"Batch Size: {batch_size}")
        print(f"Device: {device}")
        print(f"{'='*60}\n")

        # Training configuration
        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            device=device,
            project=project,
            name=name,
            patience=patience,
            lr0=lr0,
            lrf=lrf,
            resume=resume,
            augment=augment,
            mosaic=mosaic,
            mixup=mixup,
            pretrained=pretrained,
            # Additional useful settings
            save=True,
            save_period=10,  # Save checkpoint every 10 epochs
            plots=True,
            verbose=True,
            seed=42,
            deterministic=True,
            single_cls=True,  # Single class detection
            rect=False,  # Rectangular training
            cos_lr=True,  # Cosine LR scheduler
            close_mosaic=10,  # Disable mosaic for last 10 epochs
            amp=True,  # Automatic Mixed Precision
            fraction=1.0,  # Use full dataset
            overlap_mask=True,
            mask_ratio=4,
            dropout=0.0,
            val=True,
            cache=False,  # Cache images in RAM
            workers=8,
            optimizer='auto',
            warmup_epochs=3.0,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            box=7.5,  # Box loss gain
            cls=0.5,  # Class loss gain
            dfl=1.5,  # DFL loss gain
            nbs=64,  # Nominal batch size
            hsv_h=0.015,  # HSV-Hue augmentation
            hsv_s=0.7,  # HSV-Saturation augmentation
            hsv_v=0.4,  # HSV-Value augmentation
            degrees=0.0,  # Rotation augmentation (degrees)
            translate=0.1,  # Translation augmentation
            scale=0.5,  # Scale augmentation
            shear=0.0,  # Shear augmentation
            perspective=0.0,  # Perspective augmentation
            flipud=0.0,  # Flip up-down probability
            fliplr=0.5,  # Flip left-right probability
            bgr=0.0,  # BGR augmentation
            erasing=0.4,  # Random erasing augmentation
            crop_fraction=1.0,  # Crop fraction
        )

        # Get best model path
        best_model_path = Path(project) / name / 'weights' / 'best.pt'
        print(f"\nTraining complete! Best model saved to: {best_model_path}")

        return str(best_model_path)

    def detect(self, image: Union[str, np.ndarray],
               conf_threshold: float = 0.25,
               iou_threshold: float = 0.45,
               max_det: int = 1000,
               agnostic_nms: bool = False) -> List[SymbolDetection]:
        """
        Detect symbols in an image and return center points.

        Args:
            image: Image path or numpy array
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            max_det: Maximum number of detections
            agnostic_nms: Class-agnostic NMS

        Returns:
            List of SymbolDetection objects with center points
        """
        # Run inference
        results = self.model.predict(
            image,
            conf=conf_threshold,
            iou=iou_threshold,
            max_det=max_det,
            agnostic_nms=agnostic_nms,
            verbose=False
        )

        detections = []

        for result in results:
            boxes = result.boxes

            if boxes is None or len(boxes) == 0:
                continue

            for i in range(len(boxes)):
                # Get bounding box coordinates (xyxy format)
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].cpu().numpy())

                # Calculate center point
                x_center = (x1 + x2) / 2
                y_center = (y1 + y2) / 2

                detection = SymbolDetection(
                    x_center=float(x_center),
                    y_center=float(y_center),
                    confidence=conf,
                    bbox=(float(x1), float(y1), float(x2), float(y2))
                )
                detections.append(detection)

        return detections

    def detect_batch(self, images: List[Union[str, np.ndarray]],
                     conf_threshold: float = 0.25,
                     iou_threshold: float = 0.45) -> List[List[SymbolDetection]]:
        """
        Batch detection on multiple images.

        Args:
            images: List of image paths or numpy arrays
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold for NMS

        Returns:
            List of detection lists for each image
        """
        all_detections = []

        # Run batch inference
        results = self.model.predict(
            images,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False
        )

        for result in results:
            image_detections = []
            boxes = result.boxes

            if boxes is not None and len(boxes) > 0:
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                    conf = float(boxes.conf[i].cpu().numpy())

                    x_center = (x1 + x2) / 2
                    y_center = (y1 + y2) / 2

                    detection = SymbolDetection(
                        x_center=float(x_center),
                        y_center=float(y_center),
                        confidence=conf,
                        bbox=(float(x1), float(y1), float(x2), float(y2))
                    )
                    image_detections.append(detection)

            all_detections.append(image_detections)

        return all_detections

    def get_sam2_prompts(self, image: Union[str, np.ndarray],
                         conf_threshold: float = 0.25) -> List[Tuple[float, float]]:
        """
        Get center points formatted as SAM2 point prompts.

        Args:
            image: Image path or numpy array
            conf_threshold: Confidence threshold

        Returns:
            List of (x, y) center point tuples for SAM2 prompts
        """
        detections = self.detect(image, conf_threshold=conf_threshold)
        return [det.to_sam2_prompt() for det in detections]

    def export_points_json(self, image: Union[str, np.ndarray],
                           output_path: str,
                           conf_threshold: float = 0.25) -> dict:
        """
        Export detection results to JSON for SAM2 integration.

        Args:
            image: Image path or numpy array
            output_path: Output JSON file path
            conf_threshold: Confidence threshold

        Returns:
            Dictionary with detection results
        """
        detections = self.detect(image, conf_threshold=conf_threshold)

        result = {
            'image': str(image) if isinstance(image, (str, Path)) else 'numpy_array',
            'num_detections': len(detections),
            'detections': [det.to_dict() for det in detections],
            'sam2_prompts': {
                'points': [[det.x_center, det.y_center] for det in detections],
                'labels': [1] * len(detections)  # All positive labels
            }
        }

        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)

        return result

    def visualize(self, image: Union[str, np.ndarray],
                  output_path: Optional[str] = None,
                  conf_threshold: float = 0.25,
                  show_bbox: bool = True,
                  show_center: bool = True,
                  point_radius: int = 5,
                  point_color: Tuple[int, int, int] = (0, 255, 0),
                  bbox_color: Tuple[int, int, int] = (255, 0, 0)) -> np.ndarray:
        """
        Visualize detections with center points.

        Args:
            image: Image path or numpy array
            output_path: Optional output path to save visualization
            conf_threshold: Confidence threshold
            show_bbox: Draw bounding boxes
            show_center: Draw center points
            point_radius: Radius of center point circles
            point_color: BGR color for center points
            bbox_color: BGR color for bounding boxes

        Returns:
            Annotated image as numpy array
        """
        # Load image if path
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
        else:
            img = image.copy()

        detections = self.detect(image, conf_threshold=conf_threshold)

        for det in detections:
            if show_bbox and det.bbox:
                x1, y1, x2, y2 = [int(v) for v in det.bbox]
                cv2.rectangle(img, (x1, y1), (x2, y2), bbox_color, 2)

                # Add confidence label
                label = f"{det.confidence:.2f}"
                cv2.putText(img, label, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, bbox_color, 2)

            if show_center:
                cx, cy = int(det.x_center), int(det.y_center)
                cv2.circle(img, (cx, cy), point_radius, point_color, -1)
                # Add cross marker for visibility
                cv2.line(img, (cx - point_radius*2, cy),
                        (cx + point_radius*2, cy), point_color, 2)
                cv2.line(img, (cx, cy - point_radius*2),
                        (cx, cy + point_radius*2), point_color, 2)

        if output_path:
            cv2.imwrite(output_path, img)

        return img

    def validate(self, data_yaml: str) -> dict:
        """
        Validate the model on the validation set.

        Args:
            data_yaml: Path to data.yaml file

        Returns:
            Validation metrics
        """
        results = self.model.val(data=data_yaml)
        return results

    def export(self, format: str = 'onnx',
               imgsz: int = 640,
               half: bool = False,
               dynamic: bool = False) -> str:
        """
        Export model to different formats.

        Args:
            format: Export format ('onnx', 'torchscript', 'coreml', etc.)
            imgsz: Image size
            half: FP16 quantization
            dynamic: Dynamic input shape

        Returns:
            Path to exported model
        """
        path = self.model.export(
            format=format,
            imgsz=imgsz,
            half=half,
            dynamic=dynamic
        )
        return path


def export_all_points(detector: YOLOSymbolDetector,
                      data_root: str,
                      output_dir: str,
                      conf_threshold: float = 0.25):
    """
    Export center points for all P&ID images for SAM2 processing.

    Args:
        detector: Trained YOLOSymbolDetector
        data_root: Path to DigitizePID_Dataset
        output_dir: Output directory for JSON files
    """
    data_root = Path(data_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_dir = data_root / 'image_2'

    all_results = {}

    for img_path in tqdm(sorted(image_dir.glob('*.jpg')), desc="Exporting points"):
        pid_id = img_path.stem

        detections = detector.detect(str(img_path), conf_threshold=conf_threshold)

        result = {
            'image_path': str(img_path),
            'num_detections': len(detections),
            'sam2_prompts': {
                'points': [[det.x_center, det.y_center] for det in detections],
                'labels': [1] * len(detections)
            },
            'detections': [det.to_dict() for det in detections]
        }

        all_results[pid_id] = result

        # Save individual JSON
        json_path = output_dir / f"{pid_id}_points.json"
        with open(json_path, 'w') as f:
            json.dump(result, f, indent=2)

    # Save combined JSON
    combined_path = output_dir / 'all_points.json'
    with open(combined_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nExported points for {len(all_results)} images to {output_dir}")
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description='YOLO Fine-tuning for P&ID Symbol Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Convert dataset to YOLO format
    python finetune_yolo_symbols.py --mode convert --data_root ./DigitizePID_Dataset

    # Train model
    python finetune_yolo_symbols.py --mode train --data_root ./DigitizePID_Dataset --epochs 100

    # Inference on single image
    python finetune_yolo_symbols.py --mode inference --model ./runs/train/pid_symbols/weights/best.pt --image ./test.jpg

    # Export points for SAM2
    python finetune_yolo_symbols.py --mode export_points --model ./runs/train/pid_symbols/weights/best.pt --data_root ./DigitizePID_Dataset

    # Visualize predictions
    python finetune_yolo_symbols.py --mode visualize --model ./runs/train/pid_symbols/weights/best.pt --image ./test.jpg
        """
    )

    parser.add_argument('--mode', type=str, required=True,
                       choices=['convert', 'train', 'inference', 'export_points',
                               'visualize', 'validate', 'export'],
                       help='Operation mode')
    parser.add_argument('--data_root', type=str,
                       default='/Users/kangdawei/Desktop/Research/INL/MultiAgentsFT-v2/P&IDAgent/DigitizePID_Dataset',
                       help='Path to DigitizePID_Dataset')
    parser.add_argument('--output_dir', type=str, default='./yolo_dataset',
                       help='Output directory for YOLO dataset')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to trained model weights')
    parser.add_argument('--model_size', type=str, default='yolov8n',
                       choices=['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x',
                               'yolov10n', 'yolov10s', 'yolov10m', 'yolov10l', 'yolov10x',
                               'yolo11n', 'yolo11s', 'yolo11m', 'yolo11l', 'yolo11x'],
                       help='YOLO model size')
    parser.add_argument('--image', type=str, help='Image path for inference/visualization')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (auto, cpu, 0, 0,1, etc.)')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45, help='IoU threshold for NMS')
    parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--project', type=str, default='./runs/train',
                       help='Project directory')
    parser.add_argument('--name', type=str, default='pid_symbols',
                       help='Experiment name')
    parser.add_argument('--export_format', type=str, default='onnx',
                       choices=['onnx', 'torchscript', 'coreml', 'tflite', 'engine'],
                       help='Export format')
    parser.add_argument('--resume', action='store_true', help='Resume training')
    parser.add_argument('--no_augment', action='store_true', help='Disable augmentation')

    args = parser.parse_args()

    if args.mode == 'convert':
        # Convert dataset to YOLO format
        converter = PIDDatasetConverter(args.data_root, args.output_dir)
        data_yaml = converter.convert()
        print(f"\nData YAML saved to: {data_yaml}")

    elif args.mode == 'train':
        # Convert dataset first
        converter = PIDDatasetConverter(args.data_root, args.output_dir)
        data_yaml = converter.convert()

        # Initialize and train model
        detector = YOLOSymbolDetector(model_size=args.model_size)
        best_model = detector.train(
            data_yaml=data_yaml,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch_size=args.batch_size,
            device=args.device,
            project=args.project,
            name=args.name,
            patience=args.patience,
            lr0=args.lr,
            resume=args.resume,
            augment=not args.no_augment
        )
        print(f"\nBest model saved to: {best_model}")

    elif args.mode == 'inference':
        if not args.model:
            print("Error: --model is required for inference mode")
            sys.exit(1)
        if not args.image:
            print("Error: --image is required for inference mode")
            sys.exit(1)

        detector = YOLOSymbolDetector(model_path=args.model)
        detections = detector.detect(args.image, conf_threshold=args.conf)

        print(f"\nDetected {len(detections)} symbols:")
        for i, det in enumerate(detections):
            print(f"  {i+1}. Center: ({det.x_center:.1f}, {det.y_center:.1f}), "
                  f"Conf: {det.confidence:.3f}")

        # Print SAM2 prompts
        prompts = detector.get_sam2_prompts(args.image, conf_threshold=args.conf)
        print(f"\nSAM2 Point Prompts:")
        print(f"  Points: {prompts}")
        print(f"  Labels: {[1] * len(prompts)}")

    elif args.mode == 'export_points':
        if not args.model:
            print("Error: --model is required for export_points mode")
            sys.exit(1)

        detector = YOLOSymbolDetector(model_path=args.model)
        output_dir = Path(args.output_dir) / 'sam2_prompts'
        export_all_points(detector, args.data_root, str(output_dir),
                         conf_threshold=args.conf)

    elif args.mode == 'visualize':
        if not args.model:
            print("Error: --model is required for visualize mode")
            sys.exit(1)
        if not args.image:
            print("Error: --image is required for visualize mode")
            sys.exit(1)

        detector = YOLOSymbolDetector(model_path=args.model)
        output_path = str(Path(args.image).stem) + '_detected.jpg'
        detector.visualize(args.image, output_path=output_path,
                          conf_threshold=args.conf)
        print(f"\nVisualization saved to: {output_path}")

    elif args.mode == 'validate':
        if not args.model:
            print("Error: --model is required for validate mode")
            sys.exit(1)

        detector = YOLOSymbolDetector(model_path=args.model)
        data_yaml = Path(args.output_dir) / 'data.yaml'

        if not data_yaml.exists():
            print(f"Error: data.yaml not found at {data_yaml}")
            print("Run --mode convert first")
            sys.exit(1)

        results = detector.validate(str(data_yaml))
        print(f"\nValidation Results:")
        print(results)

    elif args.mode == 'export':
        if not args.model:
            print("Error: --model is required for export mode")
            sys.exit(1)

        detector = YOLOSymbolDetector(model_path=args.model)
        export_path = detector.export(format=args.export_format, imgsz=args.imgsz)
        print(f"\nModel exported to: {export_path}")


if __name__ == "__main__":
    main()
