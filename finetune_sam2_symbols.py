#!/usr/bin/env python3

import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler
from torch.amp import autocast
import cv2
from pathlib import Path
import argparse
from tqdm import tqdm
from typing import Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
from PIL import Image

def collate_fn(batch):
    """Custom collate function to handle multiple symbols per image"""
    original_size = len(batch)
    # Filter out None samples
    batch = [item for item in batch if item is not None]
    filtered_size = len(batch)

    if original_size != filtered_size:
        print(f"Filtered out {original_size - filtered_size} None samples from batch")

    if len(batch) == 0:
        print("Warning: All samples in batch were None, returning None batch")
        return None

    # Handle both single sample and batch cases
    if len(batch) == 1:
        return batch[0]
    else:
        # For batch_size > 1, return list of samples
        # Each sample contains multiple symbols per image
        return batch

# SAM2 imports from Hugging Face only
try:
    from transformers import Sam2Model, Sam2Processor
    print("Hugging Face SAM2 imported successfully")
except ImportError as e:
    print(f"Error importing SAM2 from Hugging Face: {e}")
    print("Please install transformers with SAM2 support:")
    print("pip install transformers>=4.38.0")
    sys.exit(1)


class PIDSymbolDataset(Dataset):
    """Dataset for P&ID symbols with bounding boxes or point prompts"""

    def __init__(self, data_root: str, split: str = 'train', transform=None,
                 train_ratio: float = 0.9, max_samples: Optional[int] = None,
                 split_samples: tuple = None, random_seed: int = 42, prompt_type: str = 'points',
                 num_negative_points: int = 2, skip_validation: bool = False):
        """
        Args:
            data_root: Path to DigitizePID_Dataset
            split: 'train' or 'val'
            transform: Optional transforms
            train_ratio: Ratio of training data
            max_samples: Maximum number of samples to use (for debugging)
            split_samples: Pre-computed (train_samples, val_samples) tuple to avoid data leakage
            random_seed: Random seed for reproducible splits
            prompt_type: Type of prompt to use ('points' or 'boxes')
        """
        self.data_root = Path(data_root)
        self.image_dir = self.data_root / 'image_2'
        self.transform = transform
        self.split = split
        self.prompt_type = prompt_type
        self.num_negative_points = num_negative_points
        self.skip_validation = skip_validation

        # Collect all valid P&IDs with symbols
        if split_samples is not None:
            # Use pre-computed split samples to avoid data leakage
            train_samples, val_samples = split_samples
            if split == 'train':
                self.samples = train_samples
            else:
                self.samples = val_samples
        else:
            # Fallback to original method (with fixed seed for consistency)
            self.samples = []
            self._load_samples(train_ratio, max_samples, random_seed)
        print(f"Dataset {split}: {len(self.samples)} samples loaded")

    def _load_samples(self, train_ratio: float, max_samples: Optional[int], random_seed: int = 42):
        """Load and split samples - group symbols by image for efficiency"""
        all_image_samples = []

        # Iterate through all P&ID directories
        pid_dirs = sorted([d for d in self.data_root.iterdir()
                          if d.is_dir() and d.name.isdigit()])

        print(f"Found {len(pid_dirs)} P&ID directories")

        valid_pids = 0
        total_symbols = 0

        for pid_dir in tqdm(pid_dirs, desc="Loading P&ID data"):
            pid_id = int(pid_dir.name)
            image_path = self.image_dir / f"{pid_id}.jpg"
            symbols_path = pid_dir / f"{pid_id}_symbols.npy"

            if not image_path.exists():
                print(f"Missing image for P&ID {pid_id}: {image_path}")
                continue
            if not symbols_path.exists():
                print(f"Missing symbols data for P&ID {pid_id}: {symbols_path}")
                continue

            try:
                # Load symbols data
                symbols_data = np.load(symbols_path, allow_pickle=True)
                image_symbols = []

                if symbols_data.dtype == object:
                    # Extract all bounding boxes for this image
                    for row in symbols_data:
                        bbox = None
                        symbol_type = None

                        for item in row:
                            if isinstance(item, (list, np.ndarray)) and len(item) == 4:
                                bbox = item
                            elif isinstance(item, str) and not item.startswith('symbol_'):
                                symbol_type = item

                        if bbox is not None:
                            image_symbols.append({
                                'bbox': np.array(bbox, dtype=np.float32),
                                'symbol_type': symbol_type
                            })

                if len(image_symbols) > 0:
                    # Group all symbols from this image together
                    all_image_samples.append({
                        'pid_id': pid_id,
                        'image_path': str(image_path),
                        'symbols': image_symbols  # List of symbols in this image
                    })
                    valid_pids += 1
                    total_symbols += len(image_symbols)

            except Exception as e:
                print(f"Error loading P&ID {pid_id}: {e}")
                continue

        print(f"Loaded {total_symbols} symbols from {valid_pids} P&IDs")

        # Limit samples if specified (limit by number of images, not individual symbols)
        if max_samples and len(all_image_samples) > max_samples:
            print(f"Limiting image samples from {len(all_image_samples)} to {max_samples}")
            all_image_samples = random.sample(all_image_samples, max_samples)

        # Split into train/val with fixed seed for reproducibility
        random.seed(random_seed)
        random.shuffle(all_image_samples)
        split_idx = int(len(all_image_samples) * train_ratio)

        if self.split == 'train':
            self.samples = all_image_samples[:split_idx]
        else:
            self.samples = all_image_samples[split_idx:]

        # Count total symbols after split
        total_symbols_split = sum(len(sample['symbols']) for sample in self.samples)
        print(f"Split: {len(self.samples)} images with {total_symbols_split} {self.split} symbols (train_ratio={train_ratio})")

    def __len__(self):
        return len(self.samples)

    def _bbox_to_point(self, bbox):
        """Convert bounding box to center point"""
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        return [center_x, center_y]

    def _generate_negative_points_fast(self, positive_point, bbox, image_shape, combined_mask, num_negative=2):
        """Fast generation of negative points using vectorized operations"""
        x1, y1, x2, y2 = bbox
        center_x, center_y = positive_point
        h, w = image_shape[:2]

        # Calculate bbox dimensions for sampling range
        bbox_size = max(x2 - x1, y2 - y1)

        # Pre-generate many candidate points at once (vectorized)
        num_candidates = num_negative * 20  # Generate many candidates at once

        # Generate random distances and angles
        distances = np.random.uniform(1.2 * bbox_size, 4.0 * bbox_size, num_candidates)
        angles = np.random.uniform(0, 2 * np.pi, num_candidates)

        # Calculate all candidate points at once (vectorized)
        candidate_x = center_x + distances * np.cos(angles)
        candidate_y = center_y + distances * np.sin(angles)

        # Filter points within image bounds (vectorized)
        valid_bounds = (candidate_x >= 0) & (candidate_x < w) & (candidate_y >= 0) & (candidate_y < h)

        # Get valid candidates
        valid_x = candidate_x[valid_bounds].astype(int)
        valid_y = candidate_y[valid_bounds].astype(int)

        # Check mask values (vectorized)
        if len(valid_x) > 0:
            mask_values = combined_mask[valid_y, valid_x]
            background_points = ~mask_values  # Points in background

            if np.sum(background_points) >= num_negative:
                # Take first num_negative valid points
                valid_indices = np.where(background_points)[0][:num_negative]
                negative_points = [[float(valid_x[i]), float(valid_y[i])] for i in valid_indices]
                return negative_points

        # Fallback: simple edge points if vectorized approach fails
        negative_points = []
        edge_candidates = [
            [w * 0.1, h * 0.1], [w * 0.9, h * 0.1],  # Top corners
            [w * 0.1, h * 0.9], [w * 0.9, h * 0.9],  # Bottom corners
            [w * 0.5, 0], [w * 0.5, h-1],            # Top/bottom center
            [0, h * 0.5], [w-1, h * 0.5]             # Left/right center
        ]

        for point in edge_candidates:
            if len(negative_points) >= num_negative:
                break
            x, y = int(point[0]), int(point[1])
            if not combined_mask[y, x]:
                negative_points.append(point)

        # Absolute fallback
        while len(negative_points) < num_negative:
            negative_points.append([w//4, h//4])

        return negative_points[:num_negative]

    def _validate_negative_points(self, points_data, labels_data, masks, image_shape):
        """Validate that negative points are truly outside all mask areas"""
        h, w = image_shape[:2]

        # Combine all masks
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        for mask in masks:
            if isinstance(mask, np.ndarray):
                combined_mask = np.logical_or(combined_mask, mask > 0)

        violations = 0
        total_negative = 0

        for symbol_idx, (symbol_points, symbol_labels) in enumerate(zip(points_data, labels_data)):
            for point, label in zip(symbol_points, symbol_labels):
                if label == 0:  # Negative point
                    total_negative += 1
                    x, y = int(point[0]), int(point[1])
                    if 0 <= x < w and 0 <= y < h:
                        if combined_mask[y, x]:  # Point is inside a mask
                            violations += 1

        if violations > 0:
            print(f"WARNING: {violations}/{total_negative} negative points are inside mask areas!")

        return violations == 0

    def _prepare_for_sam2(self, img, masks, bboxes):
        """Prepare image and labels for SAM2 processing with HuggingFace format"""
        # Convert CV2 BGR to PIL RGB
        if len(img.shape) == 3:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = img

        # Convert to PIL Image for HuggingFace processor
        pil_image = Image.fromarray(img_rgb)

        # Convert bboxes to points if using point prompts
        if self.prompt_type == 'points':
            h, w = img.shape[:2]

            # Create combined mask once for all symbols (major optimization)
            combined_mask = np.zeros((h, w), dtype=bool)
            for mask in masks:
                combined_mask |= (mask > 0)

            all_points = []
            all_labels = []

            # Process all symbols at once when possible
            for bbox in bboxes:
                # Positive point (center of bbox)
                positive_point = self._bbox_to_point(bbox)

                # Generate negative points using fast vectorized method
                negative_points = self._generate_negative_points_fast(
                    positive_point, bbox, (h, w), combined_mask, num_negative=self.num_negative_points
                )

                # Combine points for this symbol: [positive, negative1, negative2]
                symbol_points = [positive_point] + negative_points
                # Labels: [1, 0, 0] (positive, negative, negative)
                symbol_labels = [1] + [0] * len(negative_points)

                all_points.append(symbol_points)
                all_labels.append(symbol_labels)

            return pil_image, masks, {'points': all_points, 'labels': all_labels}
        else:
            # Keep original dimensions for now - let the processor handle resizing
            return pil_image, masks, bboxes

    def __getitem__(self, idx):
        try:
            image_sample = self.samples[idx]
            symbols = image_sample['symbols']

            # Load image
            image = cv2.imread(image_sample['image_path'])
            if image is None:
                print(f"Failed to load image: {image_sample['image_path']}")
                return None  # Will be filtered out

            h, w = image.shape[:2]

            # Process all symbols for this image - optimized
            num_symbols = len(symbols)
            bboxes = np.zeros((num_symbols, 4), dtype=np.float32)
            masks = []

            for i, symbol in enumerate(symbols):
                # Get bbox and validate
                x1, y1, x2, y2 = symbol['bbox']

                # Ensure bbox is valid (vectorized clipping)
                x1 = max(0, min(x1, w-1))
                y1 = max(0, min(y1, h-1))
                x2 = max(x1+1, min(x2, w))
                y2 = max(y1+1, min(y2, h))

                bboxes[i] = [x1, y1, x2, y2]

                # Create binary mask for the symbol (pre-allocate)
                mask = np.zeros((h, w), dtype=np.uint8)
                mask[int(y1):int(y2), int(x1):int(x2)] = 1
                masks.append(mask)

            # Convert to list for compatibility
            bboxes = bboxes.tolist()

            # Prepare for SAM2 processing with HuggingFace format
            pil_image, masks_processed, prompts_processed = self._prepare_for_sam2(
                image, masks, bboxes
            )

            if self.prompt_type == 'points':
                points_data = prompts_processed['points']
                labels_data = prompts_processed['labels']

                if idx < 5:  # Debug first 5 samples
                    total_points = sum(len(points) for points in points_data)
                    positive_points = sum(labels.count(1) for labels in labels_data)
                    negative_points = sum(labels.count(0) for labels in labels_data)

                    # Only validate on first sample for performance (optional validation)
                    validation_status = ""
                    if idx == 0 and not self.skip_validation:
                        is_valid = self._validate_negative_points(points_data, labels_data, masks_processed, (h, w))
                        validation_status = f", validation: {'✓' if is_valid else '✗'}"

                    print(f"Sample {idx}: P&ID {image_sample['pid_id']}, "
                          f"original: {w}x{h}, "
                          f"num_symbols: {len(symbols)}, "
                          f"total_points: {total_points} (pos: {positive_points}, neg: {negative_points}), "
                          f"masks: {len(masks_processed)}{validation_status}")

                return {
                    'image': pil_image,           # PIL Image
                    'points': points_data,        # List of point lists for each symbol
                    'labels': labels_data,        # List of label lists for each symbol
                    'masks': masks_processed,     # List of masks
                    'pid_id': image_sample['pid_id'],
                    'num_symbols': len(symbols),
                    'prompt_type': 'points'
                }
            else:
                if idx < 5:  # Debug first 5 samples
                    print(f"Sample {idx}: P&ID {image_sample['pid_id']}, "
                          f"original: {w}x{h}, "
                          f"num_symbols: {len(symbols)}, "
                          f"bboxes: {len(prompts_processed)}, masks: {len(masks_processed)}")

                return {
                    'image': pil_image,           # PIL Image
                    'bboxes': np.array(prompts_processed, dtype=np.float32),  # [N, 4] where N is number of symbols
                    'masks': masks_processed,     # List of masks
                    'pid_id': image_sample['pid_id'],
                    'num_symbols': len(symbols),
                    'prompt_type': 'boxes'
                }
        except Exception as e:
            print(f"Error loading sample {idx} (P&ID {image_sample.get('pid_id', 'unknown')}): {e}")
            return None


class SAM2SymbolFineTuner:
    """Fine-tune SAM2 for P&ID symbol segmentation using Hugging Face"""

    def __init__(self, model_name: str = "facebook/sam2-hiera-base-plus",
                 device: str = 'cuda', gradient_accumulation_steps: int = 4,
                 debug: bool = False):
        """
        Args:
            model_name: Hugging Face SAM2 model name
            device: Device to use
            gradient_accumulation_steps: Number of steps to accumulate gradients
            debug: Enable debug prints for intermediate values
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.debug = debug
        print(f"Using device: {self.device}")
        if self.debug:
            print("Debug mode enabled - will print intermediate values")

        # Initialize Hugging Face SAM2 model and processor
        self.model = Sam2Model.from_pretrained(model_name).to(self.device)
        self.processor = Sam2Processor.from_pretrained(model_name)

        # Training components
        self.optimizer = None
        self.scaler = GradScaler()
        self.loss_fn = self._combined_loss
        self.gradient_accumulation_steps = gradient_accumulation_steps

    def _process_batch_data(self, batch):
        """Process batch data for Hugging Face SAM2 - handles both single samples and batches"""
        # Check if batch is a list of samples (batch_size > 1) or single sample
        if isinstance(batch, list):
            # Process each sample in the batch
            all_inputs = []
            all_gt_masks = []

            for sample in batch:
                inputs, gt_masks = self._process_single_sample(sample)
                all_inputs.append(inputs)
                all_gt_masks.append(gt_masks)

            # For simplicity, process samples one by one (can be optimized later)
            # Return the first sample's inputs and concatenate masks
            return all_inputs[0], all_gt_masks[0]  # For now, process first sample only
        else:
            # Single sample case
            return self._process_single_sample(batch)

    def _process_single_sample(self, sample):
        """Process a single sample for Hugging Face SAM2"""
        image = sample['image']
        masks = sample['masks']    # List of N masks
        prompt_type = sample.get('prompt_type', 'boxes')

        # Process input for HuggingFace SAM2
        if prompt_type == 'points':
            points_data = sample['points']  # List of point lists for each symbol
            labels_data = sample['labels']  # List of label lists for each symbol

            # Convert to 4-level nested format expected by processor:
            # [image level, object level, point level, point coordinates]
            # For N symbols, each with multiple points: [[[symbol1_points], [symbol2_points], ...]]
            input_points = [points_data]  # Add image level: [[symbol1_points, symbol2_points, ...]]
            input_labels = [labels_data]  # Add image level: [[symbol1_labels, symbol2_labels, ...]]

            # Process the inputs
            inputs = self.processor(
                image,
                input_points=input_points,
                input_labels=input_labels,
                return_tensors="pt"
            )
        else:
            bboxes = sample['bboxes']  # [N, 4]
            # Convert bboxes to list format expected by processor
            input_boxes = [bboxes.tolist()] if len(bboxes.shape) == 2 else bboxes.tolist()
            # Process the inputs
            inputs = self.processor(
                image,
                input_boxes=input_boxes,
                return_tensors="pt"
            )

        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Convert masks to proper format and scale them to match processed image size
        gt_masks = []

        # Get original image dimensions
        if hasattr(image, 'size'):  # PIL Image
            orig_width, orig_height = image.size
        else:  # numpy array
            orig_height, orig_width = image.shape[:2]

        # Get processed image dimensions from processor output
        processed_height, processed_width = inputs['pixel_values'].shape[-2:]

        # Calculate scaling factors
        scale_x = processed_width / orig_width
        scale_y = processed_height / orig_height

        if self.debug:
            print(f"Scaling gt_masks: orig({orig_width}x{orig_height}) -> processed({processed_width}x{processed_height})")
            print(f"Scale factors: x={scale_x:.3f}, y={scale_y:.3f}")

        for mask in masks:
            if isinstance(mask, np.ndarray):
                mask_tensor = torch.from_numpy(mask).float()
            else:
                mask_tensor = mask.float()

            # Scale mask to match processed image size using interpolation
            if mask_tensor.dim() == 2:  # [H, W]
                mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
                scaled_mask = torch.nn.functional.interpolate(
                    mask_tensor,
                    size=(processed_height, processed_width),
                    mode='nearest'
                )
                scaled_mask = scaled_mask.squeeze(0).squeeze(0)  # [H, W]
            else:
                raise ValueError(f"Unexpected mask dimensions: {mask_tensor.shape}")

            gt_masks.append(scaled_mask)

        gt_masks_tensor = torch.stack(gt_masks).to(self.device)

        return inputs, gt_masks_tensor

    def _process_sam2_outputs(self, outputs, target_size=(1024, 1024)):
        """
        Process SAM2 outputs to get final masks at target resolution.

        Args:
            outputs: SAM2 model outputs with pred_masks and iou_scores
            target_size: Target size for upsampling (H, W)

        Returns:
            final_masks: Tensor of shape [batch_size, num_final_masks, H, W]
            selected_iou_scores: Tensor of shape [batch_size, num_final_masks] - selected IoU scores
        """
        pred_masks = outputs.pred_masks  # [1, 101, 3, 256, 256]
        iou_scores = outputs.iou_scores  # [1, 101, 3]

        batch_size, num_queries, _, _, _ = pred_masks.shape

        if self.debug:
            print(f"SAM2 output processing - pred_masks: {pred_masks.shape}, iou_scores: {iou_scores.shape}")
            print(f"Target size: {target_size}")

        # Select best mask per query using IoU scores
        # Get indices of highest IoU score for each query
        best_mask_indices = torch.argmax(iou_scores, dim=-1)  # [1, 101]

        # Gather the best masks for each query
        batch_indices = torch.arange(batch_size).unsqueeze(1).to(pred_masks.device)  # [1, 1]
        query_indices = torch.arange(num_queries).unsqueeze(0).to(pred_masks.device)  # [1, 101]

        # Use advanced indexing to select best masks
        best_masks = pred_masks[batch_indices, query_indices, best_mask_indices]  # [1, 101, 256, 256]

        # Also select the corresponding IoU scores
        selected_iou_scores = iou_scores[batch_indices, query_indices, best_mask_indices]  # [1, 101]

        if self.debug:
            print(f"Best masks shape before upsampling: {best_masks.shape}")
            print(f"Selected IoU scores shape: {selected_iou_scores.shape}")

        # Upsample masks to target resolution using bilinear interpolation
        final_masks = torch.nn.functional.interpolate(
            best_masks,  # [1, 101, 256, 256]
            size=target_size,  # (1024, 1024)
            mode='bilinear',
            align_corners=False
        )  # [1, 101, 1024, 1024]

        if self.debug:
            print(f"Final masks shape after upsampling: {final_masks.shape}")

        return final_masks, selected_iou_scores

    def _combined_loss(self, pred_masks, gt_masks, iou_predictions=None):
        """Combined loss for mask prediction with memory optimization"""
        if self.debug:
            print(f"Loss calculation - pred_masks shape: {pred_masks.shape}, gt_masks shape: {gt_masks.shape}")
            if iou_predictions is not None:
                print(f"Loss calculation - iou_predictions shape: {iou_predictions.shape}")

        # Ensure tensors have the same shape
        if pred_masks.shape != gt_masks.shape:
            # If gt_masks has fewer dimensions, add them
            while len(gt_masks.shape) < len(pred_masks.shape):
                gt_masks = gt_masks.unsqueeze(1)
            # If pred_masks has fewer dimensions, add them
            while len(pred_masks.shape) < len(gt_masks.shape):
                pred_masks = pred_masks.unsqueeze(1)

        # Binary cross entropy loss
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            pred_masks, gt_masks.float(), reduction='mean'
        )

        # Dice loss for better boundary learning
        pred_masks_sigmoid = torch.sigmoid(pred_masks)

        # Determine the spatial dimensions based on tensor shape
        # Could be [N, H, W] or [N, C, H, W] or other shapes
        if len(pred_masks.shape) == 4:  # [N, C, H, W]
            spatial_dims = (2, 3)
        elif len(pred_masks.shape) == 3:  # [N, H, W]
            spatial_dims = (1, 2)
        elif len(pred_masks.shape) == 2:  # [H, W]
            spatial_dims = (0, 1)
        else:
            # For other dimensions, sum over the last two dimensions
            spatial_dims = tuple(range(-2, 0))

        # Compute dice loss more efficiently
        with torch.cuda.amp.autocast(enabled=False):  # Disable autocast for stability
            intersection = (pred_masks_sigmoid * gt_masks).sum(dim=spatial_dims)
            pred_sum = pred_masks_sigmoid.sum(dim=spatial_dims)
            gt_sum = gt_masks.sum(dim=spatial_dims)
            dice_loss = 1 - (2 * intersection + 1) / (pred_sum + gt_sum + 1)
            dice_loss = dice_loss.mean()

        # IoU loss if predictions available
        iou_loss = 0
        if iou_predictions is not None:
            with torch.no_grad():  # Don't need gradients for IoU calculation
                pred_masks_binary = (pred_masks_sigmoid > 0.5).float()
                intersection = (pred_masks_binary * gt_masks).sum(dim=spatial_dims)
                union = pred_masks_binary.sum(dim=spatial_dims) + gt_masks.sum(dim=spatial_dims) - intersection
                actual_iou = intersection / (union + 1e-6)

                if self.debug:
                    print(f"IoU calculation - spatial_dims: {spatial_dims}")
                    print(f"IoU calculation - intersection shape: {intersection.shape}, values: {intersection}")
                    print(f"IoU calculation - union shape: {union.shape}, values: {union}")
                    print(f"IoU calculation - actual_iou shape before processing: {actual_iou.shape}, values: {actual_iou}")

                # Handle different shapes for iou_predictions
                if len(actual_iou.shape) > 1:
                    actual_iou = actual_iou.mean(dim=tuple(range(1, len(actual_iou.shape))))

            iou_pred_squeezed = iou_predictions.squeeze()
            if iou_pred_squeezed.numel() == 1:
                iou_pred_squeezed = iou_pred_squeezed.unsqueeze(0)
            if actual_iou.numel() == 1:
                actual_iou = actual_iou.unsqueeze(0)

            if self.debug:
                print(f"Final IoU tensors - iou_pred_squeezed shape: {iou_pred_squeezed.shape}, values: {iou_pred_squeezed}")
                print(f"Final IoU tensors - actual_iou shape: {actual_iou.shape}, values: {actual_iou}")
                print(f"Tensor element counts - iou_pred_squeezed: {iou_pred_squeezed.numel()}, actual_iou: {actual_iou.numel()}")

            iou_loss = nn.functional.mse_loss(iou_pred_squeezed, actual_iou)

        return bce_loss + dice_loss + 0.1 * iou_loss

    def setup_training(self, learning_rate: float = 1e-5,
                      weight_decay: float = 0.01,
                      freeze_image_encoder: bool = False):
        """Setup training parameters"""

        # Optionally freeze image encoder for faster training
        if freeze_image_encoder:
            for param in self.model.image_encoder.parameters():
                param.requires_grad = False
            print("Image encoder frozen")
        else:
            print("Image encoder unfrozen - full model training")

        # Get trainable parameters
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())

        # Setup optimizer
        self.optimizer = optim.AdamW(trainable_params,
                                     lr=learning_rate,
                                     weight_decay=weight_decay)

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=1e-7
        )

        print(f"Training setup complete. LR: {learning_rate}")

    def train_epoch(self, dataloader: DataLoader, epoch: int):
        """Train for one epoch using Hugging Face SAM2 with gradient accumulation"""
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
        accumulated_loss = 0

        for batch_idx, batch in enumerate(progress_bar):
            # Skip None batches (when all samples in batch failed to load)
            if batch is None:
                print(f"Skipping None batch {batch_idx}")
                continue

            if self.debug or batch_idx < 2:  # Debug mode or first 2 batches
                # Handle both single samples and batch lists
                if isinstance(batch, list):
                    print(f"Batch {batch_idx}: Processing {len(batch)} samples")
                    for i, sample in enumerate(batch[:3]):  # Show first 3 samples
                        prompt_type = sample.get('prompt_type', 'boxes')
                        print(f"  Sample {i}: P&ID {sample['pid_id']}, symbols: {sample['num_symbols']}, prompt_type: {prompt_type}")
                else:
                    prompt_type = batch.get('prompt_type', 'boxes')
                    print(f"Batch {batch_idx}: Processing P&ID {batch['pid_id']}")
                    print(f"Number of symbols: {batch['num_symbols']}")
                    print(f"Prompt type: {prompt_type}")
                    if prompt_type == 'points':
                        points_info = f"Points: {len(batch['points'])} symbols, "
                        total_points = sum(len(symbol_points) for symbol_points in batch['points'])
                        points_info += f"{total_points} total points"
                        print(points_info)
                    else:
                        print(f"Bboxes shape: {batch['bboxes'].shape}")

            # Clear cache periodically to prevent memory buildup
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()

            # Forward pass with mixed precision
            with autocast('cuda'):
                # Process batch data for HuggingFace SAM2
                inputs, gt_masks = self._process_batch_data(batch)

                if self.debug or batch_idx < 2:
                    print(f"DEBUG: inputs keys: {inputs.keys()}")
                    for k, v in inputs.items():
                        if hasattr(v, 'shape'):
                            print(f"DEBUG: {k} shape: {v.shape}")

                # Forward pass through HuggingFace SAM2
                outputs = self.model(**inputs)

                # Process SAM2 outputs to get final masks at target resolution
                target_size = (gt_masks.shape[-2], gt_masks.shape[-1])  # Match gt_masks resolution
                pred_masks, iou_predictions = self._process_sam2_outputs(outputs, target_size=target_size)

                if self.debug or batch_idx < 2:
                    print(f"DEBUG: Final pred_masks shape: {pred_masks.shape}")
                    print(f"DEBUG: gt_masks shape: {gt_masks.shape}")
                    if iou_predictions is not None:
                        print(f"DEBUG: iou_predictions shape: {iou_predictions.shape}")
                    else:
                        print("DEBUG: iou_predictions is None")

                # Reshape pred_masks to match gt_masks dimensions for loss calculation
                if pred_masks.shape[0] == 1:  # Remove batch dimension if present
                    pred_masks = pred_masks.squeeze(0)  # [num_queries, H, W]
                if iou_predictions is not None and iou_predictions.shape[0] == 1:
                    iou_predictions = iou_predictions.squeeze(0)  # [num_queries]

                loss = self.loss_fn(pred_masks, gt_masks, iou_predictions)

                # Scale loss for gradient accumulation
                loss = loss / self.gradient_accumulation_steps

            # Backward pass with gradient accumulation
            self.scaler.scale(loss).backward()
            accumulated_loss += loss.item()
            gpu_mem = torch.cuda.memory_allocated()/1e9

            # Update weights every gradient_accumulation_steps
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                
                # Clean up memory
                del pred_masks, gt_masks
                if 'inputs' in locals():
                    del inputs
                torch.cuda.empty_cache()

            # Update metrics
            total_loss += loss.item() * self.gradient_accumulation_steps
            avg_loss = total_loss / (batch_idx + 1)
            progress_bar.set_postfix({'loss': f'{avg_loss:.4f}', 'gpu_mem': f'{gpu_mem:.1f}GB'})

        # Final optimizer step if there are remaining gradients
        if (batch_idx + 1) % self.gradient_accumulation_steps != 0:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

        self.scheduler.step()
        torch.cuda.empty_cache()
        return avg_loss

    def evaluate(self, dataloader: DataLoader):
        """Evaluate model performance using Hugging Face SAM2"""
        self.model.eval()
        total_loss = 0
        total_iou = 0
        first_batch = True

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                # Skip None batches
                if batch is None:
                    continue

                # Debug print for first batch to confirm negative points
                if first_batch:
                    prompt_type = batch.get('prompt_type', 'boxes')
                    if prompt_type == 'points':
                        # Count positive and negative points
                        if 'labels' in batch:
                            labels = batch['labels']
                            total_pos = sum(label_list.count(1) for label_list in labels)
                            total_neg = sum(label_list.count(0) for label_list in labels)
                            print(f"Evaluation using point prompts with {total_pos} positive and {total_neg} negative points")
                    else:
                        print(f"Evaluation using box prompts")
                    first_batch = False

                with autocast('cuda'):
                    # Process batch data for HuggingFace SAM2
                    inputs, gt_masks = self._process_batch_data(batch)

                    # Forward pass through HuggingFace SAM2
                    outputs = self.model(**inputs)

                    # Process SAM2 outputs to get final masks at target resolution
                    target_size = (gt_masks.shape[-2], gt_masks.shape[-1])  # Match gt_masks resolution
                    pred_masks, iou_predictions = self._process_sam2_outputs(outputs, target_size=target_size)

                    # Remove batch dimension if present
                    if pred_masks.shape[0] == 1:
                        pred_masks = pred_masks.squeeze(0)  # [num_queries, H, W]
                    if iou_predictions is not None and iou_predictions.shape[0] == 1:
                        iou_predictions = iou_predictions.squeeze(0)  # [num_queries]

                    loss = self.loss_fn(pred_masks, gt_masks, iou_predictions)

                total_loss += loss.item()

                # Calculate actual IoU for all symbols
                pred_masks_binary = (torch.sigmoid(pred_masks) > 0.5).float()

                # Ensure masks have same number of dimensions for IoU calculation
                if pred_masks_binary.shape != gt_masks.shape:
                    # Match dimensions
                    while len(gt_masks.shape) < len(pred_masks_binary.shape):
                        gt_masks = gt_masks.unsqueeze(1)
                    while len(pred_masks_binary.shape) < len(gt_masks.shape):
                        pred_masks_binary = pred_masks_binary.unsqueeze(1)

                # Determine spatial dimensions based on shape
                if len(pred_masks_binary.shape) == 4:  # [N, C, H, W]
                    spatial_dims = (2, 3)
                elif len(pred_masks_binary.shape) == 3:  # [N, H, W]
                    spatial_dims = (1, 2)
                elif len(pred_masks_binary.shape) == 2:  # [H, W]
                    spatial_dims = (0, 1)
                else:
                    spatial_dims = tuple(range(-2, 0))

                intersection = (pred_masks_binary * gt_masks).sum(dim=spatial_dims)
                union = (pred_masks_binary + gt_masks).clamp(0, 1).sum(dim=spatial_dims)
                iou = (intersection / (union + 1e-6)).mean()
                total_iou += iou.item()

        avg_loss = total_loss / len(dataloader)
        avg_iou = total_iou / len(dataloader)

        print(f"Evaluation completed - Avg Loss: {avg_loss:.4f}, Avg IoU: {avg_iou:.4f}")

        return avg_loss, avg_iou

    def save_checkpoint(self, path: str, epoch: int, best_iou: float):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_iou': best_iou
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """Load training checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if self.optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint['epoch'], checkpoint.get('best_iou', 0)

    def visualize_predictions(self, dataloader: DataLoader, num_samples: int = 5, save_dir: str = None):
        """Visualize model predictions using Hugging Face SAM2

        Args:
            dataloader: DataLoader for visualization
            num_samples: Number of samples to visualize
            save_dir: Directory to save individual visualizations. If None, doesn't save

        Returns:
            fig: The matplotlib figure object
        """
        self.model.eval()
        samples_shown = 0

        # Create save directory if specified
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_subdir = os.path.join(save_dir, f"visualizations_{timestamp}")
            os.makedirs(save_subdir, exist_ok=True)
            print(f"Saving visualizations to: {save_subdir}")

        fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)

        with torch.no_grad():
            for batch in dataloader:
                if samples_shown >= num_samples:
                    break

                # Skip None batches
                if batch is None:
                    continue

                # Process batch data (same as training)
                inputs, gt_masks = self._process_batch_data(batch)

                # Get predictions (same as training)
                with autocast('cuda'):
                    outputs = self.model(**inputs)

                    # Process SAM2 outputs to get final masks at target resolution
                    target_size = (gt_masks.shape[-2], gt_masks.shape[-1])
                    pred_masks, iou_predictions = self._process_sam2_outputs(outputs, target_size=target_size)

                    # Remove batch dimension if present (same as training)
                    if pred_masks.shape[0] == 1:
                        pred_masks = pred_masks.squeeze(0)  # [num_queries, H, W]
                    if iou_predictions is not None and iou_predictions.shape[0] == 1:
                        iou_predictions = iou_predictions.squeeze(0)  # [num_queries]

                # Convert to binary masks for visualization
                pred_masks_binary = (torch.sigmoid(pred_masks) > 0.5).cpu().numpy()

                # Get original image for visualization - handle both single samples and batches
                if isinstance(batch, list):
                    # For batch data, visualize the first sample
                    sample = batch[0]
                    image_array = np.array(sample['image'])
                    prompt_type = sample.get('prompt_type', 'boxes')
                    if prompt_type == 'points':
                        prompts = sample['points']
                        prompt_labels = sample.get('labels', None)
                    else:
                        prompts = sample['bboxes']
                        prompt_labels = None
                    pid_id = sample['pid_id']
                    num_symbols = sample['num_symbols']

                    # Get original dimensions for proper overlay
                    if hasattr(sample['image'], 'size'):
                        orig_width, orig_height = sample['image'].size
                    else:
                        orig_height, orig_width = image_array.shape[:2]
                else:
                    # Single sample case
                    image_array = np.array(batch['image'])
                    prompt_type = batch.get('prompt_type', 'boxes')
                    if prompt_type == 'points':
                        prompts = batch['points']
                        prompt_labels = batch.get('labels', None)
                    else:
                        prompts = batch['bboxes']
                        prompt_labels = None
                    pid_id = batch['pid_id']
                    num_symbols = batch['num_symbols']

                    # Get original dimensions for proper overlay
                    if hasattr(batch['image'], 'size'):
                        orig_width, orig_height = batch['image'].size
                    else:
                        orig_height, orig_width = image_array.shape[:2]

                row_idx = samples_shown

                # Original image with prompts (bounding boxes or points)
                axes[row_idx, 0].imshow(image_array)
                axes[row_idx, 0].set_title(f"P&ID {pid_id} ({num_symbols} symbols, {prompt_type})")
                axes[row_idx, 0].axis('off')

                # Draw all prompts with different colors
                colors = plt.cm.tab20(np.linspace(0, 1, num_symbols))
                if prompt_type == 'points':
                    # Draw points (both positive and negative)
                    for symbol_idx, symbol_points in enumerate(prompts):
                        symbol_labels = prompt_labels[symbol_idx] if prompt_labels else [1] * len(symbol_points)
                        for point_idx, point in enumerate(symbol_points):
                            x, y = int(point[0]), int(point[1])
                            label = symbol_labels[point_idx]
                            if label == 1:  # Positive point
                                axes[row_idx, 0].scatter(x, y, c=[colors[symbol_idx % len(colors)]], s=80, marker='+', linewidths=4)
                            else:  # Negative point
                                axes[row_idx, 0].scatter(x, y, c=[colors[symbol_idx % len(colors)]], s=60, marker='x', linewidths=3)
                else:
                    # Draw bounding boxes
                    for idx, bbox in enumerate(prompts):
                        x1, y1, x2, y2 = bbox.astype(int)
                        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                                linewidth=2, edgecolor=colors[idx % len(colors)],
                                                facecolor='none')
                        axes[row_idx, 0].add_patch(rect)

                # Combine all ground truth masks
                combined_gt_mask = np.zeros((gt_masks.shape[-2], gt_masks.shape[-1]), dtype=np.float32)
                for idx in range(min(num_symbols, gt_masks.shape[0])):
                    gt_mask_single = gt_masks[idx].cpu().numpy()
                    # Add each mask with a different intensity level for visualization
                    combined_gt_mask = np.maximum(combined_gt_mask, gt_mask_single * ((idx + 1) / num_symbols))

                # Resize combined GT mask to original dimensions
                combined_gt_resized = torch.nn.functional.interpolate(
                    torch.from_numpy(combined_gt_mask).unsqueeze(0).unsqueeze(0).float(),
                    size=(orig_height, orig_width),
                    mode='nearest'
                ).squeeze().numpy()

                axes[row_idx, 1].imshow(combined_gt_resized, cmap='hot')
                axes[row_idx, 1].set_title(f"Ground Truth ({num_symbols} masks)")
                axes[row_idx, 1].axis('off')

                # Combine all predicted masks
                combined_pred_mask = np.zeros((pred_masks_binary.shape[-2], pred_masks_binary.shape[-1]), dtype=np.float32)
                avg_iou = 0.0
                valid_predictions = min(num_symbols, pred_masks_binary.shape[0])

                for idx in range(valid_predictions):
                    pred_mask_single = pred_masks_binary[idx]
                    # Add each mask with a different intensity level
                    combined_pred_mask = np.maximum(combined_pred_mask, pred_mask_single * ((idx + 1) / num_symbols))
                    if iou_predictions is not None and idx < len(iou_predictions):
                        avg_iou += iou_predictions[idx].item()

                avg_iou = avg_iou / valid_predictions if valid_predictions > 0 else 0.0

                # Resize combined prediction mask to original dimensions
                combined_pred_resized = torch.nn.functional.interpolate(
                    torch.from_numpy(combined_pred_mask).unsqueeze(0).unsqueeze(0).float(),
                    size=(orig_height, orig_width),
                    mode='nearest'
                ).squeeze().numpy()

                axes[row_idx, 2].imshow(combined_pred_resized, cmap='hot')
                axes[row_idx, 2].set_title(f"Predictions (Avg IoU: {avg_iou:.3f})")
                axes[row_idx, 2].axis('off')

                # Create overlay with all masks on original image
                overlay = image_array.copy().astype(np.float32)

                # Create colored overlay for each mask
                for idx in range(valid_predictions):
                    pred_mask_single = pred_masks_binary[idx]
                    # Resize individual mask
                    mask_resized = torch.nn.functional.interpolate(
                        torch.from_numpy(pred_mask_single).unsqueeze(0).unsqueeze(0).float(),
                        size=(orig_height, orig_width),
                        mode='nearest'
                    ).squeeze().numpy() > 0.5

                    # Apply different color for each mask
                    color = colors[idx % len(colors)][:3] * 255
                    alpha = 0.3
                    overlay[mask_resized] = (1 - alpha) * overlay[mask_resized] + alpha * color

                axes[row_idx, 3].imshow(overlay.astype(np.uint8))
                axes[row_idx, 3].set_title(f"Overlay ({valid_predictions} masks)")
                axes[row_idx, 3].axis('off')

                # Save individual visualization if save_dir is specified
                if save_dir:
                    # Create individual figure for this sample
                    individual_fig, individual_axes = plt.subplots(1, 4, figsize=(16, 4))

                    # Copy visualizations to individual figure
                    individual_axes[0].imshow(image_array)
                    individual_axes[0].set_title(f"P&ID {pid_id} ({num_symbols} symbols, {prompt_type})")
                    individual_axes[0].axis('off')
                    if prompt_type == 'points':
                        # Draw points (both positive and negative)
                        for symbol_idx, symbol_points in enumerate(prompts):
                            symbol_labels = prompt_labels[symbol_idx] if prompt_labels else [1] * len(symbol_points)
                            for point_idx, point in enumerate(symbol_points):
                                x, y = int(point[0]), int(point[1])
                                label = symbol_labels[point_idx]
                                if label == 1:  # Positive point
                                    individual_axes[0].scatter(x, y, c=[colors[symbol_idx % len(colors)]], s=80, marker='+', linewidths=4)
                                else:  # Negative point
                                    individual_axes[0].scatter(x, y, c=[colors[symbol_idx % len(colors)]], s=60, marker='x', linewidths=3)
                    else:
                        # Draw bounding boxes
                        for idx, bbox in enumerate(prompts):
                            x1, y1, x2, y2 = bbox.astype(int)
                            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                                    linewidth=2, edgecolor=colors[idx % len(colors)],
                                                    facecolor='none')
                            individual_axes[0].add_patch(rect)

                    individual_axes[1].imshow(combined_gt_resized, cmap='hot')
                    individual_axes[1].set_title(f"Ground Truth ({num_symbols} masks)")
                    individual_axes[1].axis('off')

                    individual_axes[2].imshow(combined_pred_resized, cmap='hot')
                    individual_axes[2].set_title(f"Predictions (Avg IoU: {avg_iou:.3f})")
                    individual_axes[2].axis('off')

                    individual_axes[3].imshow(overlay.astype(np.uint8))
                    individual_axes[3].set_title(f"Overlay ({valid_predictions} masks)")
                    individual_axes[3].axis('off')

                    plt.tight_layout()

                    # Save individual figure
                    individual_filename = f"pid_{pid_id}_sample_{samples_shown:03d}_{num_symbols}symbols_avgIoU_{avg_iou:.3f}.png"
                    individual_path = os.path.join(save_subdir, individual_filename)
                    individual_fig.savefig(individual_path, dpi=150, bbox_inches='tight')
                    plt.close(individual_fig)

                    # Also save individual components
                    components_dir = os.path.join(save_subdir, f"pid_{pid_id}_components")
                    os.makedirs(components_dir, exist_ok=True)

                    # Save original image
                    Image.fromarray(image_array).save(os.path.join(components_dir, "original.png"))

                    # Save combined ground truth masks
                    Image.fromarray((combined_gt_resized * 255).astype(np.uint8), mode='L').save(
                        os.path.join(components_dir, "gt_masks_combined.png"))

                    # Save combined predicted masks
                    Image.fromarray((combined_pred_resized * 255).astype(np.uint8), mode='L').save(
                        os.path.join(components_dir, "pred_masks_combined.png"))

                    # Save overlay
                    Image.fromarray(overlay.astype(np.uint8)).save(os.path.join(components_dir, "overlay.png"))

                    # Save individual masks
                    masks_dir = os.path.join(components_dir, "individual_masks")
                    os.makedirs(masks_dir, exist_ok=True)

                    # Save each ground truth mask
                    for idx in range(min(num_symbols, gt_masks.shape[0])):
                        gt_mask_single = gt_masks[idx].cpu().numpy()
                        gt_mask_single_resized = torch.nn.functional.interpolate(
                            torch.from_numpy(gt_mask_single).unsqueeze(0).unsqueeze(0).float(),
                            size=(orig_height, orig_width),
                            mode='nearest'
                        ).squeeze().numpy()
                        Image.fromarray((gt_mask_single_resized * 255).astype(np.uint8), mode='L').save(
                            os.path.join(masks_dir, f"gt_mask_{idx:02d}.png"))

                    # Save each predicted mask
                    for idx in range(valid_predictions):
                        pred_mask_single = pred_masks_binary[idx]
                        pred_mask_single_resized = torch.nn.functional.interpolate(
                            torch.from_numpy(pred_mask_single).unsqueeze(0).unsqueeze(0).float(),
                            size=(orig_height, orig_width),
                            mode='nearest'
                        ).squeeze().numpy()
                        iou = iou_predictions[idx].item() if iou_predictions is not None and idx < len(iou_predictions) else 0.0
                        Image.fromarray((pred_mask_single_resized * 255).astype(np.uint8), mode='L').save(
                            os.path.join(masks_dir, f"pred_mask_{idx:02d}_iou_{iou:.3f}.png"))

                samples_shown += 1

        plt.tight_layout()

        # Save the combined figure if save_dir is specified
        if save_dir:
            combined_path = os.path.join(save_subdir, "all_samples_combined.png")
            fig.savefig(combined_path, dpi=150, bbox_inches='tight')
            print(f"Saved combined visualization to: {combined_path}")
            print(f"Saved {samples_shown} individual visualizations to: {save_subdir}")

        return fig


def create_datasets_with_proper_split(data_root: str, train_ratio: float = 0.9,
                                    max_samples: Optional[int] = None,
                                    random_seed: int = 42, prompt_type: str = 'points',
                                    num_negative_points: int = 2, skip_validation: bool = False):
    """
    Create train/val datasets with proper splitting to avoid data leakage.

    Args:
        data_root: Path to DigitizePID_Dataset
        train_ratio: Ratio of training data
        max_samples: Maximum number of samples to use (for debugging)
        random_seed: Random seed for reproducible splits
        prompt_type: Type of prompt to use ('points' or 'boxes')

    Returns:
        tuple: (train_dataset, val_dataset)
    """
    # Create a temporary dataset to load and split all samples once
    temp_dataset = PIDSymbolDataset(data_root, split='train', train_ratio=1.0,
                                   max_samples=max_samples, random_seed=random_seed, prompt_type=prompt_type,
                                   num_negative_points=num_negative_points, skip_validation=skip_validation)

    # Get all samples and create proper split
    all_samples = temp_dataset.samples

    # Re-split with same seed to ensure consistency
    random.seed(random_seed)
    random.shuffle(all_samples)
    split_idx = int(len(all_samples) * train_ratio)

    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]

    print(f"Total samples: {len(all_samples)}, Train: {len(train_samples)}, Val: {len(val_samples)}")

    # Create final datasets with pre-split samples
    train_dataset = PIDSymbolDataset(data_root, split='train',
                                   split_samples=(train_samples, val_samples), prompt_type=prompt_type,
                                   num_negative_points=num_negative_points, skip_validation=skip_validation)
    val_dataset = PIDSymbolDataset(data_root, split='val',
                                 split_samples=(train_samples, val_samples), prompt_type=prompt_type,
                                 num_negative_points=num_negative_points, skip_validation=skip_validation)

    return train_dataset, val_dataset


def main():
    parser = argparse.ArgumentParser(description='Fine-tune SAM2 for P&ID symbol segmentation')
    parser.add_argument('--data_root', type=str,
                       default='/Users/kangdawei/Desktop/Research/INL/MultiAgentsFT-v2/P&IDAgent/DigitizePID_Dataset',
                       help='Path to DigitizePID_Dataset')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--checkpoint', type=str, help='Path to SAM2 checkpoint')
    parser.add_argument('--resume', type=str, help='Resume training from checkpoint')
    parser.add_argument('--output_dir', type=str, default='./sam2_finetuned',
                       help='Directory to save checkpoints')
    parser.add_argument('--max_samples', type=int, help='Max samples for debugging')
    parser.add_argument('--visualize', action='store_true', help='Visualize predictions')
    parser.add_argument('--eval_only', action='store_true', help='Only evaluate model')
    parser.add_argument('--freeze_encoder', action='store_true', help='Freeze image encoder for faster training')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4, help='Gradient accumulation steps')
    parser.add_argument('--mixed_precision', action='store_true', default=True, help='Use mixed precision training')
    parser.add_argument('--debug', action='store_true', help='Enable debug prints for intermediate values')
    parser.add_argument('--prompt_type', type=str, default='points', choices=['points', 'boxes'],
                       help='Type of prompts to use: points (center of bbox) or boxes (default: points)')
    parser.add_argument('--num_negative_points', type=int, default=2,
                       help='Number of negative points per symbol (default: 2, reduce for speed)')
    parser.add_argument('--skip_validation', action='store_true',
                       help='Skip negative point validation for faster training')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create datasets with proper splitting to avoid data leakage
    print(f"Loading datasets with prompt type: {args.prompt_type}")
    if args.prompt_type == 'points':
        print(f"  - Using {args.num_negative_points} negative points per symbol")
        print(f"  - Validation: {'Disabled' if args.skip_validation else 'Enabled'}")
    train_dataset, val_dataset = create_datasets_with_proper_split(
        args.data_root,
        train_ratio=0.9,
        max_samples=args.max_samples,
        random_seed=42,
        prompt_type=args.prompt_type,
        num_negative_points=args.num_negative_points,
        skip_validation=args.skip_validation
    )

    # Create dataloaders with specified batch_size
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_fn
    )

    # Initialize model with gradient accumulation for memory efficiency
    print("Initializing Hugging Face SAM2...")
    model_name = "facebook/sam2-hiera-base-plus"
    gradient_accumulation_steps = args.gradient_accumulation_steps if not args.eval_only else 1
    trainer = SAM2SymbolFineTuner(model_name=model_name, gradient_accumulation_steps=gradient_accumulation_steps, debug=args.debug)
    print(f"Using gradient accumulation with {gradient_accumulation_steps} steps")

    # Setup training
    if not args.eval_only:
        trainer.setup_training(learning_rate=args.lr, freeze_image_encoder=args.freeze_encoder)

    # Resume from checkpoint if specified
    start_epoch = 0
    best_iou = 0
    if args.resume:
        print(f"Resuming from {args.resume}")
        start_epoch, best_iou = trainer.load_checkpoint(args.resume)

    # Evaluation only mode
    if args.eval_only:
        print("Evaluation mode")
        val_loss, val_iou = trainer.evaluate(val_loader)
        print(f"Validation Loss: {val_loss:.4f}, IoU: {val_iou:.4f}")

        if args.visualize:
            # Save visualizations to directory
            visualize_dir = output_dir / "visualizations"
            fig = trainer.visualize_predictions(val_loader, num_samples=5, save_dir=str(visualize_dir))
            plt.close(fig)
            print(f"Visualizations saved to {visualize_dir}")
        return

    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    train_losses = []
    val_losses = []
    val_ious = []

    for epoch in range(start_epoch, args.epochs):
        # Train
        train_loss = trainer.train_epoch(train_loader, epoch)
        train_losses.append(train_loss)

        # Evaluate
        val_loss, val_iou = trainer.evaluate(val_loader)
        val_losses.append(val_loss)
        val_ious.append(val_iou)

        print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}")

        # Save best model
        if val_iou > best_iou:
            best_iou = val_iou
            trainer.save_checkpoint(
                output_dir / "best_model.pth",
                epoch,
                best_iou
            )

        # Save latest checkpoint
        trainer.save_checkpoint(
            output_dir / "latest_checkpoint.pth",
            epoch,
            best_iou
        )

        # Visualize periodically
        if args.visualize and (epoch + 1) % 10 == 0:
            visualize_dir = output_dir / "visualizations"
            fig = trainer.visualize_predictions(val_loader, num_samples=3, save_dir=str(visualize_dir))
            plt.close(fig)

    # Plot training curves
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')

    plt.subplot(1, 2, 2)
    plt.plot(val_ious, label='Val IoU')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    plt.title('Validation IoU')

    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.png", dpi=150)
    print(f"\nTraining complete! Best IoU: {best_iou:.4f}")
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()