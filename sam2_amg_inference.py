#!/usr/bin/env python3
"""
SAM 2 Automatic Mask Generation Inference Script

This script loads a fine-tuned SAM 2 model and uses Automatic Mask Generation (AMG)
to generate all masks for symbols in P&ID images. It includes visualization of
predicted masks overlaid on the original images.

Usage:
    python sam2_amg_inference.py --image_path /path/to/pid/image.jpg --model_path /path/to/best_model.pth
"""

import sys
import os
import argparse
import numpy as np
import torch
import cv2
from pathlib import Path
from typing import Dict, Any, Tuple
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import json

# SAM2 imports from Hugging Face
try:
    from transformers import Sam2Model, Sam2Processor
    print("Hugging Face SAM2 imported successfully")
except ImportError as e:
    print(f"Error importing SAM2 from Hugging Face: {e}")
    print("Please install transformers with SAM2 support:")
    print("pip install transformers>=4.38.0")
    sys.exit(1)


class SAM2AutomaticMaskGenerator:
    """SAM 2 Automatic Mask Generation using fine-tuned model"""

    def __init__(self, model_path: str, model_name: str = "facebook/sam2-hiera-base-plus",
                 device: str = 'cuda', confidence_threshold: float = 0.5):
        """
        Initialize the SAM2 Automatic Mask Generator

        Args:
            model_path: Path to the fine-tuned model checkpoint
            model_name: Base SAM2 model name from Hugging Face
            device: Device to run inference on
            confidence_threshold: Minimum confidence threshold for mask predictions
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.confidence_threshold = confidence_threshold

        print(f"Using device: {self.device}")

        # Initialize model and processor
        self.model = Sam2Model.from_pretrained(model_name).to(self.device)
        self.processor = Sam2Processor.from_pretrained(model_name)

        # Load fine-tuned weights
        self._load_finetuned_weights(model_path)

        # Set model to evaluation mode
        self.model.eval()

        print(f"SAM2 Automatic Mask Generator initialized with confidence threshold: {confidence_threshold}")

    def _load_finetuned_weights(self, model_path: str):
        """Load fine-tuned model weights"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

        checkpoint = torch.load(model_path, map_location=self.device)

        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        # Check for key compatibility
        model_keys = set(self.model.state_dict().keys())
        checkpoint_keys = set(state_dict.keys())

        print(f"Model has {len(model_keys)} parameters")
        print(f"Checkpoint has {len(checkpoint_keys)} parameters")

        missing_keys = model_keys - checkpoint_keys
        unexpected_keys = checkpoint_keys - model_keys

        if missing_keys:
            print(f"WARNING: {len(missing_keys)} missing keys in checkpoint")
            if len(missing_keys) <= 10:
                print(f"Missing keys: {list(missing_keys)[:10]}")

        if unexpected_keys:
            print(f"WARNING: {len(unexpected_keys)} unexpected keys in checkpoint")
            if len(unexpected_keys) <= 10:
                print(f"Unexpected keys: {list(unexpected_keys)[:10]}")

        # Try loading with strict=False to see what happens
        try:
            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"Missing keys during loading: {len(missing)}")
            if unexpected:
                print(f"Unexpected keys during loading: {len(unexpected)}")
        except Exception as e:
            print(f"ERROR loading state dict: {e}")
            print("Attempting to load without fine-tuned weights...")
            return

        print(f"Loaded fine-tuned weights from: {model_path}")

        # Print checkpoint info if available
        if 'epoch' in checkpoint:
            print(f"Model trained for {checkpoint['epoch']} epochs")
        if 'best_iou' in checkpoint:
            print(f"Best validation IoU: {checkpoint['best_iou']:.4f}")

    def generate_grid_points(self, image_shape: Tuple[int, int],
                           points_per_side: int = 32) -> np.ndarray:
        """
        Generate a grid of points for automatic mask generation

        Args:
            image_shape: (height, width) of the image
            points_per_side: Number of points per side of the grid

        Returns:
            Array of points with shape [N, 2] where each point is [x, y]
        """
        h, w = image_shape

        # Create grid of points
        y_coords = np.linspace(0, h - 1, points_per_side, dtype=np.int32)
        x_coords = np.linspace(0, w - 1, points_per_side, dtype=np.int32)

        # Create meshgrid and flatten
        xx, yy = np.meshgrid(x_coords, y_coords)
        points = np.column_stack([xx.ravel(), yy.ravel()])

        return points

    def generate_grid_boxes(self, image_shape: Tuple[int, int],
                          boxes_per_side: int = 16,
                          min_box_size: int = 20,
                          max_box_size: int = 200,
                          size_variations: int = 3) -> np.ndarray:
        """
        Generate a grid of bounding boxes for automatic mask generation

        Args:
            image_shape: (height, width) of the image
            boxes_per_side: Number of box centers per side of the grid
            min_box_size: Minimum box size in pixels
            max_box_size: Maximum box size in pixels
            size_variations: Number of different box sizes per location

        Returns:
            Array of boxes with shape [N, 4] where each box is [x1, y1, x2, y2]
        """
        h, w = image_shape

        # Create grid of box centers
        y_coords = np.linspace(min_box_size, h - min_box_size, boxes_per_side, dtype=np.int32)
        x_coords = np.linspace(min_box_size, w - min_box_size, boxes_per_side, dtype=np.int32)

        # Create different box sizes
        box_sizes = np.linspace(min_box_size, max_box_size, size_variations, dtype=np.int32)

        boxes = []
        for x_center in x_coords:
            for y_center in y_coords:
                for box_size in box_sizes:
                    half_size = box_size // 2
                    x1 = max(0, x_center - half_size)
                    y1 = max(0, y_center - half_size)
                    x2 = min(w, x_center + half_size)
                    y2 = min(h, y_center + half_size)
                    boxes.append([x1, y1, x2, y2])

        return np.array(boxes, dtype=np.float32)

    def generate_masks_from_points(self, image: np.ndarray,
                                 points: np.ndarray) -> Dict[str, Any]:
        """
        Generate masks from a set of prompt points using SAM2

        Args:
            image: Input image as numpy array (H, W, 3)
            points: Array of points with shape [N, 2]

        Returns:
            Dictionary containing masks, scores, and other information
        """
        # Convert image to PIL format for processor
        if len(image.shape) == 3:
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            pil_image = Image.fromarray(image)

        all_masks = []
        all_scores = []
        all_points_used = []

        # Process points in smaller batches for efficiency
        batch_size = 64  # Reduced batch size for stability
        num_batches = (len(points) + batch_size - 1) // batch_size

        print(f"Processing {len(points)} points in {num_batches} batches (batch size: {batch_size})...")

        with torch.no_grad():
            for i in tqdm(range(num_batches), desc="Generating masks"):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(points))
                batch_points = points[start_idx:end_idx]

                # Create input for multiple separate objects (one per point)
                # Each point should be its own object for automatic mask generation
                # Format: [image][object][point][coordinates]
                batch_inputs = []
                for point in batch_points:
                    # Each point as a separate image input
                    point_input = [[[int(point[0]), int(point[1])]]]  # [object][point][coords]
                    batch_inputs.append(point_input)

                batch_scores = []

                # Process each point separately (but in the same batch iteration)
                for j, point_input in enumerate(batch_inputs):
                    point = batch_points[j]

                    # Process single point
                    inputs = self.processor(
                        pil_image,
                        input_points=[point_input],  # Wrap in image-level list
                        return_tensors="pt"
                    )

                    # Move to device
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}

                    # Forward pass
                    outputs = self.model(**inputs)

                    # Process outputs
                    pred_masks = outputs.pred_masks  # [1, 1, 3, 256, 256]
                    iou_scores = outputs.iou_scores  # [1, 1, 3]

                    # Select best mask based on IoU scores
                    best_mask_idx = torch.argmax(iou_scores[0, 0]).item()  # Best of 3 masks
                    mask = pred_masks[0, 0, best_mask_idx]  # [256, 256]
                    score = iou_scores[0, 0, best_mask_idx].item()
                    batch_scores.append(score)

                    # Only keep masks above confidence threshold
                    if score >= self.confidence_threshold:
                        # Upsample mask to original image size
                        orig_h, orig_w = image.shape[:2]
                        mask_upsampled = torch.nn.functional.interpolate(
                            mask.unsqueeze(0).unsqueeze(0).float(),
                            size=(orig_h, orig_w),
                            mode='bilinear',
                            align_corners=False
                        )
                        mask_binary = (torch.sigmoid(mask_upsampled) > 0.5).squeeze().cpu().numpy()

                        all_masks.append(mask_binary)
                        all_scores.append(score)
                        all_points_used.append(point)

                    # Clear intermediate tensors
                    del pred_masks, iou_scores, outputs, inputs

                # Print batch statistics
                if len(batch_scores) > 0:
                    print(f"Batch {i+1}/{num_batches}: {len(batch_scores)} points processed")
                    print(f"  Score range: {min(batch_scores):.4f} - {max(batch_scores):.4f}")
                    print(f"  Scores above threshold ({self.confidence_threshold}): {sum(1 for s in batch_scores if s >= self.confidence_threshold)}")

                # Periodic garbage collection
                if (i + 1) % 5 == 0:
                    torch.cuda.empty_cache()
                    import gc
                    gc.collect()

        print(f"Generated {len(all_masks)} masks above confidence threshold {self.confidence_threshold}")

        return {
            'masks': all_masks,
            'scores': all_scores,
            'points': all_points_used
        }

    def generate_masks_from_boxes(self, image: np.ndarray,
                                boxes: np.ndarray) -> Dict[str, Any]:
        """
        Generate masks from a set of bounding box prompts using SAM2

        Args:
            image: Input image as numpy array (H, W, 3)
            boxes: Array of boxes with shape [N, 4] where each box is [x1, y1, x2, y2]

        Returns:
            Dictionary containing masks, scores, and other information
        """
        # Convert image to PIL format for processor
        if len(image.shape) == 3:
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            pil_image = Image.fromarray(image)

        all_masks = []
        all_scores = []
        all_boxes_used = []

        # Process boxes in batches to avoid memory issues
        batch_size = 64  # Keep original batch size
        num_batches = (len(boxes) + batch_size - 1) // batch_size

        print(f"Processing {len(boxes)} boxes in {num_batches} batches...")

        with torch.no_grad():
            for i in tqdm(range(num_batches), desc="Generating masks from boxes"):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(boxes))
                batch_boxes = boxes[start_idx:end_idx]

                # Convert boxes to the format expected by processor
                # SAM2 expects boxes in format: [[[box1], [box2], ...]] for batch processing
                # Convert numpy types to native Python types
                input_boxes = [[[int(box[0]), int(box[1]), int(box[2]), int(box[3])] for box in batch_boxes]]

                # Process inputs
                inputs = self.processor(
                    pil_image,
                    input_boxes=input_boxes,
                    return_tensors="pt"
                )

                # Move to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Forward pass
                outputs = self.model(**inputs)

                # Process outputs
                pred_masks = outputs.pred_masks  # [1, N, 3, 256, 256]
                iou_scores = outputs.iou_scores  # [1, N, 3]

                # Select best mask for each box based on IoU scores
                best_mask_indices = torch.argmax(iou_scores, dim=-1)  # [1, N]

                # Get the masks and scores for this batch
                batch_size_actual = pred_masks.shape[1]
                for j in range(batch_size_actual):
                    best_idx = best_mask_indices[0, j].item()
                    mask = pred_masks[0, j, best_idx]  # [256, 256]
                    score = iou_scores[0, j, best_idx].item()

                    # Only keep masks above confidence threshold
                    if score >= self.confidence_threshold:
                        # Upsample mask to original image size
                        orig_h, orig_w = image.shape[:2]
                        mask_upsampled = torch.nn.functional.interpolate(
                            mask.unsqueeze(0).unsqueeze(0).float(),
                            size=(orig_h, orig_w),
                            mode='bilinear',
                            align_corners=False
                        )
                        mask_binary = (torch.sigmoid(mask_upsampled) > 0.5).squeeze().cpu().numpy()

                        all_masks.append(mask_binary)
                        all_scores.append(score)
                        all_boxes_used.append(batch_boxes[j])

                # Clear intermediate tensors to free memory
                del pred_masks, iou_scores, best_mask_indices, outputs, inputs
                torch.cuda.empty_cache()

                # Periodic garbage collection for large datasets
                if (i + 1) % 10 == 0:
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()

        print(f"Generated {len(all_masks)} masks above confidence threshold {self.confidence_threshold}")

        return {
            'masks': all_masks,
            'scores': all_scores,
            'boxes': all_boxes_used
        }

    def filter_masks_by_area(self, masks_data: Dict[str, Any],
                           min_area: int = 100, max_area: int = 50000) -> Dict[str, Any]:
        """
        Filter masks by area to remove very small or very large masks

        Args:
            masks_data: Dictionary containing masks, scores, and prompts (points or boxes)
            min_area: Minimum mask area in pixels
            max_area: Maximum mask area in pixels

        Returns:
            Filtered masks dictionary
        """
        filtered_masks = []
        filtered_scores = []

        # Handle both points and boxes
        has_points = 'points' in masks_data
        has_boxes = 'boxes' in masks_data
        filtered_prompts = []

        prompts = masks_data.get('points', masks_data.get('boxes', []))

        areas = []
        for i, (mask, score) in enumerate(zip(masks_data['masks'], masks_data['scores'])):
            area = np.sum(mask)
            areas.append(area)
            if i < 10:  # Debug first 10 masks
                print(f"Mask {i}: area = {area}, score = {score:.4f}, keep = {min_area <= area <= max_area}")

            if min_area <= area <= max_area:
                filtered_masks.append(mask)
                filtered_scores.append(score)
                if i < len(prompts):
                    filtered_prompts.append(prompts[i])

        if len(areas) > 0:
            print(f"Area statistics: min = {min(areas)}, max = {max(areas)}, mean = {np.mean(areas):.1f}")
        print(f"Filtered {len(masks_data['masks'])} -> {len(filtered_masks)} masks by area ({min_area}-{max_area} pixels)")

        result = {
            'masks': filtered_masks,
            'scores': filtered_scores
        }

        if has_points:
            result['points'] = filtered_prompts
        elif has_boxes:
            result['boxes'] = filtered_prompts

        return result

    def filter_masks_by_content(self, image: np.ndarray, masks_data: Dict[str, Any],
                               white_threshold: float = 0.85,
                               line_threshold: float = 0.7,
                               min_std_dev: float = 10.0) -> Dict[str, Any]:
        """
        Filter out masks that contain mostly white space or only simple lines.
        This removes low-quality masks that don't represent actual components.

        Args:
            image: Original image as numpy array (H, W, 3) or (H, W)
            masks_data: Dictionary containing masks, scores, and prompts
            white_threshold: Fraction of white/near-white pixels to consider mask as whitespace (0-1)
            line_threshold: If mask is very elongated (aspect ratio), consider it a line
            min_std_dev: Minimum standard deviation of pixel values (detects uniform regions)

        Returns:
            Filtered masks dictionary
        """
        filtered_masks = []
        filtered_scores = []

        # Handle both points and boxes
        has_points = 'points' in masks_data
        has_boxes = 'boxes' in masks_data
        filtered_prompts = []

        prompts = masks_data.get('points', masks_data.get('boxes', []))

        # Convert image to grayscale if needed
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image

        print(f"Filtering masks by content (whitespace and line detection)...")

        whitespace_count = 0
        line_count = 0
        low_variance_count = 0

        for i, (mask, score) in enumerate(zip(masks_data['masks'], masks_data['scores'])):
            keep_mask = True
            reason = ""

            # Extract the masked region
            masked_pixels = gray_image[mask]

            if len(masked_pixels) == 0:
                keep_mask = False
                reason = "empty"
            else:
                # Check 1: Is it mostly whitespace?
                # Count pixels that are near white (> 240 for 8-bit images)
                white_pixels = np.sum(masked_pixels > 240)
                white_ratio = white_pixels / len(masked_pixels)

                if white_ratio > white_threshold:
                    keep_mask = False
                    reason = "whitespace"
                    whitespace_count += 1

                # Check 2: Is it just a simple line? (very elongated mask)
                if keep_mask:
                    bbox = get_mask_bbox(mask)
                    width = bbox[2] - bbox[0]
                    height = bbox[3] - bbox[1]

                    # Calculate aspect ratio and bbox fill ratio
                    aspect_ratio = max(width, height) / max(min(width, height), 1)
                    mask_area = np.sum(mask)
                    bbox_area = width * height
                    fill_ratio = mask_area / max(bbox_area, 1)

                    # Very elongated with low fill ratio = likely a line
                    if aspect_ratio > 10 and fill_ratio < line_threshold:
                        keep_mask = False
                        reason = "line"
                        line_count += 1

                # Check 3: Does it have sufficient variance? (not uniform/blank)
                if keep_mask:
                    std_dev = np.std(masked_pixels)

                    if std_dev < min_std_dev:
                        keep_mask = False
                        reason = "low_variance"
                        low_variance_count += 1

            # Debug first 10 filtered masks
            if not keep_mask and (whitespace_count + line_count + low_variance_count) <= 10:
                print(f"  Filtered mask {i}: {reason} (score={score:.4f})")

            if keep_mask:
                filtered_masks.append(mask)
                filtered_scores.append(score)
                if i < len(prompts):
                    filtered_prompts.append(prompts[i])

        print(f"Filtered {len(masks_data['masks'])} -> {len(filtered_masks)} masks by content")
        print(f"  Removed: {whitespace_count} whitespace, {line_count} lines, {low_variance_count} low-variance")

        result = {
            'masks': filtered_masks,
            'scores': filtered_scores
        }

        if has_points:
            result['points'] = filtered_prompts
        elif has_boxes:
            result['boxes'] = filtered_prompts

        return result

    def remove_duplicate_masks(self, masks_data: Dict[str, Any],
                             iou_threshold: float = 0.8,
                             proximity_threshold: int = 20) -> Dict[str, Any]:
        """
        Combine overlapping masks and nearby masks that belong to the same symbol.
        Uses multiple criteria to determine if masks should be merged:
        1. High IoU overlap (duplicate/overlapping masks)
        2. Spatial proximity (nearby masks with small gaps)
        3. Bounding box containment (one mask inside another)

        This handles cases where symbol parts are disconnected but belong together.

        Args:
            masks_data: Dictionary containing masks, scores, and prompts (points or boxes)
            iou_threshold: IoU threshold for combining overlapping masks
            proximity_threshold: Maximum distance (pixels) between bboxes to consider merging

        Returns:
            Dictionary with combined masks
        """
        masks = masks_data['masks']
        scores = masks_data['scores']

        # Handle both points and boxes
        has_points = 'points' in masks_data
        has_boxes = 'boxes' in masks_data
        prompts = masks_data.get('points', masks_data.get('boxes', []))

        if len(masks) == 0:
            return masks_data

        # Pre-compute bounding boxes and areas for spatial reasoning
        print("Pre-computing mask properties...")
        mask_bboxes = []
        mask_areas = []
        for mask in masks:
            bbox = get_mask_bbox(mask)
            area = np.sum(mask)
            mask_bboxes.append(bbox)
            mask_areas.append(area)

        # Sort by score (descending)
        sorted_indices = np.argsort(scores)[::-1]

        keep_masks = []
        keep_scores = []
        keep_prompts = []
        keep_bboxes = []
        keep_areas = []

        print(f"Combining masks with IoU threshold: {iou_threshold}, proximity: {proximity_threshold}px")

        for i in tqdm(sorted_indices, desc="Combining masks"):
            current_mask = masks[i]
            current_score = scores[i]
            current_bbox = mask_bboxes[i]
            current_area = mask_areas[i]
            merged = False

            # Check against all kept masks
            for j, kept_mask in enumerate(keep_masks):
                kept_bbox = keep_bboxes[j]
                kept_area = keep_areas[j]

                # Criterion 1: Calculate IoU for overlapping masks
                intersection = np.logical_and(current_mask, kept_mask).sum()
                union = current_area + kept_area - intersection
                iou = intersection / union if union > 0 else 0

                should_merge = False

                # Check if masks should be merged based on IoU
                if iou > iou_threshold:
                    should_merge = True

                # Criterion 2: Check spatial proximity for nearby non-overlapping masks
                elif intersection == 0 or iou < 0.1:  # Little to no overlap
                    # Calculate distance between bounding boxes
                    bbox_distance = self._bbox_distance(current_bbox, kept_bbox)

                    if bbox_distance <= proximity_threshold:
                        # Additional checks to avoid merging unrelated nearby masks
                        # Check if masks are similar in size (relaxed to 5x for more aggressive merging)
                        size_ratio = max(current_area, kept_area) / max(min(current_area, kept_area), 1)

                        # Check bounding box aspect ratio compatibility
                        current_aspect = self._bbox_aspect_ratio(current_bbox)
                        kept_aspect = self._bbox_aspect_ratio(kept_bbox)
                        aspect_ratio = max(current_aspect, kept_aspect) / max(min(current_aspect, kept_aspect), 0.01)

                        # Check alignment: are the masks roughly aligned (horizontally or vertically)?
                        # This helps distinguish between parts of the same symbol vs separate symbols
                        alignment_score = self._compute_alignment(current_bbox, kept_bbox)

                        # Check if merged bbox would be unreasonably large
                        # This prevents merging distant components
                        merged_bbox = self._merge_bboxes(current_bbox, kept_bbox)
                        merged_bbox_area = (merged_bbox[2] - merged_bbox[0]) * (merged_bbox[3] - merged_bbox[1])
                        combined_mask_area = current_area + kept_area
                        # If merged bbox is much larger than combined mask area, they're likely separate
                        bbox_efficiency = combined_mask_area / max(merged_bbox_area, 1)

                        # More aggressive merging conditions:
                        # Option 1: Very good alignment with reasonable size/aspect ratio
                        if (alignment_score > 0.5 and  # Reduced from 0.6
                            size_ratio < 5.0 and       # Increased from 3.0
                            aspect_ratio < 4.0 and     # Increased from 2.5
                            bbox_efficiency > 0.2):    # Reduced from 0.3
                            should_merge = True
                        # Option 2: Very close proximity, even with moderate alignment
                        elif (bbox_distance < proximity_threshold * 0.6 and  # Increased from 0.5
                              alignment_score > 0.4 and    # More lenient
                              size_ratio < 6.0 and         # Even more lenient for close masks
                              bbox_efficiency > 0.15):     # Lower threshold for very close masks
                            should_merge = True
                        # Option 3: Good alignment with very close proximity (almost touching)
                        elif (alignment_score > 0.7 and
                              bbox_distance < proximity_threshold * 0.8 and
                              size_ratio < 8.0):  # Very lenient for well-aligned masks
                            should_merge = True

                # Criterion 3: Check if one bbox is largely contained in another (expanded)
                # More aggressive: reduced threshold from 0.7 to 0.6
                elif self._bbox_containment(current_bbox, kept_bbox, expansion=proximity_threshold) > 0.6:
                    should_merge = True

                if should_merge:
                    # Combine the masks using logical OR
                    merged_mask = np.logical_or(current_mask, kept_mask)
                    keep_masks[j] = merged_mask

                    # Keep the higher score
                    keep_scores[j] = max(keep_scores[j], current_score)

                    # Update bounding box to encompass both masks
                    keep_bboxes[j] = self._merge_bboxes(current_bbox, kept_bbox)
                    keep_areas[j] = np.sum(merged_mask)

                    merged = True
                    break

            # If not merged with any existing mask, add as new mask
            if not merged:
                keep_masks.append(current_mask)
                keep_scores.append(current_score)
                keep_bboxes.append(current_bbox)
                keep_areas.append(current_area)
                if i < len(prompts):
                    keep_prompts.append(prompts[i])

        print(f"Combined {len(masks)} -> {len(keep_masks)} masks")

        result = {
            'masks': keep_masks,
            'scores': keep_scores
        }

        if has_points:
            result['points'] = keep_prompts
        elif has_boxes:
            result['boxes'] = keep_prompts

        return result

    def _bbox_distance(self, bbox1: Tuple[int, int, int, int],
                       bbox2: Tuple[int, int, int, int]) -> float:
        """
        Calculate minimum distance between two bounding boxes.
        Returns 0 if boxes overlap, otherwise returns the closest distance.

        Args:
            bbox1: First bounding box (x1, y1, x2, y2)
            bbox2: Second bounding box (x1, y1, x2, y2)

        Returns:
            Minimum distance in pixels
        """
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        # Calculate horizontal distance
        if x2_1 < x1_2:
            dx = x1_2 - x2_1
        elif x2_2 < x1_1:
            dx = x1_1 - x2_2
        else:
            dx = 0

        # Calculate vertical distance
        if y2_1 < y1_2:
            dy = y1_2 - y2_1
        elif y2_2 < y1_1:
            dy = y1_1 - y2_2
        else:
            dy = 0

        # Return Euclidean distance
        return np.sqrt(dx**2 + dy**2)

    def _bbox_aspect_ratio(self, bbox: Tuple[int, int, int, int]) -> float:
        """
        Calculate aspect ratio of a bounding box.

        Args:
            bbox: Bounding box (x1, y1, x2, y2)

        Returns:
            Aspect ratio (width / height)
        """
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        return width / max(height, 1)

    def _compute_alignment(self, bbox1: Tuple[int, int, int, int],
                          bbox2: Tuple[int, int, int, int]) -> float:
        """
        Compute alignment score between two bounding boxes.
        Higher score means better alignment (horizontally or vertically).
        This helps identify if masks are parts of the same symbol vs separate components.

        Returns a score from 0 to 1:
        - 1.0: Perfect alignment (centers aligned horizontally or vertically)
        - 0.5: Moderate alignment
        - 0.0: No alignment (diagonal arrangement)

        Args:
            bbox1: First bounding box (x1, y1, x2, y2)
            bbox2: Second bounding box (x1, y1, x2, y2)

        Returns:
            Alignment score (0 to 1)
        """
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        # Calculate centers
        cx1, cy1 = (x1_1 + x2_1) / 2, (y1_1 + y2_1) / 2
        cx2, cy2 = (x1_2 + x2_2) / 2, (y1_2 + y2_2) / 2

        # Calculate dimensions
        w1, h1 = x2_1 - x1_1, y2_1 - y1_1
        w2, h2 = x2_2 - x1_2, y2_2 - y1_2

        # Calculate horizontal and vertical offsets
        dx = abs(cx1 - cx2)
        dy = abs(cy1 - cy2)

        # Normalize by average dimensions
        avg_width = (w1 + w2) / 2
        avg_height = (h1 + h2) / 2

        # Calculate horizontal alignment (good if dy is small relative to dx)
        # This detects masks side-by-side horizontally
        if dx > 0:
            horizontal_alignment = 1.0 - min(dy / max(avg_height, 1), 1.0)
        else:
            horizontal_alignment = 1.0

        # Calculate vertical alignment (good if dx is small relative to dy)
        # This detects masks stacked vertically
        if dy > 0:
            vertical_alignment = 1.0 - min(dx / max(avg_width, 1), 1.0)
        else:
            vertical_alignment = 1.0

        # Return the best alignment score
        alignment_score = max(horizontal_alignment, vertical_alignment)

        # Also check for overlap in at least one dimension
        # If bboxes don't overlap in either x or y dimension, reduce score
        x_overlap = not (x2_1 < x1_2 or x2_2 < x1_1)
        y_overlap = not (y2_1 < y1_2 or y2_2 < y1_1)

        # If there's overlap in one dimension, it's more likely the same component
        if x_overlap or y_overlap:
            alignment_score = min(alignment_score * 1.2, 1.0)  # Boost score
        else:
            # No overlap in either dimension - they might be truly separate
            alignment_score = alignment_score * 0.8  # Reduce score

        return alignment_score

    def _bbox_containment(self, bbox1: Tuple[int, int, int, int],
                          bbox2: Tuple[int, int, int, int],
                          expansion: int = 0) -> float:
        """
        Calculate how much bbox1 is contained within bbox2 (with optional expansion).
        Useful for detecting if smaller masks are parts of larger symbols.

        Args:
            bbox1: First bounding box (x1, y1, x2, y2)
            bbox2: Second bounding box (x1, y1, x2, y2), optionally expanded
            expansion: Number of pixels to expand bbox2 by

        Returns:
            Containment ratio (0 to 1), where 1 means bbox1 is fully inside expanded bbox2
        """
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        # Expand bbox2
        x1_2_exp = x1_2 - expansion
        y1_2_exp = y1_2 - expansion
        x2_2_exp = x2_2 + expansion
        y2_2_exp = y2_2 + expansion

        # Calculate intersection with expanded bbox2
        x1_i = max(x1_1, x1_2_exp)
        y1_i = max(y1_1, y1_2_exp)
        x2_i = min(x2_1, x2_2_exp)
        y2_i = min(y2_1, y2_2_exp)

        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0

        intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
        bbox1_area = (x2_1 - x1_1) * (y2_1 - y1_1)

        return intersection_area / max(bbox1_area, 1)

    def _merge_bboxes(self, bbox1: Tuple[int, int, int, int],
                      bbox2: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """
        Merge two bounding boxes by taking the minimum and maximum coordinates.

        Args:
            bbox1: First bounding box (x1, y1, x2, y2)
            bbox2: Second bounding box (x1, y1, x2, y2)

        Returns:
            Merged bounding box
        """
        x1 = min(bbox1[0], bbox2[0])
        y1 = min(bbox1[1], bbox2[1])
        x2 = max(bbox1[2], bbox2[2])
        y2 = max(bbox1[3], bbox2[3])
        return (x1, y1, x2, y2)

    def generate_automatic_masks(self, image: np.ndarray,
                               prompt_type: str = 'points',
                               points_per_side: int = 32,
                               boxes_per_side: int = 16,
                               min_box_size: int = 20,
                               max_box_size: int = 200,
                               size_variations: int = 3,
                               min_area: int = 100,
                               max_area: int = 50000,
                               iou_threshold: float = 0.8,
                               proximity_threshold: int = 20,
                               filter_content: bool = True,
                               white_threshold: float = 0.85,
                               line_threshold: float = 0.7,
                               min_std_dev: float = 10.0) -> Dict[str, Any]:
        """
        Generate automatic masks for the entire image

        Args:
            image: Input image as numpy array
            prompt_type: Type of prompt to use ('points' or 'boxes')
            points_per_side: Number of points per side for the grid (for points mode)
            boxes_per_side: Number of box centers per side (for boxes mode)
            min_box_size: Minimum box size in pixels (for boxes mode)
            max_box_size: Maximum box size in pixels (for boxes mode)
            size_variations: Number of different box sizes per location (for boxes mode)
            min_area: Minimum mask area in pixels
            max_area: Maximum mask area in pixels
            iou_threshold: IoU threshold for deduplication
            proximity_threshold: Maximum distance (pixels) between bboxes to consider merging
            filter_content: Whether to filter masks by content (whitespace/lines)
            white_threshold: Threshold for whitespace filtering (0-1)
            line_threshold: Threshold for line detection (0-1)
            min_std_dev: Minimum standard deviation for content filtering

        Returns:
            Dictionary containing final masks and metadata
        """
        print(f"Starting automatic mask generation with {prompt_type} prompts...")

        if prompt_type == 'points':
            # Generate grid of prompt points
            prompts = self.generate_grid_points(image.shape[:2], points_per_side)
            print(f"Generated {len(prompts)} grid points")

            # Generate masks from points
            masks_data = self.generate_masks_from_points(image, prompts)
        elif prompt_type == 'boxes':
            # Generate grid of bounding boxes
            prompts = self.generate_grid_boxes(image.shape[:2], boxes_per_side,
                                              min_box_size, max_box_size, size_variations)
            print(f"Generated {len(prompts)} bounding boxes")

            # Generate masks from boxes
            masks_data = self.generate_masks_from_boxes(image, prompts)
        else:
            raise ValueError(f"Invalid prompt_type: {prompt_type}. Must be 'points' or 'boxes'")

        # Filter by area
        if min_area > 0 or max_area < float('inf'):
            masks_data = self.filter_masks_by_area(masks_data, min_area, max_area)

        # Filter by content (whitespace and lines) - NEW!
        if filter_content:
            masks_data = self.filter_masks_by_content(image, masks_data,
                                                     white_threshold, line_threshold, min_std_dev)

        # Remove duplicates and combine nearby masks
        if iou_threshold < 1.0:
            masks_data = self.remove_duplicate_masks(masks_data, iou_threshold, proximity_threshold)

        print(f"Final result: {len(masks_data['masks'])} masks")

        return masks_data


def visualize_masks(image: np.ndarray, masks_data: Dict[str, Any],
                   save_path: str = None, show_prompts: bool = True,
                   alpha: float = 0.4) -> plt.Figure:
    """
    Visualize masks overlaid on the original image

    Args:
        image: Original image
        masks_data: Dictionary containing masks, scores, and prompts (points or boxes)
        save_path: Path to save the visualization
        show_prompts: Whether to show the prompt points/boxes
        alpha: Transparency of mask overlay

    Returns:
        Matplotlib figure object
    """
    masks = masks_data['masks']
    scores = masks_data['scores']

    # Handle both points and boxes
    has_points = 'points' in masks_data
    has_boxes = 'boxes' in masks_data
    prompts = masks_data.get('points', masks_data.get('boxes', []))

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'SAM2 Automatic Mask Generation Results ({len(masks)} masks)', fontsize=16)

    # Convert BGR to RGB for display
    if len(image.shape) == 3 and image.shape[2] == 3:
        display_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        display_image = image

    # 1. Original image
    axes[0, 0].imshow(display_image)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    # 2. Image with prompts (points or boxes)
    axes[0, 1].imshow(display_image)
    if show_prompts and len(prompts) > 0:
        if has_points:
            points_array = np.array(prompts)
            axes[0, 1].scatter(points_array[:, 0], points_array[:, 1],
                              c='red', s=10, alpha=0.7)
            axes[0, 1].set_title(f'Prompt Points ({len(prompts)} points)')
        elif has_boxes:
            # Draw bounding boxes
            import matplotlib.patches as patches
            for box in prompts[:100]:  # Show first 100 boxes to avoid clutter
                x1, y1, x2, y2 = box
                rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                        linewidth=1, edgecolor='red',
                                        facecolor='none', alpha=0.5)
                axes[0, 1].add_patch(rect)
            axes[0, 1].set_title(f'Prompt Boxes ({len(prompts)} boxes, showing first 100)')
    else:
        axes[0, 1].set_title('No prompts')
    axes[0, 1].axis('off')

    # 3. All masks combined
    if len(masks) > 0:
        # Create combined mask with different colors
        combined_mask = np.zeros((*image.shape[:2], 3), dtype=np.float32)

        # Generate colors for masks
        colors = plt.cm.tab20(np.linspace(0, 1, min(len(masks), 20)))
        if len(masks) > 20:
            # Use random colors for more than 20 masks
            np.random.seed(42)
            colors = np.random.rand(len(masks), 3)

        for i, mask in enumerate(masks):
            color = colors[i % len(colors)][:3]
            combined_mask[mask] = color

        # Show combined masks
        axes[1, 0].imshow(combined_mask)
        axes[1, 0].set_title('All Masks')
        axes[1, 0].axis('off')

        # 4. Overlay on original image
        overlay = display_image.copy().astype(np.float32)

        for i, mask in enumerate(masks):
            color = colors[i % len(colors)][:3] * 255
            overlay[mask] = (1 - alpha) * overlay[mask] + alpha * color

        axes[1, 1].imshow(overlay.astype(np.uint8))
        axes[1, 1].set_title(f'Overlay (α={alpha})')
        axes[1, 1].axis('off')
    else:
        # No masks found
        axes[1, 0].text(0.5, 0.5, 'No masks found', ha='center', va='center',
                       transform=axes[1, 0].transAxes, fontsize=14)
        axes[1, 0].axis('off')
        axes[1, 1].imshow(display_image)
        axes[1, 1].set_title('No overlay (no masks)')
        axes[1, 1].axis('off')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")

    return fig


def save_combined_masks(image: np.ndarray, masks_data: Dict[str, Any],
                       output_dir: str, image_name: str):
    """
    Save combined masks, overlay, and mask data for later use

    Args:
        image: Original image
        masks_data: Dictionary containing masks and metadata
        output_dir: Output directory
        image_name: Base name for the image (without extension)
    """
    masks = masks_data['masks']
    scores = masks_data['scores']

    if len(masks) == 0:
        print("No masks to save")
        return

    print(f"Saving {len(masks)} masks as combined images and data...")

    # Convert image for overlay
    if len(image.shape) == 3 and image.shape[2] == 3:
        display_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        display_image = image

    # Generate colors
    colors = plt.cm.tab20(np.linspace(0, 1, min(len(masks), 20)))
    if len(masks) > 20:
        np.random.seed(42)
        colors = np.random.rand(len(masks), 3)

    # 1. Save combined mask visualization (colored masks)
    combined_mask = np.zeros((*image.shape[:2], 3), dtype=np.float32)
    for i, mask in enumerate(masks):
        color = colors[i % len(colors)][:3]
        combined_mask[mask] = color

    combined_mask_path = Path(output_dir) / f"{image_name}_all_masks.png"
    Image.fromarray((combined_mask * 255).astype(np.uint8)).save(combined_mask_path)
    print(f"Combined masks visualization saved to: {combined_mask_path}")

    # 2. Save overlay on original image
    overlay = display_image.copy().astype(np.float32)
    alpha = 0.4
    for i, mask in enumerate(masks):
        color = colors[i % len(colors)][:3] * 255
        overlay[mask] = (1 - alpha) * overlay[mask] + alpha * color

    overlay_path = Path(output_dir) / f"{image_name}_overlay.png"
    Image.fromarray(overlay.astype(np.uint8)).save(overlay_path)
    print(f"Combined overlay saved to: {overlay_path}")

    # 3. Save all masks as a numpy file for later processing
    masks_array = np.array(masks)  # Shape: (num_masks, height, width)
    masks_path = Path(output_dir) / f"{image_name}_masks.npz"

    # Save with metadata
    np.savez_compressed(
        masks_path,
        masks=masks_array,
        scores=np.array(scores),
        image_shape=image.shape,
        num_masks=len(masks)
    )
    print(f"Masks data saved to: {masks_path}")
    print(f"  - {len(masks)} masks with shape {masks_array.shape}")
    print(f"  - Use: data = np.load('{masks_path.name}'); masks = data['masks']; scores = data['scores']")


def save_results_json(masks_data: Dict[str, Any], output_path: str,
                     image_path: str, processing_params: Dict[str, Any]):
    """
    Save results in JSON format for further processing

    Args:
        masks_data: Dictionary containing masks and metadata
        output_path: Path to save JSON file
        image_path: Path of the processed image
        processing_params: Parameters used for processing
    """
    # Convert numpy arrays to lists for JSON serialization
    results = {
        'image_path': str(image_path),
        'num_masks': len(masks_data['masks']),
        'processing_params': processing_params,
        'masks_info': []
    }

    # Handle both points and boxes
    has_points = 'points' in masks_data
    has_boxes = 'boxes' in masks_data
    prompts = masks_data.get('points', masks_data.get('boxes', []))

    for i, (mask, score) in enumerate(zip(masks_data['masks'], masks_data['scores'])):
        # Calculate mask properties
        area = np.sum(mask)
        bbox = get_mask_bbox(mask)

        mask_info = {
            'id': i,
            'score': float(score),
            'area': int(area),
            'bbox': [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
        }

        # Add prompt information
        if i < len(prompts):
            if has_points:
                mask_info['prompt_point'] = [int(prompts[i][0]), int(prompts[i][1])]
            elif has_boxes:
                mask_info['prompt_box'] = [int(prompts[i][0]), int(prompts[i][1]),
                                          int(prompts[i][2]), int(prompts[i][3])]

        results['masks_info'].append(mask_info)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to JSON: {output_path}")


def get_mask_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Get bounding box of a binary mask

    Args:
        mask: Binary mask array

    Returns:
        Tuple of (x1, y1, x2, y2)
    """
    if not np.any(mask):
        return (0, 0, 0, 0)

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]

    return (x1, y1, x2 + 1, y2 + 1)


def main():
    parser = argparse.ArgumentParser(description='SAM2 Automatic Mask Generation for P&ID Images')
    parser.add_argument('--image_path', type=str, required=True,
                       help='Path to input P&ID image')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to fine-tuned SAM2 model checkpoint')
    parser.add_argument('--output_dir', type=str, default='./sam2_amg_results',
                       help='Output directory for results')
    parser.add_argument('--prompt_type', type=str, default='points', choices=['points', 'boxes'],
                       help='Type of prompt to use: "points" or "boxes"')
    parser.add_argument('--points_per_side', type=int, default=16,
                       help='Number of points per side for grid generation (for points mode)')
    parser.add_argument('--boxes_per_side', type=int, default=16,
                       help='Number of box centers per side (for boxes mode)')
    parser.add_argument('--min_box_size', type=int, default=20,
                       help='Minimum box size in pixels (for boxes mode)')
    parser.add_argument('--max_box_size', type=int, default=200,
                       help='Maximum box size in pixels (for boxes mode)')
    parser.add_argument('--size_variations', type=int, default=3,
                       help='Number of different box sizes per location (for boxes mode)')
    parser.add_argument('--confidence_threshold', type=float, default=0.5,
                       help='Confidence threshold for mask filtering')
    parser.add_argument('--min_area', type=int, default=9,
                       help='Minimum mask area in pixels')
    parser.add_argument('--max_area', type=int, default=50000,
                       help='Maximum mask area in pixels')
    parser.add_argument('--iou_threshold', type=float, default=0.8,
                       help='IoU threshold for duplicate removal')
    parser.add_argument('--proximity_threshold', type=int, default=20,
                       help='Maximum distance (pixels) between bounding boxes to consider merging nearby masks')
    parser.add_argument('--filter_content', action='store_true', default=True,
                       help='Filter masks containing mostly whitespace or only lines')
    parser.add_argument('--white_threshold', type=float, default=0.85,
                       help='Threshold for whitespace filtering (0-1), fraction of white pixels')
    parser.add_argument('--line_threshold', type=float, default=0.7,
                       help='Threshold for line detection (0-1), bbox fill ratio for elongated masks')
    parser.add_argument('--min_std_dev', type=float, default=10.0,
                       help='Minimum standard deviation of pixel values for content filtering')
    parser.add_argument('--save_masks', action='store_true',
                       help='Save combined masks visualization, overlay, and mask data')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to run inference on')

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.image_path):
        raise FileNotFoundError(f"Image not found: {args.image_path}")

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model not found: {args.model_path}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load image
    print(f"Loading image: {args.image_path}")
    image = cv2.imread(args.image_path)
    if image is None:
        raise ValueError(f"Could not load image: {args.image_path}")

    print(f"Image shape: {image.shape}")

    # Initialize SAM2 AMG
    print("Initializing SAM2 Automatic Mask Generator...")
    amg = SAM2AutomaticMaskGenerator(
        model_path=args.model_path,
        device=args.device,
        confidence_threshold=args.confidence_threshold
    )

    # Generate masks
    print(f"Generating automatic masks using {args.prompt_type} prompts...")
    masks_data = amg.generate_automatic_masks(
        image=image,
        prompt_type=args.prompt_type,
        points_per_side=args.points_per_side,
        boxes_per_side=args.boxes_per_side,
        min_box_size=args.min_box_size,
        max_box_size=args.max_box_size,
        size_variations=args.size_variations,
        min_area=args.min_area,
        max_area=args.max_area,
        iou_threshold=args.iou_threshold,
        proximity_threshold=args.proximity_threshold,
        filter_content=args.filter_content,
        white_threshold=args.white_threshold,
        line_threshold=args.line_threshold,
        min_std_dev=args.min_std_dev
    )

    # Get image name for output files
    image_name = Path(args.image_path).stem

    # Create visualization
    print("Creating visualization...")
    viz_path = output_dir / f"{image_name}_amg_results.png"
    fig = visualize_masks(image, masks_data, str(viz_path))
    plt.close(fig)

    # Save combined masks and data if requested
    if args.save_masks:
        save_combined_masks(image, masks_data, str(output_dir), image_name)

    # Save results as JSON
    processing_params = {
        'prompt_type': args.prompt_type,
        'confidence_threshold': args.confidence_threshold,
        'min_area': args.min_area,
        'max_area': args.max_area,
        'iou_threshold': args.iou_threshold
    }

    # Add prompt-specific parameters
    if args.prompt_type == 'points':
        processing_params['points_per_side'] = args.points_per_side
    else:
        processing_params['boxes_per_side'] = args.boxes_per_side
        processing_params['min_box_size'] = args.min_box_size
        processing_params['max_box_size'] = args.max_box_size
        processing_params['size_variations'] = args.size_variations

    json_path = output_dir / f"{image_name}_results.json"
    save_results_json(masks_data, str(json_path), args.image_path, processing_params)

    print(f"\nProcessing complete!")
    print(f"Generated {len(masks_data['masks'])} masks")
    print(f"Results saved to: {output_dir}")
    print(f"Main visualization: {viz_path}")
    print(f"Results JSON: {json_path}")


if __name__ == "__main__":
    main()