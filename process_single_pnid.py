#!/usr/bin/env python3
"""
Comprehensive P&ID Processing Pipeline

This script processes a single P&ID image through the complete pipeline:
1. Symbol Detection (SAM2 AMG or YOLO)
2. Symbol Classification & Editing (Interactive)
3. Text Detection (PaddleOCR)
4. Text Editing (Interactive)
5. Line Detection (Morphology + Hough + LSD)
6. Line Editing (Interactive)
7. Final Digitization to LLM-friendly JSON format
8. Visualization Generation (Automatic)

Usage:
    # Full pipeline with SAM2 detection (default)
    python process_single_pnid.py --image pnid.jpg --out results/ --sam2_model path/to/model.pth --interactive

    # Full pipeline with YOLO detection
    python process_single_pnid.py --image pnid.jpg --out results/ --detector yolo --yolo_model path/to/best.pt --interactive

    # Non-interactive with SAM2
    python process_single_pnid.py --image pnid.jpg --out results/ --sam2_model path/to/model.pth

    # Non-interactive with YOLO
    python process_single_pnid.py --image pnid.jpg --out results/ --detector yolo --yolo_model path/to/best.pt

    # Skip visualization
    python process_single_pnid.py --image pnid.jpg --out results/ --detector yolo --yolo_model path/to/best.pt --no-visualization
"""

import sys
import os
import argparse
import json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive P&ID Processing Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required
    parser.add_argument("--image", required=True, help="Path to input P&ID image")
    parser.add_argument("--out", required=True, help="Output directory")

    # Symbol Detection Model Selection
    parser.add_argument("--detector", type=str, default="yolo", choices=["sam2", "yolo"],
                       help="Detection model to use: 'sam2' for SAM2 AMG, 'yolo' for YOLO")

    # SAM2 Symbol Detection
    parser.add_argument("--sam2_model", default=None, help="Path to fine-tuned SAM2 model checkpoint (required if --detector=sam2)")
    parser.add_argument("--sam2_base", default="facebook/sam2-hiera-base-plus",
                       help="Base SAM2 model name from Hugging Face")
    parser.add_argument("--confidence", type=float, default=0.5,
                       help="Confidence threshold for symbol detection")

    # YOLO Symbol Detection
    parser.add_argument("--yolo_model", default="/scratch/user/u.kw178339/INL/PID_Agent/runs/train/pid_symbols_20251204_110352/weights/best.pt",
                       help="Path to trained YOLO model checkpoint (required if --detector=yolo)")

    # Symbol Classification
    parser.add_argument("--embedding_model", type=str, default='clip',
                       choices=['clip', 'dinov2', 'vit'],
                       help="Model to use for symbol embeddings")
    parser.add_argument("--clustering_method", type=str, default='hdbscan',
                       choices=['hdbscan', 'kmeans'],
                       help="Clustering method for symbols")
    parser.add_argument("--n_clusters", type=int, default=None,
                       help="Number of clusters (for kmeans)")
    parser.add_argument("--sensitivity", type=str, default='high',
                       choices=['low', 'medium', 'high', 'very_high'],
                       help="Clustering sensitivity")

    # Text Detection (from process_text_lines.py)
    parser.add_argument("--target-width", type=int, default=7168,
                       help="Resize image to this width for text detection")
    parser.add_argument("--lang", default="en",
                       help="PaddleOCR language")
    parser.add_argument("--gpu", action="store_true",
                       help="Use GPU for PaddleOCR")
    parser.add_argument("--nms-iou", type=float, default=0.3,
                       help="IoU threshold for merging text boxes")

    # Line Detection
    parser.add_argument("--suppress-text", action="store_true", default=True,
                       help="Remove text regions before line detection")
    parser.add_argument("--suppress-symbols", action="store_true",
                       help="Remove large symbols before line detection")
    parser.add_argument("--min-len", type=int, default=22,
                       help="Minimum line segment length")
    parser.add_argument("--line-mode", choices=["solid", "solid_dashed"], default="solid",
                       help="Line detection mode: solid only or solid+dashed")
    parser.add_argument("--notes-right-frac", type=float, default=0.0,
                       help="Fraction of right side to ignore (notes panel)")

    # Interactive Editing
    parser.add_argument("--interactive", action="store_true",
                       help="Launch interactive editors for symbols, text, and lines")
    parser.add_argument("--skip-symbol-edit", action="store_true",
                       help="Skip interactive symbol mask editing")
    parser.add_argument("--skip-text-edit", action="store_true",
                       help="Skip interactive text editing")
    parser.add_argument("--skip-line-edit", action="store_true",
                       help="Skip interactive line editing")

    # Digitization Parameters
    parser.add_argument("--max-text-distance", type=float, default=200.0,
                       help="Maximum distance (pixels) for text-to-symbol association")
    parser.add_argument("--max-line-distance", type=float, default=300.0,
                       help="Maximum distance (pixels) for line-to-symbol connection (lenient)")

    # Visualization Parameters
    parser.add_argument("--no-visualization", action="store_true",
                       help="Skip visualization generation")
    parser.add_argument("--no-labels", action="store_true",
                       help="Don't show category labels on symbols in visualization")
    parser.add_argument("--vis-line-width", type=int, default=2,
                       help="Width of connection lines in visualization")
    parser.add_argument("--vis-bbox-width", type=int, default=3,
                       help="Width of bounding box lines in visualization")

    # General
    parser.add_argument("--device", type=str, default='cuda',
                       help="Device to run on (cuda or cpu)")

    args = parser.parse_args()

    # Validate input
    if not os.path.exists(args.image):
        print(f"Error: Image not found: {args.image}")
        sys.exit(1)

    # Create output directory
    os.makedirs(args.out, exist_ok=True)
    img_name = Path(args.image).stem

    # Check if detection model exists (only needed if Stage 1 will run)
    results_json = os.path.join(args.out, f"{img_name}_sam2_results.json")
    masks_output = os.path.join(args.out, f"{img_name}_masks.npz")

    if not (os.path.exists(results_json) and os.path.exists(masks_output)):
        # Stage 1 will run, so we need the appropriate model
        if args.detector == "sam2":
            if not args.sam2_model or not os.path.exists(args.sam2_model):
                print(f"Error: SAM2 model not found: {args.sam2_model}")
                print("Stage 1 (Symbol Detection) needs to run but SAM2 model is missing.")
                print("Either provide --sam2_model or use --detector=yolo with --yolo_model")
                sys.exit(1)
        elif args.detector == "yolo":
            if not args.yolo_model or not os.path.exists(args.yolo_model):
                print(f"Error: YOLO model not found: {args.yolo_model}")
                print("Stage 1 (Symbol Detection) needs to run but YOLO model is missing.")
                print("Either provide --yolo_model or use --detector=sam2 with --sam2_model")
                sys.exit(1)

    print("\n" + "="*80)
    print("COMPREHENSIVE P&ID PROCESSING PIPELINE")
    print("="*80)
    print(f"Image: {args.image}")
    print(f"Output: {args.out}")
    print(f"Detector: {args.detector.upper()}")
    if args.detector == "sam2":
        print(f"SAM2 Model: {args.sam2_model}")
    else:
        print(f"YOLO Model: {args.yolo_model}")
    print(f"Interactive Mode: {args.interactive}")
    print("="*80 + "\n")

    # =========================================================================
    # STEP 1: Symbol Detection (SAM2 or YOLO)
    # =========================================================================
    print("\n" + "="*80)
    if args.detector == "sam2":
        print("STEP 1: SYMBOL DETECTION (SAM2 AMG)")
    else:
        print("STEP 1: SYMBOL DETECTION (YOLO)")
    print("="*80)

    # Define output files (masks_output and results_json already defined above for validation)
    viz_output = os.path.join(args.out, f"{img_name}_sam2_visualization.png")

    # Check if already exists
    if os.path.exists(results_json) and os.path.exists(masks_output):
        print(f"✓ Symbol detection results already exist:")
        print(f"  - {results_json}")
        print(f"  - {masks_output}")
        print("  Skipping Stage 1...")
    else:
        import cv2
        import matplotlib.pyplot as plt
        import numpy as np

        # Load the image
        image = cv2.imread(args.image)
        if image is None:
            print(f"Error: Could not load image: {args.image}")
            sys.exit(1)

        if args.detector == "sam2":
            # =====================
            # SAM2 Detection Mode
            # =====================
            try:
                from sam2_amg_inference import (
                    SAM2AutomaticMaskGenerator,
                    save_combined_masks,
                    save_results_json,
                    visualize_masks
                )
            except ImportError as e:
                print(f"Error importing SAM2 inference: {e}")
                print("Make sure sam2_amg_inference.py is in the same directory.")
                sys.exit(1)

            # Run SAM2 automatic mask generation
            print("Initializing SAM2 model...")
            mask_generator = SAM2AutomaticMaskGenerator(
                model_path=args.sam2_model,
                model_name=args.sam2_base,
                device=args.device,
                confidence_threshold=args.confidence
            )

            print(f"Generating masks for: {args.image}")
            masks_data = mask_generator.generate_automatic_masks(image)

            # Save masks
            save_combined_masks(image, masks_data, args.out, img_name)
            print(f"Saved {len(masks_data['masks'])} masks to: {masks_output}")

            # Save results JSON
            processing_params = {
                'confidence_threshold': args.confidence,
                'prompt_type': 'points',
                'detector': 'sam2'
            }
            save_results_json(masks_data, results_json, args.image, processing_params)
            print(f"Saved results to: {results_json}")

            # Visualize
            fig = visualize_masks(image, masks_data, viz_output)
            plt.close(fig)
            print(f"Saved visualization to: {viz_output}")

        else:
            # =====================
            # YOLO Detection Mode
            # =====================
            try:
                from finetune_yolo_symbols import YOLOSymbolDetector
            except ImportError as e:
                print(f"Error importing YOLO detector: {e}")
                print("Make sure finetune_yolo_symbols.py is in the same directory.")
                sys.exit(1)

            # Initialize YOLO detector
            print(f"Initializing YOLO model from: {args.yolo_model}")
            detector = YOLOSymbolDetector(model_path=args.yolo_model)

            # Run YOLO detection
            print(f"Running YOLO detection for: {args.image}")
            detections = detector.detect(image, conf_threshold=args.confidence)
            print(f"Detected {len(detections)} symbols")

            # Convert YOLO detections to masks (rectangular masks from bboxes)
            # This matches the output format of SAM2 for compatibility
            masks = []
            scores = []
            masks_info = []
            h, w = image.shape[:2]

            for i, det in enumerate(detections):
                # Create a rectangular mask from the bounding box
                mask = np.zeros((h, w), dtype=bool)
                x1, y1, x2, y2 = [int(v) for v in det.bbox]
                # Clamp to image bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                mask[y1:y2, x1:x2] = True

                masks.append(mask)
                scores.append(det.confidence)

                # Store mask info for JSON
                area = (x2 - x1) * (y2 - y1)
                masks_info.append({
                    'id': i,
                    'score': float(det.confidence),
                    'area': int(area),
                    'bbox': [x1, y1, x2, y2],
                    'center': [int(det.x_center), int(det.y_center)]
                })

            # Save masks in the same format as SAM2
            if len(masks) > 0:
                masks_array = np.array(masks)
                np.savez_compressed(
                    masks_output,
                    masks=masks_array,
                    scores=np.array(scores),
                    image_shape=image.shape,
                    num_masks=len(masks)
                )
                print(f"Saved {len(masks)} masks to: {masks_output}")

                # Save results JSON in SAM2-compatible format
                results = {
                    'image_path': str(args.image),
                    'num_masks': len(masks),
                    'processing_params': {
                        'confidence_threshold': args.confidence,
                        'detector': 'yolo',
                        'model_path': args.yolo_model
                    },
                    'masks_info': masks_info
                }

                with open(results_json, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"Saved results to: {results_json}")

                # Create visualization similar to SAM2
                from PIL import Image as PILImage

                # Generate colors for masks
                colors = plt.cm.tab20(np.linspace(0, 1, min(len(masks), 20)))
                if len(masks) > 20:
                    np.random.seed(42)
                    colors = np.random.rand(len(masks), 3)

                # Create visualization
                fig, axes = plt.subplots(2, 2, figsize=(16, 12))
                fig.suptitle(f'YOLO Symbol Detection Results ({len(masks)} detections)', fontsize=16)

                # Convert BGR to RGB
                display_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # 1. Original image
                axes[0, 0].imshow(display_image)
                axes[0, 0].set_title('Original Image')
                axes[0, 0].axis('off')

                # 2. Image with bounding boxes
                axes[0, 1].imshow(display_image)
                import matplotlib.patches as patches
                for i, det in enumerate(detections):
                    x1, y1, x2, y2 = det.bbox
                    rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                            linewidth=2, edgecolor='red',
                                            facecolor='none')
                    axes[0, 1].add_patch(rect)
                    # Add center point
                    axes[0, 1].plot(det.x_center, det.y_center, 'g+', markersize=10)
                axes[0, 1].set_title(f'YOLO Detections ({len(detections)} boxes)')
                axes[0, 1].axis('off')

                # 3. All masks combined
                combined_mask = np.zeros((*image.shape[:2], 3), dtype=np.float32)
                for i, mask in enumerate(masks):
                    color = colors[i % len(colors)][:3]
                    combined_mask[mask] = color
                axes[1, 0].imshow(combined_mask)
                axes[1, 0].set_title('All Masks (from bboxes)')
                axes[1, 0].axis('off')

                # 4. Overlay
                overlay = display_image.copy().astype(np.float32)
                alpha = 0.4
                for i, mask in enumerate(masks):
                    color = colors[i % len(colors)][:3] * 255
                    overlay[mask] = (1 - alpha) * overlay[mask] + alpha * color
                axes[1, 1].imshow(overlay.astype(np.uint8))
                axes[1, 1].set_title(f'Overlay (α={alpha})')
                axes[1, 1].axis('off')

                plt.tight_layout()
                fig.savefig(viz_output, dpi=150, bbox_inches='tight')
                plt.close(fig)
                print(f"Saved visualization to: {viz_output}")

                # Also save combined masks visualization like SAM2 does
                combined_mask_path = os.path.join(args.out, f"{img_name}_all_masks.png")
                PILImage.fromarray((combined_mask * 255).astype(np.uint8)).save(combined_mask_path)

                overlay_path = os.path.join(args.out, f"{img_name}_overlay.png")
                PILImage.fromarray(overlay.astype(np.uint8)).save(overlay_path)
            else:
                print("Warning: No symbols detected by YOLO")
                # Create empty outputs
                np.savez_compressed(
                    masks_output,
                    masks=np.array([]),
                    scores=np.array([]),
                    image_shape=image.shape,
                    num_masks=0
                )
                results = {
                    'image_path': str(args.image),
                    'num_masks': 0,
                    'processing_params': {
                        'confidence_threshold': args.confidence,
                        'detector': 'yolo',
                        'model_path': args.yolo_model
                    },
                    'masks_info': []
                }
                with open(results_json, 'w') as f:
                    json.dump(results, f, indent=2)

    # =========================================================================
    # STEP 2: Symbol Mask Editing (Interactive)
    # =========================================================================
    if args.interactive and not args.skip_symbol_edit:
        print("\n" + "="*80)
        print("STEP 2: SYMBOL MASK EDITING (Interactive)")
        print("="*80)
        print("Launching interactive mask editor...")
        print("You can delete or modify detected symbol masks.")
        print("This cleans up the masks before classification.")

        # Run interactive mask editor
        os.system(f"""python interactive_mask_editor.py \
            --results_json {results_json}""")

        print("✓ Symbol mask editing complete")

    # =========================================================================
    # STEP 3: Symbol Classification
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 3: SYMBOL CLASSIFICATION (Interactive)")
    print("="*80)

    classification_json = os.path.join(args.out, f"{img_name}_classification.json")

    if args.interactive and not args.skip_symbol_edit:
        if os.path.exists(classification_json):
            print(f"✓ Existing classification found: {classification_json}")
            print("Launching interactive classifier to review/modify classifications...")
        else:
            print("Launching interactive symbol classifier...")
            print("This will cluster similar symbols and allow you to label them.")

        # Run interactive symbol classifier (always runs if --interactive is set)
        # If classification exists, it will be loaded and user can review/modify
        os.system(f"""python interactive_symbol_classifier.py \
            --results_json {results_json} \
            --embedding_model {args.embedding_model} \
            --clustering_method {args.clustering_method} \
            --sensitivity {args.sensitivity} \
            --output_dir {args.out} \
            --device {args.device}""")

        if os.path.exists(classification_json):
            print(f"✓ Symbol classification saved to: {classification_json}")
        else:
            print("Warning: Symbol classification not saved. Continuing without it.")
    else:
        print("Skipping interactive symbol classification (use --interactive to enable)")
        if os.path.exists(classification_json):
            print(f"  Using existing classification: {classification_json}")

    # =========================================================================
    # STEP 4: Text Detection
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 4: TEXT DETECTION (PaddleOCR)")
    print("="*80)

    text_json = os.path.join(args.out, f"{img_name}_step3_text.json")
    lines_json = os.path.join(args.out, f"{img_name}_step4_lines.json")

    # Check if text and line detection already complete
    if os.path.exists(text_json) and os.path.exists(lines_json):
        print(f"✓ Text and line detection results already exist:")
        print(f"  - {text_json}")
        print(f"  - {lines_json}")
        print("  Skipping Stage 4...")
    else:
        # Run text detection from process_text_lines.py
        print("Running text and line detection pipeline...")
        cmd = f"""python process_text_lines.py \
            --image {args.image} \
            --out {args.out} \
            --target-width {args.target_width} \
            --lang {args.lang} \
            {"--gpu" if args.gpu else ""} \
            --nms-iou {args.nms_iou} \
            {"--suppress-text" if args.suppress_text else ""} \
            {"--suppress-symbols" if args.suppress_symbols else ""} \
            --line-mode {args.line_mode} \
            --notes-right-frac {args.notes_right_frac} \
            --min-len {args.min_len}"""
        os.system(cmd)

        if os.path.exists(text_json):
            print(f"✓ Text detection complete: {text_json}")
        else:
            print(f"Error: Text detection failed. Expected output: {text_json}")
            sys.exit(1)

    # =========================================================================
    # STEP 5: Text Editing (Interactive)
    # =========================================================================
    if args.interactive and not args.skip_text_edit:
        print("\n" + "="*80)
        print("STEP 5: TEXT EDITING (Interactive)")
        print("="*80)
        print("Launching interactive text editor...")
        print("You can:")
        print("  - Delete text boxes")
        print("  - Combine text boxes")
        print("  - Edit detected text")

        try:
            from interactive_text_editor import TextEditor
            text_editor = TextEditor(json_path=text_json)
            text_editor.run()
            print("✓ Text editing complete")
        except Exception as e:
            print(f"Warning: Text editor failed: {e}")
            print("Continuing without text editing...")

    # =========================================================================
    # STEP 6: Line Detection (completed in Step 4)
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 6: LINE DETECTION (Completed in Step 4)")
    print("="*80)

    # Line detection is performed together with text detection in Step 4
    if os.path.exists(lines_json):
        print(f"✓ Line detection complete: {lines_json}")
        # Load and show summary
        with open(lines_json, 'r') as f:
            lines_data = json.load(f)
        print(f"  Solid lines: {len(lines_data.get('solid', []))}")
        print(f"  Dashed lines: {len(lines_data.get('dashed', []))}")
    else:
        print("Warning: Line detection output not found.")
        print(f"Expected output: {lines_json}")
        print("Line detection should have been completed in Step 4 (process_text_lines.py)")

    # =========================================================================
    # STEP 7: Line Editing (Interactive)
    # =========================================================================
    if args.interactive and not args.skip_line_edit:
        print("\n" + "="*80)
        print("STEP 7: LINE EDITING (Interactive)")
        print("="*80)
        print("Launching interactive line editor...")
        print("You can:")
        print("  - Delete lines")
        print("  - Add new lines")
        print("  - Toggle line types (solid/dashed)")

        try:
            from interactive_line_editor import LineEditor
            line_editor = LineEditor(json_path=lines_json)
            line_editor.run()
            print("✓ Line editing complete")
        except Exception as e:
            print(f"Warning: Line editor failed: {e}")
            print("Continuing without line editing...")

    # =========================================================================
    # STEP 8: Final Digitization - Graph Structure (Nodes & Links)
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 8: FINAL DIGITIZATION - Creating Graph Structure")
    print("="*80)

    # Define output paths
    digitized_full_json = os.path.join(args.out, f"{img_name}_digitized.json")
    digitized_llm_json = os.path.join(args.out, f"{img_name}_digitized_llm.json")
    vis_output_path = os.path.join(args.out, f"{img_name}_digitized_visualization.png")

    # Check if digitization already complete
    if os.path.exists(digitized_full_json) and os.path.exists(digitized_llm_json):
        print(f"✓ Digitization results already exist:")
        print(f"  - {digitized_full_json}")
        print(f"  - {digitized_llm_json}")
        print("  Skipping Stage 8...")
    else:
        print("Combining symbols, text, and lines into graph structure...")

        # Import the digitization function
        try:
            from digitize_pnid import digitize_pnid as digitize_func
        except ImportError as e:
            print(f"Error importing digitize_pnid: {e}")
            print("Make sure digitize_pnid.py is in the same directory.")
            sys.exit(1)

        # Run digitization
        try:
            full_json, llm_json = digitize_func(
                classification_path=classification_json,
                text_path=text_json,
                lines_path=lines_json,
                sam2_path=results_json,
                max_text_distance=args.max_text_distance,
                max_line_distance=args.max_line_distance
            )

            # Save outputs
            with open(digitized_full_json, 'w', encoding='utf-8') as f:
                json.dump(full_json, f, ensure_ascii=False, indent=2)
            print(f"✓ Saved full digitized JSON: {digitized_full_json}")

            with open(digitized_llm_json, 'w', encoding='utf-8') as f:
                json.dump(llm_json, f, ensure_ascii=False, indent=2)
            print(f"✓ Saved LLM-friendly JSON: {digitized_llm_json}")

            # Print summary
            print(f"\nDigitization Summary:")
            print(f"  Nodes (symbols):     {len(full_json['nodes'])}")
            print(f"  Links (connections): {len(full_json['links'])}")
            print(f"  Unconnected lines:   {full_json['metadata']['unconnected_lines']}")

        except Exception as e:
            print(f"Error during digitization: {e}")
            print("Skipping final digitization...")
            import traceback
            traceback.print_exc()

    # =========================================================================
    # Generate Visualization (runs regardless of whether digitization was skipped)
    # =========================================================================
    if not args.no_visualization:
        # Check if visualization already exists
        if os.path.exists(vis_output_path):
            print(f"\n✓ Visualization already exists: {vis_output_path}")
            print("  Delete it to regenerate, or use --no-visualization to skip")
        else:
            print("\n" + "="*80)
            print("GENERATING VISUALIZATION")
            print("="*80)

            try:
                from digitize_pnid import visualize_digitized_pnid

                # Load digitized JSON for visualization
                if os.path.exists(digitized_full_json):
                    with open(digitized_full_json, 'r', encoding='utf-8') as f:
                        full_json = json.load(f)

                    # Load text data for visualization
                    if os.path.exists(text_json):
                        with open(text_json, 'r', encoding='utf-8') as f:
                            text_data = json.load(f)
                        text_detections = text_data if isinstance(text_data, list) else []
                    else:
                        print(f"Warning: Text JSON not found: {text_json}")
                        text_detections = []

                    # Load lines data to get resized_shape (for coordinate space matching)
                    target_width = None
                    if os.path.exists(lines_json):
                        with open(lines_json, 'r', encoding='utf-8') as f:
                            lines_data = json.load(f)
                        resized_shape = lines_data.get('resized_shape')  # [height, width]
                        target_width = resized_shape[1] if resized_shape else args.target_width

                    visualize_digitized_pnid(
                        image_path=args.image,
                        full_json=full_json,
                        text_detections=text_detections,
                        output_path=vis_output_path,
                        show_labels=not args.no_labels,
                        line_width=args.vis_line_width,
                        bbox_width=args.vis_bbox_width,
                        target_width=target_width
                    )

                    print(f"✓ Visualization saved: {vis_output_path}")
                else:
                    print(f"Warning: Cannot generate visualization - digitized JSON not found: {digitized_full_json}")

            except Exception as e:
                print(f"Warning: Visualization generation failed: {e}")
                print("Continuing without visualization...")
                import traceback
                traceback.print_exc()

    # =========================================================================
    # PIPELINE COMPLETE
    # =========================================================================
    print("\n" + "="*80)
    print("PIPELINE COMPLETE!")
    print("="*80)

    # Load and display final summary
    if os.path.exists(digitized_llm_json):
        with open(digitized_llm_json, 'r') as f:
            llm_data = json.load(f)

        print(f"\nFinal Results:")
        print(f"  Nodes (symbols):     {len(llm_data.get('nodes', []))}")
        print(f"  Links (connections): {len(llm_data.get('links', []))}")

        # Show category breakdown
        if os.path.exists(classification_json):
            with open(classification_json, 'r') as f:
                class_data = json.load(f)
            categories = class_data.get('categories', {})
            if categories:
                print(f"\n  Symbol Categories:")
                for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
                    if count > 0:
                        print(f"    - {cat}: {count}")

        print(f"\nOutput Files:")
        print(f"  Full JSON:         {digitized_full_json}")
        print(f"  LLM-friendly JSON: {digitized_llm_json}")

        # Check if visualization was generated
        vis_output_path = os.path.join(args.out, f"{img_name}_digitized_visualization.png")
        if os.path.exists(vis_output_path):
            print(f"  Visualization:     {vis_output_path}")

        print(f"\nAll intermediate results saved to: {args.out}")
        print("\nThe LLM-friendly JSON is ready for downstream processing!")
    else:
        print(f"\nAll intermediate results saved to: {args.out}")

    print("="*80 + "\n")


if __name__ == "__main__":
    main()
