#!/usr/bin/env python3
"""
P&ID Digitization - Final Stage

This script combines symbol classification, text detection, and line detection
into a structured graph representation with nodes (symbols) and links (connections).

Features:
- Extracts symbols (nodes) with associated text captions
- Detects connections (links) between symbols via lines
- Supports line direction information (forward, backward, bidirectional, none)
- Automatically adjusts 'from' and 'to' based on flow direction
- Handles both solid and dashed lines

Outputs:
1. Full JSON with all original features including direction (for debugging/visualization)
2. LLM-friendly JSON with 'from/to' relationships (direction applied to determine correct flow)
3. Visualization image showing bboxes for symbols/text and line connections with direction arrows (optional)

Usage:
    python digitize_pnid.py \
        --classification example_results/higher_resolution_classification.json \
        --text example_results/higher_resolution_step3_text.json \
        --lines example_results/higher_resolution_step4_lines.json \
        --output example_results/higher_resolution_digitized.json \
        --llm-output example_results/higher_resolution_digitized_llm.json \
        --image input_image.png \
        --vis-output example_results/visualization.png
"""

import argparse
import json
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection


def load_json(path: str) -> Any:
    """Load JSON file"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Any, path: str):
    """Save JSON file"""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Saved: {path}")


def compute_bbox_center(bbox: List[float]) -> Tuple[float, float]:
    """Compute center point of a bounding box [x1, y1, x2, y2]"""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def compute_bbox_from_mask(segmentation: Dict) -> List[float]:
    """
    Compute bounding box from segmentation mask

    Args:
        segmentation: Dict with 'size' [h, w] and 'counts' (RLE encoded mask)

    Returns:
        Bounding box [x1, y1, x2, y2]
    """
    # For now, if bbox is not directly available, we'll use a placeholder
    # In a full implementation, you would decode the RLE mask
    # This is a simplified version - you may need pycocotools for full RLE decoding
    return None


def point_to_bbox_distance(point: Tuple[float, float], bbox: List[float]) -> float:
    """
    Compute minimum distance from a point to a bounding box

    Args:
        point: (x, y) coordinates
        bbox: [x1, y1, x2, y2] bounding box

    Returns:
        Minimum distance
    """
    px, py = point
    x1, y1, x2, y2 = bbox

    # Clamp point to bbox
    closest_x = max(x1, min(px, x2))
    closest_y = max(y1, min(py, y2))

    # Compute distance
    dx = px - closest_x
    dy = py - closest_y

    return np.sqrt(dx**2 + dy**2)


def find_nearby_text(symbol_bbox: List[float],
                     text_detections: List[Dict],
                     max_distance: float = 100.0,
                     max_texts: int = 3) -> List[Dict]:
    """
    Find text detections near a symbol

    Args:
        symbol_bbox: Symbol bounding box [x1, y1, x2, y2]
        text_detections: List of text detection dicts
        max_distance: Maximum distance to consider text as "nearby"
        max_texts: Maximum number of texts to return

    Returns:
        List of nearby text dicts with distance info
    """
    symbol_center = compute_bbox_center(symbol_bbox)
    nearby_texts = []

    for text_det in text_detections:
        text_bbox = text_det.get('bbox')
        text_content = text_det.get('text', '').strip()

        if not text_bbox or not text_content:
            continue

        # Compute distance from symbol center to text bbox
        distance = point_to_bbox_distance(symbol_center, text_bbox)

        if distance <= max_distance:
            nearby_texts.append({
                'text': text_content,
                'distance': distance,
                'bbox': text_bbox,
                'confidence': text_det.get('confidence', 1.0)
            })

    # Sort by distance and return top N
    nearby_texts.sort(key=lambda x: x['distance'])
    return nearby_texts[:max_texts]


def point_to_line_distance(point: Tuple[float, float],
                           line: List[float]) -> float:
    """
    Compute minimum distance from a point to a line segment

    Args:
        point: (x, y) coordinates
        line: [x1, y1, x2, y2] line endpoints

    Returns:
        Minimum distance
    """
    px, py = point
    x1, y1, x2, y2 = line

    # Vector from line start to end
    dx = x2 - x1
    dy = y2 - y1

    # Line length squared
    length_sq = dx**2 + dy**2

    if length_sq == 0:
        # Line is a point
        return np.sqrt((px - x1)**2 + (py - y1)**2)

    # Project point onto line (clamped to segment)
    t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / length_sq))

    # Closest point on line segment
    closest_x = x1 + t * dx
    closest_y = y1 + t * dy

    # Distance to closest point
    return np.sqrt((px - closest_x)**2 + (py - closest_y)**2)


def find_multiple_connected_symbols(line: List[float],
                                   symbols: List[Dict],
                                   max_distance: float = 50.0,
                                   max_symbols: int = 10) -> List[Tuple[int, str, float]]:
    """
    Find all symbols near a line (for coming_in and going_out directions)

    Args:
        line: [x1, y1, x2, y2] line endpoints
        symbols: List of symbol dicts with 'id' and 'bbox'
        max_distance: Maximum distance from line to symbol bbox
        max_symbols: Maximum number of symbols to return

    Returns:
        List of tuples (symbol_id, endpoint_type, distance) where endpoint_type is 'start' or 'end'
        sorted by distance (closest first)
    """
    x1, y1, x2, y2 = line
    line_start = (x1, y1)
    line_end = (x2, y2)

    connections = []

    for symbol in symbols:
        symbol_id = symbol['id']
        symbol_bbox = symbol.get('bbox')

        if not symbol_bbox:
            continue

        # Calculate distance from both endpoints to this symbol
        dist_to_start = point_to_bbox_distance(line_start, symbol_bbox)
        dist_to_end = point_to_bbox_distance(line_end, symbol_bbox)

        # If either endpoint is close to this symbol, record it
        if dist_to_start <= max_distance:
            connections.append((symbol_id, 'start', dist_to_start))

        if dist_to_end <= max_distance:
            connections.append((symbol_id, 'end', dist_to_end))

    # Sort by distance (closest first)
    connections.sort(key=lambda x: x[2])

    return connections[:max_symbols]


def find_connected_symbols(line: List[float],
                           symbols: List[Dict],
                           direction: str = 'none',
                           max_distance: float = 50.0) -> Tuple[Optional[int], Optional[int], Dict]:
    """
    Find symbols connected by a line with improved matching using direction

    Args:
        line: [x1, y1, x2, y2] line endpoints
        symbols: List of symbol dicts with 'id' and 'bbox'
        direction: Line direction ('forward', 'backward', 'bidirectional', 'coming_in', 'going_out', 'none')
        max_distance: Maximum distance from line endpoint to symbol bbox

    Returns:
        Tuple of (source_symbol_id, target_symbol_id, connection_info)
        where connection_info contains distances and confidence scores

    Note:
        For 'coming_in' and 'going_out' directions, this function finds the primary
        connections. The calling code should handle creating multiple separate connections.
    """
    x1, y1, x2, y2 = line
    line_start = (x1, y1)
    line_end = (x2, y2)

    source_id = None
    target_id = None
    min_start_dist = float('inf')
    min_end_dist = float('inf')

    for symbol in symbols:
        symbol_id = symbol['id']
        symbol_bbox = symbol.get('bbox')

        if not symbol_bbox:
            continue

        # Use point-to-bbox distance instead of point-to-center distance
        # This is more accurate for lines that touch the edge of symbols
        dist_to_start = point_to_bbox_distance(line_start, symbol_bbox)
        dist_to_end = point_to_bbox_distance(line_end, symbol_bbox)

        # Update source (closest to line start)
        if dist_to_start < min_start_dist and dist_to_start <= max_distance:
            min_start_dist = dist_to_start
            source_id = symbol_id

        # Update target (closest to line end)
        if dist_to_end < min_end_dist and dist_to_end <= max_distance:
            min_end_dist = dist_to_end
            target_id = symbol_id

    # Connection info with distances and confidence
    connection_info = {
        'source_distance': min_start_dist if min_start_dist != float('inf') else None,
        'target_distance': min_end_dist if min_end_dist != float('inf') else None,
        'connection_type': 'full' if (source_id and target_id) else 'partial',
        'direction_used': direction
    }

    # Calculate confidence score (0-1) based on distance
    if min_start_dist != float('inf'):
        connection_info['source_confidence'] = max(0, 1 - (min_start_dist / max_distance))
    if min_end_dist != float('inf'):
        connection_info['target_confidence'] = max(0, 1 - (min_end_dist / max_distance))

    # Only return connection if we found at least one endpoint
    # Handle different cases based on direction
    if direction == 'none':
        # No direction information - require both endpoints
        if source_id is not None and target_id is not None:
            return source_id, target_id, connection_info
    elif direction == 'forward':
        # Forward direction: line flows from start to end
        # Require at least source (start), target (end) is optional but preferred
        if source_id is not None:
            return source_id, target_id, connection_info
    elif direction == 'backward':
        # Backward direction: line flows from end to start
        # Require at least target (end), source (start) is optional but preferred
        if target_id is not None:
            return source_id, target_id, connection_info
    elif direction == 'bidirectional':
        # Bidirectional: can connect in either direction
        # Require at least one endpoint
        if source_id is not None or target_id is not None:
            return source_id, target_id, connection_info

    return None, None, connection_info


def digitize_pnid(classification_path: str,
                  text_path: str,
                  lines_path: str,
                  sam2_path: Optional[str] = None,
                  max_text_distance: float = 100.0,
                  max_line_distance: float = 50.0) -> Tuple[Dict, Dict]:
    """
    Digitize P&ID into graph structure

    Args:
        classification_path: Path to symbol classification JSON
        text_path: Path to text detection JSON
        lines_path: Path to line detection JSON
        sam2_path: Optional path to SAM2 results JSON (for correct bbox coordinates)
        max_text_distance: Max distance for text-to-symbol association
        max_line_distance: Max distance for line-to-symbol connection (lenient)

    Returns:
        Tuple of (full_json, llm_json)
    """
    # Load input files
    print(f"Loading classification data: {classification_path}")
    classification_data = load_json(classification_path)

    print(f"Loading text data: {text_path}")
    text_data = load_json(text_path)

    print(f"Loading line data: {lines_path}")
    lines_data = load_json(lines_path)

    # Load SAM2 results if provided (for correct bbox coordinates)
    sam2_bboxes = {}
    if sam2_path:
        print(f"Loading SAM2 results: {sam2_path}")
        sam2_data = load_json(sam2_path)
        for mask_info in sam2_data.get('masks_info', []):
            mask_id = mask_info.get('id')
            bbox = mask_info.get('bbox')
            if mask_id is not None and bbox:
                sam2_bboxes[mask_id] = bbox
        print(f"Loaded {len(sam2_bboxes)} SAM2 bboxes")

    # Extract symbols from classification
    symbols = classification_data.get('symbols', [])
    print(f"Found {len(symbols)} classified symbols")

    # Update symbol bboxes with SAM2 results if available
    if sam2_bboxes:
        for symbol in symbols:
            mask_id = symbol.get('mask_id')
            if mask_id is not None and mask_id in sam2_bboxes:
                symbol['bbox'] = sam2_bboxes[mask_id]
        print(f"Updated symbol bboxes from SAM2 results")

    # Extract text detections
    text_detections = text_data if isinstance(text_data, list) else []
    print(f"Found {len(text_detections)} text detections")

    # Extract lines - handle both old format and new format with directions
    solid_lines_raw = lines_data.get('solid', [])
    dashed_lines_raw = lines_data.get('dashed', [])

    # Convert to unified format: extract line coordinates AND direction
    solid_lines = []
    for item in solid_lines_raw:
        if isinstance(item, dict):
            # New format with direction
            solid_lines.append({
                'line': item['line'],
                'direction': item.get('direction', 'none')
            })
        else:
            # Old format without direction
            solid_lines.append({
                'line': item,
                'direction': 'none'
            })

    dashed_lines = []
    for item in dashed_lines_raw:
        if isinstance(item, dict):
            # New format with direction
            dashed_lines.append({
                'line': item['line'],
                'direction': item.get('direction', 'none')
            })
        else:
            # Old format without direction
            dashed_lines.append({
                'line': item,
                'direction': 'none'
            })

    all_lines = solid_lines + dashed_lines

    # Count lines with direction
    solid_with_dir = sum(1 for line in solid_lines if line['direction'] != 'none')
    dashed_with_dir = sum(1 for line in dashed_lines if line['direction'] != 'none')
    print(f"Found {len(solid_lines)} solid and {len(dashed_lines)} dashed lines")
    print(f"  - Solid lines with direction: {solid_with_dir}")
    print(f"  - Dashed lines with direction: {dashed_with_dir}")

    # Check if we need to scale symbol coordinates to match line/text coordinates
    # Lines and text are in resized image space, symbols may be in original image space
    resized_shape = lines_data.get('resized_shape')  # [height, width]
    scale_factor = lines_data.get('scale')  # Scale factor used for resizing

    if resized_shape and symbols:
        # Get average symbol bbox coordinate magnitude
        avg_symbol_coord = np.mean([np.mean(s.get('bbox', [0])) for s in symbols if s.get('bbox')])
        # Get average line coordinate magnitude (extract 'line' from dict)
        avg_line_coord = np.mean([np.mean(line_item['line']) for line_item in all_lines[:10] if line_item]) if all_lines else resized_shape[1] / 2

        # If lines have much larger coordinates, we need to scale symbols
        if avg_line_coord > avg_symbol_coord * 2:
            # Assume symbols are in original image space
            # Need to scale to resized image space
            print(f"Detected coordinate space mismatch:")
            print(f"  Average symbol coordinate: {avg_symbol_coord:.1f}")
            print(f"  Average line coordinate: {avg_line_coord:.1f}")
            print(f"  Resized shape: {resized_shape}")

            # Use the scale factor from the lines JSON if available
            if scale_factor:
                scale_x = scale_y = scale_factor
                print(f"  Using scale factor from lines JSON: {scale_factor:.3f}")
            else:
                # Fallback: estimate scale factor from coordinate comparison
                scale_x = scale_y = avg_line_coord / avg_symbol_coord
                print(f"  Using estimated scale factor: {scale_x:.3f}")

            print(f"  Scaling symbols from original image space to resized space...")

            # For each symbol, scale its bbox
            for symbol in symbols:
                bbox = symbol.get('bbox')
                if bbox and len(bbox) == 4:
                    x1, y1, x2, y2 = bbox
                    symbol['bbox'] = [
                        int(x1 * scale_x),
                        int(y1 * scale_y),
                        int(x2 * scale_x),
                        int(y2 * scale_y)
                    ]
                    symbol['bbox_scaled'] = True

            print(f"  Scaled {len(symbols)} symbol bboxes")


    # Build nodes (symbols with nearby text as captions)
    nodes_full = []
    nodes_llm = []

    for idx, symbol in enumerate(symbols):
        symbol_id = idx  # Assign sequential ID
        category = symbol.get('category', 'unknown')
        bbox = symbol.get('bbox')

        if not bbox:
            print(f"Warning: Symbol {idx} has no bbox, skipping")
            continue

        # Find nearby text
        nearby_texts = find_nearby_text(bbox, text_detections, max_text_distance)
        captions = [t['text'] for t in nearby_texts]

        # Full node (all features)
        node_full = {
            'id': symbol_id,
            'category': category,
            'bbox': bbox,
            'area': symbol.get('area'),
            'captions': captions,
            'nearby_text_details': nearby_texts,
            'confidence': symbol.get('confidence'),
            'cluster_id': symbol.get('cluster_id'),
            'original_index': idx
        }
        nodes_full.append(node_full)

        # LLM node (essential only)
        node_llm = {
            'id': symbol_id,
            'category': category,
            'captions': captions
        }
        nodes_llm.append(node_llm)

    print(f"Created {len(nodes_full)} nodes")

    # Build links (lines connecting symbols)
    links_full = []
    links_llm = []

    # Statistics counters
    skipped_no_connection = 0
    skipped_self_loops = 0

    for idx, line_item in enumerate(all_lines):
        # Extract line coordinates and direction
        line_coords = line_item['line']
        line_direction = line_item['direction']
        line_type = 'solid' if line_item in solid_lines else 'dashed'

        # Handle coming_in and going_out directions specially - they create multiple connections
        if line_direction == 'coming_in' or line_direction == 'going_out':
            # Find all symbols near this line
            connected_symbols = find_multiple_connected_symbols(
                line_coords, nodes_full, max_line_distance
            )

            if not connected_symbols:
                skipped_no_connection += 1
                continue

            # Separate symbols by endpoint
            start_symbols = [s for s in connected_symbols if s[1] == 'start']
            end_symbols = [s for s in connected_symbols if s[1] == 'end']

            if line_direction == 'coming_in':
                # Multiple branches merge into one: start symbols → end symbol
                # Multiple sources connecting to one target
                if not end_symbols:
                    skipped_no_connection += 1
                    continue

                # Use the closest end symbol as the merge point
                target_id = end_symbols[0][0]

                # Create a separate connection for each start symbol to the target
                for source_id, _, distance in start_symbols:
                    if source_id == target_id:  # Skip self-loops
                        skipped_self_loops += 1
                        continue

                    conn_info = {
                        'source_distance': distance,
                        'target_distance': end_symbols[0][2],
                        'connection_type': 'full',
                        'direction_used': line_direction,
                        'source_confidence': max(0, 1 - (distance / max_line_distance)),
                        'target_confidence': max(0, 1 - (end_symbols[0][2] / max_line_distance))
                    }

                    # Full link
                    link_full = {
                        'id': len(links_full),
                        'source': source_id,
                        'target': target_id,
                        'type': line_type,
                        'direction': line_direction,
                        'line': line_coords,
                        'length': np.sqrt((line_coords[2]-line_coords[0])**2 + (line_coords[3]-line_coords[1])**2),
                        'connection_quality': conn_info
                    }
                    links_full.append(link_full)

                    # LLM link
                    link_llm = {
                        'from': source_id,
                        'to': target_id,
                        'type': line_type
                    }
                    links_llm.append(link_llm)

                continue  # Skip normal processing

            elif line_direction == 'going_out':
                # One line branches to multiple: start symbol → end symbols
                # One source connecting to multiple targets
                if not start_symbols:
                    skipped_no_connection += 1
                    continue

                # Use the closest start symbol as the branch point
                source_id = start_symbols[0][0]

                # Create a separate connection from the source to each end symbol
                for target_id, _, distance in end_symbols:
                    if source_id == target_id:  # Skip self-loops
                        skipped_self_loops += 1
                        continue

                    conn_info = {
                        'source_distance': start_symbols[0][2],
                        'target_distance': distance,
                        'connection_type': 'full',
                        'direction_used': line_direction,
                        'source_confidence': max(0, 1 - (start_symbols[0][2] / max_line_distance)),
                        'target_confidence': max(0, 1 - (distance / max_line_distance))
                    }

                    # Full link
                    link_full = {
                        'id': len(links_full),
                        'source': source_id,
                        'target': target_id,
                        'type': line_type,
                        'direction': line_direction,
                        'line': line_coords,
                        'length': np.sqrt((line_coords[2]-line_coords[0])**2 + (line_coords[3]-line_coords[1])**2),
                        'connection_quality': conn_info
                    }
                    links_full.append(link_full)

                    # LLM link
                    link_llm = {
                        'from': source_id,
                        'to': target_id,
                        'type': line_type
                    }
                    links_llm.append(link_llm)

                continue  # Skip normal processing

        # Normal processing for other directions
        # Find connected symbols (improved matching with direction)
        source_id, target_id, conn_info = find_connected_symbols(
            line_coords, nodes_full, line_direction, max_line_distance
        )

        # Filter out invalid connections
        if source_id is None or target_id is None:
            # Line doesn't connect to symbols at both ends (skip partial connections)
            skipped_no_connection += 1
            continue

        # Skip self-loops (source and target are the same symbol)
        if source_id == target_id:
            skipped_self_loops += 1
            continue

        # Full link (all features including connection quality info)
        link_full = {
            'id': idx,
            'source': source_id,
            'target': target_id,
            'type': line_type,
            'direction': line_direction,  # Include direction info
            'line': line_coords,  # [x1, y1, x2, y2]
            'length': np.sqrt((line_coords[2]-line_coords[0])**2 + (line_coords[3]-line_coords[1])**2),
            'connection_quality': conn_info  # Add connection quality information
        }
        links_full.append(link_full)

        # LLM link (essential only - use direction to determine from/to relationship)
        # If direction is 'backward', reverse the from/to relationship
        # If 'bidirectional', we could create two links or just use from/to
        # If 'none' or 'forward', use normal from/to

        if line_direction == 'backward':
            # Reverse: flow goes from target to source
            llm_from = target_id
            llm_to = source_id
        elif line_direction == 'bidirectional':
            # For bidirectional, keep original order but we could also create two separate links
            # For now, just use from/to to represent the physical connection
            llm_from = source_id
            llm_to = target_id
        else:
            # 'forward' or 'none': normal direction
            llm_from = source_id
            llm_to = target_id

        link_llm = {
            'from': llm_from,
            'to': llm_to,
            'type': line_type
        }

        # For bidirectional, optionally add a note or create a second reverse link
        if line_direction == 'bidirectional':
            link_llm['bidirectional'] = True

        links_llm.append(link_llm)

    # Connection statistics
    # All connections in links_full are now full connections (both endpoints required)
    full_connections = len(links_full)
    partial_connections = 0  # We now filter out partial connections

    print(f"Created {len(links_full)} links from {len(all_lines)} lines")
    print(f"  - Full connections (both endpoints): {full_connections}")
    print(f"  - Skipped (no/partial connection): {skipped_no_connection}")
    print(f"  - Skipped (self-loops): {skipped_self_loops}")

    # Assemble final outputs
    full_json = {
        'nodes': nodes_full,
        'links': links_full,
        'metadata': {
            'total_symbols': len(nodes_full),
            'total_connections': len(links_full),
            'full_connections': full_connections,
            'total_text_detections': len(text_detections),
            'total_lines': len(all_lines),
            'skipped_no_partial_connection': skipped_no_connection,
            'skipped_self_loops': skipped_self_loops,
            'lines_with_direction': sum(1 for line in all_lines if line['direction'] != 'none')
        }
    }

    llm_json = {
        'nodes': nodes_llm,
        'links': links_llm
    }

    return full_json, llm_json


def visualize_digitized_pnid(image_path: str,
                             full_json: Dict,
                             text_detections: List[Dict],
                             output_path: str,
                             show_labels: bool = True,
                             line_width: int = 2,
                             bbox_width: int = 3,
                             target_width: int = None):
    """
    Create visualization of digitized P&ID over the original image

    Args:
        image_path: Path to original P&ID image
        full_json: Full digitized JSON with nodes and links
        text_detections: List of text detection dicts
        output_path: Path to save visualization
        show_labels: Whether to show category labels on symbols
        line_width: Width of connection lines
        bbox_width: Width of bounding box lines
        target_width: Target width for resizing (if None, will try to infer from coordinates)
    """
    print(f"\nCreating visualization...")

    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return

    original_height, original_width = img.shape[:2]
    print(f"Original image size: {original_width} x {original_height}")

    # Determine if we need to resize the image to match the coordinate space
    # Check the coordinate range from the data
    nodes = full_json.get('nodes', [])
    links = full_json.get('links', [])

    if nodes or links:
        # Get max coordinates from bboxes and lines
        max_x = 0
        max_y = 0

        for node in nodes:
            bbox = node.get('bbox', [])
            if len(bbox) == 4:
                max_x = max(max_x, bbox[2])
                max_y = max(max_y, bbox[3])

        for link in links:
            line = link.get('line', [])
            if len(line) == 4:
                max_x = max(max_x, line[0], line[2])
                max_y = max(max_y, line[1], line[3])

        for text_det in text_detections:
            bbox = text_det.get('bbox', [])
            if len(bbox) == 4:
                max_x = max(max_x, bbox[2])
                max_y = max(max_y, bbox[3])

        print(f"Coordinate space: max_x={max_x}, max_y={max_y}")
        print(f"Image dimensions: width={original_width}, height={original_height}")

        # If coordinates are much larger than image, we need to resize the image
        # Allow 10% tolerance
        if max_x > original_width * 1.1 or max_y > original_height * 1.1:
            # Coordinates are in a different space - resize image to match
            if target_width is None:
                # Use the max coordinate as a hint for target size
                target_width = int(max_x * 1.05)  # Add 5% margin

            # Maintain aspect ratio
            aspect_ratio = original_height / original_width
            target_height = int(target_width * aspect_ratio)

            print(f"Resizing image to match coordinate space: {target_width} x {target_height}")
            img = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        elif max_x < original_width * 0.5 or max_y < original_height * 0.5:
            # Coordinates are much smaller - they might be in a smaller space
            print(f"Warning: Coordinates seem smaller than image dimensions.")
            print(f"This might cause visualization issues.")

    # Convert BGR to RGB for matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(f"Final visualization size: {img_rgb.shape[1]} x {img_rgb.shape[0]}")

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(20, 20))
    ax.imshow(img_rgb)

    # Color schemes
    symbol_colors = {
        'pump': '#FF6B6B',      # Red
        'valve': '#4ECDC4',     # Teal
        'tank': '#45B7D1',      # Blue
        'heat_exchanger': '#FFA07A',  # Light coral
        'instrument': '#98D8C8', # Mint
        'pipe': '#F7DC6F',      # Yellow
        'fitting': '#BB8FCE',   # Purple
        'unknown': '#95A5A6'    # Gray
    }

    # Draw symbols (nodes) with bounding boxes
    nodes = full_json.get('nodes', [])
    print(f"Drawing {len(nodes)} symbol bounding boxes...")

    for node in nodes:
        bbox = node.get('bbox')
        category = node.get('category', 'unknown')
        node_id = node.get('id')
        captions = node.get('captions', [])

        if not bbox or len(bbox) != 4:
            continue

        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1

        # Get color for category
        color = symbol_colors.get(category, symbol_colors['unknown'])

        # Draw bounding box
        rect = patches.Rectangle(
            (x1, y1), width, height,
            linewidth=bbox_width,
            edgecolor=color,
            facecolor='none',
            alpha=0.8
        )
        ax.add_patch(rect)

        # Add label with category and ID
        if show_labels:
            label = f"{category}\n[{node_id}]"
            if captions:
                label += f"\n{', '.join(captions[:2])}"  # Show first 2 captions

            # Add text background for readability
            ax.text(
                x1, y1 - 10,
                label,
                fontsize=8,
                color='white',
                weight='bold',
                bbox=dict(
                    boxstyle='round,pad=0.3',
                    facecolor=color,
                    alpha=0.7,
                    edgecolor='none'
                ),
                verticalalignment='bottom'
            )

    # Draw text detections
    print(f"Drawing {len(text_detections)} text bounding boxes...")

    for text_det in text_detections:
        bbox = text_det.get('bbox')
        text_content = text_det.get('text', '').strip()

        if not bbox or len(bbox) != 4 or not text_content:
            continue

        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1

        # Draw text bounding box in green
        rect = patches.Rectangle(
            (x1, y1), width, height,
            linewidth=1,
            edgecolor='#2ECC71',  # Green
            facecolor='none',
            alpha=0.5,
            linestyle='--'
        )
        ax.add_patch(rect)

    # Draw connections (links) - draw lines between symbol centers
    links = full_json.get('links', [])
    print(f"Drawing {len(links)} connection lines...")

    # Create a lookup table for node centers by ID
    node_centers = {}
    for node in nodes:
        node_id = node.get('id')
        bbox = node.get('bbox')
        if bbox and len(bbox) == 4:
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            node_centers[node_id] = (cx, cy)

    # Separate solid and dashed lines
    solid_lines = []
    dashed_lines = []

    for link in links:
        source_id = link.get('source')
        target_id = link.get('target')
        line_type = link.get('type', 'solid')
        direction = link.get('direction', 'none')

        # Get centers of source and target symbols
        if source_id not in node_centers or target_id not in node_centers:
            continue

        x1, y1 = node_centers[source_id]
        x2, y2 = node_centers[target_id]

        if line_type == 'dashed':
            dashed_lines.append([(x1, y1), (x2, y2)])
        else:
            solid_lines.append([(x1, y1), (x2, y2)])

        # Draw arrows based on direction (only if source != target to avoid self-loops)
        if source_id != target_id:
            color = '#E74C3C' if line_type == 'dashed' else '#3498DB'

            # Determine actual flow direction for visualization
            # This should match what's in the LLM JSON (from → to)
            if direction == 'backward':
                # Flow is reversed: target → source
                flow_from_x, flow_from_y = x2, y2  # target
                flow_to_x, flow_to_y = x1, y1      # source
            else:
                # Flow is normal: source → target (forward, none, bidirectional)
                flow_from_x, flow_from_y = x1, y1  # source
                flow_to_x, flow_to_y = x2, y2      # target

            if direction == 'forward' or direction == 'backward':
                # Single arrow pointing in the actual flow direction
                ax.annotate(
                    '',
                    xy=(flow_to_x, flow_to_y),
                    xytext=(flow_from_x, flow_from_y),
                    arrowprops=dict(
                        arrowstyle='-|>',
                        color=color,
                        lw=line_width + 1,
                        alpha=0.8,
                        mutation_scale=20
                    )
                )
            elif direction == 'bidirectional':
                # Two arrows pointing both directions
                # Calculate midpoint
                mx, my = (x1 + x2) / 2, (y1 + y2) / 2

                # Forward arrow (source to mid)
                ax.annotate(
                    '',
                    xy=(mx + (x2 - x1) * 0.1, my + (y2 - y1) * 0.1),
                    xytext=(x1, y1),
                    arrowprops=dict(
                        arrowstyle='-|>',
                        color=color,
                        lw=line_width + 1,
                        alpha=0.8,
                        mutation_scale=20
                    )
                )

                # Backward arrow (mid to target)
                ax.annotate(
                    '',
                    xy=(x2, y2),
                    xytext=(mx - (x2 - x1) * 0.1, my - (y2 - y1) * 0.1),
                    arrowprops=dict(
                        arrowstyle='-|>',
                        color=color,
                        lw=line_width + 1,
                        alpha=0.8,
                        mutation_scale=20
                    )
                )
            elif direction == 'coming_in':
                # Two arrows pointing toward center (merging: →←)
                dx_line, dy_line = x2 - x1, y2 - y1
                pos1_x, pos1_y = x1 + dx_line * 0.33, y1 + dy_line * 0.33
                pos2_x, pos2_y = x1 + dx_line * 0.67, y1 + dy_line * 0.67
                mx, my = (x1 + x2) / 2, (y1 + y2) / 2

                # First arrow pointing forward (toward center)
                ax.annotate(
                    '',
                    xy=(mx - (x2 - x1) * 0.05, my - (y2 - y1) * 0.05),
                    xytext=(pos1_x, pos1_y),
                    arrowprops=dict(
                        arrowstyle='-|>',
                        color=color,
                        lw=line_width + 1,
                        alpha=0.8,
                        mutation_scale=20
                    )
                )

                # Second arrow pointing backward (toward center)
                ax.annotate(
                    '',
                    xy=(mx + (x2 - x1) * 0.05, my + (y2 - y1) * 0.05),
                    xytext=(pos2_x, pos2_y),
                    arrowprops=dict(
                        arrowstyle='<|-',
                        color=color,
                        lw=line_width + 1,
                        alpha=0.8,
                        mutation_scale=20
                    )
                )
            elif direction == 'going_out':
                # Two arrows pointing away from center (branching: ←→)
                dx_line, dy_line = x2 - x1, y2 - y1
                pos1_x, pos1_y = x1 + dx_line * 0.33, y1 + dy_line * 0.33
                pos2_x, pos2_y = x1 + dx_line * 0.67, y1 + dy_line * 0.67
                mx, my = (x1 + x2) / 2, (y1 + y2) / 2

                # First arrow pointing backward (away from center)
                ax.annotate(
                    '',
                    xy=(pos1_x, pos1_y),
                    xytext=(mx - (x2 - x1) * 0.05, my - (y2 - y1) * 0.05),
                    arrowprops=dict(
                        arrowstyle='<|-',
                        color=color,
                        lw=line_width + 1,
                        alpha=0.8,
                        mutation_scale=20
                    )
                )

                # Second arrow pointing forward (away from center)
                ax.annotate(
                    '',
                    xy=(pos2_x, pos2_y),
                    xytext=(mx + (x2 - x1) * 0.05, my + (y2 - y1) * 0.05),
                    arrowprops=dict(
                        arrowstyle='-|>',
                        color=color,
                        lw=line_width + 1,
                        alpha=0.8,
                        mutation_scale=20
                    )
                )
            else:
                # No direction specified - draw simple arrow from source to target
                ax.annotate(
                    '',
                    xy=(x2, y2),
                    xytext=(x1, y1),
                    arrowprops=dict(
                        arrowstyle='->',
                        color=color,
                        lw=line_width,
                        alpha=0.4
                    )
                )

    # Draw solid lines as line collection
    if solid_lines:
        lc_solid = LineCollection(
            solid_lines,
            colors='#3498DB',  # Blue
            linewidths=line_width,
            alpha=0.6,
            label='Solid connections'
        )
        ax.add_collection(lc_solid)

    # Draw dashed lines as line collection
    if dashed_lines:
        lc_dashed = LineCollection(
            dashed_lines,
            colors='#E74C3C',  # Red
            linewidths=line_width,
            linestyles='dashed',
            alpha=0.6,
            label='Dashed connections'
        )
        ax.add_collection(lc_dashed)

    # Add legend
    legend_elements = [
        patches.Patch(facecolor='none', edgecolor='#2ECC71', linestyle='--',
                     label='Text Detection'),
        patches.Patch(facecolor='none', edgecolor='#3498DB',
                     label='Solid Connection'),
        patches.Patch(facecolor='none', edgecolor='#E74C3C', linestyle='--',
                     label='Dashed Connection')
    ]

    # Add symbol categories to legend
    for category, color in sorted(symbol_colors.items()):
        # Check if this category exists in the nodes
        if any(node.get('category') == category for node in nodes):
            legend_elements.append(
                patches.Patch(facecolor='none', edgecolor=color, linewidth=2,
                             label=f'Symbol: {category}')
            )

    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    # Set title with statistics
    metadata = full_json.get('metadata', {})
    title = (f"Digitized P&ID Visualization\n"
            f"Symbols: {metadata.get('total_symbols', len(nodes))} | "
            f"Connections: {metadata.get('total_connections', len(links))} | "
            f"Text: {metadata.get('total_text_detections', len(text_detections))}")
    ax.set_title(title, fontsize=14, weight='bold', pad=20)

    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved: {output_path}")

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Digitize P&ID into graph structure (nodes and links)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--classification", required=True,
                       help="Path to symbol classification JSON")
    parser.add_argument("--text", required=True,
                       help="Path to text detection JSON")
    parser.add_argument("--lines", required=True,
                       help="Path to line detection JSON")
    parser.add_argument("--sam2", required=False,
                       help="Path to SAM2 results JSON (optional, for correct bbox coordinates)")
    parser.add_argument("--output", required=True,
                       help="Path to save full digitized JSON")
    parser.add_argument("--llm-output", required=True,
                       help="Path to save LLM-friendly digitized JSON")
    parser.add_argument("--max-text-distance", type=float, default=100.0,
                       help="Maximum distance (pixels) for text-to-symbol association")
    parser.add_argument("--max-line-distance", type=float, default=50.0,
                       help="Maximum distance (pixels) for line-to-symbol connection (lenient)")
    parser.add_argument("--image", required=False,
                       help="Path to original P&ID image (for visualization)")
    parser.add_argument("--vis-output", required=False,
                       help="Path to save visualization image (e.g., output.png)")
    parser.add_argument("--show-labels", action='store_true', default=True,
                       help="Show category labels on symbols in visualization")
    parser.add_argument("--line-width", type=int, default=2,
                       help="Width of connection lines in visualization")
    parser.add_argument("--bbox-width", type=int, default=3,
                       help="Width of bounding box lines in visualization")

    args = parser.parse_args()

    # Digitize P&ID
    full_json, llm_json = digitize_pnid(
        args.classification,
        args.text,
        args.lines,
        args.sam2,
        args.max_text_distance,
        args.max_line_distance
    )

    # Save outputs
    save_json(full_json, args.output)
    save_json(llm_json, args.llm_output)

    # Create visualization if image and output path provided
    if args.image and args.vis_output:
        # Load text data for visualization
        text_data = load_json(args.text)
        text_detections = text_data if isinstance(text_data, list) else []

        # Load lines data to get resized_shape
        lines_data = load_json(args.lines)
        resized_shape = lines_data.get('resized_shape')  # [height, width]
        target_width = resized_shape[1] if resized_shape else None

        visualize_digitized_pnid(
            image_path=args.image,
            full_json=full_json,
            text_detections=text_detections,
            output_path=args.vis_output,
            show_labels=args.show_labels,
            line_width=args.line_width,
            bbox_width=args.bbox_width,
            target_width=target_width
        )
    elif args.image or args.vis_output:
        print("\nWarning: Both --image and --vis-output must be provided for visualization")

    # Print summary
    print("\n" + "="*80)
    print("DIGITIZATION SUMMARY")
    print("="*80)
    print(f"Nodes (symbols):           {len(full_json['nodes'])}")
    print(f"Links (connections):       {len(full_json['links'])}")
    print(f"Skipped (no/partial):      {full_json['metadata']['skipped_no_partial_connection']}")
    print(f"Skipped (self-loops):      {full_json['metadata']['skipped_self_loops']}")
    print("="*80)
    print(f"\nFull JSON:        {args.output}")
    print(f"LLM-friendly JSON: {args.llm_output}")
    print("\nDigitization complete!")


if __name__ == "__main__":
    main()
