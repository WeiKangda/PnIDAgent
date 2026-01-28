#!/usr/bin/env python3
"""
Interactive Line Editor for P&ID Line Detection Results

This script provides an interactive GUI to edit line detections from process_single_pnid.py.
Users can:
- Delete line segments by clicking on them
- Add new line segments by drawing (click and drag)
- Combine multiple line segments into a single line
- Toggle between solid and dashed line types
- Set and visualize flow direction (forward, backward, bidirectional)
- Undo/redo operations
- Save modifications back to the JSON file in-place

The edited results are saved in-place and can be used for downstream processing.
Direction arrows are visualized both in the UI and in the saved overlay image.

Usage:
    python interactive_line_editor.py --json /path/to/step4_lines.json
"""

import sys
import os
import argparse
import numpy as np
import cv2
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image


class LineEditor:
    """Interactive line editor with GUI"""

    def __init__(self, json_path: str):
        """
        Initialize the line editor

        Args:
            json_path: Path to the step4_lines.json file
        """
        self.json_path = json_path

        # Load JSON
        self._load_json()

        # Derive image path
        self._derive_image_path()

        # Load image
        self._load_image()

        # State management
        self.selected_solid_indices = set()  # Indices of selected solid lines
        self.selected_dashed_indices = set()  # Indices of selected dashed lines
        self.history = []  # For undo functionality
        self.redo_stack = []  # For redo functionality
        self.modified = False  # Track if changes have been made

        # UI state
        self.show_solid = True
        self.show_dashed = True
        self.line_width_multiplier = 1.0

        # Drawing mode
        self.drawing_mode = False  # Whether we're in drawing mode
        self.drawing_line_type = 'solid'  # 'solid' or 'dashed'
        self.drawing_start = None  # Start point of line being drawn
        self.drawing_end = None  # End point of line being drawn

        # Initialize GUI
        self._setup_gui()

    def _load_json(self):
        """Load line detection JSON file"""
        if not os.path.exists(self.json_path):
            raise FileNotFoundError(f"JSON file not found: {self.json_path}")

        with open(self.json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        # Extract line data
        solid_lines_raw = self.data.get('solid', [])
        dashed_lines_raw = self.data.get('dashed', [])
        self.resized_shape = self.data.get('resized_shape', None)
        self.notes_xmin = self.data.get('notes_xmin', None)

        # Handle both old format [x1,y1,x2,y2] and new format {"line": [...], "direction": ...}
        solid_lines = []
        for item in solid_lines_raw:
            if isinstance(item, dict):
                solid_lines.append(item)
            else:
                # Convert old format to new format
                solid_lines.append({"line": item, "direction": "none"})

        dashed_lines = []
        for item in dashed_lines_raw:
            if isinstance(item, dict):
                dashed_lines.append(item)
            else:
                # Convert old format to new format
                dashed_lines.append({"line": item, "direction": "none"})

        # Process solid lines: remove duplicates, merge overlapping/connected lines
        solid_original_count = len(solid_lines)
        print(f"Loaded from: {self.json_path}")
        print(f"Processing solid lines...")
        self.solid_lines, solid_stats = self._process_lines(solid_lines, min_length=100)

        # Process dashed lines: remove duplicates, merge overlapping/connected lines
        dashed_original_count = len(dashed_lines)
        print(f"Processing dashed lines...")
        self.dashed_lines, dashed_stats = self._process_lines(dashed_lines, min_length=100)

        print(f"\n  Solid lines: {len(self.solid_lines)} (from {solid_original_count} original)")
        if solid_stats['removed'] > 0:
            print(f"    Removed: {solid_stats['short']} too short, {solid_stats['duplicates']} duplicates")
        if solid_stats['merged'] > 0:
            print(f"    Merged: {solid_stats['merged']} overlapping/connected lines")

        print(f"  Dashed lines: {len(self.dashed_lines)} (from {dashed_original_count} original)")
        if dashed_stats['removed'] > 0:
            print(f"    Removed: {dashed_stats['short']} too short, {dashed_stats['duplicates']} duplicates")
        if dashed_stats['merged'] > 0:
            print(f"    Merged: {dashed_stats['merged']} overlapping/connected lines")

    def _process_lines(self, lines, min_length=10, merge_distance=15, angle_tolerance=5):
        """
        Process lines: remove duplicates, short lines, and merge overlapping/connected lines

        Args:
            lines: List of line dicts with {"line": [x1, y1, x2, y2], "direction": ...}
            min_length: Minimum line length in pixels
            merge_distance: Max distance between line endpoints to consider them connected
            angle_tolerance: Angle tolerance in degrees for considering lines collinear or perpendicular

        Returns:
            Tuple of (processed_lines, stats_dict)
        """
        stats = {
            'short': 0,
            'duplicates': 0,
            'removed': 0,
            'merged': 0
        }

        # Step 1: Remove short lines and duplicates
        seen = set()
        filtered_lines = []

        for line_item in lines:
            line = line_item["line"]
            x1, y1, x2, y2 = line
            direction = line_item.get("direction", "none")

            # Calculate line length
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

            # Skip lines that are too short
            if length < min_length:
                stats['short'] += 1
                continue

            # Normalize line for duplicate detection
            normalized = tuple(sorted([(x1, y1), (x2, y2)]))

            if normalized not in seen:
                seen.add(normalized)
                filtered_lines.append({"line": list(line), "direction": direction})
            else:
                stats['duplicates'] += 1

        stats['removed'] = stats['short'] + stats['duplicates']

        # Step 2: Merge overlapping and connected lines
        merged_lines = self._merge_connected_lines(filtered_lines, merge_distance, angle_tolerance)
        stats['merged'] = len(filtered_lines) - len(merged_lines)

        return merged_lines, stats

    def _merge_connected_lines(self, lines, merge_distance=15, angle_tolerance=5):
        """
        Merge lines that are overlapping or connected at endpoints

        Args:
            lines: List of line dicts
            merge_distance: Max distance between endpoints to consider connected
            angle_tolerance: Tolerance for angle matching (degrees)

        Returns:
            List of merged line dicts
        """
        if not lines:
            return []

        # Create a graph of lines
        merged = []
        used = set()

        for i, line_item in enumerate(lines):
            if i in used:
                continue

            # Start with this line
            current = line_item.copy()
            used.add(i)
            changed = True

            # Keep trying to merge until no more merges possible
            while changed:
                changed = False
                for j, other_item in enumerate(lines):
                    if j in used:
                        continue

                    # Try to merge current with other
                    merged_line = self._try_merge_two_lines(
                        current, other_item, merge_distance, angle_tolerance
                    )

                    if merged_line is not None:
                        current = merged_line
                        used.add(j)
                        changed = True

            merged.append(current)

        return merged

    def _try_merge_two_lines(self, line1_item, line2_item, merge_distance, angle_tolerance):
        """
        Try to merge two lines if they're overlapping or connected

        Returns:
            Merged line dict or None if not mergeable
        """
        line1 = line1_item["line"]
        line2 = line2_item["line"]
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2

        # Calculate angles
        angle1 = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        angle2 = np.degrees(np.arctan2(y4 - y3, x4 - x3))

        # Normalize angles to [0, 180)
        angle1 = angle1 % 180
        angle2 = angle2 % 180

        # Check if lines are collinear (similar angles)
        angle_diff = abs(angle1 - angle2)
        if angle_diff > 90:
            angle_diff = 180 - angle_diff

        is_collinear = angle_diff < angle_tolerance

        # Calculate distances between all endpoint pairs (used for both collinear and perpendicular checks)
        d12_34 = np.sqrt((x2 - x3)**2 + (y2 - y3)**2)  # end1 to start2
        d12_43 = np.sqrt((x2 - x4)**2 + (y2 - y4)**2)  # end1 to end2
        d21_34 = np.sqrt((x1 - x3)**2 + (y1 - y3)**2)  # start1 to start2
        d21_43 = np.sqrt((x1 - x4)**2 + (y1 - y4)**2)  # start1 to end2

        min_dist = min(d12_34, d12_43, d21_34, d21_43)

        if is_collinear:

            # If endpoints are close enough, merge the lines
            if min_dist <= merge_distance:
                # Find the two furthest points to form the merged line
                points = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]

                # Project all points onto the line direction
                # Use line1's direction
                dx = x2 - x1
                dy = y2 - y1
                length1 = np.sqrt(dx**2 + dy**2)

                if length1 > 0:
                    dx_norm = dx / length1
                    dy_norm = dy / length1

                    # Project each point onto the line
                    projections = []
                    for px, py in points:
                        # Vector from line1 start to point
                        vx, vy = px - x1, py - y1
                        # Projection length
                        proj = vx * dx_norm + vy * dy_norm
                        projections.append((proj, px, py))

                    # Sort by projection
                    projections.sort()

                    # Take the first and last points
                    _, new_x1, new_y1 = projections[0]
                    _, new_x2, new_y2 = projections[-1]

                    # Merge directions (prioritize non-none directions)
                    dir1 = line1_item.get("direction", "none")
                    dir2 = line2_item.get("direction", "none")

                    if dir1 != "none":
                        merged_direction = dir1
                    elif dir2 != "none":
                        merged_direction = dir2
                    else:
                        merged_direction = "none"

                    return {
                        "line": [int(new_x1), int(new_y1), int(new_x2), int(new_y2)],
                        "direction": merged_direction
                    }

        # Check if lines form a 90-degree angle (L-shape or T-junction)
        # This is lenient - we allow them to stay separate but could optionally combine
        angle_diff_90 = abs(angle_diff - 90)
        is_perpendicular = angle_diff_90 < angle_tolerance

        if is_perpendicular:
            # Check if they share an endpoint (T-junction or L-shape)
            distances = [d12_34, d12_43, d21_34, d21_43]
            if min(distances) <= merge_distance:
                # They form an angle - keep them separate but user can manually adjust
                # For now, we don't merge perpendicular lines
                pass

        return None

    def _derive_image_path(self):
        """Derive image path from JSON path"""
        json_path = Path(self.json_path)
        json_dir = json_path.parent

        # Try to find image path from related JSON files
        step2_json = json_dir / json_path.name.replace('_step4_lines.json', '_step2_merged.json')
        step1_json = json_dir / json_path.name.replace('_step4_lines.json', '_step1_dets.json')

        image_path = None
        scale = 1.0

        # Try to find image path from step2 or step1 JSON
        for related_json in [step2_json, step1_json]:
            if related_json.exists():
                with open(related_json, 'r') as f:
                    related_data = json.load(f)
                    if 'image_path' in related_data:
                        image_path = related_data['image_path']
                        scale = related_data.get('scale', 1.0)
                        break

        # Fallback: Check if image_path is in the line JSON itself
        if not image_path and 'image_path' in self.data:
            image_path = self.data['image_path']
            scale = self.data.get('scale', 1.0)

        if not image_path or not os.path.exists(image_path):
            # Try to find any image in the parent directory
            for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
                candidates = list(json_dir.parent.glob(f"*{ext}"))
                if candidates:
                    image_path = str(candidates[0])
                    print(f"Warning: Using guessed image path: {image_path}")
                    break

        if not image_path or not os.path.exists(image_path):
            raise FileNotFoundError(f"Could not find image file. Please ensure the image is in the expected location.")

        self.image_path = image_path
        self.scale = scale
        print(f"Image path: {self.image_path}")
        print(f"Scale: {self.scale}")

    def _load_image(self):
        """Load the image"""
        self.image = cv2.imread(self.image_path)
        if self.image is None:
            raise ValueError(f"Could not load image: {self.image_path}")

        # Resize if needed (to match the processed resolution)
        if self.scale != 1.0:
            target_width = int(self.image.shape[1] * self.scale)
            target_height = int(self.image.shape[0] * self.scale)
            self.image = cv2.resize(self.image, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

        # Convert BGR to RGB for display
        self.image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        print(f"Loaded image: {self.image.shape}")

    def _setup_gui(self):
        """Setup the GUI interface"""
        self.root = tk.Tk()
        self.root.title("Interactive Line Editor - Stage 2")
        self.root.geometry("1400x900")

        # Create main layout
        self._create_menu_bar()
        self._create_toolbar()
        self._create_main_layout()
        self._create_status_bar()

        # Initial render
        self.render_visualization()

        # Bind keyboard shortcuts
        self.root.bind('<Control-z>', lambda e: self.undo())
        self.root.bind('<Control-y>', lambda e: self.redo())
        self.root.bind('<Control-s>', lambda e: self.save())
        self.root.bind('<Delete>', lambda e: self.delete_selected())
        self.root.bind('<Escape>', lambda e: self.exit_drawing_mode() if self.drawing_mode else self.clear_selection())
        self.root.bind('d', lambda e: self.enter_drawing_mode())
        self.root.bind('c', lambda e: self.combine_selected())
        self.root.bind('s', lambda e: self.toggle_solid_visibility())
        self.root.bind('h', lambda e: self.toggle_dashed_visibility())
        self.root.bind('r', lambda e: self.apply_direction_to_selected())

    def _create_menu_bar(self):
        """Create menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Save", command=self.save, accelerator="Ctrl+S")
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.on_closing)

        # Edit menu
        edit_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Undo", command=self.undo, accelerator="Ctrl+Z")
        edit_menu.add_command(label="Redo", command=self.redo, accelerator="Ctrl+Y")
        edit_menu.add_separator()
        edit_menu.add_command(label="Draw New Line", command=self.enter_drawing_mode, accelerator="D")
        edit_menu.add_separator()
        edit_menu.add_command(label="Combine Selected", command=self.combine_selected, accelerator="C")
        edit_menu.add_command(label="Delete Selected", command=self.delete_selected, accelerator="Del")
        edit_menu.add_command(label="Clear Selection", command=self.clear_selection, accelerator="Esc")

        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Toggle Solid Lines", command=self.toggle_solid_visibility, accelerator="S")
        view_menu.add_command(label="Toggle Dashed Lines", command=self.toggle_dashed_visibility, accelerator="H")

    def _create_toolbar(self):
        """Create toolbar with action buttons"""
        toolbar = ttk.Frame(self.root)
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        # Drawing controls
        self.draw_button = ttk.Button(toolbar, text="Draw Line (D)", command=self.enter_drawing_mode)
        self.draw_button.pack(side=tk.LEFT, padx=2)

        # Line type selection
        ttk.Label(toolbar, text="Type:").pack(side=tk.LEFT, padx=(10, 2))
        self.line_type_var = tk.StringVar(value='solid')
        ttk.Radiobutton(toolbar, text="Solid", variable=self.line_type_var, value='solid').pack(side=tk.LEFT, padx=2)
        ttk.Radiobutton(toolbar, text="Dashed", variable=self.line_type_var, value='dashed').pack(side=tk.LEFT, padx=2)

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)

        # Direction selection
        ttk.Label(toolbar, text="Direction:").pack(side=tk.LEFT, padx=(10, 2))
        self.direction_var = tk.StringVar(value='none')
        direction_combo = ttk.Combobox(toolbar, textvariable=self.direction_var,
                                       values=['none', 'forward', 'backward', 'bidirectional', 'coming_in', 'going_out'],
                                       state='readonly', width=12)
        direction_combo.pack(side=tk.LEFT, padx=2)

        ttk.Button(toolbar, text="Apply Direction (R)", command=self.apply_direction_to_selected).pack(side=tk.LEFT, padx=2)

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)

        # Action buttons
        ttk.Button(toolbar, text="Combine Selected", command=self.combine_selected).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Delete Selected", command=self.delete_selected).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Clear Selection", command=self.clear_selection).pack(side=tk.LEFT, padx=2)

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)

        ttk.Button(toolbar, text="Undo", command=self.undo).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Redo", command=self.redo).pack(side=tk.LEFT, padx=2)

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)

        ttk.Button(toolbar, text="Save", command=self.save).pack(side=tk.LEFT, padx=2)

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)

        # Toggle visibility
        self.show_solid_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(toolbar, text="Show Solid", command=self.toggle_solid_visibility,
                       variable=self.show_solid_var).pack(side=tk.LEFT, padx=2)

        self.show_dashed_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(toolbar, text="Show Dashed", command=self.toggle_dashed_visibility,
                       variable=self.show_dashed_var).pack(side=tk.LEFT, padx=2)

    def _create_main_layout(self):
        """Create main layout with canvas and line list"""
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left side: Canvas
        canvas_frame = ttk.Frame(main_frame)
        canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Create matplotlib figure
        self.fig = Figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=canvas_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Add matplotlib toolbar
        toolbar_frame = ttk.Frame(canvas_frame)
        toolbar_frame.pack(fill=tk.X)
        NavigationToolbar2Tk(self.canvas, toolbar_frame)

        # Connect mouse events
        self.fig.canvas.mpl_connect('button_press_event', self.on_canvas_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_canvas_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_canvas_motion)

        # Right side: Line list
        right_frame = ttk.Frame(main_frame, width=350)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(5, 0))
        right_frame.pack_propagate(False)

        # Line list label
        ttk.Label(right_frame, text="Detected Lines", font=('Arial', 12, 'bold')).pack(pady=5)

        # Info label
        self.info_label = ttk.Label(right_frame,
                                    text=f"Solid: {len(self.solid_lines)} | Dashed: {len(self.dashed_lines)}")
        self.info_label.pack(pady=2)

        # Tabs for solid and dashed lines
        self.notebook = ttk.Notebook(right_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Solid lines tab
        solid_frame = ttk.Frame(self.notebook)
        self.notebook.add(solid_frame, text=f"Solid ({len(self.solid_lines)})")

        solid_list_frame = ttk.Frame(solid_frame)
        solid_list_frame.pack(fill=tk.BOTH, expand=True)

        solid_scrollbar = ttk.Scrollbar(solid_list_frame)
        solid_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.solid_listbox = tk.Listbox(solid_list_frame, yscrollcommand=solid_scrollbar.set,
                                       selectmode=tk.EXTENDED, font=('Courier', 9))
        self.solid_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        solid_scrollbar.config(command=self.solid_listbox.yview)
        self.solid_listbox.bind('<<ListboxSelect>>', self.on_solid_list_selection)

        # Dashed lines tab
        dashed_frame = ttk.Frame(self.notebook)
        self.notebook.add(dashed_frame, text=f"Dashed ({len(self.dashed_lines)})")

        dashed_list_frame = ttk.Frame(dashed_frame)
        dashed_list_frame.pack(fill=tk.BOTH, expand=True)

        dashed_scrollbar = ttk.Scrollbar(dashed_list_frame)
        dashed_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.dashed_listbox = tk.Listbox(dashed_list_frame, yscrollcommand=dashed_scrollbar.set,
                                        selectmode=tk.EXTENDED, font=('Courier', 9))
        self.dashed_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        dashed_scrollbar.config(command=self.dashed_listbox.yview)
        self.dashed_listbox.bind('<<ListboxSelect>>', self.on_dashed_list_selection)

        # Populate lists
        self.update_line_lists()

        # Details panel
        details_frame = ttk.LabelFrame(right_frame, text="Selection Details", padding=5)
        details_frame.pack(fill=tk.X, padx=5, pady=5)

        self.details_text = tk.Text(details_frame, height=6, wrap=tk.WORD, font=('Courier', 9))
        self.details_text.pack(fill=tk.BOTH, expand=True)
        self.update_details_panel()

    def _create_status_bar(self):
        """Create status bar"""
        self.status_bar = ttk.Label(self.root, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def update_line_lists(self):
        """Update the line list displays"""
        # Direction symbols for display
        dir_symbols = {
            'none': '—',
            'forward': '→',
            'backward': '←',
            'bidirectional': '↔',
            'coming_in': '→←',  # Two arrows pointing toward each other (merging)
            'going_out': '←→'   # Two arrows pointing away from each other (branching)
        }

        # Update solid lines
        self.solid_listbox.delete(0, tk.END)
        for i, line_item in enumerate(self.solid_lines):
            x1, y1, x2, y2 = line_item["line"]
            direction = line_item.get("direction", "none")
            length = int(np.sqrt((x2-x1)**2 + (y2-y1)**2))
            dir_symbol = dir_symbols.get(direction, '—')
            display_text = f"#{i:3d} {dir_symbol} | ({x1:4d},{y1:4d})->({x2:4d},{y2:4d}) | L:{length:4d}"
            self.solid_listbox.insert(tk.END, display_text)
            if i in self.selected_solid_indices:
                self.solid_listbox.itemconfig(tk.END, bg='lightblue')

        # Update dashed lines
        self.dashed_listbox.delete(0, tk.END)
        for i, line_item in enumerate(self.dashed_lines):
            x1, y1, x2, y2 = line_item["line"]
            direction = line_item.get("direction", "none")
            length = int(np.sqrt((x2-x1)**2 + (y2-y1)**2))
            dir_symbol = dir_symbols.get(direction, '—')
            display_text = f"#{i:3d} {dir_symbol} | ({x1:4d},{y1:4d})->({x2:4d},{y2:4d}) | L:{length:4d}"
            self.dashed_listbox.insert(tk.END, display_text)
            if i in self.selected_dashed_indices:
                self.dashed_listbox.itemconfig(tk.END, bg='lightblue')

        # Update tab labels
        self.notebook.tab(0, text=f"Solid ({len(self.solid_lines)})")
        self.notebook.tab(1, text=f"Dashed ({len(self.dashed_lines)})")

        # Update info label
        total_selected = len(self.selected_solid_indices) + len(self.selected_dashed_indices)
        self.info_label.config(
            text=f"Solid: {len(self.solid_lines)} | Dashed: {len(self.dashed_lines)} | Selected: {total_selected}")

    def update_details_panel(self):
        """Update the details panel with selection info"""
        self.details_text.delete(1.0, tk.END)

        total_selected = len(self.selected_solid_indices) + len(self.selected_dashed_indices)

        if total_selected == 0:
            self.details_text.insert(tk.END, "No lines selected.\n\n")
            self.details_text.insert(tk.END, "Click on lines in the image\n")
            self.details_text.insert(tk.END, "or select from the lists.")
        else:
            self.details_text.insert(tk.END, f"Selected: {total_selected} lines\n\n")
            if len(self.selected_solid_indices) > 0:
                self.details_text.insert(tk.END, f"Solid: {len(self.selected_solid_indices)} lines\n")
            if len(self.selected_dashed_indices) > 0:
                self.details_text.insert(tk.END, f"Dashed: {len(self.selected_dashed_indices)} lines\n")

    def render_visualization(self):
        """Render the current visualization"""
        self.ax.clear()

        # Show original image
        self.ax.imshow(self.image_rgb)

        # Draw solid lines
        if self.show_solid:
            for i, line_item in enumerate(self.solid_lines):
                line_coords = line_item["line"]
                x1, y1, x2, y2 = line_coords
                direction = line_item.get("direction", "none")

                if i in self.selected_solid_indices:
                    color = 'red'
                    linewidth = 3
                    alpha = 1.0
                else:
                    color = 'green'
                    linewidth = 2
                    alpha = 0.7

                self.ax.plot([x1, x2], [y1, y2], color=color, linewidth=linewidth, alpha=alpha)

                # Draw direction arrows
                if direction != "none":
                    self._draw_direction_arrow(x1, y1, x2, y2, direction, color, alpha)

        # Draw dashed lines
        if self.show_dashed:
            for i, line_item in enumerate(self.dashed_lines):
                line_coords = line_item["line"]
                x1, y1, x2, y2 = line_coords
                direction = line_item.get("direction", "none")

                if i in self.selected_dashed_indices:
                    color = 'red'
                    linewidth = 3
                    alpha = 1.0
                else:
                    color = 'blue'
                    linewidth = 2
                    alpha = 0.7

                self.ax.plot([x1, x2], [y1, y2], color=color, linewidth=linewidth,
                           linestyle='--', alpha=alpha)

                # Draw direction arrows
                if direction != "none":
                    self._draw_direction_arrow(x1, y1, x2, y2, direction, color, alpha)

        # Draw notes boundary if exists
        if self.notes_xmin is not None:
            self.ax.axvline(x=self.notes_xmin, color='orange', linewidth=2, linestyle=':', alpha=0.5)

        # Draw line being created
        if self.drawing_mode and self.drawing_start and self.drawing_end:
            x1, y1 = self.drawing_start
            x2, y2 = self.drawing_end
            linestyle = '--' if self.line_type_var.get() == 'dashed' else '-'
            self.ax.plot([x1, x2], [y1, y2], color='yellow', linewidth=3,
                       linestyle=linestyle, alpha=0.9)

            # Show length
            length = int(np.sqrt((x2-x1)**2 + (y2-y1)**2))
            mid_x, mid_y = (x1+x2)/2, (y1+y2)/2
            self.ax.text(mid_x, mid_y, f'{length}px', color='yellow', fontsize=10,
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.7))

        self.ax.axis('off')
        title = f'Line Editor - Solid: {len(self.solid_lines)} | Dashed: {len(self.dashed_lines)}'
        if self.drawing_mode:
            title += f' [DRAWING MODE - {self.line_type_var.get().upper()}]'
        total_selected = len(self.selected_solid_indices) + len(self.selected_dashed_indices)
        if total_selected > 0:
            title += f' ({total_selected} selected)'
        self.ax.set_title(title, fontsize=10)
        self.canvas.draw()

    def on_canvas_press(self, event):
        """Handle mouse button press"""
        if event.inaxes != self.ax:
            return

        x, y = int(event.xdata), int(event.ydata)

        # Check bounds
        if x < 0 or x >= self.image.shape[1] or y < 0 or y >= self.image.shape[0]:
            return

        if self.drawing_mode:
            # Start drawing a line
            self.drawing_start = (x, y)
            self.drawing_end = (x, y)
            self.update_status("Drawing line... Release to finish")
        else:
            # Select line
            self._select_line_at_position(x, y)

    def on_canvas_release(self, event):
        """Handle mouse button release"""
        if not self.drawing_mode:
            return

        if event.inaxes != self.ax:
            return

        x, y = int(event.xdata), int(event.ydata)

        # Check bounds
        if x < 0 or x >= self.image.shape[1] or y < 0 or y >= self.image.shape[0]:
            return

        if self.drawing_start:
            self.drawing_end = (x, y)
            self._create_line()

    def on_canvas_motion(self, event):
        """Handle mouse motion"""
        if not self.drawing_mode:
            return

        if event.inaxes != self.ax:
            return

        x, y = int(event.xdata), int(event.ydata)

        # Check bounds
        if x < 0 or x >= self.image.shape[1] or y < 0 or y >= self.image.shape[0]:
            return

        if self.drawing_start:
            self.drawing_end = (x, y)
            self.render_visualization()

    def _select_line_at_position(self, x, y, threshold=5):
        """Select a line near the clicked position"""
        clicked_line_idx = None
        clicked_line_type = None
        min_dist = threshold

        # Check solid lines
        if self.show_solid:
            for i, line_item in enumerate(self.solid_lines):
                x1, y1, x2, y2 = line_item["line"]
                dist = self._point_to_line_distance(x, y, x1, y1, x2, y2)
                if dist < min_dist:
                    min_dist = dist
                    clicked_line_idx = i
                    clicked_line_type = 'solid'

        # Check dashed lines
        if self.show_dashed:
            for i, line_item in enumerate(self.dashed_lines):
                x1, y1, x2, y2 = line_item["line"]
                dist = self._point_to_line_distance(x, y, x1, y1, x2, y2)
                if dist < min_dist:
                    min_dist = dist
                    clicked_line_idx = i
                    clicked_line_type = 'dashed'

        if clicked_line_idx is not None:
            # Toggle selection
            if clicked_line_type == 'solid':
                if clicked_line_idx in self.selected_solid_indices:
                    self.selected_solid_indices.remove(clicked_line_idx)
                else:
                    self.selected_solid_indices.add(clicked_line_idx)
            else:  # dashed
                if clicked_line_idx in self.selected_dashed_indices:
                    self.selected_dashed_indices.remove(clicked_line_idx)
                else:
                    self.selected_dashed_indices.add(clicked_line_idx)

            self.update_line_lists()
            self.update_details_panel()
            self.render_visualization()
            self.update_status(f"{clicked_line_type.capitalize()} line #{clicked_line_idx} selected")

    def _point_to_line_distance(self, px, py, x1, y1, x2, y2):
        """Calculate distance from point to line segment"""
        # Vector from line start to point
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            return np.sqrt((px - x1)**2 + (py - y1)**2)

        # Parameter t of closest point on line
        t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx**2 + dy**2)))

        # Closest point on line
        closest_x = x1 + t * dx
        closest_y = y1 + t * dy

        # Distance to closest point
        return np.sqrt((px - closest_x)**2 + (py - closest_y)**2)

    def _draw_direction_arrow(self, x1, y1, x2, y2, direction, color, alpha):
        """Draw direction arrow(s) on a line using matplotlib"""
        # Calculate line properties
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        dx, dy = x2 - x1, y2 - y1
        length = np.sqrt(dx**2 + dy**2)

        if length < 1:
            return

        # Normalize direction vector
        dx_norm, dy_norm = dx / length, dy / length

        # Arrow size (proportional to line length, max 100 pixels, min 40 pixels)
        arrow_size = min(100, max(40, length * 0.2))

        def draw_arrow_at(px, py, reverse=False):
            """Draw a single arrow at position (px, py)"""
            arrow_dx = -dx_norm if reverse else dx_norm
            arrow_dy = -dy_norm if reverse else dy_norm

            self.ax.arrow(px - arrow_dx * arrow_size * 0.4, py - arrow_dy * arrow_size * 0.4,
                         arrow_dx * arrow_size * 0.8, arrow_dy * arrow_size * 0.8,
                         head_width=arrow_size * 0.6, head_length=arrow_size * 0.5,
                         fc=color, ec=color, alpha=alpha, length_includes_head=True, linewidth=2)

        if direction == "forward":
            # Arrow pointing from start to end (at center)
            draw_arrow_at(cx, cy, reverse=False)

        elif direction == "backward":
            # Arrow pointing from end to start (reverse)
            draw_arrow_at(cx, cy, reverse=True)

        elif direction == "bidirectional":
            # Two arrows, one at 1/3 and one at 2/3
            pos1_x, pos1_y = x1 + dx * 0.33, y1 + dy * 0.33
            pos2_x, pos2_y = x1 + dx * 0.67, y1 + dy * 0.67

            draw_arrow_at(pos1_x, pos1_y, reverse=False)
            draw_arrow_at(pos2_x, pos2_y, reverse=False)

        elif direction == "coming_in":
            # Two arrows pointing toward center (merging: →←)
            pos1_x, pos1_y = x1 + dx * 0.33, y1 + dy * 0.33
            pos2_x, pos2_y = x1 + dx * 0.67, y1 + dy * 0.67

            draw_arrow_at(pos1_x, pos1_y, reverse=False)  # Arrow pointing toward center
            draw_arrow_at(pos2_x, pos2_y, reverse=True)   # Arrow pointing toward center (reversed)

        elif direction == "going_out":
            # Two arrows pointing away from center (branching: ←→)
            pos1_x, pos1_y = x1 + dx * 0.33, y1 + dy * 0.33
            pos2_x, pos2_y = x1 + dx * 0.67, y1 + dy * 0.67

            draw_arrow_at(pos1_x, pos1_y, reverse=True)   # Arrow pointing away from center
            draw_arrow_at(pos2_x, pos2_y, reverse=False)  # Arrow pointing away from center

    def on_solid_list_selection(self, event):
        """Handle solid line list selection"""
        selected_listbox_indices = self.solid_listbox.curselection()
        self.selected_solid_indices.clear()

        for idx in selected_listbox_indices:
            display_text = self.solid_listbox.get(idx)
            # Extract line index from format: "#  0 → | ..."
            # Split by '|', take first part, remove '#', split by space, get first number
            first_part = display_text.split('|')[0].strip().replace('#', '').strip()
            # Get just the numeric part (first token)
            line_idx = int(first_part.split()[0])
            self.selected_solid_indices.add(line_idx)

        self.update_details_panel()
        self.render_visualization()

    def on_dashed_list_selection(self, event):
        """Handle dashed line list selection"""
        selected_listbox_indices = self.dashed_listbox.curselection()
        self.selected_dashed_indices.clear()

        for idx in selected_listbox_indices:
            display_text = self.dashed_listbox.get(idx)
            # Extract line index from format: "#  0 → | ..."
            # Split by '|', take first part, remove '#', split by space, get first number
            first_part = display_text.split('|')[0].strip().replace('#', '').strip()
            # Get just the numeric part (first token)
            line_idx = int(first_part.split()[0])
            self.selected_dashed_indices.add(line_idx)

        self.update_details_panel()
        self.render_visualization()

    def toggle_solid_visibility(self):
        """Toggle solid lines visibility"""
        self.show_solid = self.show_solid_var.get()
        self.render_visualization()

    def toggle_dashed_visibility(self):
        """Toggle dashed lines visibility"""
        self.show_dashed = self.show_dashed_var.get()
        self.render_visualization()

    def clear_selection(self):
        """Clear all selected lines"""
        self.selected_solid_indices.clear()
        self.selected_dashed_indices.clear()
        self.update_line_lists()
        self.update_details_panel()
        self.render_visualization()
        self.update_status("Selection cleared")

    def enter_drawing_mode(self):
        """Enter drawing mode to create new lines"""
        self.drawing_mode = True
        self.drawing_start = None
        self.drawing_end = None
        self.drawing_line_type = self.line_type_var.get()
        self.draw_button.config(text="Cancel Drawing (Esc)")
        self.render_visualization()
        self.update_status(f"Drawing mode - Click and drag to draw a {self.drawing_line_type} line")

    def exit_drawing_mode(self):
        """Exit drawing mode"""
        if self.drawing_mode:
            self.drawing_mode = False
            self.drawing_start = None
            self.drawing_end = None
            self.draw_button.config(text="Draw Line (D)")
            self.render_visualization()
            self.update_status("Drawing mode cancelled")

    def _create_line(self):
        """Create a new line from drawing"""
        if not self.drawing_start or not self.drawing_end:
            return

        x1, y1 = self.drawing_start
        x2, y2 = self.drawing_end

        # Check minimum length
        length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        if length < 5:
            messagebox.showwarning("Line Too Short", "The line is too short. Please draw a longer line.")
            self.drawing_start = None
            self.drawing_end = None
            return

        # Save state for undo
        self._save_state()

        # Add line to appropriate list with direction from current selection
        new_line_coords = [int(x1), int(y1), int(x2), int(y2)]
        direction = self.direction_var.get()
        new_line = {"line": new_line_coords, "direction": direction}
        line_type = self.line_type_var.get()

        if line_type == 'solid':
            self.solid_lines.append(new_line)
            new_idx = len(self.solid_lines) - 1
            self.selected_solid_indices.clear()
            self.selected_solid_indices.add(new_idx)
        else:  # dashed
            self.dashed_lines.append(new_line)
            new_idx = len(self.dashed_lines) - 1
            self.selected_dashed_indices.clear()
            self.selected_dashed_indices.add(new_idx)

        self.modified = True

        # Exit drawing mode
        self.exit_drawing_mode()

        # Update UI
        self.update_line_lists()
        self.update_details_panel()
        self.render_visualization()
        self.update_status(f"Created new {line_type} line #{new_idx} (length: {int(length)}px)")

    def combine_selected(self):
        """Combine selected lines into a single line"""
        # Check if we have at least 2 lines selected
        total_solid = len(self.selected_solid_indices)
        total_dashed = len(self.selected_dashed_indices)
        total_selected = total_solid + total_dashed

        if total_selected < 2:
            messagebox.showwarning("Insufficient Selection",
                                 "Please select at least 2 lines to combine.")
            return

        # Check that all selected lines are of the same type
        if total_solid > 0 and total_dashed > 0:
            messagebox.showwarning("Mixed Line Types",
                                 "Cannot combine solid and dashed lines together.\n"
                                 "Please select lines of the same type.")
            return

        # Determine which type we're working with
        if total_solid > 0:
            line_type = 'solid'
            selected_indices = sorted(self.selected_solid_indices)
            lines_list = self.solid_lines
        else:
            line_type = 'dashed'
            selected_indices = sorted(self.selected_dashed_indices)
            lines_list = self.dashed_lines

        # Confirm combination
        result = messagebox.askyesno("Confirm Combination",
                                     f"Combine {len(selected_indices)} {line_type} line(s) into one?")
        if not result:
            return

        # Save state for undo
        self._save_state()

        # Collect all selected lines
        selected_lines = [lines_list[idx] for idx in selected_indices]

        # Find the combined endpoints (furthest points)
        all_points = []
        all_directions = []
        for line_item in selected_lines:
            x1, y1, x2, y2 = line_item["line"]
            all_points.append((x1, y1))
            all_points.append((x2, y2))
            direction = line_item.get("direction", "none")
            if direction != "none":
                all_directions.append(direction)

        # Find the two points that are furthest apart
        max_dist = 0
        best_pair = (all_points[0], all_points[1])

        for i in range(len(all_points)):
            for j in range(i + 1, len(all_points)):
                p1, p2 = all_points[i], all_points[j]
                dist = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
                if dist > max_dist:
                    max_dist = dist
                    best_pair = (p1, p2)

        # Create combined line
        (x1, y1), (x2, y2) = best_pair

        # Determine combined direction (prioritize non-none directions)
        combined_direction = "none"
        if all_directions:
            # If all directions are the same, use that
            unique_directions = set(all_directions)
            if len(unique_directions) == 1:
                combined_direction = all_directions[0]
            else:
                # Mixed directions - default to bidirectional
                combined_direction = "bidirectional"

        combined_line = {
            "line": [int(x1), int(y1), int(x2), int(y2)],
            "direction": combined_direction
        }

        # Remove selected lines (from end to start to maintain indices)
        for idx in reversed(selected_indices):
            del lines_list[idx]

        # Add combined line
        lines_list.append(combined_line)
        new_idx = len(lines_list) - 1

        # Update selection to the new combined line
        if line_type == 'solid':
            self.selected_solid_indices.clear()
            self.selected_solid_indices.add(new_idx)
        else:
            self.selected_dashed_indices.clear()
            self.selected_dashed_indices.add(new_idx)

        # Update UI
        self.modified = True
        self.update_line_lists()
        self.update_details_panel()
        self.render_visualization()
        self.update_status(f"Combined {len(selected_indices)} {line_type} lines into line #{new_idx}")

    def delete_selected(self):
        """Delete selected lines"""
        total_selected = len(self.selected_solid_indices) + len(self.selected_dashed_indices)

        if total_selected == 0:
            messagebox.showwarning("No Selection", "Please select lines to delete.")
            return

        # Confirm deletion
        result = messagebox.askyesno("Confirm Deletion",
                                     f"Delete {total_selected} selected line(s)?")
        if not result:
            return

        # Save state for undo
        self._save_state()

        # Delete solid lines (from end to start)
        for idx in sorted(self.selected_solid_indices, reverse=True):
            del self.solid_lines[idx]

        # Delete dashed lines (from end to start)
        for idx in sorted(self.selected_dashed_indices, reverse=True):
            del self.dashed_lines[idx]

        # Clear selection
        self.selected_solid_indices.clear()
        self.selected_dashed_indices.clear()

        # Update UI
        self.modified = True
        self.update_line_lists()
        self.update_details_panel()
        self.render_visualization()
        self.update_status(f"Deleted {total_selected} line(s)")

    def _save_state(self):
        """Save current state for undo"""
        import copy
        state = {
            'solid_lines': copy.deepcopy(self.solid_lines),
            'dashed_lines': copy.deepcopy(self.dashed_lines),
            'selected_solid': self.selected_solid_indices.copy(),
            'selected_dashed': self.selected_dashed_indices.copy()
        }
        self.history.append(state)

        # Clear redo stack when new action is performed
        self.redo_stack.clear()

        # Limit history size
        if len(self.history) > 50:
            self.history.pop(0)

    def undo(self):
        """Undo last action"""
        if not self.history:
            self.update_status("Nothing to undo")
            return

        # Save current state to redo stack
        import copy
        current_state = {
            'solid_lines': copy.deepcopy(self.solid_lines),
            'dashed_lines': copy.deepcopy(self.dashed_lines),
            'selected_solid': self.selected_solid_indices.copy(),
            'selected_dashed': self.selected_dashed_indices.copy()
        }
        self.redo_stack.append(current_state)

        # Restore previous state
        state = self.history.pop()
        self.solid_lines = state['solid_lines']
        self.dashed_lines = state['dashed_lines']
        self.selected_solid_indices = state['selected_solid']
        self.selected_dashed_indices = state['selected_dashed']

        # Update UI
        self.modified = True
        self.update_line_lists()
        self.update_details_panel()
        self.render_visualization()
        self.update_status("Undo successful")

    def redo(self):
        """Redo last undone action"""
        if not self.redo_stack:
            self.update_status("Nothing to redo")
            return

        # Save current state to history
        self._save_state()

        # Restore redo state
        state = self.redo_stack.pop()
        self.solid_lines = state['solid_lines']
        self.dashed_lines = state['dashed_lines']
        self.selected_solid_indices = state['selected_solid']
        self.selected_dashed_indices = state['selected_dashed']

        # Update UI
        self.modified = True
        self.update_line_lists()
        self.update_details_panel()
        self.render_visualization()
        self.update_status("Redo successful")

    def apply_direction_to_selected(self):
        """Apply the currently selected direction to selected lines"""
        if not self.selected_solid_indices and not self.selected_dashed_indices:
            self.update_status("No lines selected")
            return

        direction = self.direction_var.get()

        # Save state for undo
        self._save_state()

        # Apply direction to selected solid lines
        for idx in self.selected_solid_indices:
            if 0 <= idx < len(self.solid_lines):
                self.solid_lines[idx]['direction'] = direction

        # Apply direction to selected dashed lines
        for idx in self.selected_dashed_indices:
            if 0 <= idx < len(self.dashed_lines):
                self.dashed_lines[idx]['direction'] = direction

        # Update UI
        self.modified = True
        self.update_line_lists()
        self.render_visualization()
        self.update_status(f"Applied direction '{direction}' to {len(self.selected_solid_indices) + len(self.selected_dashed_indices)} lines")

    def cycle_selected_line_direction(self):
        """Cycle through directions for selected lines"""
        if not self.selected_solid_indices and not self.selected_dashed_indices:
            return

        # Direction cycle: none -> forward -> backward -> bidirectional -> coming_in -> going_out -> none
        direction_cycle = ['none', 'forward', 'backward', 'bidirectional', 'coming_in', 'going_out']

        # Get current direction of first selected line
        current_direction = 'none'
        if self.selected_solid_indices:
            idx = list(self.selected_solid_indices)[0]
            if 0 <= idx < len(self.solid_lines):
                current_direction = self.solid_lines[idx].get('direction', 'none')
        elif self.selected_dashed_indices:
            idx = list(self.selected_dashed_indices)[0]
            if 0 <= idx < len(self.dashed_lines):
                current_direction = self.dashed_lines[idx].get('direction', 'none')

        # Get next direction
        try:
            current_idx = direction_cycle.index(current_direction)
            next_direction = direction_cycle[(current_idx + 1) % len(direction_cycle)]
        except ValueError:
            next_direction = 'forward'

        # Set the direction in the UI and apply
        self.direction_var.set(next_direction)
        self.apply_direction_to_selected()

    def save(self):
        """Save modifications back to the JSON file and update overlay image"""
        if not self.modified:
            messagebox.showinfo("No Changes", "No modifications to save.")
            return

        result = messagebox.askyesno("Save Changes",
                                     f"Save changes to:\n{self.json_path}?")
        if not result:
            return

        try:
            # Update data
            self.data['solid'] = self.solid_lines
            self.data['dashed'] = self.dashed_lines

            # Save JSON
            with open(self.json_path, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)

            # Generate and save overlay image
            overlay_path = self.json_path.replace('.json', '_overlay.png')
            self._save_overlay_image(overlay_path)

            # Mark as not modified
            self.modified = False

            messagebox.showinfo("Save Successful",
                              f"Saved to:\n{self.json_path}\n\n"
                              f"Solid lines: {len(self.solid_lines)}\n"
                              f"Dashed lines: {len(self.dashed_lines)}\n\n"
                              f"Overlay image saved to:\n{overlay_path}")
            self.update_status(f"Saved {len(self.solid_lines)} solid and {len(self.dashed_lines)} dashed lines + overlay")

        except Exception as e:
            messagebox.showerror("Save Error", f"Error saving file:\n{str(e)}")
            self.update_status(f"Save failed: {str(e)}")

    def _save_overlay_image(self, output_path: str):
        """
        Save an overlay visualization of line detections on the original image

        Args:
            output_path: Path to save the overlay image
        """
        # Create a copy of the image for overlay
        overlay_img = self.image.copy()

        # Draw solid lines in green
        for line_item in self.solid_lines:
            x1, y1, x2, y2 = map(int, line_item["line"])
            direction = line_item.get("direction", "none")
            cv2.line(overlay_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw direction arrows
            if direction != "none":
                self._draw_direction_arrow_cv2(overlay_img, x1, y1, x2, y2, direction, (0, 255, 0))

        # Draw dashed lines in red
        for line_item in self.dashed_lines:
            x1, y1, x2, y2 = map(int, line_item["line"])
            direction = line_item.get("direction", "none")
            self._draw_dashed_line(overlay_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # Draw direction arrows
            if direction != "none":
                self._draw_direction_arrow_cv2(overlay_img, x1, y1, x2, y2, direction, (0, 0, 255))

        # Save overlay image
        cv2.imwrite(output_path, overlay_img)
        print(f"Saved overlay image to: {output_path}")

    def _draw_direction_arrow_cv2(self, img, x1, y1, x2, y2, direction, color):
        """
        Draw direction arrow(s) on a line using OpenCV

        Args:
            img: Image to draw on
            x1, y1, x2, y2: Line endpoints
            direction: Direction type ('forward', 'backward', 'bidirectional')
            color: Arrow color (BGR)
        """
        # Calculate line properties
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        dx, dy = x2 - x1, y2 - y1
        length = np.sqrt(dx**2 + dy**2)

        if length < 1:
            return

        # Normalize direction vector
        dx_norm, dy_norm = dx / length, dy / length

        # Arrow size (proportional to line length, max 80 pixels, min 30 pixels)
        arrow_size = int(min(80, max(30, length * 0.2)))

        def draw_arrow_at(px, py):
            """Draw a single arrow at position (px, py)"""
            # Arrow tip
            tip = (int(px + dx_norm * arrow_size * 0.5), int(py + dy_norm * arrow_size * 0.5))
            # Arrow base
            base = (int(px - dx_norm * arrow_size * 0.5), int(py - dy_norm * arrow_size * 0.5))

            # Perpendicular vector for arrow wings
            perp_x, perp_y = -dy_norm, dx_norm

            # Arrow wing points (increased wing size for better visibility)
            wing_size = arrow_size * 0.5
            wing1 = (int(tip[0] - dx_norm * wing_size + perp_x * wing_size * 0.6),
                    int(tip[1] - dy_norm * wing_size + perp_y * wing_size * 0.6))
            wing2 = (int(tip[0] - dx_norm * wing_size - perp_x * wing_size * 0.6),
                    int(tip[1] - dy_norm * wing_size - perp_y * wing_size * 0.6))

            # Draw filled triangle arrow
            arrow_points = np.array([tip, wing1, wing2], dtype=np.int32)
            cv2.fillPoly(img, [arrow_points], color)

            # Draw arrow shaft with thicker line
            cv2.line(img, base, tip, color, 4)

        if direction == "forward":
            # Arrow pointing from start to end (at center)
            draw_arrow_at(cx, cy)

        elif direction == "backward":
            # Arrow pointing from end to start (reverse at center)
            # Flip the direction
            cx_back = int(cx - dx_norm * arrow_size)
            cy_back = int(cy - dy_norm * arrow_size)

            # Temporarily reverse direction for this arrow
            dx_norm_rev, dy_norm_rev = -dx_norm, -dy_norm
            dx_norm_orig, dy_norm_orig = dx_norm, dy_norm
            dx_norm, dy_norm = dx_norm_rev, dy_norm_rev
            draw_arrow_at(cx, cy)
            dx_norm, dy_norm = dx_norm_orig, dy_norm_orig

        elif direction == "bidirectional":
            # Two arrows, one at 1/3 and one at 2/3
            pos1_x, pos1_y = int(x1 + dx * 0.33), int(y1 + dy * 0.33)
            pos2_x, pos2_y = int(x1 + dx * 0.67), int(y1 + dy * 0.67)

            draw_arrow_at(pos1_x, pos1_y)
            draw_arrow_at(pos2_x, pos2_y)

        elif direction == "coming_in":
            # Two arrows pointing toward center (merging: →←)
            pos1_x, pos1_y = int(x1 + dx * 0.33), int(y1 + dy * 0.33)
            pos2_x, pos2_y = int(x1 + dx * 0.67), int(y1 + dy * 0.67)

            # Draw first arrow (forward direction)
            draw_arrow_at(pos1_x, pos1_y)

            # Draw second arrow (reversed direction)
            # Temporarily reverse direction for this arrow
            dx_norm_orig, dy_norm_orig = dx_norm, dy_norm
            dx_norm, dy_norm = -dx_norm, -dy_norm
            draw_arrow_at(pos2_x, pos2_y)
            dx_norm, dy_norm = dx_norm_orig, dy_norm_orig

        elif direction == "going_out":
            # Two arrows pointing away from center (branching: ←→)
            pos1_x, pos1_y = int(x1 + dx * 0.33), int(y1 + dy * 0.33)
            pos2_x, pos2_y = int(x1 + dx * 0.67), int(y1 + dy * 0.67)

            # Draw first arrow (reversed direction)
            dx_norm_orig, dy_norm_orig = dx_norm, dy_norm
            dx_norm, dy_norm = -dx_norm, -dy_norm
            draw_arrow_at(pos1_x, pos1_y)
            dx_norm, dy_norm = dx_norm_orig, dy_norm_orig

            # Draw second arrow (forward direction)
            draw_arrow_at(pos2_x, pos2_y)

    def _draw_dashed_line(self, img, pt1, pt2, color, thickness, dash_length=10):
        """
        Draw a dashed line on the image

        Args:
            img: Image to draw on
            pt1: Start point (x, y)
            pt2: End point (x, y)
            color: Line color (BGR)
            thickness: Line thickness
            dash_length: Length of each dash
        """
        x1, y1 = pt1
        x2, y2 = pt2

        # Calculate line length and direction
        dx = x2 - x1
        dy = y2 - y1
        length = np.sqrt(dx**2 + dy**2)

        if length == 0:
            return

        # Normalize direction
        dx /= length
        dy /= length

        # Draw dashes
        current_length = 0
        draw_dash = True

        while current_length < length:
            # Calculate start and end of current segment
            start_x = int(x1 + dx * current_length)
            start_y = int(y1 + dy * current_length)

            current_length += dash_length
            if current_length > length:
                current_length = length

            end_x = int(x1 + dx * current_length)
            end_y = int(y1 + dy * current_length)

            # Draw or skip based on dash pattern
            if draw_dash:
                cv2.line(img, (start_x, start_y), (end_x, end_y), color, thickness)

            draw_dash = not draw_dash

    def update_status(self, message: str):
        """Update status bar message"""
        self.status_bar.config(text=message)
        self.root.update_idletasks()

    def on_closing(self):
        """Handle window closing event"""
        if self.modified:
            result = messagebox.askyesnocancel(
                "Unsaved Changes",
                "You have unsaved changes. Do you want to save before exiting?"
            )

            if result is None:  # Cancel
                return
            elif result:  # Yes
                self.save()

        self.root.destroy()

    def run(self):
        """Run the GUI event loop"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()


def main():
    parser = argparse.ArgumentParser(
        description='Interactive Line Editor for P&ID Line Detection Results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python interactive_line_editor.py --json output/image_step4_lines.json

Keyboard Shortcuts:
  D             - Enter drawing mode
  C             - Combine selected lines
  S             - Toggle solid lines visibility
  H             - Toggle dashed lines visibility
  R             - Apply direction to selected lines
  Escape        - Exit drawing mode or clear selection
  Ctrl+Z        - Undo
  Ctrl+Y        - Redo
  Ctrl+S        - Save
  Delete        - Delete selected lines
        """
    )

    parser.add_argument('--json', type=str, required=True,
                       help='Path to the step4_lines.json file')

    args = parser.parse_args()

    # Validate input
    if not os.path.exists(args.json):
        print(f"ERROR: JSON file not found: {args.json}")
        sys.exit(1)

    print("=" * 60)
    print("Interactive Line Editor - Stage 2")
    print("=" * 60)
    print(f"JSON: {args.json}")
    print("=" * 60)
    print("\nLoading data...")

    # Create and run editor
    try:
        editor = LineEditor(json_path=args.json)

        print("=" * 60)
        print(f"Image: {editor.image_path}")
        print(f"Lines: Solid={len(editor.solid_lines)}, Dashed={len(editor.dashed_lines)}")
        print("=" * 60)
        print("\nStarting interactive editor...")

        editor.run()

        print("\nEditor closed.")

    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
