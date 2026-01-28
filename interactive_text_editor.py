#!/usr/bin/env python3
"""
Interactive Text Editor for P&ID Text Detection Results

This script provides an interactive GUI to edit text detections from process_single_pnid.py.
Users can:
- Delete text bounding boxes by clicking on them
- Combine/merge multiple text bounding boxes (text is concatenated)
- Edit the detected text within a bounding box
- Undo/redo operations
- Save modifications back to the JSON file in-place

The edited results are saved in-place and can be used for downstream processing.

Usage:
    python interactive_text_editor.py --json /path/to/step3_text.json
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
from tkinter import ttk, messagebox, simpledialog
from PIL import Image


class TextEditor:
    """Interactive text editor with GUI"""

    def __init__(self, json_path: str):
        """
        Initialize the text editor

        Args:
            json_path: Path to the step3_text.json file
        """
        self.json_path = json_path

        # Load JSON
        self._load_json()

        # Derive image path
        self._derive_image_path()

        # Load image
        self._load_image()

        # State management
        self.selected_indices = set()  # Indices of selected text boxes
        self.selection_order = []  # Order in which text boxes were selected
        self.history = []  # For undo functionality
        self.redo_stack = []  # For redo functionality
        self.modified = False  # Track if changes have been made

        # UI state
        self.show_labels = True
        self.alpha = 0.3  # Transparency for bbox overlay

        # Edit mode
        self.editing_index = None  # Index of text being edited

        # Initialize GUI
        self._setup_gui()

    def _load_json(self):
        """Load text detection JSON file"""
        if not os.path.exists(self.json_path):
            raise FileNotFoundError(f"JSON file not found: {self.json_path}")

        with open(self.json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        # Ensure data is a list of detections
        if isinstance(self.data, dict):
            # Old format might have wrapped data
            if 'detections' in self.data:
                detections = self.data['detections']
            else:
                raise ValueError("JSON format not recognized. Expected list of detections or dict with 'detections' key.")
        elif isinstance(self.data, list):
            detections = self.data
        else:
            raise ValueError("JSON format not recognized.")

        # Filter out detections without text (empty or whitespace only)
        original_count = len(detections)
        self.detections = [det for det in detections if det.get('text', '').strip()]
        filtered_count = original_count - len(self.detections)

        print(f"Loaded {len(self.detections)} text detections from: {self.json_path}")
        if filtered_count > 0:
            print(f"  (Filtered out {filtered_count} empty text boxes)")

    def _derive_image_path(self):
        """Derive image path from JSON path"""
        json_path = Path(self.json_path)
        json_dir = json_path.parent

        # Try to find the resized image used for processing
        # Expected pattern: {name}_step3_text.json -> look for resized image
        # First check if there's an image_path in a related JSON
        step2_json = json_dir / json_path.name.replace('_step3_text.json', '_step2_merged.json')
        step1_json = json_dir / json_path.name.replace('_step3_text.json', '_step1_dets.json')

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
        self.root.title("Interactive Text Editor - Stage 1")
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
        self.root.bind('<Escape>', lambda e: self.clear_selection())
        self.root.bind('e', lambda e: self.edit_selected_text())
        self.root.bind('c', lambda e: self.combine_selected())

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
        edit_menu.add_command(label="Edit Text", command=self.edit_selected_text, accelerator="E")
        edit_menu.add_command(label="Combine Selected", command=self.combine_selected, accelerator="C")
        edit_menu.add_separator()
        edit_menu.add_command(label="Delete Selected", command=self.delete_selected, accelerator="Del")
        edit_menu.add_command(label="Clear Selection", command=self.clear_selection, accelerator="Esc")

        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Toggle Labels", command=self.toggle_labels)

    def _create_toolbar(self):
        """Create toolbar with action buttons"""
        toolbar = ttk.Frame(self.root)
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        # Action buttons
        ttk.Button(toolbar, text="Edit Text (E)", command=self.edit_selected_text).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Combine Selected (C)", command=self.combine_selected).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Delete Selected", command=self.delete_selected).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Clear Selection", command=self.clear_selection).pack(side=tk.LEFT, padx=2)

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)

        ttk.Button(toolbar, text="Undo", command=self.undo).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Redo", command=self.redo).pack(side=tk.LEFT, padx=2)

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)

        ttk.Button(toolbar, text="Save", command=self.save).pack(side=tk.LEFT, padx=2)

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)

        # Toggle labels
        self.show_labels_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(toolbar, text="Show Labels", command=self.toggle_labels,
                       variable=self.show_labels_var).pack(side=tk.LEFT, padx=2)

    def _create_main_layout(self):
        """Create main layout with canvas and text list"""
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
        self.fig.canvas.mpl_connect('button_press_event', self.on_canvas_click)

        # Right side: Text list
        right_frame = ttk.Frame(main_frame, width=350)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(5, 0))
        right_frame.pack_propagate(False)

        # Text list label
        ttk.Label(right_frame, text="Detected Text", font=('Arial', 12, 'bold')).pack(pady=5)

        # Info label
        self.info_label = ttk.Label(right_frame, text=f"Total: {len(self.detections)} texts")
        self.info_label.pack(pady=2)

        # Search/filter frame
        filter_frame = ttk.Frame(right_frame)
        filter_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(filter_frame, text="Filter:").pack(side=tk.LEFT)
        self.filter_var = tk.StringVar()
        self.filter_var.trace_add('write', lambda *args: self.update_text_list())
        filter_entry = ttk.Entry(filter_frame, textvariable=self.filter_var)
        filter_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        # Scrollable text list
        list_frame = ttk.Frame(right_frame)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.text_listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set,
                                       selectmode=tk.EXTENDED, font=('Courier', 9))
        self.text_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.text_listbox.yview)

        # Bind selection event
        self.text_listbox.bind('<<ListboxSelect>>', self.on_list_selection)
        self.text_listbox.bind('<Double-Button-1>', lambda e: self.edit_selected_text())

        # Populate text list
        self.update_text_list()

        # Details panel
        details_frame = ttk.LabelFrame(right_frame, text="Selection Details", padding=5)
        details_frame.pack(fill=tk.X, padx=5, pady=5)

        self.details_text = tk.Text(details_frame, height=8, wrap=tk.WORD, font=('Courier', 9))
        self.details_text.pack(fill=tk.BOTH, expand=True)
        self.update_details_panel()

    def _create_status_bar(self):
        """Create status bar"""
        self.status_bar = ttk.Label(self.root, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def update_text_list(self, *args):
        """Update the text list display"""
        self.text_listbox.delete(0, tk.END)

        filter_text = self.filter_var.get().lower()

        for i, det in enumerate(self.detections):
            text = det.get('text', '')
            bbox = det.get('bbox', [0, 0, 0, 0])
            score = det.get('score', 0.0)

            display_text = f"#{i:3d} | {text[:30]:30s} | Score: {score:.2f}"

            # Apply filter
            if filter_text and filter_text not in display_text.lower():
                continue

            self.text_listbox.insert(tk.END, display_text)

            # Highlight selected
            if i in self.selected_indices:
                self.text_listbox.itemconfig(tk.END, bg='lightblue')

        # Update info label
        self.info_label.config(text=f"Total: {len(self.detections)} texts | Selected: {len(self.selected_indices)}")

    def update_details_panel(self):
        """Update the details panel with selection info"""
        self.details_text.delete(1.0, tk.END)

        if not self.selected_indices:
            self.details_text.insert(tk.END, "No text selected.\n\n")
            self.details_text.insert(tk.END, "Click on text boxes in the image\n")
            self.details_text.insert(tk.END, "or select from the list.")
        else:
            self.details_text.insert(tk.END, f"Selected: {len(self.selected_indices)} texts\n\n")

            if len(self.selected_indices) == 1:
                idx = list(self.selected_indices)[0]
                det = self.detections[idx]
                text = det.get('text', '')
                bbox = det.get('bbox', [0, 0, 0, 0])
                score = det.get('score', 0.0)

                self.details_text.insert(tk.END, f"Index: #{idx}\n")
                self.details_text.insert(tk.END, f"Text: {text}\n")
                self.details_text.insert(tk.END, f"Score: {score:.4f}\n")
                self.details_text.insert(tk.END, f"BBox: {bbox}\n")
                self.details_text.insert(tk.END, f"Size: {bbox[2]-bbox[0]}x{bbox[3]-bbox[1]}\n")
            else:
                # Show combined text in selection order
                combined_text = " ".join(self.detections[i].get('text', '') for i in self.selection_order)
                avg_score = sum(self.detections[i].get('score', 0.0) for i in self.selected_indices) / len(self.selected_indices)

                self.details_text.insert(tk.END, f"Selection order: {self.selection_order}\n")
                self.details_text.insert(tk.END, f"Combined text: {combined_text}\n")
                self.details_text.insert(tk.END, f"Avg Score: {avg_score:.4f}\n")

    def render_visualization(self):
        """Render the current visualization"""
        self.ax.clear()

        # Show original image
        self.ax.imshow(self.image_rgb)

        # Draw all text bounding boxes
        for i, det in enumerate(self.detections):
            bbox = det.get('bbox', [0, 0, 0, 0])
            text = det.get('text', '')
            x1, y1, x2, y2 = bbox

            # Color based on selection
            if i in self.selected_indices:
                color = 'red'
                linewidth = 3
                alpha = 0.8
            else:
                color = 'green'
                linewidth = 2
                alpha = self.alpha

            # Draw rectangle
            from matplotlib.patches import Rectangle
            rect = Rectangle((x1, y1), x2-x1, y2-y1,
                           linewidth=linewidth, edgecolor=color, facecolor='none', alpha=alpha)
            self.ax.add_patch(rect)

            # Add label if enabled
            if self.show_labels:
                if i in self.selected_indices:
                    self.ax.text(x1, max(0, y1 - 5), f"#{i}: {text}", color='red', fontsize=8,
                               fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
                else:
                    self.ax.text(x1, max(0, y1 - 5), f"{text}", color='white', fontsize=7,
                               bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.5))

        self.ax.axis('off')
        title = f'Text Editor - {len(self.detections)} texts ({len(self.selected_indices)} selected)'
        self.ax.set_title(title, fontsize=10)
        self.canvas.draw()

    def on_canvas_click(self, event):
        """Handle canvas click events"""
        if event.inaxes != self.ax:
            return

        x, y = int(event.xdata), int(event.ydata)

        # Check if click is within image bounds
        if x < 0 or x >= self.image.shape[1] or y < 0 or y >= self.image.shape[0]:
            return

        # Find which text box was clicked (check from top to bottom for overlaps)
        clicked_idx = None
        for i in range(len(self.detections) - 1, -1, -1):
            bbox = self.detections[i].get('bbox', [0, 0, 0, 0])
            x1, y1, x2, y2 = bbox
            if x1 <= x <= x2 and y1 <= y <= y2:
                clicked_idx = i
                break

        if clicked_idx is not None:
            # Toggle selection
            if clicked_idx in self.selected_indices:
                self.selected_indices.remove(clicked_idx)
                self.selection_order.remove(clicked_idx)
            else:
                self.selected_indices.add(clicked_idx)
                self.selection_order.append(clicked_idx)

            self.update_text_list()
            self.update_details_panel()
            self.render_visualization()
            self.update_status(f"Text #{clicked_idx} {'selected' if clicked_idx in self.selected_indices else 'deselected'}")

    def on_list_selection(self, event):
        """Handle list selection events"""
        selected_listbox_indices = self.text_listbox.curselection()

        # Map listbox indices back to detection indices (accounting for filtering)
        new_selected = []
        for idx in selected_listbox_indices:
            display_text = self.text_listbox.get(idx)
            # Extract detection index from text (format: "#  i | ...")
            det_idx = int(display_text.split('|')[0].strip().replace('#', ''))
            new_selected.append(det_idx)

        # Update selection order: keep existing order, add new ones at end
        old_selected = set(self.selection_order)
        new_selected_set = set(new_selected)

        # Remove deselected items
        self.selection_order = [idx for idx in self.selection_order if idx in new_selected_set]

        # Add newly selected items
        for idx in new_selected:
            if idx not in old_selected:
                self.selection_order.append(idx)

        self.selected_indices = new_selected_set

        self.update_details_panel()
        self.render_visualization()

    def toggle_labels(self):
        """Toggle text labels on/off"""
        self.show_labels = self.show_labels_var.get()
        self.render_visualization()

    def clear_selection(self):
        """Clear all selected texts"""
        self.selected_indices.clear()
        self.selection_order.clear()
        self.update_text_list()
        self.update_details_panel()
        self.render_visualization()
        self.update_status("Selection cleared")

    def edit_selected_text(self):
        """Edit the text of the selected bounding box"""
        if len(self.selected_indices) != 1:
            messagebox.showwarning("Invalid Selection",
                                  "Please select exactly one text box to edit.")
            return

        idx = list(self.selected_indices)[0]
        det = self.detections[idx]
        current_text = det.get('text', '')

        # Show dialog to edit text
        new_text = simpledialog.askstring("Edit Text",
                                         f"Edit text for box #{idx}:",
                                         initialvalue=current_text,
                                         parent=self.root)

        if new_text is not None and new_text != current_text:
            # Save state for undo
            self._save_state()

            # Update text
            self.detections[idx]['text'] = new_text
            self.modified = True

            # Update UI
            self.update_text_list()
            self.update_details_panel()
            self.render_visualization()
            self.update_status(f"Updated text for box #{idx}")

    def combine_selected(self):
        """Combine selected text boxes into one"""
        if len(self.selected_indices) < 2:
            messagebox.showwarning("Insufficient Selection",
                                  "Please select at least 2 text boxes to combine.")
            return

        # Save state for undo
        self._save_state()

        # Use selection order (first selected comes first in combined text)
        ordered_indices = self.selection_order.copy()

        # Combine text in selection order (space-separated)
        combined_text = " ".join(self.detections[i].get('text', '') for i in ordered_indices)

        # Compute union bounding box
        all_bboxes = [self.detections[i]['bbox'] for i in ordered_indices]
        x1 = min(bbox[0] for bbox in all_bboxes)
        y1 = min(bbox[1] for bbox in all_bboxes)
        x2 = max(bbox[2] for bbox in all_bboxes)
        y2 = max(bbox[3] for bbox in all_bboxes)
        combined_bbox = [x1, y1, x2, y2]

        # Average score
        combined_score = sum(self.detections[i].get('score', 0.0) for i in ordered_indices) / len(ordered_indices)

        # Remove old detections (from end to start to preserve indices)
        # Sort by index for deletion to avoid index shifting issues
        for idx in sorted(ordered_indices, reverse=True):
            del self.detections[idx]

        # Add combined detection at the position of the first selected
        insert_position = min(ordered_indices)
        self.detections.insert(insert_position, {
            'bbox': combined_bbox,
            'text': combined_text,
            'score': combined_score
        })

        # Clear selection and select the new combined text
        self.selected_indices.clear()
        self.selection_order.clear()
        self.selected_indices.add(insert_position)
        self.selection_order.append(insert_position)

        # Update UI
        self.modified = True
        self.update_text_list()
        self.update_details_panel()
        self.render_visualization()
        self.update_status(f"Combined {len(ordered_indices)} text boxes into text #{insert_position}")

    def delete_selected(self):
        """Delete selected text boxes"""
        if not self.selected_indices:
            messagebox.showwarning("No Selection", "Please select text boxes to delete.")
            return

        # Confirm deletion
        result = messagebox.askyesno("Confirm Deletion",
                                     f"Delete {len(self.selected_indices)} selected text box(es)?")
        if not result:
            return

        # Save state for undo
        self._save_state()

        # Sort indices in descending order to delete from end
        sorted_indices = sorted(self.selected_indices, reverse=True)

        # Delete detections
        for idx in sorted_indices:
            del self.detections[idx]

        # Clear selection
        self.selected_indices.clear()
        self.selection_order.clear()

        # Update UI
        self.modified = True
        self.update_text_list()
        self.update_details_panel()
        self.render_visualization()
        self.update_status(f"Deleted {len(sorted_indices)} text box(es)")

    def _save_state(self):
        """Save current state for undo"""
        # Deep copy the current state
        import copy
        state = {
            'detections': copy.deepcopy(self.detections),
            'selected': self.selected_indices.copy(),
            'selection_order': self.selection_order.copy()
        }
        self.history.append(state)

        # Clear redo stack when new action is performed
        self.redo_stack.clear()

        # Limit history size to prevent memory issues
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
            'detections': copy.deepcopy(self.detections),
            'selected': self.selected_indices.copy(),
            'selection_order': self.selection_order.copy()
        }
        self.redo_stack.append(current_state)

        # Restore previous state
        state = self.history.pop()
        self.detections = state['detections']
        self.selected_indices = state['selected']
        self.selection_order = state.get('selection_order', list(state['selected']))

        # Update UI
        self.modified = True
        self.update_text_list()
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
        self.detections = state['detections']
        self.selected_indices = state['selected']
        self.selection_order = state.get('selection_order', list(state['selected']))

        # Update UI
        self.modified = True
        self.update_text_list()
        self.update_details_panel()
        self.render_visualization()
        self.update_status("Redo successful")

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
            # Save JSON
            with open(self.json_path, 'w', encoding='utf-8') as f:
                json.dump(self.detections, f, ensure_ascii=False, indent=2)

            # Generate and save overlay image
            overlay_path = self.json_path.replace('.json', '_overlay.png')
            self._save_overlay_image(overlay_path)

            # Mark as not modified
            self.modified = False

            messagebox.showinfo("Save Successful",
                              f"Saved {len(self.detections)} text detections to:\n{self.json_path}\n\nOverlay image saved to:\n{overlay_path}")
            self.update_status(f"Saved {len(self.detections)} text detections + overlay")

        except Exception as e:
            messagebox.showerror("Save Error", f"Error saving file:\n{str(e)}")
            self.update_status(f"Save failed: {str(e)}")

    def _save_overlay_image(self, output_path: str):
        """
        Save an overlay visualization of text detections on the original image

        Args:
            output_path: Path to save the overlay image
        """
        # Create a copy of the image for overlay
        overlay_img = self.image.copy()

        # Draw each text detection
        for det in self.detections:
            bbox = det.get('bbox', [])
            text = det.get('text', '')

            if len(bbox) == 4:
                x1, y1, x2, y2 = map(int, bbox)

                # Draw bounding box
                cv2.rectangle(overlay_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Draw text label with background
                if text:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.5
                    thickness = 1

                    # Get text size for background
                    (text_width, text_height), baseline = cv2.getTextSize(
                        text, font, font_scale, thickness
                    )

                    # Draw text background
                    text_x = x1
                    text_y = y1 - 5
                    if text_y < text_height:
                        text_y = y2 + text_height + 5

                    cv2.rectangle(
                        overlay_img,
                        (text_x, text_y - text_height - baseline),
                        (text_x + text_width, text_y + baseline),
                        (0, 255, 0),
                        -1
                    )

                    # Draw text
                    cv2.putText(
                        overlay_img,
                        text,
                        (text_x, text_y),
                        font,
                        font_scale,
                        (0, 0, 0),
                        thickness
                    )

        # Save overlay image
        cv2.imwrite(output_path, overlay_img)
        print(f"Saved overlay image to: {output_path}")

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
        description='Interactive Text Editor for P&ID Text Detection Results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python interactive_text_editor.py --json output/image_step3_text.json

Keyboard Shortcuts:
  E             - Edit selected text
  C             - Combine selected text boxes
  Escape        - Clear selection
  Ctrl+Z        - Undo
  Ctrl+Y        - Redo
  Ctrl+S        - Save
  Delete        - Delete selected text boxes
        """
    )

    parser.add_argument('--json', type=str, required=True,
                       help='Path to the step3_text.json file')

    args = parser.parse_args()

    # Validate input
    if not os.path.exists(args.json):
        print(f"ERROR: JSON file not found: {args.json}")
        sys.exit(1)

    print("=" * 60)
    print("Interactive Text Editor - Stage 1")
    print("=" * 60)
    print(f"JSON: {args.json}")
    print("=" * 60)
    print("\nLoading data...")

    # Create and run editor
    try:
        editor = TextEditor(json_path=args.json)

        print("=" * 60)
        print(f"Image:  {editor.image_path}")
        print(f"Texts:  {len(editor.detections)}")
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
