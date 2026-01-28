#!/usr/bin/env python3
"""
Interactive Symbol Clustering and Classification for P&ID Diagrams

This script builds on top of SAM2 AMG inference outputs to:
1. Extract all detected symbol patches
2. Cluster them using embeddings (CLIP, DINOv2, or ViT features)
3. Present clusters to user for labeling
4. Apply labels to all symbols in each cluster

Usage:
    python interactive_symbol_classifier.py --image_path /path/to/pid.jpg --masks_path /path/to/masks.npz

Or with JSON results:
    python interactive_symbol_classifier.py --results_json /path/to/results.json
"""

import sys
import os
import argparse
import numpy as np
import torch
import cv2
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
from PIL import Image
import json
from tqdm import tqdm
from collections import defaultdict

# Embedding model imports
try:
    from transformers import CLIPProcessor, CLIPModel
    CLIP_AVAILABLE = True
except ImportError:
    print("Warning: CLIP not available. Install with: pip install transformers")
    CLIP_AVAILABLE = False

try:
    from transformers import AutoImageProcessor, AutoModel
    DINOV2_AVAILABLE = True
except ImportError:
    print("Warning: DINOv2 not available. Install with: pip install transformers")
    DINOV2_AVAILABLE = False

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    print("Warning: HDBSCAN not available. Install with: pip install hdbscan")
    print("Falling back to KMeans clustering")
    HDBSCAN_AVAILABLE = False

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Try to import UMAP with proper error handling
UMAP_AVAILABLE = False
umap_reducer = None
try:
    # Try the correct umap-learn package first
    from umap import UMAP as umap_reducer
    UMAP_AVAILABLE = True
except (ImportError, AttributeError):
    try:
        # Try alternative import path
        from umap.umap_ import UMAP as umap_reducer
        UMAP_AVAILABLE = True
    except (ImportError, AttributeError):
        print("Warning: UMAP not available. Install with: pip install umap-learn")
        print("Note: If you have 'umap' installed, uninstall it first: pip uninstall umap")
        print("Then install the correct package: pip install umap-learn")
        print("Will use PCA for 2D visualization instead.")
        UMAP_AVAILABLE = False


class SymbolEmbedder:
    """Generate embeddings for symbol patches using various models"""

    def __init__(self, model_type: str = 'clip', device: str = 'cuda'):
        """
        Initialize the symbol embedder

        Args:
            model_type: Type of model to use ('clip', 'dinov2', 'vit')
            device: Device to run on
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model_type = model_type

        print(f"Initializing {model_type.upper()} embedder on {self.device}...")

        if model_type == 'clip':
            if not CLIP_AVAILABLE:
                raise ImportError("CLIP not available. Install transformers.")
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.embedding_dim = 512

        elif model_type == 'dinov2':
            if not DINOV2_AVAILABLE:
                raise ImportError("DINOv2 not available. Install transformers.")
            model_name = "facebook/dinov2-base"
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
            self.embedding_dim = 768

        elif model_type == 'vit':
            if not DINOV2_AVAILABLE:
                raise ImportError("ViT not available. Install transformers.")
            model_name = "google/vit-base-patch16-224"
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
            self.embedding_dim = 768

        else:
            raise ValueError(f"Unknown model type: {model_type}")

        self.model.eval()
        print(f"Embedder initialized with dimension: {self.embedding_dim}")

    def extract_embeddings(self, image_patches: List[np.ndarray],
                          batch_size: int = 32) -> np.ndarray:
        """
        Extract embeddings from image patches

        Args:
            image_patches: List of image patches as numpy arrays
            batch_size: Batch size for processing

        Returns:
            Array of embeddings with shape (N, embedding_dim)
        """
        embeddings = []

        print(f"Extracting embeddings from {len(image_patches)} patches...")

        with torch.no_grad():
            for i in tqdm(range(0, len(image_patches), batch_size)):
                batch = image_patches[i:i + batch_size]

                # Convert to PIL Images
                pil_images = []
                for patch in batch:
                    if patch.shape[-1] == 3:
                        # BGR to RGB
                        patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
                    else:
                        patch_rgb = patch
                    pil_images.append(Image.fromarray(patch_rgb))

                # Process batch
                if self.model_type == 'clip':
                    inputs = self.processor(images=pil_images, return_tensors="pt", padding=True)
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    outputs = self.model.get_image_features(**inputs)
                    batch_embeddings = outputs.cpu().numpy()
                else:
                    inputs = self.processor(images=pil_images, return_tensors="pt")
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    outputs = self.model(**inputs)
                    # Use CLS token or mean pooling
                    if hasattr(outputs, 'last_hidden_state'):
                        batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                    else:
                        batch_embeddings = outputs.pooler_output.cpu().numpy()

                embeddings.append(batch_embeddings)

        embeddings = np.vstack(embeddings)
        print(f"Extracted embeddings with shape: {embeddings.shape}")

        # Normalize embeddings
        embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)

        return embeddings


class SymbolClusterer:
    """Cluster symbol embeddings"""

    def __init__(self, method: str = 'hdbscan', n_clusters: Optional[int] = None,
                 sensitivity: str = 'medium'):
        """
        Initialize the clusterer

        Args:
            method: Clustering method ('hdbscan', 'kmeans')
            n_clusters: Number of clusters (for kmeans, or estimate for hdbscan)
            sensitivity: Clustering sensitivity ('low', 'medium', 'high', 'very_high')
                        - low: Fewer, broader clusters
                        - medium: Balanced (default)
                        - high: More fine-grained clusters
                        - very_high: Maximum sensitivity, many small clusters
        """
        self.method = method
        self.n_clusters = n_clusters
        self.sensitivity = sensitivity
        self.labels_ = None
        self.clusterer = None

    def fit(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Cluster embeddings

        Args:
            embeddings: Array of embeddings (N, D)

        Returns:
            Cluster labels (N,)
        """
        print(f"Clustering {len(embeddings)} embeddings using {self.method.upper()}...")

        if self.method == 'hdbscan' and HDBSCAN_AVAILABLE:
            # Use HDBSCAN for automatic cluster detection
            # Adjust parameters based on sensitivity
            if self.sensitivity == 'low':
                divisor = 20  # Fewer, larger clusters
                min_samples_ratio = 0.7
            elif self.sensitivity == 'medium':
                divisor = 50  # Default
                min_samples_ratio = 0.5
            elif self.sensitivity == 'high':
                divisor = 100  # More clusters
                min_samples_ratio = 0.3
            else:  # very_high
                divisor = 150  # Maximum sensitivity
                min_samples_ratio = 0.2

            min_cluster_size = max(2, len(embeddings) // divisor)
            min_samples = max(1, int(min_cluster_size * min_samples_ratio))

            print(f"Using sensitivity: {self.sensitivity}")
            print(f"  min_cluster_size: {min_cluster_size}")
            print(f"  min_samples: {min_samples}")

            self.clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric='euclidean',
                cluster_selection_method='eom',
                cluster_selection_epsilon=0.0  # More sensitive to small clusters
            )
            self.labels_ = self.clusterer.fit_predict(embeddings)

            # Count clusters (excluding noise labeled as -1)
            n_clusters = len(set(self.labels_)) - (1 if -1 in self.labels_ else 0)
            n_noise = list(self.labels_).count(-1)

            print(f"HDBSCAN found {n_clusters} clusters")
            print(f"Noise points: {n_noise} ({n_noise/len(embeddings)*100:.1f}%)")

        else:
            # Fallback to KMeans
            if self.n_clusters is None:
                # Estimate number of clusters using elbow method
                self.n_clusters = self._estimate_n_clusters(embeddings)

            print(f"Using KMeans with {self.n_clusters} clusters")
            self.clusterer = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
            self.labels_ = self.clusterer.fit_predict(embeddings)

        # Print cluster statistics
        unique_labels, counts = np.unique(self.labels_, return_counts=True)
        print("\nCluster distribution:")
        for label, count in zip(unique_labels, counts):
            if label == -1:
                print(f"  Noise: {count} symbols")
            else:
                print(f"  Cluster {label}: {count} symbols")

        return self.labels_

    def _estimate_n_clusters(self, embeddings: np.ndarray,
                            max_clusters: int = None) -> int:
        """
        Estimate optimal number of clusters using elbow method

        Args:
            embeddings: Embeddings array
            max_clusters: Maximum number of clusters to try (auto if None)

        Returns:
            Estimated number of clusters
        """
        n_samples = len(embeddings)

        # Adjust max_clusters based on sensitivity
        if max_clusters is None:
            if self.sensitivity == 'low':
                max_clusters = 20
            elif self.sensitivity == 'medium':
                max_clusters = 30
            elif self.sensitivity == 'high':
                max_clusters = 50
            else:  # very_high
                max_clusters = 80

        max_k = min(max_clusters, n_samples // 3)  # Changed from // 5 to // 3
        max_k = max(2, max_k)

        print(f"Estimating optimal number of clusters (max={max_k})...")

        inertias = []
        K_range = range(2, max_k + 1)

        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=3)
            kmeans.fit(embeddings)
            inertias.append(kmeans.inertia_)

        # Find elbow using difference method
        if len(inertias) >= 3:
            diffs = np.diff(inertias)
            second_diffs = np.diff(diffs)
            elbow_idx = np.argmax(second_diffs) + 2  # +2 because of two diffs
            optimal_k = list(K_range)[elbow_idx]
        else:
            optimal_k = max_k

        print(f"Estimated optimal clusters: {optimal_k}")
        return optimal_k


class InteractiveClusterLabeler:
    """Interactive GUI for labeling symbol clusters"""

    def __init__(self, image_patches: List[np.ndarray],
                 cluster_labels: np.ndarray,
                 masks_info: List[Dict[str, Any]]):
        """
        Initialize the interactive labeler

        Args:
            image_patches: List of symbol image patches
            cluster_labels: Cluster assignments for each patch
            masks_info: Metadata for each mask/symbol
        """
        self.image_patches = image_patches
        self.cluster_labels = cluster_labels.copy()  # Make a copy so we can modify it
        self.masks_info = masks_info

        # Organize patches by cluster
        self.clusters = defaultdict(list)
        for idx, label in enumerate(self.cluster_labels):
            self.clusters[label].append(idx)

        # Sort clusters by size (largest first)
        self.cluster_ids = sorted(self.clusters.keys(),
                                 key=lambda x: len(self.clusters[x]),
                                 reverse=True)

        # Remove noise cluster (-1) from main list if present
        if -1 in self.cluster_ids:
            self.cluster_ids.remove(-1)
            self.cluster_ids.append(-1)  # Add at end

        # Labels dictionary: cluster_id -> label_name
        self.labels = {}

        # Current cluster being labeled
        self.current_cluster_idx = 0

        # Selection state for moving symbols
        self.selected_indices = set()  # Indices of selected symbols in current cluster

        # Next available cluster ID for creating new clusters
        self.next_cluster_id = max(self.cluster_ids) + 1 if self.cluster_ids else 0

        print(f"\nReady to label {len(self.cluster_ids)} clusters")
        print(f"Total symbols: {len(image_patches)}")

    def run(self, existing_label_names: Optional[Dict[int, str]] = None) -> Dict[int, str]:
        """
        Run the interactive labeling session

        Args:
            existing_label_names: Optional dictionary of existing cluster labels to pre-populate

        Returns:
            Dictionary mapping cluster IDs to label names
        """
        # Pre-populate with existing labels if provided
        if existing_label_names:
            self.labels = existing_label_names.copy()
            print(f"\nLoaded {len(existing_label_names)} existing cluster labels")

        print("\n" + "="*60)
        print("INTERACTIVE CLUSTER LABELING")
        print("="*60)
        print("\nInstructions:")
        print("  - View representative symbols from each cluster")
        print("  - Enter a label name for each cluster")
        print("  - Click on symbols to select them (highlighted in red)")
        print("  - Press 'Move Selected' to move selected symbols to another cluster")
        print("  - Press 'New Cluster' to create a new cluster from selected symbols")
        print("  - Press 'Clear Selection' to deselect all symbols")
        print("  - Press 'Next' to move to the next cluster")
        print("  - Press 'Skip' to skip labeling a cluster")
        print("  - Press 'Done' when finished")

        if existing_label_names:
            print("\n  NOTE: Existing classifications have been loaded.")
            print("        You can review and modify them as needed.")

        print("="*60 + "\n")

        self._show_cluster_labeling_interface()

        return self.labels

    def _force_linear_scale_on_all_axes(self):
        """Force linear scale on all axes in the figure"""
        for ax in self.fig.get_axes():
            try:
                ax.set_xscale('linear')
                ax.set_yscale('linear')
            except:
                pass

    def _safe_draw(self):
        """Safely draw the figure with linear scale enforcement"""
        try:
            self._force_linear_scale_on_all_axes()
            self.fig.canvas.draw_idle()
        except ValueError as e:
            if "log-scaled" in str(e):
                # Force again and retry
                self._force_linear_scale_on_all_axes()
                try:
                    self.fig.canvas.draw_idle()
                except:
                    pass  # Suppress if it still fails
            else:
                pass  # Suppress other drawing errors

    def _show_cluster_labeling_interface(self):
        """Show matplotlib-based interactive labeling interface"""

        # Create figure
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.suptitle('Interactive Symbol Cluster Labeling', fontsize=16, fontweight='bold')

        # Create grid for symbol display (4x8 grid for up to 32 samples)
        gs = self.fig.add_gridspec(5, 8, hspace=0.3, wspace=0.3,
                                   top=0.85, bottom=0.25, left=0.05, right=0.95)

        # Create axes for symbol display
        self.symbol_axes = []
        for i in range(4):
            for j in range(8):
                ax = self.fig.add_subplot(gs[i, j])
                ax.axis('off')
                ax.set_xscale('linear')  # Prevent log scale errors
                ax.set_yscale('linear')  # Prevent log scale errors
                self.symbol_axes.append(ax)

        # Create info text area
        self.info_ax = self.fig.add_subplot(gs[4, :])
        self.info_ax.axis('off')
        self.info_ax.set_xscale('linear')
        self.info_ax.set_yscale('linear')
        self.info_text = self.info_ax.text(0.5, 0.5, '',
                                           transform=self.info_ax.transAxes,
                                           ha='center', va='center', fontsize=12)

        # Create buttons and text input with explicit linear scale
        # Row 1: Navigation buttons
        button_ax_prev = plt.axes([0.05, 0.12, 0.08, 0.04])
        button_ax_prev.set_xscale('linear')
        button_ax_prev.set_yscale('linear')

        button_ax_next = plt.axes([0.14, 0.12, 0.08, 0.04])
        button_ax_next.set_xscale('linear')
        button_ax_next.set_yscale('linear')

        button_ax_skip = plt.axes([0.23, 0.12, 0.08, 0.04])
        button_ax_skip.set_xscale('linear')
        button_ax_skip.set_yscale('linear')

        button_ax_done = plt.axes([0.32, 0.12, 0.08, 0.04])
        button_ax_done.set_xscale('linear')
        button_ax_done.set_yscale('linear')

        # Row 2: Cluster management buttons
        button_ax_clear = plt.axes([0.45, 0.12, 0.12, 0.04])
        button_ax_clear.set_xscale('linear')
        button_ax_clear.set_yscale('linear')

        button_ax_move = plt.axes([0.58, 0.12, 0.12, 0.04])
        button_ax_move.set_xscale('linear')
        button_ax_move.set_yscale('linear')

        button_ax_new = plt.axes([0.71, 0.12, 0.12, 0.04])
        button_ax_new.set_xscale('linear')
        button_ax_new.set_yscale('linear')

        textbox_ax = plt.axes([0.05, 0.05, 0.78, 0.05])
        textbox_ax.set_xscale('linear')
        textbox_ax.set_yscale('linear')

        self.btn_prev = Button(button_ax_prev, 'Previous')
        self.btn_next = Button(button_ax_next, 'Next')
        self.btn_skip = Button(button_ax_skip, 'Skip')
        self.btn_done = Button(button_ax_done, 'Done')
        self.btn_clear = Button(button_ax_clear, 'Clear Selection')
        self.btn_move = Button(button_ax_move, 'Move Selected')
        self.btn_new = Button(button_ax_new, 'New Cluster')
        self.textbox = TextBox(textbox_ax, 'Label:', initial='')

        # Connect button callbacks
        self.btn_prev.on_clicked(self._on_previous)
        self.btn_next.on_clicked(self._on_next)
        self.btn_skip.on_clicked(self._on_skip)
        self.btn_done.on_clicked(self._on_done)
        self.btn_clear.on_clicked(self._on_clear_selection)
        self.btn_move.on_clicked(self._on_move_selected)
        self.btn_new.on_clicked(self._on_new_cluster)

        # Connect click event for selecting symbols
        self.fig.canvas.mpl_connect('button_press_event', self._on_symbol_click)

        # Monkey-patch canvas draw methods to enforce linear scale
        original_draw = self.fig.canvas.draw
        original_draw_idle = self.fig.canvas.draw_idle

        def safe_canvas_draw(*args, **kwargs):
            try:
                self._force_linear_scale_on_all_axes()
                return original_draw(*args, **kwargs)
            except ValueError as e:
                if "log-scaled" in str(e):
                    self._force_linear_scale_on_all_axes()
                    try:
                        return original_draw(*args, **kwargs)
                    except:
                        pass
                pass

        def safe_canvas_draw_idle(*args, **kwargs):
            try:
                self._force_linear_scale_on_all_axes()
                return original_draw_idle(*args, **kwargs)
            except ValueError as e:
                if "log-scaled" in str(e):
                    self._force_linear_scale_on_all_axes()
                    try:
                        return original_draw_idle(*args, **kwargs)
                    except:
                        pass
                pass

        self.fig.canvas.draw = safe_canvas_draw
        self.fig.canvas.draw_idle = safe_canvas_draw_idle

        # Display first cluster
        self._display_cluster(self.current_cluster_idx)

        plt.show()

    def _display_cluster(self, cluster_idx: int):
        """Display symbols from a specific cluster"""

        if cluster_idx >= len(self.cluster_ids):
            print("All clusters labeled!")
            plt.close(self.fig)
            return

        cluster_id = self.cluster_ids[cluster_idx]
        symbol_indices = self.clusters[cluster_id]

        # Clear selection when switching clusters
        self.selected_indices.clear()

        # Clear all axes
        for ax in self.symbol_axes:
            ax.clear()
            ax.axis('off')
            ax.set_xscale('linear')  # Ensure linear scale
            ax.set_yscale('linear')  # Ensure linear scale

        # Sample up to 32 symbols from this cluster
        n_display = min(32, len(symbol_indices))
        if len(symbol_indices) > 32:
            # Sample evenly across the cluster
            step = len(symbol_indices) / 32
            display_indices = [symbol_indices[int(i * step)] for i in range(32)]
        else:
            display_indices = symbol_indices

        # Store mapping from display position to actual index
        self.display_index_map = {}

        # Display symbols
        for i, idx in enumerate(display_indices):
            if i >= len(self.symbol_axes):
                break

            self.display_index_map[i] = idx  # Map display position to actual index

            patch = self.image_patches[idx]

            # Convert BGR to RGB if needed
            if patch.shape[-1] == 3:
                patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
            else:
                patch_rgb = patch

            self.symbol_axes[i].imshow(patch_rgb)
            self.symbol_axes[i].axis('off')

            # Add red border if selected
            if idx in self.selected_indices:
                for spine in self.symbol_axes[i].spines.values():
                    spine.set_edgecolor('red')
                    spine.set_linewidth(3)
                    spine.set_visible(True)

        # Update info text
        current_label = self.labels.get(cluster_id, "")
        info_str = f"Cluster {cluster_idx + 1}/{len(self.cluster_ids)} "
        info_str += f"(ID: {cluster_id})\n"
        info_str += f"Symbols in cluster: {len(symbol_indices)}\n"
        info_str += f"Showing {n_display} representative samples\n"
        if len(self.selected_indices) > 0:
            info_str += f"Selected: {len(self.selected_indices)} symbols\n"
        if current_label:
            info_str += f"Current label: '{current_label}'"
        else:
            info_str += "Not yet labeled"

        self.info_text.set_text(info_str)

        # Update textbox with current label
        if current_label:
            self.textbox.set_val(current_label)
        else:
            self.textbox.set_val('')

        # Use safe draw method
        self._safe_draw()

    def _on_previous(self, event):
        """Handle previous button click"""
        # Save current label
        self._save_current_label()

        if self.current_cluster_idx > 0:
            self.current_cluster_idx -= 1
            self._display_cluster(self.current_cluster_idx)

    def _on_next(self, event):
        """Handle next button click"""
        # Save current label
        self._save_current_label()

        if self.current_cluster_idx < len(self.cluster_ids) - 1:
            self.current_cluster_idx += 1
            self._display_cluster(self.current_cluster_idx)
        else:
            print("Reached last cluster. Press 'Done' to finish.")

    def _on_skip(self, event):
        """Handle skip button click"""
        if self.current_cluster_idx < len(self.cluster_ids) - 1:
            self.current_cluster_idx += 1
            self._display_cluster(self.current_cluster_idx)
        else:
            print("Reached last cluster. Press 'Done' to finish.")

    def _on_done(self, event):
        """Handle done button click"""
        # Save current label
        self._save_current_label()

        print(f"\nLabeling complete! Labeled {len(self.labels)} clusters.")
        plt.close(self.fig)

    def _save_current_label(self):
        """Save the current textbox content as label"""
        label_text = self.textbox.text.strip()
        if label_text:
            cluster_id = self.cluster_ids[self.current_cluster_idx]
            self.labels[cluster_id] = label_text
            print(f"Cluster {cluster_id} labeled as: '{label_text}'")

    def _on_symbol_click(self, event):
        """Handle clicking on a symbol to select/deselect it"""
        if event.inaxes in self.symbol_axes:
            # Find which symbol was clicked
            clicked_idx = self.symbol_axes.index(event.inaxes)

            # Check if this position has a symbol
            if clicked_idx in self.display_index_map:
                actual_idx = self.display_index_map[clicked_idx]

                # Toggle selection
                if actual_idx in self.selected_indices:
                    self.selected_indices.remove(actual_idx)
                    print(f"Deselected symbol {actual_idx}")
                    # Remove red border
                    for spine in event.inaxes.spines.values():
                        spine.set_visible(False)
                else:
                    self.selected_indices.add(actual_idx)
                    print(f"Selected symbol {actual_idx}")
                    # Add red border
                    for spine in event.inaxes.spines.values():
                        spine.set_edgecolor('red')
                        spine.set_linewidth(3)
                        spine.set_visible(True)

                # Update info text
                cluster_id = self.cluster_ids[self.current_cluster_idx]
                symbol_indices = self.clusters[cluster_id]
                current_label = self.labels.get(cluster_id, "")
                n_display = min(32, len(symbol_indices))

                info_str = f"Cluster {self.current_cluster_idx + 1}/{len(self.cluster_ids)} "
                info_str += f"(ID: {cluster_id})\n"
                info_str += f"Symbols in cluster: {len(symbol_indices)}\n"
                info_str += f"Showing {n_display} representative samples\n"
                if len(self.selected_indices) > 0:
                    info_str += f"Selected: {len(self.selected_indices)} symbols\n"
                if current_label:
                    info_str += f"Current label: '{current_label}'"
                else:
                    info_str += "Not yet labeled"

                self.info_text.set_text(info_str)
                self._safe_draw()

    def _on_clear_selection(self, event):
        """Clear all selected symbols"""
        if len(self.selected_indices) > 0:
            self.selected_indices.clear()
            print("Selection cleared")
            self._display_cluster(self.current_cluster_idx)
        else:
            print("No symbols selected")

    def _on_new_cluster(self, event):
        """Create a new cluster from selected symbols"""
        if len(self.selected_indices) == 0:
            print("No symbols selected. Please select symbols first.")
            return

        current_cluster_id = self.cluster_ids[self.current_cluster_idx]
        new_cluster_id = self.next_cluster_id
        self.next_cluster_id += 1

        # Initialize the new cluster in the clusters dictionary
        self.clusters[new_cluster_id] = []

        # Move selected symbols to new cluster
        for idx in self.selected_indices:
            self.cluster_labels[idx] = new_cluster_id
            self.clusters[current_cluster_id].remove(idx)
            self.clusters[new_cluster_id].append(idx)

        # Add new cluster to cluster_ids list (insert after current cluster)
        insert_pos = self.current_cluster_idx + 1
        self.cluster_ids.insert(insert_pos, new_cluster_id)

        # Clear selection
        num_moved = len(self.selected_indices)
        self.selected_indices.clear()

        print(f"\n✓ Created new cluster {new_cluster_id} with {num_moved} symbols")
        print(f"  Cluster {current_cluster_id} now has {len(self.clusters[current_cluster_id])} symbols")
        print(f"  Moving to new cluster for labeling...")

        # Move to the new cluster so user can label it
        self.current_cluster_idx = insert_pos

        # Display the new cluster
        self._display_cluster(self.current_cluster_idx)

    def _on_move_selected(self, event):
        """Move selected symbols to another cluster"""
        if len(self.selected_indices) == 0:
            print("No symbols selected. Please select symbols first.")
            return

        current_cluster_id = self.cluster_ids[self.current_cluster_idx]

        # Show available clusters in console
        print("\n" + "="*60)
        print("MOVE SYMBOLS TO CLUSTER")
        print("="*60)
        print(f"Selected {len(self.selected_indices)} symbols from cluster {current_cluster_id}")
        print("\nAvailable clusters:")
        for i, cid in enumerate(self.cluster_ids):
            label = self.labels.get(cid, "unlabeled")
            size = len(self.clusters[cid])
            marker = " (current)" if cid == current_cluster_id else ""
            print(f"  {i}: Cluster {cid} - '{label}' ({size} symbols){marker}")

        # Create a dialog window for input
        self._create_move_dialog(current_cluster_id)

    def _create_move_dialog(self, current_cluster_id):
        """Create a dialog for selecting target cluster"""
        # Create a new figure for the dialog
        dialog_fig = plt.figure(figsize=(6, 4))
        dialog_fig.suptitle('Move Symbols to Cluster', fontsize=14, fontweight='bold')

        # Create text area showing cluster options
        ax_text = dialog_fig.add_subplot(111)
        ax_text.axis('off')

        # Build cluster list text
        cluster_text = f"Selected: {len(self.selected_indices)} symbols\n"
        cluster_text += f"From cluster: {current_cluster_id}\n\n"
        cluster_text += "Available target clusters:\n"
        for i, cid in enumerate(self.cluster_ids):
            if cid == current_cluster_id:
                continue
            label = self.labels.get(cid, "unlabeled")
            size = len(self.clusters[cid])
            cluster_text += f"  {i}: Cluster {cid} - '{label}' ({size} symbols)\n"

        ax_text.text(0.05, 0.95, cluster_text, transform=ax_text.transAxes,
                    verticalalignment='top', fontsize=10, family='monospace')

        # Create text input box
        textbox_ax = plt.axes([0.3, 0.15, 0.4, 0.08])
        textbox_ax.set_xscale('linear')
        textbox_ax.set_yscale('linear')
        target_textbox = TextBox(textbox_ax, 'Target cluster index:', initial='')

        # Create OK and Cancel buttons
        ok_ax = plt.axes([0.3, 0.05, 0.15, 0.08])
        ok_ax.set_xscale('linear')
        ok_ax.set_yscale('linear')
        ok_button = Button(ok_ax, 'OK')

        cancel_ax = plt.axes([0.55, 0.05, 0.15, 0.08])
        cancel_ax.set_xscale('linear')
        cancel_ax.set_yscale('linear')
        cancel_button = Button(cancel_ax, 'Cancel')

        # Store dialog state
        dialog_state = {'confirmed': False, 'target_idx': None}

        def on_ok(event):
            try:
                target_idx = int(target_textbox.text.strip())
                if target_idx < 0 or target_idx >= len(self.cluster_ids):
                    print(f"Invalid cluster index. Must be 0-{len(self.cluster_ids)-1}")
                    return

                target_cluster_id = self.cluster_ids[target_idx]

                if target_cluster_id == current_cluster_id:
                    print("Cannot move to the same cluster")
                    return

                dialog_state['confirmed'] = True
                dialog_state['target_idx'] = target_idx
                plt.close(dialog_fig)

            except ValueError:
                print("Invalid input. Please enter a number")

        def on_cancel(event):
            print("Move cancelled")
            plt.close(dialog_fig)

        ok_button.on_clicked(on_ok)
        cancel_button.on_clicked(on_cancel)

        # Show dialog and wait
        plt.show(block=True)

        # Process the move if confirmed
        if dialog_state['confirmed'] and dialog_state['target_idx'] is not None:
            target_cluster_id = self.cluster_ids[dialog_state['target_idx']]

            # Move selected symbols
            for idx in self.selected_indices:
                self.cluster_labels[idx] = target_cluster_id
                self.clusters[current_cluster_id].remove(idx)
                self.clusters[target_cluster_id].append(idx)

            num_moved = len(self.selected_indices)
            self.selected_indices.clear()

            print(f"\nMoved {num_moved} symbols to cluster {target_cluster_id}")
            print(f"Cluster {current_cluster_id} now has {len(self.clusters[current_cluster_id])} symbols")
            print(f"Cluster {target_cluster_id} now has {len(self.clusters[target_cluster_id])} symbols")

            # Refresh display
            self._display_cluster(self.current_cluster_idx)


def extract_symbol_patches(image: np.ndarray,
                          masks: Optional[List[np.ndarray]] = None,
                          bboxes: Optional[List[List[int]]] = None,
                          padding: int = 5) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
    """
    Extract image patches for each detected symbol

    Args:
        image: Original P&ID image
        masks: List of binary masks (optional if bboxes provided)
        bboxes: List of bounding boxes [x1, y1, x2, y2] (optional if masks provided)
        padding: Pixels to pad around each mask/bbox

    Returns:
        Tuple of (patches, metadata)
    """
    patches = []
    metadata = []

    if masks is not None:
        # Extract from masks
        print(f"Extracting patches from {len(masks)} masks...")

        for i, mask in enumerate(tqdm(masks)):
            # Get bounding box
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)

            if not np.any(rows) or not np.any(cols):
                continue

            y1, y2 = np.where(rows)[0][[0, -1]]
            x1, x2 = np.where(cols)[0][[0, -1]]

            # Add padding
            h, w = image.shape[:2]
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding + 1)
            y2 = min(h, y2 + padding + 1)

            # Extract patch
            patch = image[y1:y2, x1:x2].copy()

            # Skip very small patches
            if patch.shape[0] < 5 or patch.shape[1] < 5:
                continue

            patches.append(patch)
            metadata.append({
                'mask_id': i,
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'size': [int(x2-x1), int(y2-y1)],
                'area': int(np.sum(mask))
            })

    elif bboxes is not None:
        # Extract from bounding boxes
        print(f"Extracting patches from {len(bboxes)} bounding boxes...")

        h, w = image.shape[:2]

        for i, bbox in enumerate(tqdm(bboxes)):
            x1, y1, x2, y2 = bbox

            # Add padding
            x1_padded = max(0, x1 - padding)
            y1_padded = max(0, y1 - padding)
            x2_padded = min(w, x2 + padding)
            y2_padded = min(h, y2 + padding)

            # Extract patch
            patch = image[y1_padded:y2_padded, x1_padded:x2_padded].copy()

            # Skip very small patches
            if patch.shape[0] < 5 or patch.shape[1] < 5:
                continue

            patches.append(patch)
            metadata.append({
                'mask_id': i,
                'bbox': [int(x1_padded), int(y1_padded), int(x2_padded), int(y2_padded)],
                'original_bbox': [int(x1), int(y1), int(x2), int(y2)],
                'size': [int(x2_padded - x1_padded), int(y2_padded - y1_padded)],
                'area': (x2 - x1) * (y2 - y1)  # Approximate area from bbox
            })

    else:
        raise ValueError("Either masks or bboxes must be provided")

    print(f"Extracted {len(patches)} valid patches")
    return patches, metadata


def visualize_clusters_2d(embeddings: np.ndarray,
                         cluster_labels: np.ndarray,
                         label_names: Dict[int, str],
                         save_path: str,
                         method: str = 'umap'):
    """
    Visualize clusters in 2D using dimensionality reduction

    Args:
        embeddings: High-dimensional embeddings
        cluster_labels: Cluster assignments
        label_names: Mapping from cluster ID to label name
        save_path: Path to save visualization
        method: Dimensionality reduction method ('umap', 'tsne', 'pca')
    """
    # Fallback to PCA if UMAP is not available
    if method == 'umap' and not UMAP_AVAILABLE:
        print("Warning: UMAP not available, falling back to PCA")
        method = 'pca'

    print(f"Creating 2D visualization using {method.upper()}...")

    # Reduce to 2D
    if method == 'umap':
        reducer = umap_reducer(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        embeddings_2d = reducer.fit_transform(embeddings)
    elif method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
        embeddings_2d = reducer.fit_transform(embeddings)
    else:  # pca
        reducer = PCA(n_components=2, random_state=42)
        embeddings_2d = reducer.fit_transform(embeddings)

    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 10))

    # Get unique clusters
    unique_labels = np.unique(cluster_labels)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

    # Plot each cluster
    for i, label in enumerate(unique_labels):
        mask = cluster_labels == label
        label_name = label_names.get(label, f"Cluster {label}")
        if label == -1:
            label_name = "Noise/Unlabeled"

        ax.scatter(embeddings_2d[mask, 0],
                  embeddings_2d[mask, 1],
                  c=[colors[i]],
                  label=label_name,
                  alpha=0.6,
                  s=50)

    ax.set_xlabel(f'{method.upper()} Dimension 1', fontsize=12)
    ax.set_ylabel(f'{method.upper()} Dimension 2', fontsize=12)
    ax.set_title('Symbol Clusters Visualization', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Cluster visualization saved to: {save_path}")
    plt.close()


def is_discarded_category(label: str) -> bool:
    """
    Check if a category should be discarded (text, lines, etc.)

    Args:
        label: Category label name

    Returns:
        True if should be discarded, False otherwise
    """
    if not label:
        return False

    # Normalize to lowercase for comparison
    label_lower = label.lower().strip()

    # List of patterns to discard
    discard_patterns = [
        'text', 'txt', 'label', 'annotation',
        'line', 'lines', 'pipe', 'pipes', 'piping',
        'connector', 'connection',
        'border', 'frame',
        'title', 'heading',
        'noise', 'background', 'other', 'unknown'
    ]

    # Check if label matches any discard pattern
    for pattern in discard_patterns:
        if pattern in label_lower or label_lower in pattern:
            return True

    return False


def save_classification_results(image_path: str,
                               patches_metadata: List[Dict[str, Any]],
                               cluster_labels: np.ndarray,
                               label_names: Dict[int, str],
                               embeddings: np.ndarray,
                               output_path: str):
    """
    Save classification results to JSON, filtering out discarded categories

    Args:
        image_path: Path to original image
        patches_metadata: Metadata for each patch
        cluster_labels: Cluster assignments
        label_names: Cluster label names
        embeddings: Symbol embeddings
        output_path: Path to save results
    """
    results = {
        'image_path': str(image_path),
        'num_symbols_detected': len(patches_metadata),
        'num_clusters': len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0),
        'categories': {},
        'discarded_categories': {},
        'symbols': [],
        'discarded_symbols': []
    }

    # Separate valid and discarded categories
    valid_categories = {}
    discarded_categories = {}

    for cluster_id, label_name in label_names.items():
        count = np.sum(cluster_labels == cluster_id)
        if is_discarded_category(label_name):
            discarded_categories[label_name] = int(count)
        else:
            valid_categories[label_name] = int(count)

    results['categories'] = valid_categories
    results['discarded_categories'] = discarded_categories

    # Add unlabeled count
    unlabeled_count = np.sum([cluster_labels[i] not in label_names
                             for i in range(len(cluster_labels))])
    if unlabeled_count > 0:
        results['categories']['unlabeled'] = int(unlabeled_count)

    # Save per-symbol information
    num_valid = 0
    num_discarded = 0

    for i, (metadata, cluster_id) in enumerate(zip(patches_metadata, cluster_labels)):
        label_name = label_names.get(cluster_id, 'unlabeled')

        symbol_info = {
            'id': i,
            'mask_id': metadata['mask_id'],
            'bbox': metadata['bbox'],
            'size': metadata['size'],
            'area': metadata['area'],
            'cluster_id': int(cluster_id),
            'category': label_name,
            'embedding': embeddings[i].tolist()  # Save for future use
        }

        # Separate into valid symbols and discarded symbols
        if is_discarded_category(label_name):
            results['discarded_symbols'].append(symbol_info)
            num_discarded += 1
        else:
            results['symbols'].append(symbol_info)
            num_valid += 1

    # Add summary statistics
    results['num_symbols_valid'] = num_valid
    results['num_symbols_discarded'] = num_discarded

    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nClassification results saved to: {output_path}")
    print(f"\n{'='*60}")
    print(f"CLASSIFICATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total symbols detected: {len(patches_metadata)}")
    print(f"Valid symbols: {num_valid}")
    print(f"Discarded symbols: {num_discarded}")
    print(f"\nValid categories:")
    for category, count in results['categories'].items():
        print(f"  ✓ {category}: {count} symbols")

    if results['discarded_categories']:
        print(f"\nDiscarded categories (text/lines/etc):")
        for category, count in results['discarded_categories'].items():
            print(f"  ✗ {category}: {count} symbols (filtered out)")


def create_category_visualization(image: np.ndarray,
                                 patches_metadata: List[Dict[str, Any]],
                                 cluster_labels: np.ndarray,
                                 label_names: Dict[int, str],
                                 save_path: str,
                                 show_discarded: bool = False):
    """
    Create visualization of categorized symbols on original image

    Args:
        image: Original image
        patches_metadata: Metadata for each patch
        cluster_labels: Cluster assignments
        label_names: Cluster label names
        save_path: Path to save visualization
        show_discarded: If True, show discarded categories in gray
    """
    # Convert BGR to RGB
    if len(image.shape) == 3 and image.shape[2] == 3:
        display_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).copy()
    else:
        display_image = image.copy()

    # Separate valid and discarded labels
    valid_labels = [label for label in set(label_names.values())
                    if not is_discarded_category(label)]
    discarded_labels = [label for label in set(label_names.values())
                        if is_discarded_category(label)]

    # Generate colors for valid categories only
    colors = plt.cm.tab20(np.linspace(0, 1, len(valid_labels)))
    color_map = {label: colors[i] for i, label in enumerate(valid_labels)}

    # Discarded categories get gray color
    discarded_color = [0.7, 0.7, 0.7, 0.3]  # Light gray, transparent

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.imshow(display_image)
    ax.axis('off')

    # Draw bounding boxes
    import matplotlib.patches as patches_mpl

    num_valid = 0
    num_discarded = 0

    for metadata, cluster_id in zip(patches_metadata, cluster_labels):
        label_name = label_names.get(cluster_id, 'unlabeled')

        # Determine color and whether to show
        is_discarded = is_discarded_category(label_name)

        if is_discarded:
            if not show_discarded:
                continue  # Skip discarded symbols
            color = discarded_color
            num_discarded += 1
        elif label_name == 'unlabeled':
            color = [0.5, 0.5, 0.5, 0.5]
        else:
            color = color_map.get(label_name, [0.5, 0.5, 0.5, 0.5])
            num_valid += 1

        bbox = metadata['bbox']
        x1, y1, x2, y2 = bbox

        # Use thinner lines for discarded
        linewidth = 1 if is_discarded else 2
        alpha = 0.3 if is_discarded else 0.8

        rect = patches_mpl.Rectangle((x1, y1), x2-x1, y2-y1,
                                    linewidth=linewidth, edgecolor=color,
                                    facecolor='none', alpha=alpha)
        ax.add_patch(rect)

    # Create legend (only valid categories)
    legend_elements = []
    for label_name in valid_labels:
        if label_name in color_map:
            count = sum(1 for cl in cluster_labels
                       if label_names.get(cl) == label_name)
            legend_elements.append(plt.Line2D([0], [0], marker='s', color='w',
                                             markerfacecolor=color_map[label_name],
                                             markersize=10,
                                             label=f'{label_name} ({count})'))

    # Add discarded categories info if shown
    if show_discarded and len(discarded_labels) > 0:
        total_discarded = sum(1 for cl in cluster_labels
                             if is_discarded_category(label_names.get(cl, '')))
        legend_elements.append(plt.Line2D([0], [0], marker='s', color='w',
                                         markerfacecolor=discarded_color,
                                         markersize=10,
                                         label=f'Discarded ({total_discarded})'))

    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    title = 'Classified Symbols on P&ID'
    if not show_discarded and num_discarded > 0:
        title += f' ({num_valid} valid symbols shown)'
    ax.set_title(title, fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Category visualization saved to: {save_path}")
    if not show_discarded:
        print(f"  Note: Discarded categories (text/lines) are hidden from visualization")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Interactive Symbol Clustering and Classification for P&ID Diagrams'
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--results_json', type=str,
                           help='Path to SAM2 AMG results JSON file')
    input_group.add_argument('--masks_path', type=str,
                           help='Path to masks NPZ file (requires --image_path)')

    parser.add_argument('--image_path', type=str,
                       help='Path to original P&ID image (required if using --masks_path)')

    # Model options
    parser.add_argument('--embedding_model', type=str, default='clip',
                       choices=['clip', 'dinov2', 'vit'],
                       help='Model to use for embeddings')
    parser.add_argument('--clustering_method', type=str, default='hdbscan',
                       choices=['hdbscan', 'kmeans'],
                       help='Clustering method')
    parser.add_argument('--n_clusters', type=int, default=None,
                       help='Number of clusters (for kmeans, optional)')
    parser.add_argument('--sensitivity', type=str, default='high',
                       choices=['low', 'medium', 'high', 'very_high'],
                       help='Clustering sensitivity: low (fewer clusters), medium (balanced), '
                            'high (more fine-grained, recommended), very_high (maximum sensitivity)')

    # Processing options
    parser.add_argument('--padding', type=int, default=5,
                       help='Padding around symbol patches in pixels')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for embedding extraction')

    # Output options
    parser.add_argument('--output_dir', type=str, default='./symbol_classification_results',
                       help='Output directory for results')
    parser.add_argument('--visualization_method', type=str, default='umap',
                       choices=['umap', 'tsne', 'pca'],
                       help='Method for 2D visualization')

    # Device
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to run on (cuda or cpu)')

    args = parser.parse_args()

    # Validate inputs
    if args.masks_path and not args.image_path:
        parser.error("--image_path is required when using --masks_path")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("="*60)
    print("STEP 1: Loading Data")
    print("="*60)

    masks = None
    bboxes = None

    if args.results_json:
        # Load from JSON results
        print(f"Loading results from: {args.results_json}")
        with open(args.results_json, 'r') as f:
            sam_results = json.load(f)

        image_path = sam_results['image_path']
        print(f"Loading image: {image_path}")
        image = cv2.imread(image_path)

        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        # Try to load masks from NPZ file
        json_path = Path(args.results_json)
        npz_path = json_path.parent / f"{json_path.stem.replace('_results', '_masks')}.npz"

        if not npz_path.exists():
            # Try alternative naming
            npz_path = json_path.parent / f"{json_path.stem}_masks.npz"

        if npz_path.exists():
            print(f"Loading masks from: {npz_path}")
            masks_data = np.load(npz_path)
            masks = [masks_data['masks'][i] for i in range(len(masks_data['masks']))]
            print(f"Loaded {len(masks)} masks from NPZ file")
        else:
            # No masks file - use bboxes from JSON
            print("WARNING: Masks NPZ file not found.")
            print(f"Expected at: {npz_path}")
            print("Falling back to bounding boxes from JSON for patch extraction.")
            print("Note: This is less accurate than using full masks.\n")

            # Extract bboxes from JSON
            bboxes = [mask_info['bbox'] for mask_info in sam_results['masks_info']]
            print(f"Loaded {len(bboxes)} bounding boxes from JSON")

    else:
        # Load from separate files
        image_path = args.image_path
        print(f"Loading image: {image_path}")
        image = cv2.imread(image_path)

        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        print(f"Loading masks from: {args.masks_path}")
        masks_data = np.load(args.masks_path)
        masks = [masks_data['masks'][i] for i in range(len(masks_data['masks']))]
        print(f"Loaded {len(masks)} masks from NPZ file")

    print(f"Image shape: {image.shape}")

    # Extract symbol patches
    print("\n" + "="*60)
    print("STEP 2: Extracting Symbol Patches")
    print("="*60)

    patches, patches_metadata = extract_symbol_patches(
        image, masks=masks, bboxes=bboxes, padding=args.padding
    )

    if len(patches) == 0:
        print("No valid patches extracted. Exiting.")
        return

    # Generate embeddings
    print("\n" + "="*60)
    print("STEP 3: Generating Embeddings")
    print("="*60)

    embedder = SymbolEmbedder(
        model_type=args.embedding_model,
        device=args.device
    )

    embeddings = embedder.extract_embeddings(patches, batch_size=args.batch_size)

    # Check if classification already exists
    image_name = Path(image_path).stem
    classification_path = output_dir / f"{image_name}_classification.json"

    existing_labels = None
    existing_label_names = None

    if classification_path.exists():
        print("\n" + "="*60)
        print("EXISTING CLASSIFICATION FOUND")
        print("="*60)
        print(f"Found existing classification: {classification_path}")

        try:
            with open(classification_path, 'r') as f:
                existing_data = json.load(f)

            # Extract existing cluster labels and names
            existing_symbols = existing_data.get('symbols', [])
            if existing_symbols:
                # Map mask_id to cluster_id and category
                mask_id_to_cluster = {}
                mask_id_to_category = {}
                cluster_to_category = {}

                for symbol in existing_symbols:
                    mask_id = symbol.get('mask_id')
                    cluster_id = symbol.get('cluster_id')
                    category = symbol.get('category', 'unknown')

                    if mask_id is not None and cluster_id is not None:
                        mask_id_to_cluster[mask_id] = cluster_id
                        mask_id_to_category[mask_id] = category
                        cluster_to_category[cluster_id] = category

                print(f"Loaded existing classifications for {len(existing_symbols)} symbols")
                print(f"Found {len(cluster_to_category)} existing clusters")

                # Create label arrays matching current patches
                existing_labels = np.zeros(len(patches_metadata), dtype=int)
                for i, meta in enumerate(patches_metadata):
                    mask_id = meta.get('mask_id', i)
                    if mask_id in mask_id_to_cluster:
                        existing_labels[i] = mask_id_to_cluster[mask_id]

                existing_label_names = cluster_to_category

                print("\nExisting category distribution:")
                for category, count in existing_data.get('categories', {}).items():
                    if count > 0:
                        print(f"  - {category}: {count}")

                print("\nLoading existing classifications into UI...")
                print("You can review and modify the labels if needed.")

        except Exception as e:
            print(f"Warning: Could not load existing classification: {e}")
            print("Will proceed with new clustering...")
            existing_labels = None
            existing_label_names = None

    # Cluster embeddings (or use existing)
    if existing_labels is not None:
        print("\n" + "="*60)
        print("STEP 4: Using Existing Classifications")
        print("="*60)
        cluster_labels = existing_labels
        print(f"Loaded {len(np.unique(cluster_labels))} existing clusters")
    else:
        print("\n" + "="*60)
        print("STEP 4: Clustering Symbols")
        print("="*60)

        clusterer = SymbolClusterer(
            method=args.clustering_method,
            n_clusters=args.n_clusters,
            sensitivity=args.sensitivity
        )

        cluster_labels = clusterer.fit(embeddings)

    # Interactive labeling
    print("\n" + "="*60)
    print("STEP 5: Interactive Cluster Labeling")
    print("="*60)

    labeler = InteractiveClusterLabeler(patches, cluster_labels, patches_metadata)
    label_names = labeler.run(existing_label_names=existing_label_names)

    # Save results
    print("\n" + "="*60)
    print("STEP 6: Saving Results")
    print("="*60)

    # Save classification results (classification_path and image_name already defined above)
    # classification_path = output_dir / f"{image_name}_classification.json"
    save_classification_results(
        image_path, patches_metadata, cluster_labels,
        label_names, embeddings, str(classification_path)
    )

    # Create visualizations
    print("\nCreating visualizations...")

    # 2D cluster visualization
    cluster_viz_path = output_dir / f"{image_name}_clusters_2d.png"
    visualize_clusters_2d(
        embeddings, cluster_labels, label_names,
        str(cluster_viz_path), method=args.visualization_method
    )

    # Category visualization on image
    category_viz_path = output_dir / f"{image_name}_categorized.png"
    create_category_visualization(
        image, patches_metadata, cluster_labels,
        label_names, str(category_viz_path)
    )

    print("\n" + "="*60)
    print("PROCESSING COMPLETE!")
    print("="*60)
    print(f"\nResults saved to: {output_dir}")
    print(f"  - Classification JSON: {classification_path.name}")
    print(f"  - Cluster visualization: {cluster_viz_path.name}")
    print(f"  - Category visualization: {category_viz_path.name}")
    print("\nYou can use the classification JSON for downstream tasks.")


if __name__ == "__main__":
    main()
