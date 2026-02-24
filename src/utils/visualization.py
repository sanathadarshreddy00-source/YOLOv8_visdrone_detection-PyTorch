"""Visualization utilities for data and results."""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from src.utils import paths
from typing import List, Tuple
import numpy as np
from PIL import Image


def visualize_annotations(img_path: Path,
                         visdrone_ann_path: Path = None,
                         yolo_ann_path: Path = None,
                         save_path: Path = None,
                         show: bool = True) -> None:
    """Visualize VisDrone and/or YOLO annotations on image.
    
    Args:
        img_path: Path to image file
        visdrone_ann_path: Path to VisDrone annotation file
        yolo_ann_path: Path to YOLO annotation file
        save_path: Path to save visualization
        show: Whether to display the plot
    """
    img = Image.open(img_path)
    img_width, img_height = img.size
    
    fig, axes = plt.subplots(1, 2 if (visdrone_ann_path and yolo_ann_path) else 1, 
                            figsize=(16, 8))
    if not isinstance(axes, np.ndarray):
        axes = [axes]
    
    # VisDrone annotations
    if visdrone_ann_path:
        ax = axes[0] if len(axes) > 1 else axes[0]
        ax.imshow(img)
        ax.set_title('VisDrone Annotations (Original)', fontsize=14)
        ax.axis('off')
        
        with open(visdrone_ann_path, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) != 8:
                    continue
                left, top, width, height, score, class_id, _, _ = map(int, parts)
                
                # Skip ignored regions
                if score == 0 or class_id == 0 or class_id == 11:
                    color = 'gray'
                    alpha = 0.3
                else:
                    color = 'red'
                    alpha = 0.7
                
                rect = patches.Rectangle((left, top), width, height,
                                        linewidth=2, edgecolor=color,
                                        facecolor='none', alpha=alpha)
                ax.add_patch(rect)
    
    # YOLO annotations
    if yolo_ann_path:
        ax = axes[1] if len(axes) > 1 else axes[0]
        ax.imshow(img)
        ax.set_title('YOLO Annotations (Converted)', fontsize=14)
        ax.axis('off')
        
        with open(yolo_ann_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                class_id, x_center, y_center, norm_width, norm_height = map(float, parts)
                
                # Convert back to absolute coordinates
                left = (x_center - norm_width / 2) * img_width
                top = (y_center - norm_height / 2) * img_height
                width = norm_width * img_width
                height = norm_height * img_height
                
                rect = patches.Rectangle((left, top), width, height,
                                        linewidth=2, edgecolor='green',
                                        facecolor='none', alpha=0.7)
                ax.add_patch(rect)
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Visualization saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_class_distribution(class_counts: dict, 
                           class_names: List[str],
                           save_path: Path = None,
                           show: bool = True) -> None:
    """Plot class distribution bar chart.
    
    Args:
        class_counts: Dictionary mapping class IDs to counts
        class_names: List of class names
        save_path: Path to save plot
        show: Whether to display the plot
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    classes = sorted(class_counts.keys())
    counts = [class_counts[c] for c in classes]
    names = [class_names[c] if c < len(class_names) else f"Class {c}" for c in classes]
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(classes)))
    bars = ax.bar(names, counts, color=colors, alpha=0.8)
    
    # Add count labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{int(height)}',
               ha='center', va='bottom', fontsize=10)
    
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Class Distribution', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Class distribution saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
