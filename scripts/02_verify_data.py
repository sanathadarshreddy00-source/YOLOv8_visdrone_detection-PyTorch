"""
Script 2: Verify annotation conversion visually

Usage:
    python scripts/02_verify_data.py
"""
import os
# Fix OpenMP conflict
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
from pathlib import Path
import yaml
import random

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.visualization import visualize_annotations
from src.utils.reproducibility import set_seed
from src.utils import paths


def main():
    """Verify conversion by visualizing random samples."""
    
    print("="*70)
    print("STEP 2: VERIFY ANNOTATION CONVERSION")
    print("="*70)
    
    # Load config
    config_path = Path("configs/experiment_config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Set seed
    set_seed(config['seed'])
    
    # Paths (centralized)
    images_dir = paths.IMAGES
    visdrone_dir = paths.ANNOTATIONS
    yolo_dir = paths.LABELS
    output_dir = paths.VERIFICATION
    paths.ensure_dirs(output_dir)
    
    # Get random sample
    image_files = sorted(images_dir.glob("*.jpg"))
    sample_size = min(10, len(image_files))
    sample_images = random.sample(image_files, sample_size)
    
    print(f"\nVisualizing {sample_size} random images...")
    print(f"Plots will be saved to: {output_dir}\n")
    
    for i, img_path in enumerate(sample_images, 1):
        print(f"[{i}/{sample_size}] Processing {img_path.name}...")
        
        visdrone_ann = visdrone_dir / (img_path.stem + ".txt")
        yolo_ann = yolo_dir / (img_path.stem + ".txt")
        save_path = output_dir / f"verify_{img_path.stem}.png"
        
        # Check if annotation files exist
        if not visdrone_ann.exists():
            print(f"  ⚠ VisDrone annotation not found")
            continue
        if not yolo_ann.exists():
            print(f"  ⚠ YOLO annotation not found")
            continue
        
        # Visualize
        try:
            visualize_annotations(
                img_path=img_path,
                visdrone_ann_path=visdrone_ann,
                yolo_ann_path=yolo_ann,
                save_path=save_path,
                show=False  # Don't display, just save
            )
            print(f"  ✓ Saved to {save_path.name}")
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    print("\n" + "="*70)
    print("VERIFICATION COMPLETE")
    print("="*70)
    print(f"Check visualizations in: {output_dir}")
    print("\nLegend:")
    print("  - Red boxes:   VisDrone annotations (original)")
    print("  - Green boxes: YOLO annotations (converted)")
    print("  - Gray boxes:  Filtered objects (ignored regions)")
    print("\nIf red and green boxes align perfectly, conversion is correct!")
    print("="*70)


if __name__ == "__main__":
    main()
