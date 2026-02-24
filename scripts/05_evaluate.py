"""
Script 5: Evaluate trained model and visualize results

Usage:
    python scripts/05_evaluate.py --weights runs/detect/experiment/weights/best.pt
"""
import os
# Fix OpenMP conflict
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
from pathlib import Path
import yaml
import argparse
import random

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.reproducibility import set_seed
from src.utils import paths

# Import YOLOv8
try:
    from ultralytics import YOLO
except ImportError:
    print("ERROR: ultralytics not installed!")
    print("Install with: pip install ultralytics")
    sys.exit(1)


def main(args):
    """Evaluate trained model on validation set."""
    
    print("="*70)
    print("STEP 5: EVALUATE MODEL")
    print("="*70)
    
    # Load config
    config_path = Path("configs/experiment_config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Set seed
    set_seed(config['seed'])
    
    # Weights path
    if args.weights:
        weights_path = Path(args.weights)
    else:
        # Find latest run (use centralized runs path)
        runs_dir = paths.RUNS_PROJECT
        if not runs_dir.exists():
            print(f"ERROR: No training runs found in {runs_dir}")
            print("Run script 04_train.py first!")
            sys.exit(1)

        # Get latest experiment
        experiments = sorted(runs_dir.glob("yolov8*"))
        if not experiments:
            print(f"ERROR: No experiments found in {runs_dir}")
            sys.exit(1)
        
        latest_exp = experiments[-1]
        weights_path = latest_exp / "weights" / "best.pt"
    
    if not weights_path.exists():
        print(f"ERROR: Weights not found: {weights_path}")
        sys.exit(1)
    
    print(f"\nLoading model from: {weights_path}")
    model = YOLO(str(weights_path))
    print("✓ Model loaded")
    
    # Dataset config
    data_yaml = Path(config['data']['yaml_path'])
    
    # Evaluate on validation set
    print("\nRunning evaluation on validation set...")
    print("="*70)
    # Use command-line conf if provided, else config default
    conf_threshold = args.conf if args.conf is not None else config['inference']['conf']
    print(f"Confidence threshold: {conf_threshold}")
    
    metrics = model.val(
        data=str(data_yaml),
        imgsz=config['training']['imgsz'],
        batch=config['training']['batch'],
        conf=conf_threshold,
        iou=config['inference']['iou'],
        max_det=config['inference']['max_det'],
        plots=True,
        save_json=True,
        verbose=True
    )
    
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    
    # Overall metrics
    print("\nOverall Metrics:")
    print(f"  mAP@0.5:      {metrics.box.map50:.4f} ({metrics.box.map50*100:.2f}%)")
    print(f"  mAP@0.5:0.95: {metrics.box.map:.4f} ({metrics.box.map*100:.2f}%)")
    print(f"  Precision:    {metrics.box.mp:.4f}")
    print(f"  Recall:       {metrics.box.mr:.4f}")
    
    # Per-class metrics
    print("\nPer-Class mAP@0.5:")
    class_names = [
        'pedestrian', 'people', 'bicycle', 'car', 'van',
        'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor'
    ]
    
    for i, (name, ap) in enumerate(zip(class_names, metrics.box.ap50)):
        print(f"  {name:18s}: {ap:.4f} ({ap*100:.2f}%)")
    
    # Compare to baseline
    baseline_map = 0.0723  # 7.23%
    your_map = metrics.box.map50
    
    print("\n" + "="*70)
    print("BASELINE COMPARISON")
    print("="*70)
    print(f"  MATLAB baseline: {baseline_map*100:.2f}%")
    print(f"  Your result:     {your_map*100:.2f}%")
    
    if your_map > baseline_map:
        improvement = (your_map - baseline_map) / baseline_map * 100
        print(f"\n  🎉 SUCCESS! You beat the baseline by {improvement:.1f}%!")
    elif your_map > 0:
        gap = (baseline_map - your_map) / baseline_map * 100
        print(f"\n  📊 Close! You're {gap:.1f}% away from beating the baseline.")
        print(f"  Try: larger model (yolov8s/m), more epochs, or higher imgsz")
    else:
        print(f"\n  ⚠ Warning: mAP is 0%. Check:")
        print(f"    - Dataset paths in configs/visdrone.yaml")
        print(f"    - Confidence threshold (try lower)")
        print(f"    - Model training logs")
    
    print("="*70)
    
    # Visualize predictions on random validation images
    if args.visualize:
        print("\nGenerating prediction visualizations...")
        
        val_images_dir = paths.DATASET / "images" / "val"
        if not val_images_dir.exists():
            print(f"WARNING: Validation images not found at {val_images_dir}")
            return

        val_images = list(val_images_dir.glob("*.jpg"))
        sample_images = random.sample(val_images, min(10, len(val_images)))

        output_dir = paths.PREDICTIONS
        paths.ensure_dirs(output_dir)
        
        for img_path in sample_images:
            results = model.predict(
                source=str(img_path),
                conf=conf_threshold,
                iou=config['inference']['iou'],
                max_det=config['inference']['max_det'],
                save=True,
                project=str(output_dir),
                name='visualizations',
                exist_ok=True
            )
        
        print(f"✓ Predictions saved to: {output_dir}/visualizations")
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate YOLOv8 model")
    parser.add_argument('--weights', type=str, help='Path to weights file')
    parser.add_argument('--visualize', action='store_true', 
                       help='Generate prediction visualizations')
    parser.add_argument('--conf', type=float, default=None,
                       help='Confidence threshold (overrides config)')
    
    args = parser.parse_args()
    main(args)
