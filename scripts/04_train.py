"""
Script 4: Train YOLOv8 model on VisDrone dataset
]
"""
import os
# Fix OpenMP conflict
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
from pathlib import Path
import yaml
import argparse
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.reproducibility import set_seed
from src.utils.logging import save_experiment_config
from src.utils import paths

# Import YOLOv8
try:
    from ultralytics import YOLO
except ImportError:
    print("ERROR: ultralytics not installed!")
    print("Install with: pip install ultralytics")
    sys.exit(1)


def main(args):
    """Train YOLOv8 model on VisDrone dataset."""
    
    print("="*70)
    print("STEP 4: TRAIN YOLOV8 MODEL")
    print("="*70)
    
    # Load config
    config_path = Path("configs/experiment_config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Override config with CLI args
    if args.model:
        config['model']['name'] = args.model
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.imgsz:
        config['training']['imgsz'] = args.imgsz
    if args.batch:
        config['training']['batch'] = args.batch
    
    # Set seed for reproducibility
    set_seed(config['seed'])
    
    # Print configuration
    print("\nTraining Configuration:")
    print(f"  Model:        {config['model']['name']}")
    print(f"  Epochs:       {config['training']['epochs']}")
    print(f"  Image size:   {config['training']['imgsz']}")
    print(f"  Batch size:   {config['training']['batch']}")
    print(f"  Device:       {config['training']['device']}")
    print(f"  AMP:          {config['training']['amp']}")
    print(f"  Seed:         {config['seed']}")
    
    # Check dataset
    data_yaml = Path(config['data']['yaml_path'])
    if not data_yaml.exists():
        print(f"\nERROR: Dataset config not found: {data_yaml}")
        print("Run scripts 01-03 first to prepare the dataset!")
        sys.exit(1)
    
    # Load pretrained model
    model_name = config['model']['name']
    print(f"\nLoading {model_name} model...")
    model = YOLO(f"{model_name}.pt")
    print(f"✓ Model loaded: {model_name}")
    
    # Create experiment name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{model_name}_{timestamp}"
    
    # Save config for this run (use centralized runs path)
    run_dir = paths.RUNS_PROJECT / experiment_name
    paths.ensure_dirs(run_dir, paths.LOGS)
    save_experiment_config(config, run_dir)
    
    print(f"\nStarting training...")
    print(f"Results will be saved to: {run_dir}")
    print("="*70)
    
    # Train
    results = model.train(
        # Data
        data=str(data_yaml),
        
        # Training settings
        epochs=config['training']['epochs'],
        imgsz=config['training']['imgsz'],
        batch=config['training']['batch'],
        device=config['training']['device'],
        
        # Optimization
        optimizer=config['optimizer']['name'],
        lr0=config['optimizer']['lr0'],
        lrf=config['optimizer']['lrf'],
        momentum=config['optimizer']['momentum'],
        weight_decay=config['optimizer']['weight_decay'],
        warmup_epochs=config['optimizer']['warmup_epochs'],
        warmup_momentum=config['optimizer']['warmup_momentum'],
        warmup_bias_lr=config['optimizer']['warmup_bias_lr'],
        
        # Augmentation
        hsv_h=config['augmentation']['hsv_h'],
        hsv_s=config['augmentation']['hsv_s'],
        hsv_v=config['augmentation']['hsv_v'],
        degrees=config['augmentation']['degrees'],
        translate=config['augmentation']['translate'],
        scale=config['augmentation']['scale'],
        shear=config['augmentation']['shear'],
        perspective=config['augmentation']['perspective'],
        flipud=config['augmentation']['flipud'],
        fliplr=config['augmentation']['fliplr'],
        mosaic=config['augmentation']['mosaic'],
        mixup=config['augmentation']['mixup'],
        copy_paste=config['augmentation']['copy_paste'],
        
        # Loss weights
        box=config['loss']['box'],
        cls=config['loss']['cls'],
        dfl=config['loss']['dfl'],
        
        # Inference
        conf=config['inference']['conf'],
        iou=config['inference']['iou'],
        max_det=config['inference']['max_det'],
        
        # Logging
        project=str(paths.RUNS_PROJECT),
        name=experiment_name,
        exist_ok=config['logging']['exist_ok'],
        plots=False,  # Disable to save RAM during validation
        save=config['training']['save'],
        save_period=config['training']['save_period'],
        
        # Performance
        amp=config['training']['amp'],
        patience=config['training']['patience'],
        workers=0,  # Single-threaded to save RAM
        cache=config['data']['cache'],
        
        # Misc
        verbose=config['logging']['verbose'],
        seed=config['seed'],
        deterministic=config['deterministic'],
    )
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    
    # Print metrics
    metrics = results.results_dict
    print("\nFinal Metrics:")
    print(f"  mAP@0.5:     {metrics.get('metrics/mAP50(B)', 0):.4f}")
    print(f"  mAP@0.5:0.95: {metrics.get('metrics/mAP50-95(B)', 0):.4f}")
    print(f"  Precision:   {metrics.get('metrics/precision(B)', 0):.4f}")
    print(f"  Recall:      {metrics.get('metrics/recall(B)', 0):.4f}")
    
    # Compare to baseline
    baseline_map = 0.0723  # 7.23%
    final_map = metrics.get('metrics/mAP50(B)', 0)
    
    print(f"\nBaseline Comparison:")
    print(f"  MATLAB baseline: {baseline_map*100:.2f}%")
    print(f"  Your result:     {final_map*100:.2f}%")
    
    if final_map > baseline_map:
        improvement = (final_map - baseline_map) / baseline_map * 100
        print(f"  🎉 IMPROVEMENT: +{improvement:.1f}%")
    elif final_map > 0:
        print(f"  📊 Keep tuning to beat baseline!")
    else:
        print(f"  ⚠ Check evaluation settings if mAP is 0%")
    
    print(f"\nWeights saved to:")
    print(f"  Best:  {run_dir}/weights/best.pt")
    print(f"  Last:  {run_dir}/weights/last.pt")
    print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 on VisDrone")
    parser.add_argument('--model', type=str, help='Model size (yolov8n/s/m/l/x)')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--imgsz', type=int, help='Image size')
    parser.add_argument('--batch', type=int, help='Batch size')
    
    args = parser.parse_args()
    main(args)
