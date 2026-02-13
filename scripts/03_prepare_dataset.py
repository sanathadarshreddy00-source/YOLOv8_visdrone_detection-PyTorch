"""
Script 3: Prepare YOLO dataset structure with train/val split

Usage:
    python scripts/03_prepare_dataset.py
"""
import sys
from pathlib import Path
import yaml
import random
import shutil
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.reproducibility import set_seed
from src.utils.logging import setup_logger


def main():
    """Organize images and labels into YOLO dataset structure."""
    
    print("="*70)
    print("STEP 3: PREPARE YOLO DATASET (TRAIN/VAL SPLIT)")
    print("="*70)
    
    # Load config
    config_path = Path("configs/experiment_config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Set seed for reproducible split
    set_seed(config['seed'])
    
    # Setup logger
    logger = setup_logger('prepare', 'logs')
    logger.info("Preparing dataset...")
    
    # Paths
    images_dir = Path("images1/images")
    labels_dir = Path("labels_converted")
    output_dir = Path("dataset")
    
    # Create output structure
    (output_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
    (output_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)
    
    # Get all images
    image_files = sorted(images_dir.glob("*.jpg"))
    logger.info(f"Found {len(image_files)} images")
    
    # Filter images that have corresponding labels
    valid_images = []
    for img in image_files:
        label_file = labels_dir / (img.stem + ".txt")
        if label_file.exists():
            valid_images.append(img)
        else:
            logger.warning(f"No label for {img.name}")
    
    logger.info(f"Valid images with labels: {len(valid_images)}")
    
    # Shuffle and split
    random.shuffle(valid_images)
    
    train_ratio = config['split']['train_ratio']
    split_idx = int(len(valid_images) * train_ratio)
    
    train_images = valid_images[:split_idx]
    val_images = valid_images[split_idx:]
    
    print(f"\nDataset split (seed={config['seed']}):")
    print(f"  Training:   {len(train_images)} images ({len(train_images)/len(valid_images)*100:.1f}%)")
    print(f"  Validation: {len(val_images)} images ({len(val_images)/len(valid_images)*100:.1f}%)")
    
    # Save split lists for reproducibility
    split_dir = Path("splits")
    split_dir.mkdir(exist_ok=True)
    
    with open(split_dir / "train_images.txt", 'w') as f:
        f.write("\n".join([img.name for img in train_images]))
    
    with open(split_dir / "val_images.txt", 'w') as f:
        f.write("\n".join([img.name for img in val_images]))
    
    print(f"\n✓ Split lists saved to: {split_dir}")
    
    # Copy files
    def copy_files(image_list, split_name):
        print(f"\nCopying {split_name} files...")
        for img in tqdm(image_list, desc=f"{split_name}"):
            # Copy image
            src_img = img
            dst_img = output_dir / "images" / split_name / img.name
            shutil.copy2(src_img, dst_img)
            
            # Copy label
            src_label = labels_dir / (img.stem + ".txt")
            dst_label = output_dir / "labels" / split_name / (img.stem + ".txt")
            shutil.copy2(src_label, dst_label)
    
    copy_files(train_images, "train")
    copy_files(val_images, "val")
    
    print("\n" + "="*70)
    print("DATASET PREPARATION COMPLETE")
    print("="*70)
    print(f"Dataset structure:")
    print(f"  {output_dir}/")
    print(f"    ├── images/")
    print(f"    │   ├── train/ ({len(train_images)} images)")
    print(f"    │   └── val/   ({len(val_images)} images)")
    print(f"    └── labels/")
    print(f"        ├── train/ ({len(train_images)} labels)")
    print(f"        └── val/   ({len(val_images)} labels)")
    print("\n✓ Ready for YOLOv8 training!")
    print("="*70)
    
    logger.info("Dataset preparation complete!")


if __name__ == "__main__":
    main()
