"""
Script 1: Convert VisDrone annotations to YOLO format

Usage:
    python scripts/01_convert_annotations.py
"""
import sys
from pathlib import Path
import yaml
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.convert import VisDroneConverter
from src.utils.logging import setup_logger
from src.utils.reproducibility import set_seed
from src.utils import paths


def main():
    """Convert all VisDrone annotations to YOLO format."""
    
    # Setup
    print("="*70)
    print("STEP 1: CONVERT VISDRONE ANNOTATIONS TO YOLO FORMAT")
    print("="*70)
    
    # Load config
    config_path = Path("configs/experiment_config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Set seed for reproducibility
    set_seed(config['seed'])
    
    # Setup logger
    logger = setup_logger('convert', 'logs')
    logger.info("Starting annotation conversion...")
    
    # Paths (centralized)
    annotations_dir = paths.ANNOTATIONS
    images_dir = paths.IMAGES
    output_dir = paths.LABELS
    
    # Validate paths
    if not annotations_dir.exists():
        logger.error(f"Annotations directory not found: {annotations_dir}")
        return
    if not images_dir.exists():
        logger.error(f"Images directory not found: {images_dir}")
        return
    
    # Initialize converter
    converter = VisDroneConverter(
        filter_ignored=config['data_filtering']['filter_ignored'],
        min_box_area=config['data_filtering']['min_box_area'],
        min_dimension=config['data_filtering']['min_dimension']
    )
    
    # Get all annotation files
    ann_files = sorted(annotations_dir.glob("*.txt"))
    logger.info(f"Found {len(ann_files)} annotation files")
    
    # Convert
    total_stats = {
        'total': 0,
        'kept': 0,
        'filtered_ignored': 0,
        'filtered_class': 0,
        'filtered_size': 0
    }
    
    failed_files = []
    
    for ann_file in tqdm(ann_files, desc="Converting annotations"):
        img_file = images_dir / (ann_file.stem + ".jpg")
        out_file = output_dir / ann_file.name
        
        if not img_file.exists():
            logger.warning(f"Image not found: {img_file}")
            failed_files.append(ann_file.name)
            continue
        
        try:
            stats = converter.convert_annotation_file(ann_file, img_file, out_file)
            for key in total_stats:
                total_stats[key] += stats[key]
        except Exception as e:
            logger.error(f"Error converting {ann_file.name}: {e}")
            failed_files.append(ann_file.name)
    
    # Summary
    print("\n" + "="*70)
    print("CONVERSION SUMMARY")
    print("="*70)
    print(f"Files processed:      {len(ann_files) - len(failed_files)}/{len(ann_files)}")
    print(f"Total objects:        {total_stats['total']}")
    print(f"Kept objects:         {total_stats['kept']} ({total_stats['kept']/total_stats['total']*100:.1f}%)")
    print(f"Filtered (ignored):   {total_stats['filtered_ignored']} ({total_stats['filtered_ignored']/total_stats['total']*100:.1f}%)")
    print(f"Filtered (class):     {total_stats['filtered_class']} ({total_stats['filtered_class']/total_stats['total']*100:.1f}%)")
    print(f"Filtered (size):      {total_stats['filtered_size']} ({total_stats['filtered_size']/total_stats['total']*100:.1f}%)")
    
    if failed_files:
        print(f"\nFailed files ({len(failed_files)}): {', '.join(failed_files[:5])}")
    
    print(f"\n✓ Converted annotations saved to: {output_dir}")
    print("="*70)
    
    logger.info("Conversion complete!")


if __name__ == "__main__":
    main()
