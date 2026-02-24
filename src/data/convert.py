"""Convert VisDrone annotations to YOLO format."""
from typing import Tuple, Optional, Dict
from pathlib import Path
import numpy as np
from PIL import Image
from src.utils import paths


class VisDroneConverter:
    """Convert VisDrone annotations to YOLO format.
    
    VisDrone format: <left>,<top>,<width>,<height>,<score>,<class_id>,<truncation>,<occlusion>
    YOLO format: <class_id> <x_center> <y_center> <width> <height> (all normalized 0-1)
    """
    
    # Class mapping (VisDrone 1-10 → YOLO 0-9)
    CLASS_MAPPING = {
        1: 0,   # pedestrian
        2: 1,   # people
        3: 2,   # bicycle
        4: 3,   # car
        5: 4,   # van
        6: 5,   # truck
        7: 6,   # tricycle
        8: 7,   # awning-tricycle
        9: 8,   # bus
        10: 9   # motor
    }
    
    CLASS_NAMES = [
        'pedestrian', 'people', 'bicycle', 'car', 'van',
        'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor'
    ]
    
    def __init__(self, 
                 filter_ignored: bool = True,
                 min_box_area: int = 4,
                 min_dimension: int = 2):
        """Initialize converter.
        
        Args:
            filter_ignored: Filter objects with score=0 (ignored regions)
            min_box_area: Minimum bounding box area in pixels
            min_dimension: Minimum width or height dimension
        """
        self.filter_ignored = filter_ignored
        self.min_box_area = min_box_area
        self.min_dimension = min_dimension
        
    def parse_visdrone_line(self, line: str) -> Optional[Dict]:
        """Parse one line of VisDrone annotation.
        
        Args:
            line: Annotation line in VisDrone format
            
        Returns:
            Dictionary with parsed fields or None if invalid
        """
        parts = line.strip().split(',')
        if len(parts) != 8:
            return None
            
        try:
            left, top, width, height, score, class_id, truncation, occlusion = map(int, parts)
        except ValueError:
            return None
        
        return {
            'left': left,
            'top': top,
            'width': width,
            'height': height,
            'score': score,
            'class_id': class_id,
            'truncation': truncation,
            'occlusion': occlusion
        }
    
    def should_keep_object(self, obj: Dict) -> bool:
        """Apply filtering rules to determine if object should be kept.
        
        Args:
            obj: Parsed annotation dictionary
            
        Returns:
            True if object should be kept, False otherwise
        """
        # Filter ignored regions (score=0)
        if self.filter_ignored and obj['score'] == 0:
            return False
        
        # Filter invalid class IDs (0=background, 11=other)
        if obj['class_id'] == 0 or obj['class_id'] == 11:
            return False
        
        # Filter valid class range (1-10)
        if obj['class_id'] not in self.CLASS_MAPPING:
            return False
        
        # Filter by minimum size
        area = obj['width'] * obj['height']
        if area < self.min_box_area:
            return False
        
        if obj['width'] < self.min_dimension or obj['height'] < self.min_dimension:
            return False
        
        return True
    
    def convert_to_yolo(self, 
                       obj: Dict, 
                       img_width: int, 
                       img_height: int) -> Tuple[int, float, float, float, float]:
        """Convert VisDrone bbox to YOLO format (normalized).
        
        Args:
            obj: Parsed annotation dictionary
            img_width: Image width in pixels
            img_height: Image height in pixels
            
        Returns:
            Tuple of (class_id, x_center, y_center, width, height) normalized to [0, 1]
        """
        # Convert to center coordinates
        x_center = (obj['left'] + obj['width'] / 2) / img_width
        y_center = (obj['top'] + obj['height'] / 2) / img_height
        norm_width = obj['width'] / img_width
        norm_height = obj['height'] / img_height
        
        # Map class ID (VisDrone 1-10 → YOLO 0-9)
        yolo_class = self.CLASS_MAPPING[obj['class_id']]
        
        # Clamp to [0, 1] to handle edge cases
        x_center = np.clip(x_center, 0, 1)
        y_center = np.clip(y_center, 0, 1)
        norm_width = np.clip(norm_width, 0, 1)
        norm_height = np.clip(norm_height, 0, 1)
        
        return yolo_class, x_center, y_center, norm_width, norm_height
    
    def convert_annotation_file(self, 
                                ann_path: Path, 
                                img_path: Path,
                                output_path: Path) -> Dict:
        """Convert single annotation file from VisDrone to YOLO format.
        
        Args:
            ann_path: Path to VisDrone annotation file
            img_path: Path to corresponding image file
            output_path: Path to save YOLO format annotation
            
        Returns:
            Dictionary with conversion statistics
            
        Raises:
            ValueError: If image cannot be read
        """
        # Read image to get dimensions
        try:
            img = Image.open(img_path)
            img_width, img_height = img.size
        except Exception as e:
            raise ValueError(f"Could not read image {img_path}: {e}")
        
        # Parse annotations
        with open(ann_path, 'r') as f:
            lines = f.readlines()
        
        stats = {
            'total': 0,
            'kept': 0,
            'filtered_ignored': 0,
            'filtered_class': 0,
            'filtered_size': 0
        }
        
        yolo_annotations = []
        
        for line in lines:
            if not line.strip():
                continue
                
            stats['total'] += 1
            obj = self.parse_visdrone_line(line)
            
            if obj is None:
                continue
            
            # Track filtering reasons
            if self.filter_ignored and obj['score'] == 0:
                stats['filtered_ignored'] += 1
                continue
            
            if obj['class_id'] == 0 or obj['class_id'] == 11:
                stats['filtered_class'] += 1
                continue
            
            if obj['class_id'] not in self.CLASS_MAPPING:
                stats['filtered_class'] += 1
                continue
            
            area = obj['width'] * obj['height']
            if (area < self.min_box_area or 
                obj['width'] < self.min_dimension or 
                obj['height'] < self.min_dimension):
                stats['filtered_size'] += 1
                continue
            
            # Convert and add
            yolo_bbox = self.convert_to_yolo(obj, img_width, img_height)
            yolo_annotations.append(yolo_bbox)
            stats['kept'] += 1
        
        # Write YOLO format
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            for class_id, x, y, w, h in yolo_annotations:
                f.write(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
        
        return stats
