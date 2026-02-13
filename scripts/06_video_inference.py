"""
Script 6: Run inference on videos and generate demo outputs

Usage:
    python scripts/06_video_inference.py --video path/to/video.mp4 --conf 0.3
    python scripts/06_video_inference.py --video sequences/folder --conf 0.3
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

# Import YOLOv8
try:
    from ultralytics import YOLO
except ImportError:
    print("ERROR: ultralytics not installed!")
    print("Install with: pip install ultralytics")
    sys.exit(1)

# Import OpenCV for video compilation
try:
    import cv2
except ImportError:
    print("ERROR: opencv-python not installed!")
    print("Install with: pip install opencv-python")
    sys.exit(1)


def main(args):
    """Run inference on video(s) and generate demo outputs."""
    
    print("="*70)
    print("STEP 6: VIDEO INFERENCE - DEMO GENERATION")
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
        # Use the latest trained model
        weights_path = Path("runs/detect/runs/detect/yolov8s_20260213_185302/weights/best.pt")
    
    if not weights_path.exists():
        print(f"ERROR: Weights not found: {weights_path}")
        print("\nPlease provide weights with --weights argument")
        return
    
    print(f"\nLoading model from: {weights_path}")
    model = YOLO(str(weights_path))
    print("✓ Model loaded")
    
    # Get video source
    if args.video:
        video_source = Path(args.video)
    else:
        # Default to VisDrone validation sequences
        video_source = Path("VisDrone2019-VID-val/sequences")
    
    if not video_source.exists():
        print(f"ERROR: Video source not found: {video_source}")
        print("\nPlease provide valid --video path")
        return
    
    # Confidence threshold
    conf_threshold = args.conf if args.conf is not None else 0.3
    
    print(f"\nConfiguration:")
    print(f"  Video source:  {video_source}")
    print(f"  Confidence:    {conf_threshold}")
    print(f"  Image size:    {args.imgsz}")
    print(f"  Save videos:   {args.save}")
    print(f"  Show live:     {args.show}")
    
    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("runs/videos") / f"demo_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nOutput directory: {output_dir}")
    print("="*70)
    
    # Process videos
    if video_source.is_file():
        # Single video
        print(f"\nProcessing video: {video_source.name}")
        process_video(model, video_source, output_dir, conf_threshold, args)
    elif video_source.is_dir():
        # Directory of videos or image sequences
        video_files = list(video_source.glob("*.mp4")) + \
                     list(video_source.glob("*.avi")) + \
                     list(video_source.glob("*.mov"))
        
        # Also check for VisDrone sequences (folders with images)
        sequence_folders = [d for d in video_source.iterdir() if d.is_dir()]
        
        if video_files:
            print(f"\nFound {len(video_files)} video files")
            for video_file in video_files:
                print(f"\n{'='*70}")
                print(f"Processing: {video_file.name}")
                print('='*70)
                process_video(model, video_file, output_dir, conf_threshold, args)
        
        if sequence_folders:
            print(f"\nFound {len(sequence_folders)} sequence folders")
            for seq_folder in sequence_folders:
                print(f"\n{'='*70}")
                print(f"Processing sequence: {seq_folder.name}")
                print('='*70)
                process_sequence(model, seq_folder, output_dir, conf_threshold, args)
        
        if not video_files and not sequence_folders:
            print(f"ERROR: No videos or sequences found in {video_source}")
            return
    else:
        print(f"ERROR: Invalid video source: {video_source}")
        return
    
    print("\n" + "="*70)
    print("VIDEO INFERENCE COMPLETE!")
    print("="*70)
    print(f"✓ Results saved to: {output_dir}")
    print("\nGenerated demo videos are ready for presentation!")


def process_video(model, video_path, output_dir, conf_threshold, args):
    """Process a single video file."""
    results = model.predict(
        source=str(video_path),
        imgsz=args.imgsz,
        conf=conf_threshold,
        iou=0.7,
        max_det=500,
        save=args.save,
        show=args.show,
        project=str(output_dir.parent),
        name=output_dir.name,
        exist_ok=True,
        stream=True,  # Stream mode for videos
        verbose=True
    )
    
    # Process results
    frame_count = 0
    for result in results:
        frame_count += 1
    
    print(f"✓ Processed {frame_count} frames")


def process_sequence(model, seq_folder, output_dir, conf_threshold, args):
    """Process an image sequence folder (VisDrone format)."""
    # Find images in sequence
    images = sorted(list(seq_folder.glob("*.jpg")) + list(seq_folder.glob("*.png")))
    
    if not images:
        print(f"WARNING: No images found in {seq_folder.name}")
        return
    
    print(f"Found {len(images)} frames in sequence")
    
    # Create output subfolder for this sequence
    seq_output = output_dir / seq_folder.name
    seq_output.mkdir(parents=True, exist_ok=True)
    
    # Process sequence
    results = model.predict(
        source=str(seq_folder),
        imgsz=args.imgsz,
        conf=conf_threshold,
        iou=0.7,
        max_det=500,
        save=args.save,
        show=args.show,
        project=str(output_dir.parent),
        name=output_dir.name + "/" + seq_folder.name,
        exist_ok=True,
        stream=True,
        verbose=False  # Less verbose for sequences
    )
    
    # Process results
    frame_count = 0
    for result in results:
        frame_count += 1
        if frame_count % 50 == 0:
            print(f"  Processed {frame_count}/{len(images)} frames...")
    
    print(f"✓ Processed {frame_count} frames from {seq_folder.name}")
    
    # Compile frames into video
    if args.save:
        print(f"  Compiling frames into video...")
        video_path = compile_frames_to_video(
            output_dir / seq_folder.name,
            seq_folder.name,
            fps=args.fps
        )
        if video_path:
            print(f"✓ Video saved: {video_path.name}")


def compile_frames_to_video(frames_dir, video_name, fps=30):
    """Compile annotated frames into a video file."""
    # Find all annotated frames
    frames = sorted(list(frames_dir.glob("*.jpg")) + list(frames_dir.glob("*.png")))
    
    if not frames:
        print(f"  WARNING: No frames found in {frames_dir}")
        return None
    
    # Read first frame to get dimensions
    first_frame = cv2.imread(str(frames[0]))
    if first_frame is None:
        print(f"  ERROR: Could not read first frame")
        return None
    
    height, width = first_frame.shape[:2]
    
    # Create video writer
    video_path = frames_dir.parent / f"{video_name}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"  ERROR: Could not create video writer")
        return None
    
    # Write frames to video
    for frame_path in frames:
        frame = cv2.imread(str(frame_path))
        if frame is not None:
            out.write(frame)
    
    out.release()
    
    return video_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video inference for YOLOv8 model")
    parser.add_argument('--weights', type=str, default=None,
                       help='Path to weights file (default: latest trained model)')
    parser.add_argument('--video', type=str, default=None,
                       help='Path to video file or directory (default: VisDrone val sequences)')
    parser.add_argument('--conf', type=float, default=0.3,
                       help='Confidence threshold (default: 0.3)')
    parser.add_argument('--imgsz', type=int, default=384,
                       help='Inference image size (default: 384)')
    parser.add_argument('--fps', type=int, default=30,
                       help='Output video FPS (default: 30)')
    parser.add_argument('--save', action='store_true', default=True,
                       help='Save output videos/images (default: True)')
    parser.add_argument('--show', action='store_true',
                       help='Display results in real-time')
    parser.add_argument('--no-save', dest='save', action='store_false',
                       help='Do not save output videos')
    
    args = parser.parse_args()
    main(args)
