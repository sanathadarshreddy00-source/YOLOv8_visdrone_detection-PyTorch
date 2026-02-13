"""
Quick script to compile existing annotated frames into video files
"""
import cv2
from pathlib import Path
import argparse


def compile_frames_to_video(frames_dir, output_path, fps=30):
    """Compile frames into a video file."""
    # Find all frames
    frames = sorted(list(frames_dir.glob("*.jpg")) + list(frames_dir.glob("*.png")))
    
    if not frames:
        print(f"  No frames found in {frames_dir.name}")
        return False
    
    print(f"  Found {len(frames)} frames")
    
    # Read first frame to get dimensions
    first_frame = cv2.imread(str(frames[0]))
    if first_frame is None:
        print(f"  ERROR: Could not read first frame")
        return False
    
    height, width = first_frame.shape[:2]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"  ERROR: Could not create video writer")
        return False
    
    # Write frames to video
    for i, frame_path in enumerate(frames):
        frame = cv2.imread(str(frame_path))
        if frame is not None:
            out.write(frame)
        if (i + 1) % 100 == 0:
            print(f"    Writing frame {i+1}/{len(frames)}...")
    
    out.release()
    print(f"✓ Video saved: {output_path.name}")
    return True


def main(args):
    """Compile all sequence folders in the demo directory."""
    demo_dir = Path(args.demo_dir)
    
    if not demo_dir.exists():
        print(f"ERROR: Directory not found: {demo_dir}")
        return
    
    print("="*70)
    print("COMPILING FRAMES TO VIDEOS")
    print("="*70)
    print(f"Demo directory: {demo_dir}")
    print(f"Output FPS: {args.fps}")
    print("="*70)
    
    # Find all sequence folders
    sequence_folders = [d for d in demo_dir.iterdir() if d.is_dir()]
    
    if not sequence_folders:
        print("No sequence folders found!")
        return
    
    print(f"\nFound {len(sequence_folders)} sequences to compile\n")
    
    success_count = 0
    for seq_folder in sequence_folders:
        print(f"Processing: {seq_folder.name}")
        video_path = demo_dir / f"{seq_folder.name}.mp4"
        if compile_frames_to_video(seq_folder, video_path, fps=args.fps):
            success_count += 1
        print()
    
    print("="*70)
    print(f"COMPLETE! Compiled {success_count}/{len(sequence_folders)} videos")
    print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile annotated frames into videos")
    parser.add_argument('--demo-dir', type=str, 
                       default='runs/videos/demo_20260213_211950',
                       help='Path to demo directory with frame folders')
    parser.add_argument('--fps', type=int, default=30,
                       help='Output video FPS (default: 30)')
    
    args = parser.parse_args()
    main(args)
