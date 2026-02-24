# YOLOv8 — VisDrone Detection (Canonical README)

One-line: Reproducible YOLOv8 detection pipeline for the VisDrone 2019 dataset — training, evaluation, and demo video generation.

---

## Quickstart

- Create and activate environment (example):

```powershell
conda create -n cv310 python=3.10 -y
conda activate cv310
```

- Install dependencies (if `requirements.txt` present):

```powershell
pip install -r requirements.txt
# or at minimum:
pip install ultralytics opencv-python pyyaml
```

- If Python cannot import `src`, set `PYTHONPATH` at project root before running scripts:

```powershell
$env:PYTHONPATH = "$(Get-Location)"
```

Run the three main steps (copy/paste):

```powershell
# Train (example)
python scripts/04_train.py

# Evaluate canonical model (conf=0.001)
python scripts/05_evaluate.py --weights runs/detect/yolov8s_20260224_1745502/weights/best.pt --conf 0.001

# Generate demo video for a sequence
python scripts/06_video_inference.py --video VisDrone2019-VID-val/sequences/uav0000137_00458_v --weights runs/detect/yolov8s_20260224_1745502/weights/best.pt --conf 0.3 --save
```

---

## Canonical experiment (this README is written around this run)

- Experiment name: `yolov8s_20260224_1745502`
- Weights: `runs/detect/yolov8s_20260224_1745502/weights/best.pt`
- Training args (excerpt): `imgsz=640`, `batch=4`, `workers=0`, `seed=42`
- Eval command used for reported metrics:

```bash
python scripts/05_evaluate.py --weights runs/detect/yolov8s_20260224_1745502/weights/best.pt --conf 0.001
```

Canonical metrics (validation, conf=0.001):

- mAP@0.5: **36.35%**
- mAP@0.5:0.95: **21.21%**
- Precision: **47.10%**
- Recall: **36.35%**

Per-class mAP@0.5 (conf=0.001):

| Class | mAP@0.5 |
|---|---:|
| pedestrian | 21.76% |
| people | 13.75% |
| bicycle | 10.69% |
| car | 73.78% |
| van | 42.95% |
| truck | 56.66% |
| tricycle | 20.58% |
| awning-tricycle | 29.25% |
| bus | 66.97% |
| motor | 27.14% |

Demo examples (produced under `runs/detect/videos/`):

- Example demo folder: `runs/detect/videos/demo_20260224_224656/`
- Example MP4: `runs/detect/videos/demo_20260224_224656/uav0000137_00458_v.mp4` (233 frames processed)

---

## Project layout (key folders)

- `configs/` — dataset and experiment YAMLs (`visdrone.yaml`, `experiment_config.yaml`, `paths.yaml`).
- `src/` — reusable utilities (paths loader, reproducibility, visualization).
- `scripts/` — runnable scripts: conversion, verification, prepare dataset, train, evaluate, video inference.
- `tools/` — helper tools (inventory generation, maintenance).
- `runs/detect/` — training experiment folders and demo outputs (weights and demo videos).
- `VisDrone2019-VID-val/` — video sequences used for demo inference.

---

## Script usage (short)

- `scripts/01_convert_annotations.py`: Convert VisDrone annotations to YOLO format.
- `scripts/02_verify_data.py`: Quick visual checks of conversion (saves verification plots).
- `scripts/03_prepare_dataset.py`: Creates `dataset/` structure and train/val splits.
- `scripts/04_train.py`: Trains YOLOv8 (reads `configs/experiment_config.yaml`).
- `scripts/05_evaluate.py`: Evaluates a weights file on validation set and prints metrics (use `--conf` to change threshold).
- `scripts/06_video_inference.py`: Runs inference on video files or VisDrone sequence folders, saves per-frame outputs and compiles MP4s.

Examples:

```bash
python scripts/05_evaluate.py --weights runs/detect/yolov8s_20260224_1745502/weights/best.pt --conf 0.001

python scripts/06_video_inference.py --video VisDrone2019-VID-val/sequences/uav0000137_00458_v --weights runs/detect/yolov8s_20260224_1745502/weights/best.pt --conf 0.3 --save
```

---

## Artifacts & what NOT to commit

- Do NOT commit:
  - Large model files (`*.pt`, `*.pth`) — use Git LFS, DVC, S3, or GitHub Releases.
  - Full datasets and image folders (e.g., `images1/`, `VisDrone2019-*`) — provide download instructions instead.
  - Runtime outputs: `runs/`, `logs/`, `backups/`.

- Recommended: commit small manifests that point to artifacts (e.g., `configs/last_experiment_args.yaml`, `DIRECTORY_RECORD_SUMMARY.json`) and an `artifacts/weights_manifest.json` with checksums/URLs for large weights.

---

## Reproducibility

To reproduce the canonical evaluation exactly:

1. Ensure the environment matches `requirements.txt` (or install dependencies above).
2. Set `PYTHONPATH` to project root if needed:

```powershell
$env:PYTHONPATH = "$(Get-Location)"
```

3. Run evaluation:

```bash
python scripts/05_evaluate.py --weights runs/detect/yolov8s_20260224_1745502/weights/best.pt --conf 0.001
```

Optional: regenerate demo video for a chosen sequence:

```bash
python scripts/06_video_inference.py --video VisDrone2019-VID-val/sequences/uav0000137_00458_v --weights runs/detect/yolov8s_20260224_1745502/weights/best.pt --conf 0.3 --save
```

---

## References

- VisDrone 2019 Detection challenge: http://aiskyeye.com/
- Ultralytics YOLOv8: https://github.com/ultralytics/ultralytics

---