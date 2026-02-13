# VisDrone Object Detection with YOLOv8

Real-time object detection pipeline for aerial imagery using YOLOv8 on the VisDrone 2019 dataset.

**Project Status:** ✅ Complete  
**Achieved:** **28.1% mAP@0.5** (3.9× baseline improvement)  
**Model:** YOLOv8s (11.1M parameters)

---

## 🎯 Project Summary

Successfully trained and deployed YOLOv8s for detecting 10 object classes in dense aerial imagery. The project demonstrates end-to-end pipeline development from data conversion to video inference, with emphasis on handling extreme small-object detection challenges under hardware constraints.

**Key Achievements:**
- 📈 28.1% mAP@0.5 (vs 7.23% MATLAB baseline)
- 🎥 Generated 7 annotated demo videos from validation sequences
- ⚡ Optimized for 6GB VRAM laptop GPU (RTX 3060)
- 🔬 Systematic precision-recall trade-off analysis

---

## 📁 Project Structure

```
Second Project/
├── configs/
│   ├── visdrone.yaml              # YOLOv8 dataset config
│   └── experiment_config.yaml     # Training hyperparameters
├── src/
│   ├── data/
│   │   ├── convert.py             # VisDrone → YOLO conversion
│   │   └── __init__.py
│   └── utils/
│       ├── logging.py             # Experiment logging
│       ├── reproducibility.py     # Seed management
│       ├── visualization.py       # Plotting utilities
│       └── __init__.py
├── scripts/
│   ├── 01_convert_annotations.py  # Convert VisDrone to YOLO format
│   ├── 02_verify_data.py          # Visual verification
│   ├── 03_prepare_dataset.py      # Create train/val split
│   ├── 04_train.py                # Train YOLOv8
│   ├── 05_evaluate.py             # Evaluate and compare
│   ├── 06_video_inference.py      # Video sequence inference
│   └── compile_videos.py          # Frame-to-video compilation
├── Annotations/_train/            # Original VisDrone annotations
├── images1/images/                # Original images (1610 files)
├── dataset/                       # Processed YOLO format dataset
│   ├── images/train/              # 1288 training images
│   ├── images/val/                # 322 validation images
│   ├── labels/train/              # Training annotations
│   └── labels/val/                # Validation annotations
├── runs/detect/                   # Training outputs
│   ├── yolov8s_20260213_185302/   # Best model checkpoint
│   └── runs/videos/               # Demo videos
├── VisDrone2019-VID-val/          # Video sequences for inference
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Activate your conda environment
conda activate cv310

# Install requirements
pip install -r requirements.txt
```

### 2. Run Pipeline

Execute scripts in order:

```bash
# Step 1: Convert annotations (VisDrone → YOLO format)
python scripts/01_convert_annotations.py

# Step 2: Verify conversion visually
python scripts/02_verify_data.py

# Step 3: Prepare dataset (train/val split + file organization)
python scripts/03_prepare_dataset.py

# Step 4: Train YOLOv8 model
python scripts/04_train.py

# Step 5: Evaluate and compare to baseline
python scripts/05_evaluate.py

# Step 6: Generate demo videos
python scripts/06_video_inference.py --conf 0.3
python scripts/compile_videos.py --demo-dir "runs/detect/runs/videos/demo_TIMESTAMP" --fps 30
```

---

## 🏆 Results

### Training Performance

**Model:** YOLOv8s  
**Training Time:** 1.42 hours (73 epochs, early stopped at epoch 63)  
**Resolution:** 384×384 pixels  
**Hardware:** RTX 3060 Laptop GPU (6GB VRAM)

| Metric | Training Best | Evaluation |
|--------|--------------|------------|
| mAP@0.5 | 23.7% | **28.1%** |
| mAP@0.5:0.95 | 12.6% | 16.1% |
| Precision | - | 33.7% |
| Recall | - | 31.0% |

**Baseline Comparison:** 3.9× improvement over MATLAB baseline (7.23% → 28.1%)

### Per-Class Performance

| Class | mAP@0.5 | Notes |
|-------|---------|-------|
| car | 59.7% | Best performing |
| bus | 55.4% | Strong detection |
| van | 42.2% | Good |
| truck | 40.3% | Good |
| motor | 32.7% | Moderate |
| tricycle | 21.0% | Challenging |
| awning-tricycle | 17.5% | Small objects |
| pedestrian | 13.8% | Dense occlusion |
| people | 7.1% | Group ambiguity |
| bicycle | 5.5% | Weakest class |

### Confidence Threshold Analysis

Different operating points trade precision for recall:

| Threshold | mAP@0.5* | Precision | Recall | Use Case |
|-----------|----------|-----------|--------|----------|
| 0.001 | 28.1% | 33.7% | 31.0% | Evaluation baseline |
| 0.3 | 35.7% | 45.6% | 24.4% | Balanced deployment |
| 0.5 | 40.3% | 61.6% | 18.4% | High-precision mode |

*Note: mAP values at conf > 0.001 are threshold-filtered metrics, not true model performance*

### Demo Videos

7 annotated sequences generated showing real-time detection:
- `uav0000086_00000_v`: 464 frames
- `uav0000117_02622_v`: 349 frames  
- `uav0000137_00458_v`: 233 frames
- `uav0000182_00000_v`: 363 frames
- `uav0000268_05773_v`: 978 frames
- `uav0000305_00000_v`: 184 frames
- `uav0000339_00001_v`: 275 frames

**Total:** 2,846 frames processed @ 30 FPS

---

## 📊 Dataset Information

### VisDrone 2019 Detection Challenge

- **Images:** 1,610 training images (1360×765 resolution)
- **Objects:** ~75,000 annotated objects (after filtering)
- **Classes (10):**
  1. pedestrian (27.1%)
  2. people (8.2%)
  3. bicycle (1.7%)
  4. car (36.2%)
  5. van (7.4%)
  6. truck (3.4%)
  7. tricycle (0.7%)
  8. awning-tricycle (0.8%)
  9. bus (3.8%)
  10. motor (7.5%)

### Key Challenges

- **67% small objects** (<32×32 pixels)
- **Median object size:** 20×23 pixels
- **42% occluded** objects
- **Aerial perspective** with high object density

---

## ⚙️ Configuration

All settings are in `configs/experiment_config.yaml`:

```yaml
# Final optimized parameters
model: yolov8s             # 11.1M parameters
epochs: 100                # Early stopping at epoch 63
imgsz: 384                 # Memory-optimized resolution
batch: 4                   # RTX 3060 6GB limit
workers: 0                 # Single-threaded (memory constraint)
plots: False               # Disabled for memory efficiency
patience: 10               # Early stopping patience
seed: 42                   # Reproducibility
```

**Memory Optimization:**
- `workers=0`: Prevents RAM fragmentation during validation
- `plots=False`: Avoids O(N²) confusion matrix memory allocation
- `imgsz=384`: Balances small-object detection with VRAM constraints

---

## 📈 Training Insights

### Memory Constraints Handling

Initial configuration (416×416, batch 8) caused OOM during validation metrics computation. Solution:
1. Reduced resolution to 384×384
2. Set `workers=0` (single-threaded dataloader)
3. Disabled plotting (`plots=False`)
4. Maintained batch size 4 for stable training

### Model Selection

| Model   | Parameters | GPU Memory | Expected mAP |
|---------|-----------|------------|--------------|
| yolov8n | 3.0M      | 2-3 GB     | 15-20%       |
| **yolov8s** | **11.1M** | **3-4 GB** | **23-28%** ✅ |
| yolov8m | 25.9M     | 5-6 GB     | 30-35%       |

**Achieved:** 28.1% mAP@0.5 with YOLOv8s

---

## 🎯 Evaluation Metrics

All metrics computed on 322 validation images (20% split):

**Primary Metrics:**
- **mAP@0.5:** 28.1% (main metric, 3.9× baseline)
- **mAP@0.5:0.95:** 16.1% (COCO-style metric)
- **Precision:** 33.7% @ conf=0.001
- **Recall:** 31.0% @ conf=0.001

**Results Location:**
- Training outputs: `runs/detect/yolov8s_20260213_185302/`
- Best weights: `weights/best.pt` (epoch 63)
- Last weights: `weights/last.pt` (epoch 73)
- Demo videos: `runs/detect/runs/videos/demo_20260213_211950/`

---

## 🔍 Key Learnings

### Technical Challenges

1. **Small Object Detection:**
   - 67% of objects <32×32 pixels → required resolution/model trade-offs
   - Best performing: Large vehicles (car 59.7%, bus 55.4%)
   - Weakest: Small/ambiguous objects (bicycle 5.5%, people 7.1%)

2. **Memory Management:**
   - Validation metrics computation bottleneck (not GPU VRAM)
   - Solution: `workers=0`, `plots=False`, resolution reduction
   - Enabled training on laptop GPU without external compute

3. **Precision-Recall Trade-offs:**
   - Higher confidence threshold → higher precision, lower recall
   - Deployment choice depends on application (false positives vs false negatives)
   - conf=0.3 provides balanced detection for most use cases

### Production Considerations

- **Inference Speed:** ~30 FPS on RTX 3060 @ 384×384
- **Model Size:** 22.5 MB (best.pt)
- **Deployment:** OpenCV-compatible video processing pipeline
- **Scalability:** Batch processing for large video datasets

---

## 🔍 Verification

After conversion, check `verification_plots/` for:
- **Red boxes:** Original VisDrone annotations
- **Green boxes:** Converted YOLO annotations

If boxes don't align, there's a conversion bug.

---

## 🐛 Common Issues & Solutions

### Out of Memory (OOM)
**Problem:** Crashes during validation metric computation  
**Solution:**
- Set `workers: 0` in config
- Set `plots: False` in config
- Reduce `imgsz` to 384 or 320
- Reduce batch size to 2

### OpenMP Runtime Conflict
**Problem:** Multiple OpenMP libraries error  
**Solution:** Add to script header:
```python
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
```

### Low mAP on Small Objects
**Problem:** Bicycle, people classes perform poorly  
**Explanation:** Insufficient visual information in 20×23 pixel median size
- Bicycles often indistinguishable from motorcycles at aerial view
- People vs pedestrian ambiguity in dataset annotations
- Consider this during deployment threshold selection

---

## 📝 Reproducibility

All experiments are fully reproducible:
- **Fixed seed:** 42 (NumPy, PyTorch, CUDA)
- **Deterministic operations:** cudnn.deterministic = True
- **Split saved:** `splits/train_images.txt`, `splits/val_images.txt` (sequence-stratified)
- **Config tracked:** Full hyperparameters saved with each run
- **Version controlled:** See `requirements.txt` for exact package versions

**To reproduce results:**
```bash
python scripts/04_train.py  # Automatically loads saved config
```

---

## 📚 Technical Stack

- **Python:** 3.10.16
- **PyTorch:** 2.7.1 (CUDA 11.8)
- **Ultralytics:** 8.4.9
- **OpenCV:** 4.x (video processing)
- **Hardware:** RTX 3060 Laptop GPU (6GB VRAM)

---

## 🎓 References

- **Dataset:** [VisDrone 2019 Detection Challenge](http://aiskyeye.com/)
- **Model:** [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- **Paper:** Zhu et al. (2020), "VisDrone-DET2019: The Vision Meets Drone Object Detection Challenge"

---

## 📧 Project Information

**Status:** Complete ✅  
**Completion Date:** February 13, 2026  
**Training Duration:** 1.42 hours  
**Total Development Time:** ~8 hours (including pipeline development, debugging, evaluation)

**Final Deliverables:**
- Trained YOLOv8s model (28.1% mAP@0.5)
- Complete training/inference pipeline
- 7 annotated demo videos
- Comprehensive documentation

For detailed project summary and technical deep-dive, see [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)

---

## 📄 License

This project uses the VisDrone dataset under their terms of use. The code is provided for educational purposes.
