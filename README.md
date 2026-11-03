# Skin Lesion Classification - Simplified Guide

**Quick reference for building this project step by step.**

> Research and education only. Not a medical diagnostic tool.

---

## 7 Classes to Predict

| Code | Full Name | Priority |
|------|-----------|----------|
| `nv` | Melanocytic nevi | 67% of data (majority class) |
| `mel` | Melanoma | 11% (most important for screening) |
| `bkl` | Benign keratosis-like | 11% |
| `bcc` | Basal cell carcinoma | 5% |
| `akiec` | Actinic keratoses | 3% |
| `vasc` | Vascular lesions | 1.5% |
| `df` | Dermatofibroma | 1% (minority class) |

**Class imbalance ratio: 63:1** (need focal loss)

---

## Project Pipeline Stages

### ✅ Stage 1: Exploratory Data Analysis (EDA)
**Status: COMPLETE**

**What we learned:**
- 10,015 images across 7 classes
- Severe imbalance: nv (67%) vs df (1%)
- **Must split by lesion_id** (not by image)
- 57 missing age values
- All images verified on disk

**Notebook:** `notebooks/01_data_inspection.ipynb`
**Documentation:** `docs/01_data_inspection.md`

---

### ✅ Stage 2: Data Validation
**Status: COMPLETE**

**What we verified:**
- ✅ All 10,015 images exist on disk
- ✅ All 7 classes present and valid
- ✅ No duplicate image_ids
- ✅ No corrupted images
- ✅ All lesion_ids have consistent labels
- ✅ Created clean metadata with image paths

**Notebook:** `notebooks/02_data_validation.ipynb`
**Documentation:** `docs/02_data_validation.md`

---

### ✅ Stage 3: Data Splitting
**Status: COMPLETE**

**Strategy:** Stratified Group Splitting (70% / 15% / 15%)

**Why this strategy:**
- **Group splitting** by `lesion_id` → Prevents data leakage
- **Stratified** by `dx` → Preserves class proportions (67% nv, 1% df)
- **Two-stage** → Achieves exact 70:15:15 ratio
- **Fixed seed** → Reproducible results

**Splitting strategies considered:**

| Strategy | Decision | Reason |
|----------|----------|--------|
| Random split (image-level) | ❌ Rejected | Data leakage from multi-image lesions |
| Stratified split alone | ❌ Rejected | Preserves classes but not leakage-safe |
| Group split alone | ⚠️ Considered | No leakage but loses class balance |
| Stratified Group K-Fold | ⚠️ Considered | Good for CV, but creates folds not 70:15:15 |
| **Stratified Shuffle Split (on grouped data)** | ✅ **Selected** | Best of both worlds |
| K-Fold Cross-Validation | 📅 Future | For robust final evaluation |

**Results:**
- Train: 5,234 lesions → 7,010 images (70%)
- Validation: 1,115 lesions → 1,502 images (15%)
- Test: 1,121 lesions → 1,503 images (15%)

**Verification:**
- ✅ No lesion overlap between splits (leakage check)
- ✅ Class proportions preserved in all splits
- ✅ All 7 classes present in all splits

**Files created:**
- `data/splits/train.csv` (7,010 rows)
- `data/splits/val.csv` (1,502 rows)
- `data/splits/test.csv` (1,503 rows)
- `data/splits/split_config.json`

**Notebook:** `notebooks/03_data_splitting.ipynb`
**Documentation:** `docs/03_data_splitting.md`

---

### ✅ Stage 4: Data Loading
**Status: COMPLETE**

**Architecture:** Three-layer design
- **Layer 1: Dataset** - Lazy loading with relative paths
- **Layer 2: Transforms** - Training augmentation + ImageNet normalization  
- **Layer 3: DataLoader** - Batching, shuffling, parallel loading

**What we built:**
- ✅ `SkinLesionDataset` - Custom PyTorch Dataset class
- ✅ Training transforms - Horizontal flip (50%), rotation (±15°), color jitter
- ✅ Validation transforms - Deterministic (resize + normalize only)
- ✅ DataLoader - Batch size 16, num_workers 4, pin_memory

**Key design decisions:**
- **Relative paths** - `data/raw/HAM10000_images_part_1/...` (portable)
- **ImageNet normalization** - Required for pretrained model compatibility
- **Different transforms** - Augmentation for training, none for val/test
- **Lazy loading** - Images loaded on-demand (memory efficient)

**Class mapping:**
```python
{'akiec': 0, 'bcc': 1, 'bkl': 2, 'df': 3, 'mel': 4, 'nv': 5, 'vasc': 6}
```

**Usage:**
```python
from src.data import get_dataloaders

train_loader, val_loader, test_loader = get_dataloaders(batch_size=16)

for images, labels in train_loader:
    print(images.shape)  # (16, 3, 224, 224)
    break
```

**Files created:**
- `src/data/__init__.py` (package exports)
- `src/data/dataset.py` (Dataset class with path resolution)
- `src/data/transforms.py` (train/val transforms)
- `src/data/dataloader.py` (DataLoader creation)

**Notebook:** Test transforms with `src/data/transforms.py`
**Documentation:** `docs/04_data_loading.md`

---

### ✅ Stage 5: Model Implementation - ResNet-18 COMPLETE
**Status: ResNet-18 DONE; VGG-16 and EfficientNet-B0 pending**

**Completed - ResNet-18:**
- ✅ `src/models/resnet18.py` - ResNet-18 with custom 7-class classifier
- ✅ ImageNet pretrained weights (`ResNet18_Weights.IMAGENET1K_V1`)
- ✅ Replaced FC layer: 512 → 7 outputs with Xavier initialization
- ✅ Grad-CAM target layer: `layer4[-1]` (final conv features)
- ✅ Tested: Input (B, 3, 224, 224) → Output (B, 7) logits
- ✅ Trained on Kaggle T4 GPU (30 epochs, ~45 minutes)
- ✅ Best validation accuracy: **85.67%**
- ✅ Model checkpoint saved with optimizer state

**ResNet-18 Features:**
```python
model = create_resnet18(
    num_classes=7,
    pretrained=True,      # ImageNet weights
    freeze_backbone=False # Full fine-tuning
)
# Output: logits (not probabilities) - apply softmax for predictions
```

**Training Results:**
- Parameters: 11,182,983 (11M)
- Batch size: 32 (T4 GPU)
- Training time: ~45 min for 30 epochs
- Best val accuracy: 85.67%
- Early stopping: patience=5

**Files created:**
- `src/models/resnet18.py` - Model implementation
- `src/models/__init__.py` - Model registry (partial)
- `tests/test_resnet18.py` - Unit tests
- `checkpoints/resnet18_best.pth` - Trained weights
- `docs/05_model_resnet18.md` - Documentation

**Next:** VGG-16, EfficientNet-B0

**All models must have same interface:**
```python
model = create_model('resnet18', num_classes=7, pretrained=True)
output = model(images)  # shape: [batch_size, 7]
```

**Key points:**
- Use `torchvision.models` with `pretrained=True`
- Replace final FC layer with 7 outputs
- Freeze early layers (optional)
- Test each model creates correct output shape

**Files to create:**
- `src/models/__init__.py`
- `src/models/resnet18.py`
- `src/models/vgg16.py`
- `src/models/efficientnet_b0.py`

---

### ✅ Stage 6: Loss Functions
**Status: COMPLETE (implemented and used in the VGG-16 runs)**

`src/losses/focal_loss.py` implements multiclass focal loss on raw logits (gamma=2.0 in the runs, optional per-class alpha not used). Cross-entropy is `torch.nn.CrossEntropyLoss`. A test checks that focal loss with gamma=0 equals cross-entropy. All 4 conditions are selectable in `train.py` (`--loss`, `--augmentation`/`--no-augmentation`). Whether focal loss helps is an empirical question; see the results table below.

**Goal:** Implement cross-entropy and focal loss

**Tasks:**
- [x] `src/losses/focal_loss.py` - focal loss implementation
  - Focus on hard examples
  - Down-weight easy examples
  - Formula: FL = -α(1-p)^γ log(p)
- [x] Support 4 experimental conditions:
  1. Cross-entropy, no augmentation
  2. Focal loss, no augmentation
  3. Cross-entropy, with augmentation
  4. Focal loss, with augmentation

**Why focal loss:**
- Handles 63:1 class imbalance
- Prevents model from ignoring minority classes (mel, df)
- Better for medical imaging with imbalanced data

**Files to create:**
- `src/losses/__init__.py`
- `src/losses/focal_loss.py`

---

### ✅ Stage 7: Training Loop
**Status: COMPLETE for VGG-16 (shared loop; other models pending)**

`src/training/trainer.py` provides the shared train/validation loop: Adam, CUDA AMP, per-epoch history, best checkpoint chosen by **validation macro F1**, early stopping (patience 5), atomic checkpoint saves, and `seed_everything`. `train.py` is the CLI entry point (settings are CLI arguments saved to `config.json`; the YAML config files and LR scheduler listed below were not implemented). Run it as described in [docs/vgg16_local_training.md](docs/vgg16_local_training.md).

**Goal:** Train models with checkpoint saving

**Tasks:**
- [x] `src/training/trainer.py` - shared train/val loop
  - Training loop with backpropagation
  - Validation loop (no grad)
  - Metric tracking (loss, accuracy, F1)
  - Checkpoint saving (best model based on val F1)
- [x] `train.py` - entry point script
  - Parse config YAML
  - Initialize model, loss, optimizer
  - Run training
  - Save results
- [ ] Config files: `configs/resnet18.yaml`, `configs/vgg16.yaml`, etc.

**Hyperparameters:**
- Learning rate: 1e-4 (start), 1e-3 (max)
- Batch size: 16 (GPU memory) or 32
- Epochs: 20-50 (with early stopping)
- Optimizer: Adam
- Scheduler: ReduceLROnPlateau or CosineAnnealing
- Early stopping: patience=5

**Checkpoint selection:**
- Based on **validation macro F1** (not accuracy!)
- Save best model only
- Save training history (loss curves)

**Files to create:**
- `src/training/__init__.py`
- `src/training/trainer.py`
- `configs/resnet18.yaml`
- `configs/vgg16.yaml`
- `configs/efficientnet_b0.yaml`
- `train.py`

---

### 🔶 Stage 8: Evaluation
**Status: PARTIAL - metrics and VGG-16 test evaluation done; cross-model comparison pending**

`src/evaluation/metrics.py` computes macro F1, balanced accuracy, per-class precision/recall/F1/support, melanoma sensitivity, one-vs-rest melanoma specificity, and the confusion matrix with the fixed class order. Evaluation is currently run by `train.py --evaluate-test` (once per finalized run, on the held-out test split); `evaluate.py` is still an empty placeholder.

**Goal:** Evaluate all models on test set

**Tasks:**
- [x] `src/evaluation/metrics.py` - metric calculations
  - Macro F1 (equal weight per class)
  - Balanced accuracy
  - Per-class precision/recall
  - Melanoma sensitivity/recall
  - Melanoma specificity
  - Confusion matrix
- [ ] `evaluate.py` - evaluation script
  - Load trained checkpoint
  - Run on test set
  - Generate metrics report
- [ ] Compare 3 architectures (ResNet-18, VGG-16, EfficientNet-B0)
- [ ] Compare 4 experimental conditions

**Required metrics (from AGENTS.md):**
- Macro F1 (not just accuracy!)
- Balanced accuracy
- Per-class precision and recall
- Melanoma sensitivity and specificity
- Confusion matrix
- Runtime and parameter count

**Files to create:**
- `src/evaluation/__init__.py`
- `src/evaluation/metrics.py`
- `evaluate.py`

---

### ⏳ Stage 9: Grad-CAM Visualization
**Status: PENDING**

**Goal:** Generate heatmaps showing what models focus on

**Tasks:**
- [ ] `src/explainability/gradcam.py` - Grad-CAM implementation
  - Compute gradients for target class
  - Generate heatmap
  - Overlay on original image
- [ ] Generate heatmaps for:
  - Correct predictions (each class)
  - Incorrect predictions (failure analysis)
  - Melanoma cases (critical!)
- [ ] Check for shortcuts (hair, rulers, borders)
  - Is model looking at lesion or artifacts?
  - Document findings
- [ ] Save visualizations to `outputs/gradcam/`

**Different target layers per architecture:**
- ResNet-18: `layer4` or `layer4.1`
- VGG-16: `features[28]` or `features[30]`
- EfficientNet-B0: `features` or specific block

**Guardrails:**
- Grad-CAM shows "where" model looks, not "why"
- Heatmap plausibility ≠ correctness
- Document limitations clearly

**Files to create:**
- `src/explainability/__init__.py`
- `src/explainability/gradcam.py`
- `notebooks/04_gradcam_analysis.ipynb`

---

### ⏳ Stage 10: Documentation & Final Report
**Status: PENDING**

**Goal:** Document results, limitations, and conclusions

**Tasks:**
- [ ] Update README.md with results
- [ ] Document all 4 experimental conditions
- [ ] Include confusion matrices and metrics
- [ ] Document Grad-CAM findings (artifacts vs lesions)
- [ ] Include educational-use disclaimer
- [ ] Report failure cases
- [ ] Compare 3 architectures with tables
- [ ] Do NOT claim clinical validation
- [ ] Document dataset limitations (HAM10000 specific)

**Required sections:**
- Executive summary
- Methodology (splits, models, training)
- Results (metrics, comparison)
- Interpretability (Grad-CAM analysis)
- Limitations (dataset, model, clinical)
- Conclusions
- References

**Files:**
- `README.md` (updated)
- `REPORT.md` (final report)
- `outputs/` (figures, tables)

---

## Quick Commands

```bash
# Run notebooks
uv run jupyter notebook notebooks/01_data_inspection.ipynb
uv run jupyter notebook notebooks/02_data_validation.ipynb
uv run jupyter notebook notebooks/03_data_splitting.ipynb

# Train a model
uv run python train.py --model resnet18 --config configs/resnet18.yaml

# Evaluate
uv run python evaluate.py --checkpoint checkpoints/resnet18_best.pth --split test

# Run tests
uv run pytest tests/

# Lint code
uv run ruff check .
```

---

## Current Progress Tracker

| Stage | Status | Files | Notes |
|-------|--------|-------|-------|
| 1. EDA | ✅ Done | `notebooks/01_data_inspection.ipynb`, `docs/01_data_inspection.md` | Class imbalance: 63:1, must split by lesion_id |
| 2. Validation | ✅ Done | `notebooks/02_data_validation.ipynb`, `docs/02_data_validation.md` | All 10,015 images verified |
| 3. Splitting | ✅ Done | `notebooks/03_data_splitting.ipynb`, `docs/03_data_splitting.md`, `data/splits/*.csv` | Stratified group split, no leakage |
| 4. Data Loading | ✅ Done | `src/data/*.py` | PyTorch Dataset, transforms, DataLoader |
| 5. Models | 🔶 Partial | `src/models/vgg16.py` | VGG-16 done; ResNet-18, EfficientNet-B0 pending |
| 6. Loss Functions | ✅ Done | `src/losses/focal_loss.py` | Focal loss + cross-entropy |
| 7. Training | ✅ Done (VGG-16) | `src/training/trainer.py`, `train.py` | Best checkpoint by val macro F1 |
| 8. Evaluation | 🔶 Partial | `src/evaluation/metrics.py`, `train.py` | VGG-16 x 4 conditions evaluated; `evaluate.py` empty |
| 9. Grad-CAM | ⏳ Pending | `src/explainability/*.py` | Inspect heatmaps |
| 10. Documentation | ⏳ Pending | `README.md`, `REPORT.md` | Final report |

**Progress: 5/10 stages complete, Stage 5 partially done (ResNet-18 trained)**

---

## Key Decisions Made

### Data & Splitting
1. **Split by lesion_id** - prevents data leakage from multiple images of same lesion
2. **Stratified splitting** - preserves 63:1 class imbalance in all splits
3. **Image size: 224×224** - standard for pretrained models
4. **Use focal loss** - handles class imbalance, focuses on hard examples

### Data Loading
5. **Lazy loading** - images loaded on-demand (memory efficient, 10K+ images)
6. **Relative paths** - `data/raw/...` not absolute (portable across machines)
7. **Different transforms** - augmentation for training, deterministic for val/test
8. **ImageNet normalization** - required for pretrained model compatibility
9. **Batch size 16** - balances GPU memory and training speed
10. **num_workers=4** - parallel loading (use 0 on Windows)

### Model Architecture & Training
10. **ResNet-18 first** - lightweight baseline (~11M params), fast training
11. **ImageNet pretrained** - transfer learning for faster convergence
12. **Replace only FC layer** - 512→7 outputs, freeze backbone optional
13. **Xavier initialization** - for new classifier layer (training stability)
14. **Batch size 32** - optimal for T4 GPU (vs 16 on smaller GPUs)
15. **Adam optimizer** - lr=1e-3 with ReduceLROnPlateau scheduler
16. **Early stopping** - patience=5 epochs (prevent overfitting)

### Training & Evaluation
17. **4 experimental conditions** - controlled ablation study (CE vs FL, Aug vs No Aug)
18. **Macro F1 for checkpoint** - better than accuracy for imbalanced data
19. **Grad-CAM for interpretability** - check for artifacts vs lesion focus

---

## Directory Structure (Updated)

```
skin-lesion-screening/
├── configs/                    # Model configs (YAML)
│   ├── resnet18.yaml
│   ├── vgg16.yaml
│   └── efficientnet_b0.yaml
├── data/
│   ├── raw/                    # Original dataset (not committed)
│   │   ├── HAM10000_images_part_1/
│   │   ├── HAM10000_images_part_2/
│   │   └── HAM10000_metadata_clean.csv
│   ├── splits/                 # Split manifests (generated)
│   │   ├── train.csv
│   │   ├── val.csv
│   │   ├── test.csv
│   │   └── split_config.json
│   └── validation_reports/     # Validation artifacts
├── docs/                       # Documentation
│   ├── 01_data_inspection.md   # ✅ Complete
│   ├── 02_data_validation.md   # ✅ Complete
│   ├── 03_data_splitting.md    # ✅ Complete
│   └── 04_data_loading.md      # ✅ Complete
├── notebooks/                  # Exploration notebooks
│   ├── 01_data_inspection.ipynb   # ✅ Complete
│   ├── 02_data_validation.ipynb   # ✅ Complete
│   └── 03_data_splitting.ipynb    # ✅ Complete
├── src/                        # Source code (build this!)
│   ├── data/                   # Stage 4: Data loading
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   ├── transforms.py
│   │   └── dataloader.py
│   ├── models/                 # Stage 5: Model architectures
│   │   ├── __init__.py
│   │   ├── resnet18.py
│   │   ├── vgg16.py
│   │   └── efficientnet_b0.py
│   ├── losses/                 # Stage 6: Loss functions
│   │   ├── __init__.py
│   │   └── focal_loss.py
│   ├── training/                 # Stage 7: Training loop
│   │   ├── __init__.py
│   │   └── trainer.py
│   ├── evaluation/               # Stage 8: Metrics
│   │   ├── __init__.py
│   │   └── metrics.py
│   └── explainability/           # Stage 9: Grad-CAM
│       ├── __init__.py
│       └── gradcam.py
├── checkpoints/                # Model weights (not committed)
├── outputs/                      # Results, plots, heatmaps
├── notebooks/                    # Analysis notebooks
├── train.py                      # Entry: training
├── evaluate.py                   # Entry: evaluation
├── predict.py                    # Entry: inference
├── pyproject.toml                # Dependencies
└── README.md                     # This file
```

---

## Local VGG16 training

One model (VGG-16) is trained so far. Measured test-set results (single seed; differences between conditions may be within noise; educational use only, not clinical evidence):

| Condition | Test macro F1 | Balanced acc. | Melanoma sensitivity | Melanoma specificity |
|---|---|---|---|---|
| Cross-entropy, no aug | 0.5763 | 0.5663 | 0.4821 | 0.9586 |
| Focal, no aug | 0.5746 | 0.5648 | 0.7500 | 0.9112 |
| Cross-entropy, aug | 0.6185 | 0.6237 | 0.5893 | 0.9533 |
| Focal, aug | 0.6233 | 0.6395 | 0.6071 | 0.9428 |

Grad-CAM, ResNet-18, and EfficientNet-B0 have not been run.

The VGG16 trainer supports the four loss/augmentation conditions, validation macro-F1 checkpoint selection, and one held-out test evaluation per run. Windows environment setup (Conda + uv with a CUDA-enabled PyTorch index), commands, artifacts, and measured run status are documented in [docs/vgg16_local_training.md](docs/vgg16_local_training.md). Only runs listed there as completed have results.

## References

- HAM10000: https://doi.org/10.1038/sdata.2018.161
- PyTorch: https://pytorch.org/docs/stable/
- Torchvision: https://pytorch.org/vision/stable/models.html
- Grad-CAM: https://arxiv.org/abs/1610.02391
- Focal Loss: https://arxiv.org/abs/1708.02002
