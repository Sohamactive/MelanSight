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

What we learned:
- 10,015 images across 7 classes
- Severe imbalance: nv (67%) vs df (1%)
- **Must split by lesion_id** (not by image)
- 57 missing age values
- All images verified on disk

**Notebook:** `notebooks/01_data_inspection.ipynb`

---

### ⏳ Stage 2: Data Preprocessing & Splitting
**Status: PENDING**

**Goal:** Create train/val/test splits without data leakage

**Key Rule:** Split by `lesion_id`, not by individual images

**Tasks:**
- [ ] Create split by lesion_id (70/15/15 or 80/10/10)
- [ ] Save split manifests to `data/splits/`
- [ ] Document random seed used
- [ ] Verify no lesion appears in multiple splits

**Files to create:**
- `src/data/split.py` - split generation
- `data/splits/train.csv`, `val.csv`, `test.csv`

---

### ⏳ Stage 3: Data Loading & Transforms
**Status: PENDING**

**Goal:** Build PyTorch Dataset and DataLoader

**Tasks:**
- [ ] Create `src/data/dataset.py` - custom Dataset class
- [ ] Create `src/data/transforms.py` - train/val transforms
- [ ] Training transforms: random flip, rotation, color jitter
- [ ] Validation transforms: resize, normalize only
- [ ] Test `DataLoader` with batch

**Important:**
- Training: augmentation ON
- Validation/Test: augmentation OFF (deterministic)
- Image size: 224x224 (standard for pretrained models)

---

### ⏳ Stage 4: Model Implementation
**Status: PENDING**

**Goal:** Build 3 models with 7-class output heads

**Tasks:**
- [ ] `src/models/resnet18.py` - ResNet-18 + 7 outputs
- [ ] `src/models/vgg16.py` - VGG-16 + 7 outputs  
- [ ] `src/models/efficientnet_b0.py` - EfficientNet-B0 + 7 outputs
- [ ] Load ImageNet pretrained weights
- [ ] Test forward pass: input (B,3,224,224) → output (B,7)

**All models must have same interface:**
```python
model = create_model('resnet18', num_classes=7, pretrained=True)
output = model(images)  # shape: [batch_size, 7]
```

---

### ⏳ Stage 5: Loss Functions
**Status: PENDING**

**Goal:** Implement cross-entropy and focal loss

**Tasks:**
- [ ] `src/losses/focal_loss.py` - focal loss implementation
- [ ] Support 4 experimental conditions:
  1. Cross-entropy, no augmentation
  2. Focal loss, no augmentation
  3. Cross-entropy, with augmentation
  4. Focal loss, with augmentation

**Why focal loss:** Handles class imbalance by down-weighting easy examples

---

### ⏳ Stage 6: Training Loop
**Status: PENDING**

**Goal:** Train models with checkpoint saving

**Tasks:**
- [ ] `src/training/trainer.py` - shared train/val loop
- [ ] `train.py` - entry point script
- [ ] Config file (YAML) for hyperparameters
- [ ] Save best checkpoint based on validation loss
- [ ] Log metrics: loss, accuracy, F1

**Hyperparameters to tune:**
- Learning rate: 1e-4 to 1e-3
- Batch size: 16 or 32
- Epochs: 20-50 (with early stopping)
- Optimizer: Adam

---

### ⏳ Stage 7: Evaluation
**Status: PENDING**

**Goal:** Evaluate all models on test set

**Tasks:**
- [ ] `src/evaluation/metrics.py` - metric calculations
- [ ] `evaluate.py` - evaluation script
- [ ] Report required metrics:
  - Macro F1
  - Balanced accuracy
  - Per-class precision/recall
  - Melanoma sensitivity/specificity
  - Confusion matrix
- [ ] Compare 3 architectures
- [ ] Compare 4 experimental conditions

---

### ⏳ Stage 8: Grad-CAM Visualization
**Status: PENDING**

**Goal:** Generate heatmaps showing what models focus on

**Tasks:**
- [ ] `src/explainability/gradcam.py` - Grad-CAM implementation
- [ ] Generate heatmaps for:
  - Correct predictions (each class)
  - Incorrect predictions
  - Melanoma cases (important!)
- [ ] Check for shortcuts (hair, rulers, borders)
- [ ] Save visualizations to `outputs/gradcam/`

**Different target layers per architecture!**

---

### ⏳ Stage 9: Documentation & Report
**Status: PENDING**

**Goal:** Document results and limitations

**Tasks:**
- [ ] Update README with results
- [ ] Document guardrails and limitations
- [ ] Include educational-use disclaimer
- [ ] Report failure cases
- [ ] Do NOT claim clinical validation

---

## Quick Commands

```bash
# Run data inspection
uv run jupyter notebook notebooks/01_data_inspection.ipynb

# Train a model
uv run python train.py --model resnet18 --config configs/resnet18.yaml

# Evaluate
uv run python evaluate.py --checkpoint checkpoints/resnet18_best.pth

# Run tests
uv run pytest tests/
```

---

## Current Progress Tracker

| Stage | Status | Notes |
|-------|--------|-------|
| 1. EDA | ✅ Done | Class imbalance: 63:1, must split by lesion_id |
| 2. Splitting | ⏳ Pending | Next priority |
| 3. Data Loading | ⏳ Pending | Depends on splitting |
| 4. Models | ⏳ Pending | Start with ResNet-18 |
| 5. Loss Functions | ⏳ Pending | Focal loss for imbalance |
| 6. Training | ⏳ Pending | Save checkpoints |
| 7. Evaluation | ⏳ Pending | All 4 conditions |
| 8. Grad-CAM | ⏳ Pending | Inspect heatmaps |
| 9. Documentation | ⏳ Pending | Final report |

---

## Key Decisions Made

1. **Split by lesion_id** - prevents data leakage from multiple images of same lesion
2. **Image size: 224x224** - standard for pretrained models
3. **Use focal loss** - handles 63:1 class imbalance
4. **4 experimental conditions** - controlled ablation study
5. **ImageNet pretrained** - faster convergence, better results

---

## References

- HAM10000: https://doi.org/10.1038/sdata.2018.161
- PyTorch: https://pytorch.org/docs/stable/
- Torchvision: https://pytorch.org/vision/stable/models.html
