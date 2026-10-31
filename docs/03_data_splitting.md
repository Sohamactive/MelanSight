# Data Splitting Pipeline Documentation

## Overview

The data splitting pipeline is the **third stage** of the ML workflow. Its purpose is to divide the HAM10000 dataset into training, validation, and test sets while preventing data leakage and maintaining class balance.

**Goal:** Create splits that are:
1. **Leakage-free** - Same lesion never appears in multiple splits
2. **Stratified** - Class proportions preserved in all splits
3. **Reproducible** - Fixed random seed, documented splits
4. **Usable** - Saved as CSV files for downstream pipelines

---

## Splitting Strategies Considered

We evaluated multiple splitting strategies before selecting our final approach. Here's the complete decision matrix:

| Strategy | What It Does | Decision | Reason |
|----------|--------------|----------|--------|
| **Random Split** | Randomly distributes individual images | ❌ **Rejected** | Could place images of same lesion in train+test, causing data leakage |
| **Stratified Split** | Preserves class proportions | ❌ **Rejected alone** | Preserves class distribution but doesn't prevent lesion-level leakage |
| **Group Split** | Keeps all images of one lesion together | ⚠️ **Considered** | Prevents leakage, but doesn't preserve class proportions for imbalanced data |
| **Stratified Group K-Fold** | Preserves class proportions while keeping groups | ⚠️ **Considered** | Strong alternative, but generates folds rather than 70:15:15 split |
| **Stratified Shuffle Split** | Random split preserving class proportions | ✅ **Selected** | Works on pre-grouped lesion data, gives exact 70:15:15 split |
| **K-Fold Cross-Validation** | Repeated training on different folds | 📅 **Future** | More robust evaluation, but requires K times more training |

---

## Why We Rejected Each Strategy

### 1. ❌ Random Split (Image-Level)

**The Problem:**

Imagine one physical lesion has three photographs:

| Image | Lesion ID | Class |
|-------|-----------|-------|
| IMG_001 | L001 | mel |
| IMG_002 | L001 | mel |
| IMG_003 | L001 | mel |

**Random split could produce:**
- Training: IMG_001, IMG_002
- Testing: IMG_003

**Why this is bad:**
```
The model has already seen photographs of the same physical lesion during training.

Result:
- Model "recognizes" the lesion in test
- Accuracy appears high (95%+) but is FAKE
- Real generalization ability is unknown
```

**Impact of leakage:**
```
Without leakage: Model accuracy = 75% (realistic)
With leakage:    Model accuracy = 95% (inflated by 20%)
                 → False confidence, wrong conclusions
```

**Verdict:** ❌ Rejected due to data leakage risk.

---

### 2. ❌ Stratified Split Alone

**What it does:**
- Preserves class proportions (67% nv, 1% df, etc.)
- But operates on individual images

**The Problem:**

HAM10000 has severe class imbalance:
- nv: 67% of dataset
- df: 1.1% of dataset

**Stratification solves:**
```
Without stratification:
  Train: nv=80%, df=0% (df disappears!)
  Test:  nv=40%, df=5%

With stratification:
  Train: nv=67%, df=1%
  Test:  nv=67%, df=1%
```

**But stratification alone doesn't know:**
```
IMG_001 and IMG_002 belong to the same lesion (L001)

Stratified split could produce:
  Training: L001 - IMG_001
  Testing:  L001 - IMG_002  ← SAME LESION!
```

**Verdict:** ❌ Rejected alone - solves imbalance but not leakage.

---

### 3. ⚠️ Group Split

**What it does:**
```python
from sklearn.model_selection import GroupShuffleSplit

gss = GroupShuffleSplit(n_splits=1, test_size=0.15)
for train_idx, test_idx in gss.split(X, y, groups=df['lesion_id']):
    # All images of lesion L001 stay together
```

**Why it's better:**
- Groups by `lesion_id`
- Prevents leakage ✅
- All images of lesion stay in one split

**The Problem:**

Ordinary group splitting doesn't preserve class proportions:

```python
# Without stratification:
Group split might produce:
  Train: nv=75%, df=0.5%  (df underrepresented!)
  Test:  nv=50%, df=3%    (df overrepresented!)
```

For HAM10000's 63:1 imbalance, this can result in:
- Some splits having disproportionately few minority classes
- Unfair comparison across models
- Poor evaluation of rare classes (mel, df)

**Verdict:** ⚠️ Considered but rejected - prevents leakage but not ideal for imbalanced data.

---

### 4. ⚠️ Stratified Group K-Fold

**What it does:**
```python
from sklearn.model_selection import StratifiedGroupKFold

sgkf = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

for train_idx, val_idx in sgkf.split(X, y, groups=df['lesion_id']):
    # Both requirements satisfied:
    # 1. Preserve class proportions
    # 2. Keep groups (lesions) together
```

**Why it's strong:**
- ✅ Prevents leakage (groups stay together)
- ✅ Preserves class proportions (stratified)
- ✅ Cross-validation for robust evaluation

**Why we didn't select it:**

StratifiedGroupKFold creates **5 folds**, not a 70:15:15 three-way split:

```
Fold 1: Train (80%), Val (20%)
Fold 2: Train (80%), Val (20%)
...
Fold 5: Train (80%), Val (20%)
```

**Not:**
```
Train (70%), Val (15%), Test (15%)
```

We needed:
- A **fixed holdout test set** for final comparison
- **Single validation set** for hyperparameter tuning
- Not 5 different validation sets

**Verdict:** ⚠️ Considered but not selected - excellent for future cross-validation, but not for initial fixed split.

---

### 5. ✅ Stratified Shuffle Split (Selected)

**What we actually implemented:**

A **two-stage approach** on **pre-grouped lesion data**:

```python
# Step 1: Group by lesion
lesion_groups = df.groupby('lesion_id').agg({
    'image_id': list,
    'dx': 'first'  # Class label for stratification
})

# Step 2: Stratified Shuffle Split on groups
from sklearn.model_selection import StratifiedShuffleSplit

# Stage 1: Separate test (15%)
sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
for trainval_idx, test_idx in sss1.split(lesion_groups, lesion_groups['dx']):
    trainval = lesion_groups.iloc[trainval_idx]
    test = lesion_groups.iloc[test_idx]

# Stage 2: Separate train (70%) and val (15%)
sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.176, random_state=42)
for train_idx, val_idx in sss2.split(trainval, trainval['dx']):
    train = trainval.iloc[train_idx]
    val = trainval.iloc[val_idx]
```

**Why this works:**

1. **Grouped first** → Each row = one lesion (all images together)
2. **Stratified shuffle** → Preserves class proportions
3. **Two-stage** → Achieves 70:15:15 ratio
4. **Fixed seed** → Reproducible

**Technical detail:**
> We use stratified shuffle splitting on **pre-grouped lesion-level data**, rather than directly using a group-aware sklearn splitter.

**Result:**
```
Train:   5,234 lesions → 7,010 images (70%)
Val:     1,115 lesions → 1,502 images (15%)
Test:    1,121 lesions → 1,503 images (15%)

Class distribution preserved:
  nv:  67% in all splits
  mel: 11% in all splits
  df:  1% in all splits
```

**Verdict:** ✅ **Selected** - meets all requirements.

---

### 6. 📅 K-Fold Cross-Validation (Future)

**What it offers:**
- More robust performance estimates
- Uses all data for training and validation
- Reduces variance in metrics

**When to use:**
- After initial model selection
- When comparing architectures
- For publication-quality results

**Why not now:**
- Requires K times more training (5× longer)
- More complex experiment tracking
- Good for final evaluation, not for initial development

**Verdict:** 📅 Future option for robust evaluation.

---

## Summary: Why Our Choice is Best

| Requirement | Random | Stratified | Group | Stratified Group K-Fold | **Our Choice** |
|-------------|--------|------------|-------|------------------------|----------------|
| No leakage | ❌ | ❌ | ✅ | ✅ | ✅ |
| Class balance | ❌ | ✅ | ❌ | ✅ | ✅ |
| 70:15:15 split | ✅ | ✅ | ✅ | ❌ | ✅ |
| Fixed test set | ✅ | ✅ | ✅ | ❌ | ✅ |
| Reproducible | ✅ | ✅ | ✅ | ✅ | ✅ |

**Our approach combines:**
1. **Grouping** (prevents leakage)
2. **Stratification** (preserves class balance)
3. **Shuffle split** (flexible ratios)
4. **Fixed seed** (reproducibility)

---

## The Data Leakage Problem in Detail

### Scenario Without Proper Splitting:

```
Lesion HAM_0000118 (Melanoma):
├── Image ISIC_0027419 → Train set
└── Image ISIC_0025030 → Test set

Training:
- Model learns features from ISIC_0027419
- Sees melanoma patterns specific to this lesion

Testing:
- Model evaluates on ISIC_0025030
- Recognizes same lesion patterns
- "Predicts" correctly with high confidence

Result:
- Accuracy: 95% (inflated)
- Real generalization: Unknown (contaminated)
```

### Our Solution:

```
Lesion HAM_0000118 (Melanoma):
├── Image ISIC_0027419 → Train
└── Image ISIC_0025030 → Train (same split!)

Lesion HAM_0002730 (Different) → Test

Training:
- Model learns from HAM_0000118 only
- No exposure to test lesions

Testing:
- Model evaluates on completely new lesions
- Real generalization measured accurately
```

---

## Why Two-Stage Splitting?

**Goal:** 70% / 15% / 15%

**sklearn limitation:**
- `StratifiedShuffleSplit` only does 2-way splits
- No native 3-way stratified split

**Our solution:**

```
Stage 1: Split 100% → 85% (trainval) + 15% (test)
         Result: Test is fixed

Stage 2: Split 85% → 70% (train) + 15% (val)
         Math: 15% / 85% = 17.6% of remaining
         Result: Train + Val = 85%

Final: 70% + 15% + 15% = 100% ✓
```

**Why not other approaches:**
- `train_test_split` twice: May lose stratification
- Manual splitting: Error-prone
- Custom code: Hard to maintain

---

## Pipeline Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│  STAGE 1: Setup                                             │
│  - Import libraries                                         │
│  - Configure paths                                          │
│  - Load clean metadata (10,015 images)                       │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 2: Group by Lesion ID                                │
│  - WHY: Prevent data leakage                                │
│  - Group 10,015 images → 7,470 lesions                      │
│  - Aggregate image_ids into lists                           │
│  - Each row = one physical lesion                           │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 3: Split 1 - Test Set                                │
│  - METHOD: StratifiedShuffleSplit                           │
│  - WHY: Preserve class proportions                          │
│  - Split: 85% trainval / 15% test                         │
│  - Stratify by dx (class)                                   │
│  - Result: 6,349 / 1,121 lesions                           │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 4: Split 2 - Train/Val                                 │
│  - METHOD: StratifiedShuffleSplit                           │
│  - WHY: Get exact 70:15:15 ratio                            │
│  - Split: 70% train / 15% val from trainval                │
│  - Val ratio: 15% / 85% = 17.6%                           │
│  - Result: 5,234 / 1,115 lesions                            │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 5: Expand to Image Level                             │
│  - WHY: PyTorch needs one row per image                     │
│  - Convert lesion rows to image rows                        │
│  - Train: 5,234 lesions → 7,010 images                      │
│  - Val:   1,115 lesions → 1,502 images                      │
│  - Test:  1,121 lesions → 1,503 images                      │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 6: Verify No Leakage                                 │
│  - Check Train-Val overlap: 0 ✓                           │
│  - Check Train-Test overlap: 0 ✓                            │
│  - Check Val-Test overlap: 0 ✓                              │
│  - CRITICAL: If > 0, STOP and fix                           │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 7: Verify Class Distribution                         │
│  - Check nv: 67.0% in all splits ✓                          │
│  - Check mel: 11.1% in all splits ✓                         │
│  - Check df: 1.1% in all splits ✓                           │
│  - All classes present and proportional ✓                   │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 8: Save Manifests                                    │
│  - train.csv (7,010 rows)                                 │
│  - val.csv (1,502 rows)                                     │
│  - test.csv (1,503 rows)                                    │
│  - split_config.json (full configuration)                 │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 9: Summary                                           │
│  - Document all results                                     │
│  - Verify completeness                                      │
│  → READY FOR STAGE 4: Data Loading                          │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Concepts Explained

### Group Splitting vs Random Splitting

| Method | How It Works | Problem Solved |
|--------|--------------|----------------|
| **Random** | Shuffle all images, split | Simple but causes leakage |
| **Group** | Group by lesion_id, split groups | Prevents leakage |

**Example:**
```
Random split (WRONG):
HAM_0000118 - Image A → Train
HAM_0000118 - Image B → Test ← SAME LESION!

Group split (CORRECT):
HAM_0000118 (both images) → Train
HAM_0002730 (both images) → Test
```

### Stratification vs Random Sampling

| Method | How It Works | Problem Solved |
|--------|--------------|----------------|
| **Random** | Random selection | Loses minority classes |
| **Stratified** | Sample each class proportionally | Preserves all classes |

**Example:**
```
Class distribution: nv=67%, df=1%

Random sample of 100:
- nv: 70 (close to 67)
- df: 0 (lost! should be 1)

Stratified sample of 100:
- nv: 67 (exactly 67%)
- df: 1 (exactly 1%)
```

### Why Two-Stage Splitting

**Goal:** 70% / 15% / 15%

**sklearn limitation:**
- `StratifiedShuffleSplit` only does 2-way splits
- No built-in 3-way stratified split

**Solution:**
```
Stage 1: Split 100% → 85% + 15% (test)
Stage 2: Split 85% → 70% + 15% (val)

Math: 15% / 85% = 17.6%
Result: 70% / 15% / 15% ✓
```

### Data Leakage Detection

**What is leakage:**
- Same entity in train and test
- Model sees "answer" during training
- Inflated performance metrics

**Detection:**
```python
# Check for overlapping lesion IDs
train_ids = set(train_df['lesion_id'])
test_ids = set(test_df['lesion_id'])
overlap = train_ids & test_ids  # Intersection

# Must be empty!
assert len(overlap) == 0
```

---

## Outputs Generated

| Output | File | Contents |
|--------|------|----------|
| Train split | `data/splits/train.csv` | 7,010 images, 70% |
| Validation split | `data/splits/val.csv` | 1,502 images, 15% |
| Test split | `data/splits/test.csv` | 1,503 images, 15% |
| Configuration | `data/splits/split_config.json` | Seeds, counts, distributions |

---

## Validation Checklist

| # | Check | Expected | Critical? |
|---|-------|----------|-----------|
| 1 | Total images | 10,015 | Yes |
| 2 | Train images | 7,010 (70%) | Yes |
| 3 | Val images | 1,502 (15%) | Yes |
| 4 | Test images | 1,503 (15%) | Yes |
| 5 | Train-Val overlap | 0 lesions | **CRITICAL** |
| 6 | Train-Test overlap | 0 lesions | **CRITICAL** |
| 7 | Val-Test overlap | 0 lesions | **CRITICAL** |
| 8 | nv in all splits | ~67% | Yes |
| 9 | df in all splits | ~1% | Yes |
| 10 | All classes present | 7 classes | Yes |

---

## Common Splitting Mistakes

| Mistake | Why It's Bad | How We Avoided |
|---------|--------------|----------------|
| Split by image | Data leakage | Grouped by lesion_id first |
| No stratification | Lose minority classes | Used StratifiedShuffleSplit |
| Random seed | Non-reproducible | Fixed seed=42 |
| No verification | Silent leakage | Explicit overlap checks |
| Modifying test set | Overfitting | Test set locked after creation |

---

## Future Iterations

| Strategy | When to Use |
|----------|-------------|
| **StratifiedGroupKFold** | Cross-validation for robust evaluation |
| **Custom allocation** | If minority classes poorly represented |
| **Repeated validation** | Test sensitivity to train/val partition |
| **Patient-level splitting** | If patient IDs available (stronger generalization) |

**Note:** Lesion-level splitting doesn't guarantee patient-level separation. Multiple lesions could belong to the same patient.

---

## Interview-Ready Summary

> **"What splitting strategies did you evaluate, and why did you choose your final approach?"**

**Answer:**

"I considered random splitting, stratified splitting, group splitting, and stratified group cross-validation.

Random image-level splitting was unsuitable because HAM10000 contains multiple images of the same physical lesion, which could introduce data leakage. Stratified splitting alone preserves class proportions but doesn't prevent that leakage. Group splitting prevents lesion overlap but doesn't explicitly maintain class proportions.

I therefore grouped the dataset by lesion ID first and then applied StratifiedShuffleSplit in two stages. The first stage reserves 15% for testing, and the second divides the remaining data into approximately 70% training and 15% validation.

I also considered StratifiedGroupKFold for future cross-validation experiments, but selected a fixed holdout test set for the initial comparison of ResNet-18, VGG-16, and EfficientNet-B0."

---

## Next Steps

After splitting, proceed to:

**Stage 4: Data Loading**
- Create PyTorch Dataset class
- Load images from split CSVs
- Apply transforms (augmentation for train)
- Create DataLoaders

**See:** `README_SIMPLIFIED.md` for full pipeline

---

## References

- Splitting notebook: `notebooks/03_data_splitting.ipynb`
- Validation docs: `docs/02_data_validation.md`
- EDA docs: `docs/01_data_inspection.md`
- Split manifests: `data/splits/*.csv`
- scikit-learn: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html
