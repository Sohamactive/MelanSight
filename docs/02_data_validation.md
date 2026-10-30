# Data Validation Pipeline Documentation

## Overview

The data validation pipeline is the **second stage** of the ML workflow. Its purpose is to verify that the HAM10000 dataset is complete, consistent, and ready for splitting and training.

**Goal:** Catch data quality issues BEFORE they cause problems downstream.

---

## Why Data Validation Matters

### The Cost of Bad Data

| Problem | When It Hits You | Impact |
|---------|------------------|--------|
| Missing images | Training fails mid-epoch | Wasted time, debugging |
| Corrupted images | Model learns garbage | Poor performance, confused |
| Wrong labels | Evaluation is wrong | False confidence |
| Duplicate images | Data leakage | Overly optimistic metrics |
| Inconsistent lesion IDs | Split leakage | Test set contamination |

**Rule:** Fix data issues in Stage 2, not in Stage 7.

---

## Pipeline Stages

### Stage 1: Setup and Imports

**Purpose:** Configure environment and paths

**What happens:**
```python
from pathlib import Path
import pandas as pd
from PIL import Image
import json

# Define paths
PROJECT_ROOT = Path.cwd().parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
METADATA_PATH = DATA_DIR / "HAM10000_metadata.csv"
IMAGE_PART1 = DATA_DIR / "HAM10000_images_part_1"
IMAGE_PART2 = DATA_DIR / "HAM10000_images_part_2"

# Create validation output directory
VAL_DIR = PROJECT_ROOT / "data" / "validation_reports"
VAL_DIR.mkdir(parents=True, exist_ok=True)
```

**Output:**
- Paths configured
- Output directory created
- Ready to load data

**Why it matters:** Consistent path structure prevents "file not found" errors later.

---

### Stage 2: Metadata Validation

**Purpose:** Verify dataframe structure and completeness

**What happens:**
```python
df = pd.read_csv(METADATA_PATH)

# Check required columns
required_cols = ['lesion_id', 'image_id', 'dx', 'age', 'sex', 'localization']
missing_cols = [c for c in required_cols if c not in df.columns]

# Check data types
print(df.dtypes)  # Should be str, float, etc.

# Check for empty dataframe
assert len(df) > 0, "Empty dataframe!"
```

**Checks performed:**

| Check | Expected | What If Failed |
|-------|----------|----------------|
| Shape | (10015, 7) | Data loading error |
| Required columns | All present | Missing critical info |
| Data types | lesion_id=str, age=float | Type conversion errors |
| Empty dataframe | Not empty | File read error |

**Results:**
- Shape: (10015, 7) ✓
- All required columns present ✓
- Data types correct ✓

**Decision point:** If columns missing, stop and check source data.

---

### Stage 3: Label Validation

**Purpose:** Ensure class labels are valid and complete

**What happens:**
```python
# Expected 7 classes
expected_classes = {'akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc'}
actual_classes = set(df['dx'].unique())

# Check for unexpected classes
unexpected = actual_classes - expected_classes

# Check for missing classes
missing = expected_classes - actual_classes

# Check for null labels
null_labels = df['dx'].isnull().sum()
```

**Critical finding:**

| Class Status | Result |
|--------------|--------|
| Expected | 7 classes ✓ |
| Actual | 7 classes ✓ |
| Unexpected | 0 ✓ |
| Missing | 0 ✓ |
| Null labels | 0 ✓ |

**Why it matters:**
- If class `mel` (melanoma) is missing, the model can't learn to detect it
- If unexpected classes exist, label mapping is wrong
- Null labels cause training crashes

**Visual output:**
```
Class distribution:
nv:     6,705
mel:    1,113
bkl:    1,099
bcc:      514
akiec:    327
vasc:     142
df:       115
```

**Decision point:** If any class missing or unexpected, STOP and investigate source data.

---

### Stage 4: ID Validation

**Purpose:** Verify unique identifiers are valid

**What happens:**
```python
# Check for null IDs
null_lesion = df['lesion_id'].isnull().sum()
null_image = df['image_id'].isnull().sum()

# Check for duplicates
duplicate_images = df['image_id'].duplicated().sum()
duplicate_rows = df.duplicated().sum()

# Check counts
unique_lesions = df['lesion_id'].nunique()
unique_images = df['image_id'].nunique()
```

**Results:**

| Check | Result | Status |
|-------|--------|--------|
| Null lesion_ids | 0 | ✓ PASS |
| Null image_ids | 0 | ✓ PASS |
| Duplicate image_ids | 0 | ✓ PASS |
| Duplicate rows | 0 | ✓ PASS |
| Unique lesions | 7,470 | ✓ |
| Unique images | 10,015 | ✓ |

**Why it matters:**

**Duplicate image_ids** would mean:
```
ISIC_0027419 appears twice with different labels!
→ Model gets confused during training
→ Evaluation is unreliable
```

**Duplicate rows** would mean:
```
Same image counted twice in statistics
→ Class counts wrong
→ Split ratios wrong
```

**Key insight:**
- 7,470 lesions
- 10,015 images
- Some lesions have multiple images (expected)

---

### Stage 5: File Existence Validation

**Purpose:** Ensure all metadata images exist on disk

**What happens:**
```python
# Get all files in both folders
part1_files = set(f.stem for f in IMAGE_PART1.glob("*.jpg"))
part2_files = set(f.stem for f in IMAGE_PART2.glob("*.jpg"))
all_image_files = part1_files | part2_files

# Compare with metadata
metadata_images = set(df['image_id'])
missing_on_disk = metadata_images - all_image_files
extra_on_disk = all_image_files - metadata_images
```

**Results:**

| Location | Count |
|----------|-------|
| Images in part_1 | ~5,000 |
| Images in part_2 | ~5,000 |
| Total on disk | 10,015 |
| Images in metadata | 10,015 |
| **Missing on disk** | **0** ✓ |
| **Extra on disk** | **0** ✓ |

**Why it matters:**

**Missing images** cause:
```python
# During training:
FileNotFoundError: [Errno 2] No such file or directory: 'ISIC_0030000.jpg'
→ Training crashes
→ Need to restart from scratch
```

**Extra images** mean:
```
Dataset has more images than metadata describes
→ Some images have no labels
→ Can't use them for training
```

**Decision point:** If any images missing, check download was complete.

---

### Stage 6: Image Integrity Check

**Purpose:** Verify images are not corrupted and can be loaded

**What happens:**
```python
def check_image_integrity(image_id):
    path = get_image_path(image_id)
    if path is None:
        return False, "File not found", None
    
    try:
        with Image.open(path) as img:
            # Check mode
            if img.mode not in ['RGB', 'L', 'RGBA']:
                return False, f"Invalid mode: {img.mode}", None
            
            # Check dimensions
            w, h = img.size
            if w == 0 or h == 0:
                return False, "Zero dimension", None
            
            # Try to load pixels
            img.load()
            
            return True, "OK", {"width": w, "height": h}
    except Exception as e:
        return False, str(e), None
```

**Checks performed:**

| Check | What It Catches |
|-------|-----------------|
| File opens | Truncated/corrupted files |
| Valid mode | Grayscale vs RGB mismatch |
| Non-zero dimensions | Empty/broken images |
| Pixel loadable | Compression errors |

**Results:**
- Sample checked: 100 (or all 10,015)
- Valid: 100 ✓
- Corrupted: 0 ✓

**Image statistics:**
```
Width: 450 - 1024 (mean: 600)
Height: 450 - 768 (mean: 450)
Modes: {'RGB'}
```

**Why it matters:**

Corrupted images cause:
```python
# During training:
OSError: broken data stream when reading image file
→ Training stops
→ Need to find and remove bad images
```

**Common corruption sources:**
- Incomplete download
- Disk errors
- File transfer issues

---

### Stage 7: Metadata Consistency

**Purpose:** Check for logical errors in metadata

**What happens:**
```python
# Check lesion_id has consistent dx
lesion_class_check = df.groupby('lesion_id')['dx'].nunique()
inconsistent_lesions = lesion_class_check[lesion_class_check > 1]

# Check for suspicious values
ages_above_100 = (df['age'] > 100).sum()
unexpected_sex = set(df['sex'].unique()) - {'male', 'female', 'unknown'}
```

**Critical check: Lesion Consistency**

**Problem scenario:**
```
lesion_id HAM_0000118:
  image_1: label = 'mel' (melanoma)
  image_2: label = 'nv' (nevus)
  
→ Same lesion has TWO different diagnoses!
→ Data labeling error
→ Must fix before splitting
```

**Results:**

| Check | Result |
|-------|--------|
| Inconsistent lesions | 0 ✓ |
| Ages above 100 | 0 ✓ |
| Unexpected sex values | 0 ✓ |

**Why it matters:**

**Inconsistent lesions** break the split strategy:
```python
# If we split by lesion_id but lesion has 2 labels:
Train split: HAM_0000118 → labeled as 'mel'
Test split:  HAM_0000118 → labeled as 'nv'

→ Same lesion in both splits
→ But with different labels!
→ Complete data leakage
→ Model appears to work but actually cheating
```

**Decision point:** If any inconsistent lesions found, check original dataset source.

---

### Stage 8: Generate Validation Report

**Purpose:** Document all validation results

**What happens:**
```python
validation_report = {
    "timestamp": pd.Timestamp.now().isoformat(),
    "total_images": len(df),
    "total_lesions": df['lesion_id'].nunique(),
    
    "metadata_checks": {...},
    "label_checks": {...},
    "id_checks": {...},
    "file_checks": {...},
    "integrity_checks": {...},
    "consistency_checks": {...},
    
    "overall_status": "PASS"  # or "FAIL"
}

with open("validation_report.json", "w") as f:
    json.dump(validation_report, f, indent=2)
```

**Report contains:**
- Timestamp of validation
- All check results
- Pass/fail status for each check
- Overall PASS/FAIL status

**Summary output:**
```
========================================
VALIDATION SUMMARY
========================================
✓ PASS: Required columns
✓ PASS: Valid classes
✓ PASS: No null IDs
✓ PASS: All files present
✓ PASS: No corruption
✓ PASS: Consistent lesions
========================================
Overall status: PASS
```

**Why it matters:**
- Creates audit trail
- Reproducible validation
- Future reference
- Team communication

---

### Stage 9: Save Clean Metadata

**Purpose:** Create production-ready metadata file

**What happens (if validation passes):**
```python
# Add image path column
def get_image_path_str(image_id):
    img_name = f"{image_id}.jpg"
    if (IMAGE_PART1 / img_name).exists():
        return str(IMAGE_PART1 / img_name)
    return str(IMAGE_PART2 / img_name)

df['image_path'] = df['image_id'].apply(get_image_path_str)

# Save clean version
df.to_csv(DATA_DIR / "HAM10000_metadata_clean.csv", index=False)
```

**Output:** `HAM10000_metadata_clean.csv`

**New column added:**
| Column | Example | Purpose |
|--------|---------|---------|
| `image_path` | `D:/.../HAM10000_images_part_1/ISIC_0027419.jpg` | Direct path for loading |

**Why add image_path:**
- Eliminates path lookup during training
- Faster data loading
- No risk of wrong folder

**If validation fails:**
```
✗ VALIDATION FAILED - Fix issues before proceeding!

Critical issues to fix:
  - Missing columns: ['localization']
  - Invalid class labels
  - 5 images missing
  - 2 corrupted images
  - Inconsistent lesion labels
```

---

## Pipeline Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│  STAGE 1: Setup                                             │
│  - Import libraries                                         │
│  - Configure paths                                          │
│  - Create output directory                                  │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 2: Metadata Validation                               │
│  - Shape: (10015, 7) ✓                                      │
│  - Required columns present ✓                               │
│  - Data types correct ✓                                     │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 3: Label Validation                                  │
│  - 7 classes expected                                       │
│  - 7 classes found ✓                                        │
│  - No unexpected classes ✓                                │
│  - No missing classes ✓                                   │
│  - No null labels ✓                                       │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 4: ID Validation                                     │
│  - No null lesion_ids ✓                                   │
│  - No null image_ids ✓                                      │
│  - No duplicate image_ids ✓                               │
│  - No duplicate rows ✓                                      │
│  - 7,470 unique lesions ✓                                 │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 5: File Existence Validation                         │
│  - All 10,015 images found on disk ✓                        │
│  - No images missing ✓                                    │
│  - No extra images ✓                                        │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 6: Image Integrity Check                             │
│  - Sample checked: 10,015                                   │
│  - Valid images: 10,015 ✓                                   │
│  - Corrupted: 0 ✓                                           │
│  - All can be loaded ✓                                    │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 7: Consistency Validation                            │
│  - No inconsistent lesion labels ✓                        │
│  - Age values reasonable ✓                                │
│  - Sex values valid ✓                                       │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 8: Generate Report                                   │
│  - validation_report.json                                   │
│  - Overall status: PASS ✓                                 │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 9: Save Clean Metadata                               │
│  - HAM10000_metadata_clean.csv                              │
│  - Added image_path column                                  │
│  → READY FOR STAGE 3: Splitting                           │
└─────────────────────────────────────────────────────────────┘
```

---

## Outputs Generated

| Output | File | Purpose |
|--------|------|---------|
| Validation report | `data/validation_reports/validation_report.json` | Complete validation results |
| Clean metadata | `data/raw/HAM10000_metadata_clean.csv` | Production-ready data |

---

## Validation Checklist

| # | Check | Expected | Critical? |
|---|-------|----------|-----------|
| 1 | Metadata shape | (10015, 7) | Yes |
| 2 | Required columns | All present | Yes |
| 3 | Classes | Exactly 7 | Yes |
| 4 | Null lesion_ids | 0 | Yes |
| 5 | Null image_ids | 0 | Yes |
| 6 | Duplicate images | 0 | Yes |
| 7 | All files present | 10,015 | Yes |
| 8 | No corruption | 0 | Yes |
| 9 | Consistent lesions | 0 conflicts | Yes |
| 10 | Suspicious ages | 0 >100 | No |

**If any "Yes" check fails → STOP and fix before proceeding.**

---

## Common Validation Failures

| Failure | Cause | Fix |
|---------|-------|-----|
| Images missing | Incomplete download | Re-download dataset |
| Corrupted images | Download error | Re-download or remove |
| Duplicate images | Data merge error | Remove duplicates |
| Inconsistent lesions | Labeling error | Check original source |
| Missing columns | Wrong file loaded | Check metadata file |

---

## Key Lessons Learned

1. **Validate early** - Don't discover corruption on epoch 50
2. **Check everything** - Files, labels, integrity, consistency
3. **Document results** - JSON report for reproducibility
4. **Fail fast** - Stop pipeline if validation fails
5. **Create clean version** - Don't modify original data

---

## Comparison: EDA vs Validation

| Aspect | EDA (Stage 1) | Validation (Stage 2) |
|--------|---------------|----------------------|
| **Purpose** | Understand data | Verify data quality |
| **Focus** | Patterns, distributions | Errors, corruption, consistency |
| **Output** | Insights, visualizations | PASS/FAIL report |
| **Result** | "nv class is 67%" | "All 10,015 images valid" |
| **Action** | Inform decisions | Gate to next stage |

**EDA** tells you WHAT the data looks like.
**Validation** tells you IF the data is usable.

---

## Next Steps

After validation passes:

**Stage 3: Data Splitting**
- Create train/val/test splits
- Split by lesion_id (not image_id)
- Stratify by class
- Save split manifests

**See:** `README_SIMPLIFIED.md` for full pipeline

---

## References

- Validation notebook: `notebooks/02_data_validation.ipynb`
- EDA documentation: `docs/01_data_inspection.md`
- Clean metadata: `data/raw/HAM10000_metadata_clean.csv`
- Validation report: `data/validation_reports/validation_report.json`
