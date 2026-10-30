# Data Inspection Pipeline Documentation

## Overview

The data inspection pipeline is the **first stage** of the ML workflow for skin lesion classification. Its purpose is to understand the HAM10000 dataset before any modeling decisions are made.

---

## What is EDA?

**EDA = Exploratory Data Analysis**

EDA is the process of investigating datasets to:
- Discover patterns
- Spot anomalies
- Test hypotheses
- Check assumptions
- Guide modeling decisions

**Why it matters:** EDA prevents costly mistakes later by catching data issues early.

---

## Pipeline Stages

### Stage 1: Setup and Imports

**Purpose:** Configure environment and load data

**What happens:**
```python
# 1. Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 2. Define paths
PROJECT_ROOT = Path.cwd().parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
METADATA_PATH = DATA_DIR / "HAM10000_metadata.csv"
IMAGE_DIRS = [DATA_DIR / "HAM10000_images_part_1", DATA_DIR / "HAM10000_images_part_2"]

# 3. Load metadata
df = pd.read_csv(METADATA_PATH)
```

**Output:**
- Loaded dataframe with 10,015 rows and 7 columns
- Verified file paths exist

**Key columns in metadata:**
| Column | Description | Example |
|--------|-------------|---------|
| `lesion_id` | Unique lesion identifier | HAM_0000118 |
| `image_id` | Unique image filename | ISIC_0027419 |
| `dx` | Diagnosis code (7 classes) | bkl |
| `dx_type` | How diagnosis was made | histo |
| `age` | Patient age | 80.0 |
| `sex` | Patient sex | male |
| `localization` | Body location | scalp |

---

### Stage 2: Schema Validation

**Purpose:** Verify data structure and quality

**What happens:**
1. Check shape: (10015, 7)
2. Check dtypes: lesion_id (str), age (float), etc.
3. Count missing values
4. Check for duplicates

**Why it matters:**
- **Missing values** (57 in age) → need to handle in preprocessing
- **Duplicates** (0 found) → data is clean
- **Data types** → confirm age is numeric for analysis

**Decision point:** If many missing values or duplicates found, must fix before proceeding.

---

### Stage 3: Class Distribution Analysis

**Purpose:** Understand target variable imbalance

**What happens:**
```python
class_counts = df['dx'].value_counts()
imbalance_ratio = class_counts.max() / class_counts.min()
```

**Results:**
| Class | Count | Percentage |
|-------|-------|------------|
| nv | 6,705 | 66.9% |
| mel | 1,113 | 11.1% |
| bkl | 1,099 | 11.0% |
| bcc | 514 | 5.1% |
| akiec | 327 | 3.3% |
| vasc | 142 | 1.4% |
| df | 115 | 1.1% |

**Critical finding:** Imbalance ratio of **63:1**

**Impact on modeling:**
- Standard cross-entropy loss will be biased toward majority class
- Need **focal loss** to handle imbalance
- Need **stratified sampling** when splitting
- Metrics must include per-class performance, not just accuracy

**Visual output:** Bar charts showing class distribution

---

### Stage 4: Lesion ID Analysis (CRITICAL)

**Purpose:** Identify data leakage risk

**What is lesion_id?**
Each `lesion_id` represents a unique skin lesion. Some lesions have **multiple images** taken from different angles or at different times.

**Example:**
```
lesion_id: HAM_0000118
├── image_id: ISIC_0027419 (image 1)
└── image_id: ISIC_0025030 (image 2)
```

**The problem:**
If we split randomly by image, both images of the same lesion could end up in train AND test. This is **data leakage** - the model has "seen" the lesion before.

**What happens:**
```python
# Count images per lesion
images_per_lesion = df.groupby('lesion_id').size()
multi_image_lesions = images_per_lesion[images_per_lesion > 1]

# Check for lesions with multiple class labels
lesion_class_count = df.groupby('lesion_id')['dx'].nunique()
multi_class_lesions = lesion_class_count[lesion_class_count > 1]
```

**Results:**
- Total unique lesions: 7,470
- Lesions with multiple images: ~2,500
- Lesions with inconsistent labels: 0 (good!)

**Critical rule:**
> **ALWAYS split by lesion_id, not by image_id**

All images from the same lesion must stay together in the same split (train/val/test).

**Visual output:** Distribution of images per lesion

---

### Stage 5: Image File Verification

**Purpose:** Ensure all metadata images exist on disk

**What happens:**
```python
# Check both image folders
part1_images = set(f.stem for f in part1_dir.glob("*.jpg"))
part2_images = set(f.stem for f in part2_dir.glob("*.jpg"))
all_images = part1_images | part2_images

# Compare with metadata
metadata_images = set(df['image_id'])
missing_images = metadata_images - all_images
```

**Results:**
- Images in part_1: ~5,000
- Images in part_2: ~5,000
- Total expected: 10,015
- Missing: 0
- Extra on disk: 0

**Why it matters:**
If images are missing, training will fail when trying to load them.

---

### Stage 6: Visual Inspection

**Purpose:** See what the actual images look like

**What happens:**
```python
# Load and display one sample from each class
for class_code in ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']:
    sample = df[df['dx'] == class_code].iloc[0]
    img_path = get_image_path(sample['image_id'])
    img = Image.open(img_path)
    display(img)
```

**What to look for:**
- Image quality (resolution, clarity)
- Color variations
- Artifacts (hair, rulers, markings)
- Visual differences between classes

**Key observations:**
- Images are dermoscopic (skin microscopy)
- Various sizes (need resizing to 224x224)
- Some have hair artifacts
- Some have ruler markings

**Impact:**
- Hair/rulers could be **shortcut features** (model learns "has ruler = nv")
- Need careful preprocessing and augmentation

---

### Stage 7: Demographic Analysis

**Purpose:** Understand patient characteristics

**What happens:**
```python
# Age statistics
print(df['age'].describe())

# Age by class
age_by_class = df.groupby('dx')['age'].agg(['mean', 'median', 'std'])

# Sex distribution
print(df['sex'].value_counts())

# Body location
print(df['localization'].value_counts().head(10))
```

**Results:**
- Mean age: ~51 years
- Melanoma more common in older patients
- Slight male majority
- Most common locations: back, face, scalp

**Why it matters:**
- Age is a risk factor for melanoma
- Location matters clinically
- Sex distribution shows dataset bias

---

### Stage 8: Summary and Key Takeaways

**Purpose:** Document findings for modeling decisions

**Critical findings:**

| Finding | Impact |
|---------|--------|
| 63:1 class imbalance | Use focal loss, weighted sampling |
| Must split by lesion_id | Prevents data leakage |
| 57 missing ages | Handle in preprocessing |
| Multiple images per lesion | Group by lesion_id for splits |
| Hair/ruler artifacts | Check Grad-CAM for shortcuts |

**Decisions made:**
1. ✅ Use focal loss (not just cross-entropy)
2. ✅ Split by lesion_id (not by image)
3. ✅ Image size: 224x224
4. ✅ Report per-class metrics
5. ✅ Check Grad-CAM for artifact focus

---

## Pipeline Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│  STAGE 1: Setup                                             │
│  - Import libraries                                         │
│  - Define paths                                             │
│  - Load metadata                                            │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 2: Schema Validation                                 │
│  - Check shape: (10015, 7)                                  │
│  - Verify dtypes                                            │
│  - Count missing: 57 in age                                 │
│  - Check duplicates: 0                                      │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 3: Class Distribution                                │
│  - 7 classes identified                                     │
│  - Severe imbalance: 63:1 ratio                           │
│  - nv dominates (67%), df rare (1%)                        │
│  → Decision: Use focal loss                                 │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 4: Lesion ID Analysis                                │
│  - 7,470 unique lesions                                     │
│  - ~2,500 lesions have multiple images                      │
│  → Decision: Split by lesion_id (critical!)                 │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 5: Image Verification                                │
│  - All 10,015 images found                                  │
│  - No missing files                                         │
│  → Ready for loading                                        │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 6: Visual Inspection                                 │
│  - Dermoscopic images                                       │
│  - Hair/ruler artifacts observed                            │
│  → Decision: Check Grad-CAM for shortcuts                   │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 7: Demographics                                      │
│  - Mean age: 51                                             │
│  - Location patterns                                        │
│  → Context for interpretability                             │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 8: Summary                                           │
│  - Document all findings                                      │
│  - Guide next stages                                         │
└─────────────────────────────────────────────────────────────┘
```

---

## Outputs Generated

| Output | File | Purpose |
|--------|------|---------|
| Class distribution | `outputs/class_distribution.png` | Visual reference |
| Images per lesion | `outputs/images_per_lesion.png` | Splitting strategy |
| Sample images | `outputs/sample_images.png` | Visual understanding |
| Demographics | Visualizations | Context |
| Summary CSV | `data/class_distribution.csv` | Reference for splits |

---

## Next Steps

After completing data inspection, proceed to:

**Stage 2: Data Splitting**
- Create train/val/test splits by lesion_id
- Ensure stratified sampling
- Save split manifests

**See:** `README_SIMPLIFIED.md` for full pipeline

---

## Key Lessons Learned

1. **Always inspect before modeling** - Caught the lesion_id issue early
2. **Class imbalance is severe** - Would fail with naive approach
3. **Visual inspection matters** - Found artifacts that could confuse model
4. **Document everything** - Future you will thank current you
5. **Split strategy is critical** - One mistake here ruins everything

---

## Common EDA Mistakes

| Mistake | Why It's Bad | How We Avoided It |
|---------|--------------|-------------------|
| Not checking for leakage | Model cheats, false high accuracy | Analyzed lesion_id |
| Ignoring class imbalance | Model predicts majority class only | Measured 63:1 ratio |
| Skipping visual inspection | Miss artifacts/quality issues | Viewed sample images |
| Not documenting findings | Forget why decisions were made | Created this doc |
| Looking at test set during EDA | Data snooping, overfitting | Only looked at metadata |

---

## References

- Original notebook: `notebooks/01_data_inspection.ipynb`
- HAM10000 paper: https://doi.org/10.1038/sdata.2018.161
- EDA best practices: https://towardsdatascience.com/exploratory-data-analysis
