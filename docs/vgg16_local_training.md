# Local VGG16 training (Windows + NVIDIA GPU)

Educational research prototype; not a medical diagnostic tool. Softmax scores are not calibrated probabilities, and no output of this pipeline should be read as clinical validation.

## Environment setup (Conda + uv)

Conda provides the isolated Python 3.12 interpreter; `uv` installs the locked project dependencies into that same environment, so `pyproject.toml` and `uv.lock` remain the only dependency source (no `requirements.txt`).

```powershell
conda create -n melansight-vgg16 python=3.12 -y
conda activate melansight-vgg16
python -m pip install uv                 # uv only; everything else comes from the lock
$env:UV_PROJECT_ENVIRONMENT = $env:CONDA_PREFIX
uv sync                                  # installs the locked deps (incl. dev tools) into the Conda env
python scripts/verify_environment.py
```

`UV_PROJECT_ENVIRONMENT` makes `uv sync` / `uv run` target the active Conda prefix instead of creating `.venv`. Without it, uv would create a competing `.venv`.

### Why `pyproject.toml` has a PyTorch index entry

PyPI's Windows `torch` wheel is CPU-only (about 127 MB). To get CUDA builds on Windows, `pyproject.toml` pins `torch` and `torchvision` to `https://download.pytorch.org/whl/cu132` on `sys_platform == 'win32'` only. Versions are unchanged (`torch 2.15.0`, `torchvision 0.29.2`); Linux/macOS (e.g. Colab/Kaggle) still resolve from PyPI. The prebuilt CUDA wheel bundles its CUDA runtime, so no local CUDA Toolkit is needed; it only needs a recent NVIDIA driver (this machine: driver 616.92, CUDA 13.4 capable).

## Verified runtime (this machine)

| Item | Value |
|---|---|
| OS | Windows 11 |
| Python | 3.12.16 (Conda env `melansight-vgg16`) |
| PyTorch / Torchvision | 2.15.0+cu132 / 0.29.2+cu132 |
| `torch.cuda.is_available()` | True, CUDA runtime 13.2 |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU, 8.00 GiB |
| uv | 0.13.6 |

## Data

Images come from the Kaggle HAM10000 mirror (`kmader/skin-cancer-mnist-ham10000`, via `kagglehub`) and must sit under `data/raw/` (gitignored):

```
data/raw/HAM10000_metadata.csv
data/raw/HAM10000_images_part_1/
data/raw/HAM10000_images_part_2/
```

`python scripts/audit_data.py` (run with `PYTHONPATH=.`) checks labels, class order, split counts vs `split_config.json`, lesion/image overlap, missing or unreadable images, label agreement with the metadata, and train-vs-eval transform determinism. It is read-only and never regenerates splits.

**Documentation discrepancy (not changed):** `README.md` Stage 3 lists 7,010 / 1,502 / 1,503 images (train/val/test), while the committed manifests and `data/splits/split_config.json` contain 7,054 / 1,464 / 1,497. The manifests were used as-is.

## Run a condition

From the repository root, with the environment active. Windows defaults to `--workers 0`.

```powershell
python train.py --loss cross_entropy --no-augmentation --batch-size 16 --epochs 20 --patience 5 --evaluate-test
python train.py --loss focal         --no-augmentation --batch-size 16 --epochs 20 --patience 5 --evaluate-test
python train.py --loss cross_entropy --augmentation    --batch-size 16 --epochs 20 --patience 5 --evaluate-test
python train.py --loss focal         --augmentation    --batch-size 16 --epochs 20 --patience 5 --evaluate-test
```

- Defaults: Adam, lr 1e-4, image size 224, seed 42, ImageNet `VGG16_Weights.IMAGENET1K_V1`, focal gamma 2.0 (no alpha weights), CUDA AMP (fp16) on GPU.
- The checkpoint is selected by **validation macro F1**; early stopping uses the same value.
- `--evaluate-test` evaluates the selected checkpoint on the held-out test split. Omit it for pilots; use it once per finalized protocol. Without it, the confusion matrix and classification report are from validation.
- `--smoke-samples N` runs a tiny subset for plumbing checks; its numbers are not results.
- CUDA is required by default; there is no CPU fallback. On a CUDA out-of-memory error the run restarts from scratch at half the batch size (up to `--max-oom-restarts`, default 2); the original sizes are recorded in `config.json`.

### Artifacts (gitignored)

```
checkpoints/vgg16/<experiment>/best.pt
outputs/vgg16/<experiment>/{config.json, history.csv, metrics.json, classification_report.json, confusion_matrix.png}
```

`<experiment>` defaults to `<loss>_aug-<true|false>`. Metric definitions: macro F1 is the unweighted mean of the seven per-class F1 scores; balanced accuracy is mean per-class recall; melanoma sensitivity is `mel` recall and melanoma specificity is one-vs-rest TN/(TN+FP) with `mel` as positive.

## Tests

```powershell
python -m pytest tests -q
python scripts/verify_environment.py
```

## Measured status

See the section below, which is updated only from completed runs.

All four conditions were trained on the RTX 4060 Laptop GPU (CUDA, fp16 AMP); no CPU fallback or out-of-memory restart occurred. Protocol for every condition: batch size 16, Adam lr 1e-4, up to 20 epochs, patience 5, seed 42, `--workers 0`.

| Condition | Status | Best val macro F1 (epoch) | Test macro F1 | Test balanced acc. | Mel sensitivity | Mel specificity |
|---|---|---|---|---|---|---|
| 1. Cross-entropy, no aug | Completed (early stop, epoch 12) | 0.6275 (7) | 0.5763 | 0.5663 | 0.4821 | 0.9586 |
| 2. Focal, no aug | Completed (early stop, epoch 18) | 0.6324 (13) | 0.5746 | 0.5648 | 0.7500 | 0.9112 |
| 3. Cross-entropy, aug | Completed (early stop, epoch 19) | 0.6744 (14) | 0.6185 | 0.6237 | 0.5893 | 0.9533 |
| 4. Focal, aug | Completed (early stop, epoch 19) | 0.6414 (14) | 0.6233 | 0.6395 | 0.6071 | 0.9428 |

Test cross-entropy and focal losses are on different scales and must not be compared with each other. Each cell is one seed (42) and one run. With melanoma n=168 in test, one melanoma is about 0.6 percentage points of sensitivity, and differences of a few points between conditions are not shown to exceed run-to-run noise. Observed pattern (not a causal claim): the two augmentation-on runs scored higher on test macro F1 and balanced accuracy than the two augmentation-off runs; focal loss raised melanoma sensitivity without augmentation (0.48 to 0.75) at lower specificity, but not clearly with augmentation.

Condition 1 details (other runs' full metrics are in their `outputs/vgg16/<experiment>/metrics.json`): 134,289,223 parameters; checkpoint 537,170,051 bytes; typical epoch about 150 s. The logged `training_seconds` (6,701 s) includes one epoch (11) that took 5,075 s because of a machine stall (probable sleep or throttling), so it overstates compute time. Test per-class results (precision/recall, support): akiec 0.435/0.588 (51), bcc 0.792/0.494 (77), bkl 0.505/0.586 (157), df 0.176/0.273 (22), mel 0.596/0.482 (168), nv 0.894/0.905 (1000), vasc 0.875/0.636 (22). Melanoma specificity is one-vs-rest. Full detail is in `outputs/vgg16/cross_entropy_aug-false/`.

Limitations: single seed and single run, so differences between conditions may be within run-to-run noise. Validation loss rose from epoch 7 while training loss kept falling, which indicates overfitting. Classes `df` and `vasc` have only 22 test images each, so their metrics are very uncertain. Roughly half of test melanomas were missed. The test split was evaluated once per condition and was not used for any decision. These numbers are not evidence of clinical utility.

Tests executed: `python -m pytest tests -q` gave 11 passed; `scripts/audit_data.py` gave 34 passed, 0 failed; GPU smoke run and 1-epoch pilots (batch 8: 186 s/epoch, 2.7 GB peak; batch 16: 159 s/epoch, 3.0 GB peak) completed. Pilot and smoke outputs (`pilot_bs8`, `pilot_bs16`, `smoke_test`) are not results. Grad-CAM generation is not implemented yet; only that the VGG16 target layer (`features[28]`, 512x14x14 output) is reachable was tested.
