# Skin Lesion and Melanoma Screening System

A PyTorch research project for seven-class skin-lesion classification using the **HAM10000** dermoscopic image dataset. The project compares **VGG-16**, **ResNet-18**, and **EfficientNet-B0**, investigates class-imbalance strategies, and uses **Grad-CAM** to inspect image regions that influence model predictions.

> **Research and education only.** This project is not a medical device and must not be used to diagnose, rule out, or make treatment decisions about skin cancer. Model predictions and Grad-CAM visualizations are not substitutes for assessment by a qualified healthcare professional.

## 1. Project goals

The project has five main requirements:

1. **Seven-class classification:** use HAM10000 and predict one of its seven diagnostic categories.
2. **Architecture comparison:** train and evaluate VGG-16, ResNet-18, and EfficientNet-B0 using comparable experimental conditions.
3. **Class-imbalance strategies:** use focal loss and training-time data augmentation, and evaluate whether they improve results.
4. **Grad-CAM:** generate class-activation heatmaps for each model pipeline and inspect both successful and failed predictions.
5. **Interpretability guardrails:** communicate uncertainty and limitations, investigate possible shortcut learning, and avoid presenting model output as a diagnosis.

## 2. Dataset and labels

HAM10000 contains approximately 10,015 dermoscopic images across seven diagnostic categories:

| Label | Category |
|---|---|
| `akiec` | Actinic keratoses and intraepithelial carcinoma |
| `bcc` | Basal cell carcinoma |
| `bkl` | Benign keratosis-like lesions |
| `df` | Dermatofibroma |
| `mel` | Melanoma |
| `nv` | Melanocytic nevi |
| `vasc` | Vascular lesions |

Use only the permitted HAM10000 dataset for model training and evaluation. Follow the dataset's terms and citation requirements. Keep downloaded dataset files out of Git.

### Data-splitting requirements

- Create training, validation, and test splits before training.
- Where lesion identifiers are available, split by **lesion ID**, not merely by image, to reduce leakage from multiple images of the same lesion.
- Reuse the same split assignments for all models and experiment conditions.
- Apply augmentation to training data only. Validation and test images must use deterministic preprocessing.
- Do not use the test set to choose hyperparameters, thresholds, or checkpoints.
- Record the split-generation seed and save split metadata in `data/splits/`.

## 3. Models and training strategy

All three models should use a consistent seven-class label mapping and a comparable evaluation protocol.

- **ResNet-18:** recommended first implementation because it is a practical baseline and is available in `torchvision`.
- **EfficientNet-B0:** a compact architecture to compare against ResNet-18.
- **VGG-16:** a conventional CNN architecture; its original fully connected classifier is comparatively large, so replace the classifier and monitor memory usage.

Use ImageNet-pretrained weights unless the assignment explicitly requires training from scratch. Replace each original classifier with a seven-output classification head. Document the selected pretrained-weight version and its license/source.

### Class-imbalance experiments

Use controlled experiments to understand the contribution of each technique. A recommended experiment matrix is:

| Condition | Loss | Training augmentation |
|---|---|---|
| Baseline | Cross-entropy | Off |
| Focal-loss ablation | Focal loss | Off |
| Augmentation ablation | Cross-entropy | On |
| Combined | Focal loss | On |

Run the four conditions for each architecture if time and compute permit (12 experiments total). If resources are limited, first establish a baseline for all three models, then prioritize the combined condition and selected ablations.

Focal loss changes the contribution of easy and difficult examples; it does **not** itself guarantee balanced class counts. Augmentation adds variation but does **not** guarantee equal class frequencies. Report the observed per-class results rather than assuming either technique helps.

## 4. Evaluation

The problem statement does not prescribe a particular metric. Report a set of complementary metrics rather than relying on accuracy alone:

- **Macro F1:** gives each class equal weight in the aggregate.
- **Balanced accuracy:** averages recall across classes.
- **Per-class precision and recall:** shows class-specific errors.
- **Melanoma sensitivity/recall and specificity:** important screening-related measures, reported with appropriate caveats.
- **Confusion matrix:** reveals which categories the model confuses.
- **Runtime, parameter count, and model/checkpoint size:** helps compare computational cost.

Use validation results to select checkpoints and tune thresholds. Evaluate the final selected models on the held-out test set once the approach is finalized. State the split method, class counts, random seeds, hardware, and model configuration in the report.

## 5. Interpretability and guardrails

Grad-CAM creates a coarse heatmap showing image regions that influence a selected class score. Generate examples for correct and incorrect predictions, including melanoma cases.

Inspect whether heatmaps overlap the lesion or focus on potential artifacts such as hair, rulers, borders, markings, or image background. A visually plausible heatmap does not prove that a model has learned valid clinical features or that its prediction is correct.

The demo/report should:
- Label results as experimental model predictions, not diagnoses.
- Explain that confidence scores are not guaranteed to be calibrated probabilities.
- Avoid reassuring users that a lesion is benign based on a model result.
- Clearly communicate dataset and generalization limitations.
- Encourage professional clinical assessment for medical concerns.
- Document false negatives and other important failure cases.

Do not claim clinical validation unless an appropriate, independently designed clinical validation study has been conducted.

## 6. Repository structure

```text
skin-lesion-screening/
├── configs/
│   ├── resnet18.yaml
│   ├── vgg16.yaml
│   ├── efficientnet_b0.yaml
│   └── experiment.yaml
├── data/
│   ├── raw/                  # Original dataset files; not committed
│   ├── processed/            # Optional cached/prepared artifacts
│   └── splits/               # Saved train/validation/test assignments
├── src/
│   ├── data/
│   │   ├── dataset.py        # Dataset and label loading
│   │   ├── transforms.py     # Training/evaluation transforms
│   │   └── dataloader.py     # DataLoader construction
│   ├── models/
│   │   ├── resnet18.py       # ResNet-18 construction and classifier head
│   │   ├── vgg16.py          # VGG-16 construction and classifier head
│   │   └── efficientnet_b0.py# EfficientNet-B0 construction and classifier head
│   ├── losses/
│   │   └── focal_loss.py     # Focal-loss implementation
│   ├── training/
│   │   └── trainer.py        # Shared train/validation loop and checkpointing
│   ├── evaluation/
│   │   ├── metrics.py        # Metrics and per-class results
│   │   └── plots.py          # Confusion matrices and learning curves
│   ├── explainability/
│   │   ├── gradcam.py        # Grad-CAM calculation
│   │   └── visualization.py  # Heatmap overlays and exports
│   └── utils/
│       ├── reproducibility.py# Seeds and deterministic settings where possible
│       ├── checkpoints.py    # Save/load helpers
│       └── logging.py        # Run logging helpers
├── experiments/              # Run configurations and metric logs
├── checkpoints/              # Saved weights; not committed
├── outputs/
│   ├── plots/
│   ├── confusion_matrices/
│   └── gradcam/
├── notebooks/                # Exploration and analysis notebooks
├── tests/                    # Unit and integration tests
├── train.py                  # Training entry point
├── evaluate.py               # Evaluation entry point
├── predict.py                # Single-image inference entry point
├── pyproject.toml             # Project metadata and dependencies managed by uv
├── uv.lock                   # Locked dependency resolution
├── .python-version            # Python version selected for the project
├── .gitignore
├── AGENTS.md
└── README.md
```

The tree describes the intended layout; files can be added as implementation proceeds. Keep reusable implementation in `src/`. Use notebooks for exploration rather than making them the only location of core training logic.

### Directory and module responsibilities

| Path | Purpose |
|---|---|
| `configs/` | Model and experiment settings such as image size, batch size, learning rate, loss, augmentation, epochs, and seed. |
| `data/raw/` | Original HAM10000 files. Keep outside version control. |
| `data/processed/` | Optional cached data or metadata; avoid saving unnecessary duplicate image copies. |
| `data/splits/` | Reusable split manifests and split-generation metadata. |
| `src/data/` | Dataset reading, label mapping, transforms, and batching. |
| `src/models/` | Constructs each architecture and replaces the original classifier with seven outputs. |
| `src/losses/` | Custom focal loss; standard cross-entropy can be provided by PyTorch. |
| `src/training/` | Training loop, validation, optimizer steps, scheduler, early stopping if used, and checkpoint selection. |
| `src/evaluation/` | Metric calculation, confusion matrices, model comparison, and learning curves. |
| `src/explainability/` | Grad-CAM and visualization. Target layers may differ between architectures. |
| `src/utils/` | Small shared utilities such as seeds, checkpoint helpers, and logging. Avoid turning this into a catch-all. |
| `experiments/` | Per-run configuration snapshots, metrics, histories, and references to artifacts. |
| `checkpoints/` | Model weights and resumable training states. Store outside Git. |
| `outputs/` | Generated plots, heatmaps, and report-ready artifacts. |
| `notebooks/` | Data exploration, debugging, and result analysis. |
| `tests/` | Checks for labels, output shapes, loss finiteness, split leakage, and Grad-CAM dimensions. |
| `train.py` | Parses configuration and starts a training run. |
| `evaluate.py` | Loads a checkpoint and evaluates a specified split. |
| `predict.py` | Runs inference on an image using the trained model. |
| `pyproject.toml` | Project metadata, dependency declarations, and development-tool configuration managed by `uv`. |
| `uv.lock` | Exact resolved dependency versions for repeatable installs. Commit this file. |
| `.python-version` | Project's selected Python version for `uv`. |
| `AGENTS.md` | Repository-specific instructions for coding agents and contributors. |

## 7. Environment and dependency management with uv

Use [`uv`](https://docs.astral.sh/uv/) to manage Python versions, the virtual environment, dependencies, and the lockfile. The intended source of truth for dependencies is `pyproject.toml` plus the committed `uv.lock`. Do not maintain a separate `requirements.txt` unless a deployment platform specifically requires one; if one is generated, document how it is kept in sync.

### Initial setup (PowerShell)

Install `uv` using its official installation instructions, then create the project:

```powershell
uv init skin-lesion-screening
cd skin-lesion-screening
uv python pin 3.11
```

If the repository has already been initialized, do not run `uv init` again. Instead, run the setup commands from the existing project root.

Add the core dependencies:

```powershell
uv add torch torchvision pandas numpy pillow scikit-learn matplotlib seaborn pyyaml tqdm
uv add --dev pytest ruff
```

These are starting dependencies, not a guarantee that every version works on every platform. Select a PyTorch/Torchvision combination appropriate to your local CPU or CUDA environment using the official PyTorch installation guidance. After dependency changes, commit both `pyproject.toml` and `uv.lock`.

### Common commands

```powershell
# Create/sync the environment from project metadata and lockfile
uv sync

# Run a training script
uv run python train.py --model resnet18

# Run tests
uv run pytest

# Lint the project
uv run ruff check .
```

The training command is an example; it works once `train.py` implements the stated CLI. Use `uv run` for project commands so they execute in the managed environment.

### Colab and Kaggle

Notebook runtimes are separate environments and their local files may be temporary. Clone the repository into the runtime, install `uv`, then use `uv sync` where compatible. Confirm the installed Python, PyTorch, Torchvision, and accelerator configuration before training. GPU/CUDA package availability can differ by runtime, so do not assume the local lockfile will work unchanged in every cloud environment. Save checkpoints and important run artifacts to persistent storage.

## 8. Collaboration workflow

For a two-person team:

1. Agree on label mapping, lesion-level split manifests, preprocessing, metrics, and experiment conditions.
2. Build and test one shared data pipeline before starting long training runs.
3. Assign model ownership, but share the same training, evaluation, and Grad-CAM interfaces.
4. Save configuration and metrics for every run so results can be reproduced.
5. Keep datasets and large model artifacts out of Git; use Git for source code, configuration, tests, and documentation.
6. Review model comparison and failure cases together before writing conclusions.

Suggested initial ownership: one member establishes the ResNet-18 baseline and shared pipeline; the other prepares EfficientNet-B0 against the agreed interfaces. Add VGG-16 after the pipeline is stable. Rebalance ownership according to experience and available compute.

## 9. Definition of done

The project is ready for submission when:

- All three models train and produce seven-class outputs.
- The train/validation/test split is documented and leakage checks pass.
- Focal loss and data augmentation are implemented and their effects are evaluated.
- All models are evaluated with the same declared metrics and test protocol.
- Grad-CAM is generated for each architecture, including failure cases.
- Interpretability and medical-use limitations are visible in the demo/report.
- Checkpoints, configurations, results, and instructions are organized so another team member can reproduce the experiments.

## 10. Limitations

HAM10000 is a research dataset and may not represent all skin tones, clinical settings, imaging devices, or patient populations. Results on this dataset alone do not establish real-world clinical safety or generalization. The system must be presented as an educational research prototype, not a diagnostic service.

## References

- HAM10000 dataset: Tschandl, P., Rosendahl, C., & Kittler, H. (2018). *The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions*. Scientific Data, 5, 180161. https://doi.org/10.1038/sdata.2018.161
- PyTorch documentation: https://pytorch.org/docs/stable/
- Torchvision models: https://pytorch.org/vision/stable/models.html
- uv documentation: https://docs.astral.sh/uv/