# AGENTS.md — Repository Working Instructions

## Mission

Build a reproducible educational research prototype for seven-class skin-lesion image classification using only the HAM10000 dataset. Compare pretrained VGG-16, ResNet-18, and EfficientNet-B0 in PyTorch; evaluate focal loss and data augmentation; generate Grad-CAM visualizations; and document interpretability guardrails.

This repository is not a medical diagnostic tool. Do not write product copy or code comments that imply clinical validation, diagnosis, or guaranteed safety.

## Source-of-truth rules

1. Read `README.md` and `pyproject.toml` before making structural or dependency changes.
2. Use `uv` as the project's Python and dependency manager. Keep dependency declarations in `pyproject.toml` and commit `uv.lock`.
3. Do not add or maintain `requirements.txt` as a second source of truth unless a deployment platform specifically requires it. If needed, document how it is generated from the `uv` project.
4. Keep reusable implementation in `src/`; notebooks are for exploration, not the only copy of core logic.
5. Use a shared seven-class label mapping in every model and evaluation path:
   - `akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, `vasc`
6. Do not silently change dataset splits, class order, preprocessing, metrics, or experiment definitions. Update documentation and configurations when a change is intentional.
7. Use only HAM10000 for the project's dataset-based training and evaluation, unless the project owner explicitly changes this requirement.
8. Never invent experiment results, performance claims, dataset counts, or clinical conclusions. Clearly mark unrun experiments as pending.

## uv and environment management

- Use `uv add <package>` to add runtime dependencies and `uv add --dev <package>` for development-only dependencies.
- Use `uv sync` to synchronize the environment and `uv run <command>` to execute project commands.
- Keep `pyproject.toml` and `uv.lock` in sync; commit both when dependencies change.
- Pin the intended Python version in `.python-version` and keep it compatible with the selected PyTorch/Torchvision versions.
- Do not mix ad hoc `pip install` steps into project instructions when the dependency should be declared with `uv`.
- PyTorch installation can depend on platform and accelerator. Follow the official PyTorch instructions for the selected CPU/CUDA runtime; verify the installation before expensive training.
- Colab and Kaggle runtimes may have different Python/CUDA setups and temporary filesystems. Verify compatibility before `uv sync`; document any runtime-specific setup rather than assuming the local lockfile works unchanged.
- Never commit credentials, tokens, `.env` secrets, dataset files, or model artifacts.

## Data integrity and leakage prevention

- Split by lesion identifier where available so images of the same lesion do not cross train/validation/test boundaries.
- Generate split manifests once, record the random seed and class counts, and reuse the same manifests for every architecture and ablation.
- Augment training data only. Validation and test transforms must be deterministic and must not include random augmentation.
- Do not tune hyperparameters or thresholds against the held-out test set.
- Check image paths, labels, duplicates, missing metadata, and class mapping before training.
- Keep dataset images, large checkpoints, temporary caches, and generated artifacts out of Git.

## Model implementation

- Use PyTorch and `torchvision` where suitable.
- Prefer ImageNet-pretrained weights unless the assignment explicitly requires training from scratch. Document the weight source/version.
- Replace each model's original classifier with a seven-class output head.
- Make the model factory or construction interface consistent across VGG-16, ResNet-18, and EfficientNet-B0.
- Keep model construction separate from the training loop.
- Do not assume all architectures use the same Grad-CAM target layer; choose and document an appropriate convolutional target layer for each.

## Training and experimental design

- Implement a shared training/validation loop where practical.
- Keep seeds, optimizer, scheduler, batch size, epochs, image size, loss parameters, augmentation, and checkpoint-selection rules in configuration rather than scattering magic constants throughout code.
- Use cross-entropy as a baseline and implement focal loss as a selectable option.
- Support controlled conditions:
  1. Cross-entropy, augmentation off.
  2. Focal loss, augmentation off.
  3. Cross-entropy, augmentation on.
  4. Focal loss, augmentation on.
- If compute limits require fewer runs, document which experiments were run and which were skipped.
- Save the configuration snapshot and training/validation history for each run.
- Select checkpoints using validation data only.
- Do not claim that focal loss or augmentation balances class counts automatically; evaluate their effects empirically.

## Evaluation requirements

Report complementary metrics rather than accuracy alone:
- Macro F1.
- Balanced accuracy.
- Per-class precision and recall.
- Melanoma sensitivity/recall and specificity.
- Confusion matrix.
- Runtime, parameter count, and checkpoint size where feasible.

Use identical split manifests and a comparable evaluation protocol across models. State metric definitions and averaging methods. Avoid presenting a single metric as proof of clinical utility.

## Grad-CAM and guardrails

- Generate Grad-CAM outputs for all three models.
- Inspect correct and incorrect predictions, including melanoma examples.
- Inspect whether heatmaps focus on the lesion or potential shortcuts/artifacts such as hair, rulers, markings, borders, or background.
- Do not state that Grad-CAM proves the model's reasoning is correct. Treat it as an exploratory explanation.
- Clearly state that softmax scores are not necessarily calibrated probabilities.
- Never tell users that a lesion is safe or benign based solely on model output.
- Include a clear educational-use disclaimer and advise seeking qualified clinical assessment for medical concerns.
- Record limitations and important failure cases.

## Code quality

- Prefer small, typed or clearly documented functions with single responsibilities.
- Use meaningful names and avoid unnecessary abstraction.
- Add or update tests when changing data handling, label mapping, loss functions, model output shapes, splitting, metrics, or Grad-CAM.
- Keep random seeds and device selection in shared utilities, while acknowledging that full determinism may not be guaranteed on every GPU.
- Handle missing files and invalid labels with actionable errors.
- Do not suppress exceptions or silently skip failed samples.
- Avoid duplicating training loops or metric implementations per model.

## Testing before expensive runs

Before a full run:
1. Check all seven labels and their mapping.
2. Confirm train/validation/test lesion IDs do not overlap.
3. Run one small batch through each model and verify output shape `[batch_size, 7]`.
4. Verify loss is finite and backward propagation works.
5. Run a tiny training/validation smoke test.
6. Verify checkpoint save/load.
7. Verify Grad-CAM produces a correctly sized heatmap for each architecture.

Use a small subset for smoke tests, but do not report smoke-test metrics as final results.

## Repository and cloud workflow

- Do not commit dataset images, checkpoints, generated artifacts, `.env` files, credentials, or tokens.
- Keep environment-specific data and output paths configurable; do not hardcode Colab, Kaggle, or local paths throughout source code.
- Notebook-local storage may be temporary. Save checkpoints and run metadata to persistent storage when using cloud notebooks.
- Do not launch expensive cloud jobs without explicit user approval when working autonomously.
- Do not install packages or alter the environment unnecessarily; update `pyproject.toml` and `uv.lock` when dependencies change.
- Preserve unrelated user changes and inspect Git status before proposing destructive cleanup.

## Collaboration and reporting

- Agree on split manifests, label order, preprocessing, metrics, and experiment conditions before members train independently.
- Keep one shared training and evaluation interface.
- Record for each run: model, condition, seed, config, split version, metrics, runtime, checkpoint path, and status.
- Separate measured results from hypotheses and planned work.
- Update `README.md` when commands, directory responsibilities, experiment conditions, or limitations change.