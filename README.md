# Diffusion-MRI Image Synthesis with a Multi-GPU 3D U-Net

A configurable 2D/3D U-Net that synthesizes one diffusion-MRI-derived map
from another, trained with leakage-safe grouped cross-validation and
evaluated on a fully held-out external test set. Cross-validation folds
train in parallel across multiple GPUs, orchestrated as a three-phase SLURM
job-dependency pipeline.

## Problem

Diffusion MRI can be processed into several different microstructural maps
(e.g. Fractional Anisotropy, restricted/isotropic diffusion signal
fractions), but not every map is acquired or successfully processed for
every participant in a large imaging study. This project trains a model to
predict one such map from another, so that missing or unreliable maps can
be estimated from a routinely-available one, and to study how well spatial
structure in these maps transfers across a learned mapping.

The specific task in the included config is:

```
input:  FA   (Fractional Anisotropy)
output: RNI, HNI  (Restricted and Hindered Normalized Isotropic diffusion signal fraction)
```

but the pipeline is not specific to these two maps -- any pair of
co-registered 3D volumes can be substituted via `config/config.yaml`.

## Dataset

The dataset used to develop and run this pipeline comes from a
restricted-access, multi-site neuroimaging study and **is not included in
this repository** due to its data use agreement. To run the pipeline
yourself, supply your own data directory (path set by `data_directory` in
`config/config.yaml`) containing:

- `<modality>.npy` -- a stacked array of registered 3D volumes for each
  modality named in `input_modality` / `output_modality`, shape
  `(n_samples, depth, height, width)`
- `<modality>_index.csv` -- one row per sample in the `.npy` array, used to
  detect empty/invalid volumes
- a participant metadata CSV (`data_frame`) with, at minimum, participant
  ID, imaging session ID, family ID (for grouped splitting), site ID, and
  scanner manufacturer columns

## Approach

**Model.** A U-Net (`src/unet.py`) built dynamically from the encoder /
bottleneck / decoder layer specs in the config -- filter counts, dropout,
pooling type, and skip connections are all data, not code, so the same
class serves both the 3D model used for the reported results and a 2D
slice-based variant for faster experimentation.

**Loss.** A weighted combination of L1 and SSIM losses by default
(`src/losses.py`); MSE and a VGG-feature perceptual loss are also
available via config.

**Splitting.** Cross-validation is grouped at the **family** level (or
site level), so related participants never appear in both the training
and test portions of a fold -- a plain random split would leak information
through shared genetics/environment. A separate pool of participants is
held out entirely as an **external test set** before cross-validation
starts, and is scored only once, after all folds have finished training
(`src/trainer.py: UNetTrainer._split_external_test`, with an explicit
overlap check in `_check_overlap`).

**Training.** 5-fold grouped cross-validation, AdamW, `ReduceLROnPlateau`
scheduling, gradient clipping, early stopping, and 3D data augmentation
(rotation, scaling, elastic deformation, intensity jitter) applied
identically to the input and target volume so spatial correspondence is
preserved.

**Evaluation.** Each fold's test split is scored with MAE, MSE, RMSE,
PSNR, Pearson correlation, and SSIM (`src/metrics.py`). The final result
is the 5-fold ensemble's performance on the external test set, plus a look
at how much those metrics vary fold-to-fold -- a model whose accuracy
swings widely depending on which participants it trained on is less
trustworthy than one with tightly clustered per-fold metrics, even at the
same average accuracy.

## Multi-GPU Training

The training pipeline is split into three phases, each a separate SLURM
job, so the 5 cross-validation folds train **in parallel on 5 GPUs**
instead of sequentially on one:

| Phase | Script | What it does | Resources |
|---|---|---|---|
| 1 | `slurm/01_preprocess.sh` | Split data, save fold indices to disk | CPU only, runs once |
| 2 | `slurm/02_train_fold.sh` | Train one cross-validation fold | 5 GPU x 5 parallel tasks (SLURM job array) |
| 3 | `slurm/03_aggregate.sh` | Load all 5 fold models, evaluate external test set as an ensemble | 1 GPU, runs once |

```
Preprocess (Phase 1)
   └─ Train folds (Phase 2)   [5 parallel GPUs, one per fold]
        └─ Aggregate (Phase 3)
```

`slurm/submit_all.sh` submits all three phases at once with SLURM
`--dependency=afterok` chains, so Phase 2 only starts once Phase 1
succeeds, and Phase 3 only starts once **all 5** Phase-2 tasks succeed.
Each fold's job checkpoints its model, optimizer, and scheduler state
after every epoch, so a fold that gets killed by the cluster's time limit
resumes from its last completed epoch on resubmission instead of starting
over -- important on shared clusters where a 48-hour job may not run to
completion in one scheduling window.

The same notebook (`notebooks/model_training.ipynb`) is used for all three
phases: a routing cell reads `TRAINING_PHASE` / `FOLD_ID` environment
variables (exported by the SLURM scripts) and calls the training entry
point accordingly. Running the notebook manually with no environment
variables set falls back to training all 5 folds sequentially in a single
process, which is convenient for local debugging on a single GPU.

## Key Results

Because the dataset is private, no result numbers are published in this
repository. Running the full pipeline against your own data produces:

- Per-fold training curves and validation metrics: `experiments/<name>/fold_<n>/`
- Per-fold test-set summary: `experiments/<name>/fold_results.csv`
- External test set ensemble metrics and predictions: `experiments/<name>/external_test/`

Section 10 of the notebook loads and summarizes these once they exist.

## Repository Structure

```
.
├── config/
│   └── config.yaml            # data, model architecture, training hyperparameters
├── notebooks/
│   └── model_training.ipynb   # the full pipeline: data loading through evaluation
├── src/
│   ├── unet.py                # U-Net architecture, built from config
│   ├── losses.py               # L1 / MSE / SSIM / perceptual / combined losses
│   ├── augmentation.py         # 2D/3D spatial + intensity augmentation
│   ├── datasets.py             # PyTorch Dataset classes (3D volumes, 2D slices)
│   ├── metrics.py              # MAE / MSE / RMSE / PSNR / correlation / SSIM
│   ├── optim.py                # optimizer / scheduler factories, early stopping
│   ├── plotting.py             # training-curve and prediction visualizations
│   └── trainer.py              # UNetTrainer: preprocess / train / aggregate phases
├── slurm/
│   ├── submit_all.sh           # submits the full 3-phase pipeline with dependencies
│   ├── 01_preprocess.sh
│   ├── 02_train_fold.sh        # job array: one GPU per fold
│   └── 03_aggregate.sh
├── requirements.txt
└── README.md
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

A CUDA-capable GPU is recommended for training; the 3D model in the
default config is memory-intensive (the original runs used 1x NVIDIA
H200 per fold). The pipeline also runs on CPU for small-scale testing by
setting `device: cpu`, or automatically falls back to CPU if no GPU is
available.

## Running

**Locally, single process (all folds sequentially):**

```bash
jupyter nbconvert --to notebook --execute notebooks/model_training.ipynb \
    --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.cwd=.
```

**On a SLURM cluster, multi-GPU:**

```bash
bash slurm/submit_all.sh
```

This submits the three-phase dependency chain described above. Edit the
`#SBATCH` directives in `slurm/*.sh` (partition, GPU type/constraint,
account, email) to match your cluster, and set `data_directory` in
`config/config.yaml` to your data.

## Notes on Correctness and Reproducibility

A few issues were found and fixed while preparing this repository for
release:

- **Fold-strategy selection bug**: the original notebook computed the
  cross-validation grouping column conditionally on `fold_strategy`
  (`'family'` vs. `'site'`), but a later cell unconditionally overwrote
  that choice with the family-ID column, so switching `fold_strategy` to
  `'site'` in the config silently had no effect. This did not change any
  results produced with the default `fold_strategy: family`, but would
  have silently invalidated a site-grouped run. Fixed in the notebook
  (Section 7) so the grouping column always matches `fold_strategy`.
- **Undefined command in the training SLURM script**: the original fold
  job called a `my_job_header` shell function that is not defined
  anywhere in the project (an artifact of the original cluster's shared
  environment). Removed, since it would fail on any other cluster.
- **Unused imports**: the notebook imported several packages
  (`tqdm`, `seaborn`, `category_encoders.BinaryEncoder`, `sklearn`'s plain
  `KFold`, and a few `typing` names) that are never used; removed for
  clarity.

All other logic -- model architecture, loss functions, data splitting,
training loop, and metrics -- is unchanged from the original
implementation; only code organization, naming, and the issues above were
touched.

## License

MIT -- see [LICENSE](LICENSE).
