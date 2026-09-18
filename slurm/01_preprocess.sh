#!/bin/bash
#==============================================================================
# PHASE 1 -- Preprocessing
# Splits the data and saves fold indices to disk. No GPU required.
# Runs once, before the 5 parallel training jobs.
#==============================================================================
#SBATCH --job-name=unet_synth_preprocess
#SBATCH --partition=general                  # CPU partition -- no GPU needed
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256gb
#SBATCH --time=2:00:00

#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your-email@example.com   # <-- set your own address

#SBATCH --output=logs/preprocess_%j.out
#SBATCH --error=logs/preprocess_%j.err

set -e

error_exit() {
    echo "ERROR: $1" >&2
    echo "Job failed at: $(date)" >&2
    exit 1
}
trap 'error_exit "Script failed at line $LINENO"' ERR

cd "$(dirname "$0")/.."   # repo root

echo "=============================================="
echo "PHASE 1 -- PREPROCESSING"
echo "Job Started : $(date)"
echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $(hostname)"
echo "=============================================="

mkdir -p logs outputs experiments

# -- Load environment ---------------------------------------------------------
source $(conda info --base)/etc/profile.d/conda.sh
conda activate unet-synth || error_exit "Failed to activate conda environment"

# -------------------------------------------------------------------------
CONFIG_PATH="config/config.yaml"
INPUT_NOTEBOOK="notebooks/model_training.ipynb"
OUTPUT_NOTEBOOK="outputs/preprocess_${SLURM_JOB_ID}.html"
OUTPUT_CONFIG="outputs/config_${SLURM_JOB_ID}.yaml"

[ -f "$CONFIG_PATH" ]    || error_exit "Config file not found: $CONFIG_PATH"
[ -f "$INPUT_NOTEBOOK" ] || error_exit "Notebook not found: $INPUT_NOTEBOOK"

cp "$CONFIG_PATH" "$OUTPUT_CONFIG"

export CONFIG_PATH="$CONFIG_PATH"
export JOB_ID="${SLURM_JOB_ID}"

# Tells the notebook's routing cell to run ONLY the preprocessing phase.
export TRAINING_PHASE="preprocess"
export FOLD_ID=""

echo "Running preprocessing notebook..."
jupyter nbconvert \
    --to html \
    --execute "$INPUT_NOTEBOOK" \
    --output  "$OUTPUT_NOTEBOOK" \
    --ExecutePreprocessor.timeout=-1 \
    --ExecutePreprocessor.cwd=. \
    --log-level=INFO \
    || error_exit "Preprocessing notebook execution failed"

echo ""
echo "=============================================="
echo "PHASE 1 complete: $(date)"
echo "Fold index files are in:  experiments/<name>/fold_splits/"
echo "Submit the 5 training jobs now with:"
echo "  sbatch --dependency=afterok:${SLURM_JOB_ID} slurm/02_train_fold.sh"
echo "=============================================="
exit 0
