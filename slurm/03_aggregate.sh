#!/bin/bash
#==============================================================================
# PHASE 3 -- Aggregation
# Loads all 5 trained fold models, evaluates the external test set as an
# ensemble, and saves the final summary. Runs after all 5 training jobs
# (Phase 2) have finished.
#
# Submit with:
#   sbatch --dependency=afterok:<ARRAY_JOB_ID> slurm/03_aggregate.sh
#==============================================================================
#SBATCH --job-name=unet_synth_aggregate
#SBATCH --partition=gpu                       # GPU speeds up inference; a CPU partition also works
#SBATCH --constraint=GPU_SKU:H200
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256gb
#SBATCH --time=4:00:00

#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your-email@example.com   # <-- set your own address

#SBATCH --output=logs/aggregate_%j.out
#SBATCH --error=logs/aggregate_%j.err

set -e

error_exit() {
    echo "ERROR: $1" >&2
    echo "Job failed at: $(date)" >&2
    exit 1
}
trap 'error_exit "Script failed at line $LINENO"' ERR

cd "$(dirname "$0")/.."   # repo root

echo "=============================================="
echo "PHASE 3 -- AGGREGATION"
echo "Job Started : $(date)"
echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $(hostname)"
echo "=============================================="

mkdir -p logs outputs

# -- Load environment ---------------------------------------------------------
source $(conda info --base)/etc/profile.d/conda.sh
conda activate unet-synth || error_exit "Failed to activate conda environment"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# -------------------------------------------------------------------------
CONFIG_PATH="config/config.yaml"
INPUT_NOTEBOOK="notebooks/model_training.ipynb"
OUTPUT_NOTEBOOK="outputs/aggregate_${SLURM_JOB_ID}.html"

[ -f "$CONFIG_PATH" ]    || error_exit "Config file not found: $CONFIG_PATH"
[ -f "$INPUT_NOTEBOOK" ] || error_exit "Notebook not found: $INPUT_NOTEBOOK"

export CONFIG_PATH="$CONFIG_PATH"
export JOB_ID="${SLURM_JOB_ID}"

# Tells the notebook's routing cell to run ONLY the aggregation phase.
export TRAINING_PHASE="aggregate"
export FOLD_ID=""

echo "Running aggregation notebook..."
jupyter nbconvert \
    --to html \
    --execute "$INPUT_NOTEBOOK" \
    --output  "$OUTPUT_NOTEBOOK" \
    --ExecutePreprocessor.timeout=-1 \
    --ExecutePreprocessor.cwd=. \
    --log-level=INFO \
    || error_exit "Aggregation notebook execution failed"

echo ""
echo "=============================================="
echo "PHASE 3 complete: $(date)"
echo ""
echo "Final outputs:"
echo "  HTML report  : $OUTPUT_NOTEBOOK"
echo "  Fold results : experiments/*/fold_results.csv"
echo "  Ext. test    : experiments/*/external_test/"
echo "  Models       : experiments/*/model_fold_*.pth"
echo "=============================================="
exit 0
