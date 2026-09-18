#!/bin/bash
#==============================================================================
# PHASE 2 -- Training (job array: 5 folds x 1 GPU each)
#
# SLURM sets $SLURM_ARRAY_TASK_ID to 0, 1, 2, 3, or 4. Each array task trains
# exactly one cross-validation fold on its own GPU, so all 5 folds train in
# parallel instead of sequentially.
#
# Submit with:
#   sbatch --dependency=afterok:<PREPROCESS_JOB_ID> slurm/02_train_fold.sh
#==============================================================================
#SBATCH --job-name=unet_synth_fold
#SBATCH --partition=gpu
#SBATCH --constraint=GPU_SKU:H200            # adjust to a GPU type available on your cluster
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=512gb
#SBATCH --time=2-00:00:00

#SBATCH --array=0-4                          # one task per fold

#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80
#SBATCH --mail-user=your-email@example.com   # <-- set your own address

# %A = array master job ID, %a = task index (0-4)
#SBATCH --output=logs/fold_%A_%a.out
#SBATCH --error=logs/fold_%A_%a.err

set -e

error_exit() {
    echo "ERROR: $1" >&2
    echo "Job failed at: $(date)" >&2
    exit 1
}
trap 'error_exit "Script failed at line $LINENO"' ERR

cd "$(dirname "$0")/.."   # repo root

echo "=============================================="
echo "PHASE 2 -- TRAINING"
echo "Job Started  : $(date)"
echo "Array Job ID : $SLURM_ARRAY_JOB_ID"
echo "Task Index   : $SLURM_ARRAY_TASK_ID   (= fold_id passed to trainer)"
echo "Node         : $(hostname)"
echo "GPU          :"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "=============================================="

mkdir -p logs outputs experiments

# -- Load environment ---------------------------------------------------------
source $(conda info --base)/etc/profile.d/conda.sh
conda activate unet-synth || error_exit "Failed to activate conda environment"

# -------------------------------------------------------------------------
CONFIG_PATH="config/config.yaml"
INPUT_NOTEBOOK="notebooks/model_training.ipynb"
OUTPUT_NOTEBOOK="outputs/fold_${SLURM_ARRAY_TASK_ID}_${SLURM_ARRAY_JOB_ID}.html"
OUTPUT_CONFIG="outputs/config_${SLURM_ARRAY_JOB_ID}.yaml"

[ -f "$CONFIG_PATH" ]    || error_exit "Config file not found: $CONFIG_PATH"
[ -f "$INPUT_NOTEBOOK" ] || error_exit "Notebook not found: $INPUT_NOTEBOOK"

cp "$CONFIG_PATH" "$OUTPUT_CONFIG" 2>/dev/null || true   # may already exist from another task

export CONFIG_PATH="$CONFIG_PATH"
export JOB_ID="${SLURM_ARRAY_JOB_ID}"

# Tells the notebook's routing cell which fold this task owns.
export TRAINING_PHASE="train_fold"
export FOLD_ID="${SLURM_ARRAY_TASK_ID}"      # 0-based, matches trainer's fold_id

echo "Training fold ${FOLD_ID} ..."

jupyter nbconvert \
    --to html \
    --execute "$INPUT_NOTEBOOK" \
    --output  "$OUTPUT_NOTEBOOK" \
    --ExecutePreprocessor.timeout=-1 \
    --ExecutePreprocessor.cwd=. \
    --log-level=INFO \
    || error_exit "Notebook execution failed for fold ${FOLD_ID}"

echo ""
echo "=============================================="
echo "PHASE 2 complete -- fold ${FOLD_ID}: $(date)"
echo "=============================================="

MODEL_FILE="experiments/*/model_fold_$((SLURM_ARRAY_TASK_ID + 1)).pth"
echo "Saved model : $(ls $MODEL_FILE 2>/dev/null || echo 'not found')"

exit 0
