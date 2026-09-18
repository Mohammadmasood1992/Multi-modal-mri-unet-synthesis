#!/bin/bash
#==============================================================================
# MASTER SUBMISSION SCRIPT
#
# Submits all 3 phases of the pipeline with the correct SLURM dependency
# chain, so they run in order without any manual babysitting:
#
#   1. Phase 1 (preprocess)     -- submitted immediately
#   2. Phase 2 (train folds)    -- 5-task job array, one GPU per fold,
#                                   waits for Phase 1 to succeed
#   3. Phase 3 (aggregate)      -- waits for ALL 5 fold jobs to succeed
#
# Run from the repository root:
#   bash slurm/submit_all.sh
#==============================================================================

set -e
cd "$(dirname "$0")/.."   # repo root, regardless of where this is called from

echo "=============================================="
echo "Submitting multi-GPU cross-validation pipeline"
echo "=============================================="

# -- Phase 1: preprocessing --------------------------------------------------
JOB_A=$(sbatch slurm/01_preprocess.sh | awk '{print $NF}')
echo "Phase 1 (preprocess)  -> Job ID: $JOB_A"

# -- Phase 2: 5 training jobs (job array), depend on Phase 1 success --------
JOB_B=$(sbatch --dependency=afterok:${JOB_A} slurm/02_train_fold.sh | awk '{print $NF}')
echo "Phase 2 (train folds) -> Array Job ID: $JOB_B  (tasks 0-4, one H200 GPU each)"

# -- Phase 3: aggregation, depends on ALL array tasks succeeding ------------
# afterok on an array job ID waits for every task in the array to complete.
JOB_C=$(sbatch --dependency=afterok:${JOB_B} slurm/03_aggregate.sh | awk '{print $NF}')
echo "Phase 3 (aggregate)   -> Job ID: $JOB_C"

echo ""
echo "=============================================="
echo "All jobs submitted. Dependency chain:"
echo "  Preprocess ($JOB_A)"
echo "    -> Train folds ($JOB_B) [5 parallel GPUs]"
echo "         -> Aggregate ($JOB_C)"
echo ""
echo "Monitor with:  squeue -u \$USER"
echo "Cancel all  :  scancel $JOB_A $JOB_B $JOB_C"
echo "=============================================="
