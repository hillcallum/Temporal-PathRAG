#!/bin/bash
# Master script for submitting training jobs to Imperial GPU cluster
# This script handles job submission when run from the cluster

echo "=== Temporal PathRAG Training Job Submission ==="
echo ""

# Check if we're on the cluster
if [[ ! "$HOSTNAME" =~ "ic.ac.uk" ]]; then
    echo "ERROR: This script must be run from the Imperial cluster"
    echo "Please SSH to gpucluster2.doc.ic.ac.uk first"
    exit 1
fi

# Configuration defaults (can be overridden by environment variables)
DATASET=${DATASET:-MultiTQ}
EPOCHS=${EPOCHS:-10}
BATCH_SIZE=${BATCH_SIZE:-32}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-4}
LR=${LR:-1e-4}
WARMUP_STEPS=${WARMUP_STEPS:-500}
QUADRUPLET_MARGIN=${QUADRUPLET_MARGIN:-0.5}
TEMPORAL_WEIGHT=${TEMPORAL_WEIGHT:-0.3}
CHECKPOINT_EVERY=${CHECKPOINT_EVERY:-1000}
EVAL_EVERY=${EVAL_EVERY:-500}
USE_WANDB=${USE_WANDB:-false}
USE_MIXED_PRECISION=${USE_MIXED_PRECISION:-true}

# Get email for notifications
if [ -z "$SLURM_EMAIL" ]; then
    echo "Enter your email for SLURM notifications:"
    read -r SLURM_EMAIL
fi

# Display configuration
echo "Training Configuration:"
echo "----------------------"
echo "Dataset: ${DATASET}"
echo "Epochs: ${EPOCHS}"
echo "Batch size: ${BATCH_SIZE}"
echo "Gradient accumulation: ${GRAD_ACCUM_STEPS}"
echo "Effective batch size: $((BATCH_SIZE * GRAD_ACCUM_STEPS))"
echo "Learning rate: ${LR}"
echo "Temporal weight: ${TEMPORAL_WEIGHT}"
echo "Mixed precision: ${USE_MIXED_PRECISION}"
echo "Email: ${SLURM_EMAIL}"
echo ""

# Create temporary job script with email
TEMP_JOB_SCRIPT="/tmp/train_temporal_embeddings_$$.sh"
sed "s/--mail-user=cih124/--mail-user=$SLURM_EMAIL/" \
    /vol/bitbucket/${USER}/Temporal_PathRAG/slurm/job_scripts/train_temporal_embeddings.sh > "$TEMP_JOB_SCRIPT"

# Build SBATCH parameters
SBATCH_PARAMS="--export=ALL"
SBATCH_PARAMS="${SBATCH_PARAMS},DATASET=${DATASET}"
SBATCH_PARAMS="${SBATCH_PARAMS},EPOCHS=${EPOCHS}"
SBATCH_PARAMS="${SBATCH_PARAMS},BATCH_SIZE=${BATCH_SIZE}"
SBATCH_PARAMS="${SBATCH_PARAMS},GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS}"
SBATCH_PARAMS="${SBATCH_PARAMS},LR=${LR}"
SBATCH_PARAMS="${SBATCH_PARAMS},WARMUP_STEPS=${WARMUP_STEPS}"
SBATCH_PARAMS="${SBATCH_PARAMS},QUADRUPLET_MARGIN=${QUADRUPLET_MARGIN}"
SBATCH_PARAMS="${SBATCH_PARAMS},TEMPORAL_WEIGHT=${TEMPORAL_WEIGHT}"
SBATCH_PARAMS="${SBATCH_PARAMS},CHECKPOINT_EVERY=${CHECKPOINT_EVERY}"
SBATCH_PARAMS="${SBATCH_PARAMS},EVAL_EVERY=${EVAL_EVERY}"

if [ "${USE_WANDB}" = "true" ]; then
    SBATCH_PARAMS="${SBATCH_PARAMS},USE_WANDB=1"
else
    SBATCH_PARAMS="${SBATCH_PARAMS},USE_WANDB="
fi

if [ "${USE_MIXED_PRECISION}" = "true" ]; then
    SBATCH_PARAMS="${SBATCH_PARAMS},USE_MIXED_PRECISION=1"
fi

# Submit job
echo "Submitting training job"
JOB_ID=$(sbatch ${SBATCH_PARAMS} "$TEMP_JOB_SCRIPT" | awk '{print $4}')

if [ -n "$JOB_ID" ]; then
    echo "✓ Job submitted successfully with ID: $JOB_ID"
    echo ""
    echo "Monitor your job:"
    echo "  squeue -j $JOB_ID"
    echo "  tail -f /vol/bitbucket/${USER}/temporal_pathrag_logs/train_embeddings_${JOB_ID}.out"
else
    echo "✗ Failed to submit job"
fi

# Clean up
rm -f "$TEMP_JOB_SCRIPT"

# Show current jobs
echo ""
echo "Your current jobs:"
squeue -u ${USER}