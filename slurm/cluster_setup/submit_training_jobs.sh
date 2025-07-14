#!/bin/bash
# Script to submit training and evaluation jobs to Imperial GPU cluster

echo "Temporal PathRAG Training Job Submission"
echo "This script will submit jobs to the Imperial GPU cluster"
echo ""

# Check if we're on the cluster
if [[ ! "$HOSTNAME" =~ "ic.ac.uk" ]]; then
    echo "Warning: Not on Imperial cluster. This script should be run from gpucluster2.doc.ic.ac.uk"
    echo "Do you want to continue anyway? (y/n)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        exit 0
    fi
fi

# Configuration
CLUSTER_USER="${CLUSTER_USER:-cih124}"
CLUSTER_HOST="${CLUSTER_HOST:-gpucluster2.doc.ic.ac.uk}"
BASE_DIR="/vol/bitbucket/${CLUSTER_USER}/Temporal_PathRAG"

# Function to sync code to cluster
sync_to_cluster() {
    echo "Syncing code to cluster"
    
    # Create directory structure on cluster
    ssh ${CLUSTER_USER}@${CLUSTER_HOST} "mkdir -p ${BASE_DIR}"
    
    # Sync code (excluding large files and caches)
    rsync -avz --progress \
        --exclude='*.pyc' \
        --exclude='__pycache__' \
        --exclude='.git' \
        --exclude='datasets/*/kg/*.txt' \
        --exclude='analysis_results' \
        --exclude='cache' \
        --exclude='.venv' \
        --exclude='*.pkl' \
        --exclude='*.pth' \
        --exclude='*.bin' \
        --exclude='timeR4_datasets/' \
        --exclude='raw_datasets/' \
        --exclude='logs/*.log' \
        --exclude='test_results/' \
        . ${CLUSTER_USER}@${CLUSTER_HOST}:${BASE_DIR}/
    
    echo "Code sync completed"
}

# Function to submit a job
submit_job() {
    local job_script=$1
    local job_name=$2
    local extra_args=${3:-""}
    
    echo "Submitting job: ${job_name}"
    
    ssh ${CLUSTER_USER}@${CLUSTER_HOST} "cd ${BASE_DIR} && sbatch ${extra_args} ${job_script}"
    
    if [ $? -eq 0 ]; then
        echo "Job submitted successfully"
    else
        echo "Failed to submit job"
    fi
}

# Function to check job status
check_jobs() {
    echo "Current jobs for ${CLUSTER_USER}:"
    ssh ${CLUSTER_USER}@${CLUSTER_HOST} "squeue -u ${CLUSTER_USER}"
}

# Default parameters (can be overridden with environment variables)
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
MAX_QUESTIONS=${MAX_QUESTIONS:-100}
BASELINES=${BASELINES:-"vanilla_llm temporal_pathrag"}

# Main execution
echo "Starting job submission"
echo ""

# Step 1: Sync code to cluster
sync_to_cluster
echo ""

# Step 2: Submit embedding training job
echo "Submitting embedding training job"
echo "Dataset: ${DATASET}"
echo "Epochs: ${EPOCHS}"
echo "Batch size: ${BATCH_SIZE}"
echo "Gradient accumulation steps: ${GRAD_ACCUM_STEPS}"
echo "Learning rate: ${LR}"
echo "Temporal weight: ${TEMPORAL_WEIGHT}"

# Build export variables
export_vars="--export=DATASET=${DATASET},EPOCHS=${EPOCHS},BATCH_SIZE=${BATCH_SIZE}"
export_vars="${export_vars},GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS},LR=${LR}"
export_vars="${export_vars},WARMUP_STEPS=${WARMUP_STEPS},QUADRUPLET_MARGIN=${QUADRUPLET_MARGIN}"
export_vars="${export_vars},TEMPORAL_WEIGHT=${TEMPORAL_WEIGHT},CHECKPOINT_EVERY=${CHECKPOINT_EVERY}"
export_vars="${export_vars},EVAL_EVERY=${EVAL_EVERY}"

# Add wandb if enabled
if [ "${USE_WANDB}" = "true" ]; then
    export_vars="${export_vars},USE_WANDB=1"
fi

submit_job "slurm/job_scripts/train_temporal_embeddings.sh" "Embedding Training" "$export_vars"
echo ""

# Step 3: Submit evaluation job
echo "Submitting evaluation job"
echo "Dataset: ${DATASET}"
echo "Max questions: ${MAX_QUESTIONS}"
echo "Baselines: ${BASELINES}"
echo "Model: llama3.2:3b"
export_vars="--export=DATASET=${DATASET},MAX_QUESTIONS=${MAX_QUESTIONS},BASELINES='${BASELINES}'"
submit_job "slurm/job_scripts/run_evaluation_with_ollama.sh" "Ollama Evaluation" "$export_vars"
echo ""

# Step 4: Check job status
check_jobs
echo ""

echo "All jobs submitted. Use 'squeue -u ${CLUSTER_USER}' to monitor progress"