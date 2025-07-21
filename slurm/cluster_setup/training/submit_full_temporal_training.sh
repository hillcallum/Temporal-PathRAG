#!/bin/bash
# Submission script for full temporal embeddings training
# This script ensures all paths and configurations are correct

echo "Temporal PathRAG Full Training Submission"
echo ""

# Configuration
DATASET="combined"  # Use combined MultiTQ + TimeQuestions dataset
EPOCHS=30          # Full training run
BATCH_SIZE=32      # Optimal for 48GB GPU memory
GRAD_ACCUM=4       # Effective batch size = 128
LR=1e-4           # Learning rate
WARMUP=1000       # Warmup steps
CHECKPOINT_EVERY=2000  # Save checkpoint every 2000 steps
EVAL_EVERY=1000   # Evaluate every 1000 steps
LOG_EVERY=100     # Log metrics every 100 steps

# Paths (relative to cluster home)
PROJECT_ROOT="/vol/bitbucket/${USER}/Temporal_PathRAG"
DATA_DIR="${PROJECT_ROOT}/data/training"
OUTPUT_BASE="/vol/bitbucket/${USER}/temporal_embeddings"
BEST_MODEL_DIR="${OUTPUT_BASE}/best_model"
CLUSTER_SCRIPT="${PROJECT_ROOT}/slurm/job_scripts/train_temporal_embeddings.sh"

echo "Configuration:"
echo "- Dataset: ${DATASET}"
echo "- Epochs: ${EPOCHS}"
echo "- Batch Size: ${BATCH_SIZE} (effective: $((BATCH_SIZE * GRAD_ACCUM)))"
echo "- Learning Rate: ${LR}"
echo "- Data Directory: ${DATA_DIR}"
echo "- Best Model Directory: ${BEST_MODEL_DIR}"
echo ""

# Check if we're on the cluster
if [[ ! -d "/vol/bitbucket" ]]; then
    echo "ERROR: This script must be run on the Imperial cluster"
    echo "Please SSH to the cluster first"
    exit 1
fi

# Verify data exists
if [[ ! -d "${DATA_DIR}/${DATASET}" ]]; then
    echo "ERROR: Training data not found at ${DATA_DIR}/${DATASET}"
    echo "Please sync the training data to the cluster first"
    echo ""
    echo "Run from local machine:"
    echo "  ./slurm/cluster_setup/sync_to_cluster.sh"
    exit 1
fi

# Check data files
echo "Checking training data files"
for split in train validation test; do
    file="${DATA_DIR}/${DATASET}/${split}.json"
    if [[ -f "$file" ]]; then
        size=$(ls -lh "$file" | awk '{print $5}')
        echo "${split}.json: ${size}"
    else
        echo "${split}.json: Not Found"
        exit 1
    fi
done
echo ""

# Create output directories
echo "Creating output directories"
mkdir -p "${BEST_MODEL_DIR}"
mkdir -p "/vol/bitbucket/${USER}/temporal_pathrag_logs"
echo "Output directories created"
echo ""

# Check if training script exists
if [[ ! -f "${CLUSTER_SCRIPT}" ]]; then
    echo "Error: Cluster training script not found at ${CLUSTER_SCRIPT}"
    exit 1
fi

# Submit the job
echo "Submitting SLURM job"
echo "Command:"
echo "sbatch \\"
echo "    --export=ALL,\\"
echo "    DATASET=${DATASET},\\"
echo "    EPOCHS=${EPOCHS},\\"
echo "    BATCH_SIZE=${BATCH_SIZE},\\"
echo "    GRAD_ACCUM_STEPS=${GRAD_ACCUM},\\"
echo "    LR=${LR},\\"
echo "    TEMPORAL_WEIGHT=0.3,\\"
echo "    QUADRUPLET_MARGIN=0.5,\\"
echo "    WARMUP_STEPS=${WARMUP},\\"
echo "    CHECKPOINT_EVERY=${CHECKPOINT_EVERY},\\"
echo "    EVAL_EVERY=${EVAL_EVERY},\\"
echo "    LOG_EVERY=${LOG_EVERY},\\"
echo "    USE_MIXED_PRECISION=1 \\"
echo "    ${CLUSTER_SCRIPT}"
echo ""

# Actually submit the job
JOB_OUTPUT=$(sbatch \
    --export=ALL,DATASET=${DATASET},EPOCHS=${EPOCHS},BATCH_SIZE=${BATCH_SIZE},GRAD_ACCUM_STEPS=${GRAD_ACCUM},LR=${LR},TEMPORAL_WEIGHT=0.3,QUADRUPLET_MARGIN=0.5,WARMUP_STEPS=${WARMUP},CHECKPOINT_EVERY=${CHECKPOINT_EVERY},EVAL_EVERY=${EVAL_EVERY},LOG_EVERY=${LOG_EVERY},USE_MIXED_PRECISION=1 \
    ${CLUSTER_SCRIPT} 2>&1)

# Extract job ID
if [[ $? -eq 0 ]]; then
    JOB_ID=$(echo "$JOB_OUTPUT" | grep -oE '[0-9]+' | head -1)
    echo "Job submitted successfully!"
    echo "Job ID: ${JOB_ID}"
    echo ""
    echo "Monitoring:"
    echo "  - Check status: squeue -u ${USER}"
    echo "  - View logs: tail -f /vol/bitbucket/${USER}/temporal_pathrag_logs/train_embeddings_${JOB_ID}.out"
    echo "  - Cancel job: scancel ${JOB_ID}"
    echo ""
    echo "Expected outcomes:"
    echo "  - Best model saved to: ${BEST_MODEL_DIR}"
    echo "  - Checkpoints saved to: /vol/bitbucket/${USER}/temporal_embeddings/runs/job_${JOB_ID}/checkpoints"
    echo ""
    echo "After training completes, run evaluation:"
    echo "  python scripts/evaluate_temporal_embeddings.py --model_path ${BEST_MODEL_DIR}"
else
    echo "Job submission failed!"
    echo "Error: ${JOB_OUTPUT}"
    exit 1
fi