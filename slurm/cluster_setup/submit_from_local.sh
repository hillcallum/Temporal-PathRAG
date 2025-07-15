#!/bin/bash
# Submit training job from local machine (handles sync and submission)

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check if a training config script was provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <training_config_script> [--clean]"
    echo ""
    echo "Examples:"
    echo "  From cluster_setup directory:"
    echo "    ./submit_from_local.sh training/submit_15_Jul_1_training.sh"
    echo ""
    echo "  From any directory:"
    echo "    ~/Temporal_PathRAG/slurm/cluster_setup/submit_from_local.sh training/submit_15_Jul_1_training.sh"
    echo ""
    echo "  With cleanup of old files:"
    echo "    ./submit_from_local.sh training/submit_15_Jul_1_training.sh --clean"
    echo ""
    echo "Available training configs:"
    ls -1 "$SCRIPT_DIR/training/" 2>/dev/null | grep "^submit_.*\.sh$" | sed 's/^/    training\//' || echo "    (none found)"
    exit 1
fi

TRAINING_CONFIG="$1"
CLEAN_FLAG=""

# Check for --clean flag
if [ "$2" == "--clean" ]; then
    CLEAN_FLAG="--clean"
fi

# Configuration
CLUSTER_USER="${CLUSTER_USER:-cih124}"
CLUSTER_HOST="${CLUSTER_HOST:-gpucluster2.doc.ic.ac.uk}"

echo "Temporal PathRAG Training Submission"
echo "Training config: $TRAINING_CONFIG"
echo ""

# Step 1: Sync code to cluster
echo "Step 1: Syncing code to cluster"
"$SCRIPT_DIR/sync_to_cluster.sh" $CLEAN_FLAG

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to sync code to cluster"
    exit 1
fi

echo ""
echo "Step 2: Submitting training job"

# Step 2: SSH to cluster and run the training config script
ssh ${CLUSTER_USER}@${CLUSTER_HOST} << EOF
    cd /vol/bitbucket/${CLUSTER_USER}/Temporal_PathRAG
    ./slurm/cluster_setup/${TRAINING_CONFIG}
EOF

echo ""
echo "Submission complete"
echo ""
echo "To download results after training completes:"
echo "  ./download_results.sh --best                    # Download global best model"
echo "  ./download_results.sh --job <JOB_ID>           # Download specific job results"