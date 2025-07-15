#!/bin/bash
# Sync local code to Imperial GPU cluster

# Configuration
CLUSTER_USER="${CLUSTER_USER:-cih124}"
CLUSTER_HOST="${CLUSTER_HOST:-gpucluster2.doc.ic.ac.uk}"
BASE_DIR="/vol/bitbucket/${CLUSTER_USER}/Temporal_PathRAG"

# Parse arguments
CLEAN_OLD_LOGS=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --clean)
            CLEAN_OLD_LOGS=true
            shift
            ;;
        *)
            shift
            ;;
    esac
done

echo "Syncing code to Imperial GPU cluster"
echo "User: ${CLUSTER_USER}"
echo "Host: ${CLUSTER_HOST}"
echo "Destination: ${BASE_DIR}"
echo ""

# Clean up old logs and temp files if requested
if [ "$CLEAN_OLD_LOGS" = true ]; then
    echo "Cleaning up old files on cluster"
    ssh ${CLUSTER_USER}@${CLUSTER_HOST} << 'EOF'
        # Remove old training logs (older than 7 days)
        find /vol/bitbucket/${USER}/temporal_pathrag_logs -name "*.out" -mtime +7 -delete 2>/dev/null
        find /vol/bitbucket/${USER}/temporal_pathrag_logs -name "*.err" -mtime +7 -delete 2>/dev/null
        
        # Remove temporary test scripts
        find /vol/bitbucket/${USER}/Temporal_PathRAG -name "test_*.py" -delete 2>/dev/null
        find /vol/bitbucket/${USER}/Temporal_PathRAG -name "debug_*.py" -delete 2>/dev/null
        find /vol/bitbucket/${USER}/Temporal_PathRAG/slurm -name "test_*.sh" -delete 2>/dev/null
        find /vol/bitbucket/${USER}/Temporal_PathRAG/slurm -name "debug_*.sh" -delete 2>/dev/null
        
        # Clean Python cache
        find /vol/bitbucket/${USER}/Temporal_PathRAG -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
        find /vol/bitbucket/${USER}/Temporal_PathRAG -name "*.pyc" -delete 2>/dev/null
        
        echo "Cleanup completed"
EOF
fi

# Create directory structure on cluster
echo "Creating directory structure"
ssh ${CLUSTER_USER}@${CLUSTER_HOST} "mkdir -p ${BASE_DIR}"

# Change to bitbucket directory before operations
ssh ${CLUSTER_USER}@${CLUSTER_HOST} "cd /vol/bitbucket/${CLUSTER_USER}"

# Sync code (excluding large files and caches)
echo "Syncing code"
rsync -avz --progress \
    --exclude='*.pyc' \
    --exclude='__pycache__' \
    --exclude='.git' \
    --exclude='datasets/*/kg/*.txt' \
    --exclude='analysis_results' \
    --exclude='cache' \
    --exclude='.venv' \
    --exclude='venv/' \
    --exclude='*.pkl' \
    --exclude='*.pth' \
    --exclude='*.bin' \
    --exclude='timeR4_datasets/' \
    --exclude='raw_datasets/' \
    --exclude='logs/*.log' \
    --exclude='test_results/' \
    --exclude='wandb/' \
    --exclude='models/' \
    --exclude='checkpoints/' \
    --include='data/training/***' \
    . ${CLUSTER_USER}@${CLUSTER_HOST}:${BASE_DIR}/

if [ $? -eq 0 ]; then
    echo "Code sync completed successfully"
else
    echo "Code sync completed with warnings"
fi