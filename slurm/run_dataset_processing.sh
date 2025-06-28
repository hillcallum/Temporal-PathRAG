#!/bin/bash

set -e

CLUSTER_HOST="gpucluster2.doc.ic.ac.uk"
CLUSTER_USER="cih124"
PROJECT_NAME="Temporal_PathRAG"
LOCAL_PROJECT_PATH="/Users/hillcallum/Temporal_PathRAG"
LOCAL_LOGS_DIR="/Users/hillcallum/Temporal_PathRAG/logs"

echo "Dataset Processing Pipeline"
echo "==========================="

# 1. Sync code to cluster
echo "1. Syncing code to cluster"
rsync -avz --delete \
    --exclude='venv/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='.git/' \
    --exclude='logs/' \
    --exclude='reports/' \
    --exclude='datasets/' \
    "$LOCAL_PROJECT_PATH/" \
    "$CLUSTER_USER@$CLUSTER_HOST:/vol/bitbucket/$CLUSTER_USER/$PROJECT_NAME/"

# 2. Submit job
echo "2. Submitting dataset processing job"
JOB_OUTPUT=$(ssh "$CLUSTER_USER@$CLUSTER_HOST" "cd /vol/bitbucket/$CLUSTER_USER/$PROJECT_NAME && sbatch slurm/process_datasets.sh")
JOB_ID=$(echo "$JOB_OUTPUT" | grep "Submitted batch job" | awk '{print $4}')

if [ -z "$JOB_ID" ]; then
    echo "Failed to submit job"
    exit 1
fi

echo "Job submitted with ID: $JOB_ID"

# 3. Show initial status
echo "3. Initial job status:"
ssh "$CLUSTER_USER@$CLUSTER_HOST" "squeue --job=$JOB_ID"

echo ""
echo "================================================"
echo "Dataset processing job $JOB_ID is running"
echo "================================================"

# 4. Ask for monitoring preference
read -p "Monitor job and auto-copy results when done? (y/n): " monitor_choice

if [[ $monitor_choice =~ ^[Yy]$ ]]; then
    echo ""
    echo "Monitoring dataset processing job $JOB_ID"
    echo "(Ctrl+C to stop monitoring - job continues running)"
    echo "======================================================================"
    
    # Monitor job until completion
    while true; do
        STATUS=$(ssh "$CLUSTER_USER@$CLUSTER_HOST" "squeue --job=$JOB_ID --noheader --format='%T' 2>/dev/null" || echo "COMPLETED")
        
        if [ "$STATUS" = "COMPLETED" ] || [ "$STATUS" = "FAILED" ] || [ "$STATUS" = "CANCELLED" ] || [ -z "$STATUS" ]; then
            echo "Job $JOB_ID finished with status: ${STATUS:-COMPLETED}"
            break
        else
            echo "$(date '+%H:%M:%S'): Job $JOB_ID status: $STATUS"
            sleep 30  # Check every 30 seconds
        fi
    done
    
    # 5. Auto-copy results
    echo ""
    echo "Copying results and logs to local machine"
    
    # Create local logs directory
    mkdir -p "$LOCAL_LOGS_DIR"
    
    # Copy logs with timestamp
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    LOCAL_OUT="$LOCAL_LOGS_DIR/dataset_processing_${JOB_ID}_${TIMESTAMP}.out"
    LOCAL_ERR="$LOCAL_LOGS_DIR/dataset_processing_${JOB_ID}_${TIMESTAMP}.err"
    
    scp "$CLUSTER_USER@$CLUSTER_HOST:/vol/bitbucket/$CLUSTER_USER/temporal_pathrag_logs/dataset_processing_${JOB_ID}.out" "$LOCAL_OUT" 2>/dev/null || echo "Warning: No output log found"
    scp "$CLUSTER_USER@$CLUSTER_HOST:/vol/bitbucket/$CLUSTER_USER/temporal_pathrag_logs/dataset_processing_${JOB_ID}.err" "$LOCAL_ERR" 2>/dev/null || echo "Warning: No error log found"
    
    # Copy processed datasets
    echo "Copying processed datasets"
    mkdir -p datasets
    rsync -avz "$CLUSTER_USER@$CLUSTER_HOST:/vol/bitbucket/$CLUSTER_USER/$PROJECT_NAME/datasets/" ./datasets/
    
    echo ""
    echo "Results copied to:"
    echo "Output log: $LOCAL_OUT"
    echo "Error log:  $LOCAL_ERR"
    echo "Datasets:   ./datasets/"
    
    # Show processing summary
    if [ -f "$LOCAL_OUT" ]; then
        echo ""
        echo "Processing Summary:"
        echo "=================="
        grep -E "(processing completed|Dataset Summary)" "$LOCAL_OUT" | tail -10
    fi
    
else
    echo ""
    echo "Job is running in background. To check later:"
    echo "ssh $CLUSTER_USER@$CLUSTER_HOST 'squeue --me'"
    echo ""
    echo "To copy results manually when complete:"
    echo "rsync -avz $CLUSTER_USER@$CLUSTER_HOST:/vol/bitbucket/$CLUSTER_USER/$PROJECT_NAME/datasets/ ./datasets/"
fi

echo ""
echo "Done!"