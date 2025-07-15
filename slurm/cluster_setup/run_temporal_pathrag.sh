#!/bin/bash

set -e

CLUSTER_HOST="gpucluster2.doc.ic.ac.uk"
CLUSTER_USER="cih124"
PROJECT_NAME="Temporal_PathRAG"
LOCAL_PROJECT_PATH="/Users/hillcallum/Temporal_PathRAG"
LOCAL_LOGS_DIR="/Users/hillcallum/Temporal_PathRAG/logs"

# Use clean project name in bitbucket 
CLUSTER_PROJECT_PATH="/vol/bitbucket/$CLUSTER_USER/${PROJECT_NAME}"

echo "Temporal PathRAG with LLM Execution"
echo ""

SCRIPT_NAME="run_temporal_pathrag_with_llm.sh"
JOB_PREFIX="temporal_pathrag_llm"

# 1. Clean up and sync code to cluster
echo ""
echo "1. Cleaning up cluster space and syncing code"

echo "Aggressive cleanup"
ssh "$CLUSTER_USER@$CLUSTER_HOST" "
    echo 'Current bitbucket usage:'
    df -h /vol/bitbucket
    
    echo 'Performing aggressive cleanup - given lack of space on bitbucket (can probably remove in future assuming that space gets cleared)'
    # Remove everything possible to free space
    rm -rf /vol/bitbucket/$CLUSTER_USER/python_packages 2>/dev/null || true
    rm -rf /vol/bitbucket/$CLUSTER_USER/hf_cache 2>/dev/null || true
    rm -rf /vol/bitbucket/$CLUSTER_USER/.cache 2>/dev/null || true
    rm -rf /vol/bitbucket/$CLUSTER_USER/.local 2>/dev/null || true
    rm -rf /vol/bitbucket/$CLUSTER_USER/temporal_pathrag_env 2>/dev/null || true
    rm -rf /vol/bitbucket/$CLUSTER_USER/Temporal_PathRAG 2>/dev/null || true
    rm -rf /vol/bitbucket/$CLUSTER_USER/temporal_pathrag_* 2>/dev/null || true
    
    # Remove old logs and checkpoints
    find /vol/bitbucket/$CLUSTER_USER -name '*.out' -delete 2>/dev/null || true
    find /vol/bitbucket/$CLUSTER_USER -name '*.err' -delete 2>/dev/null || true
    find /vol/bitbucket/$CLUSTER_USER -name '*checkpoint*' -delete 2>/dev/null || true
    find /vol/bitbucket/$CLUSTER_USER -name '*.pkl' -delete 2>/dev/null || true
    find /vol/bitbucket/$CLUSTER_USER -name '*.json' -size +1M -delete 2>/dev/null || true
    
    echo 'Creating project structure'
    mkdir -p /vol/bitbucket/$CLUSTER_USER/$PROJECT_NAME
    mkdir -p /vol/bitbucket/$CLUSTER_USER/logs
    
    echo 'Space after cleanup:'
    df -h /vol/bitbucket
"

# Sync repository including datasets
echo "Syncing repository to cluster"
rsync -avz --size-only \
    --exclude='venv/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='.git/' \
    --exclude='logs/' \
    --exclude='test_results/' \
    --exclude='raw_datasets/' \
    --exclude='timeR4_datasets/' \
    "$LOCAL_PROJECT_PATH/" \
    "$CLUSTER_USER@$CLUSTER_HOST:/vol/bitbucket/$CLUSTER_USER/$PROJECT_NAME/"

echo "Code synced to cluster"

# 2. Submit job
echo ""
echo "2. Submitting SLURM job: $SCRIPT_NAME"
JOB_OUTPUT=$(ssh "$CLUSTER_USER@$CLUSTER_HOST" "cd /vol/bitbucket/$CLUSTER_USER/$PROJECT_NAME/slurm/job_scripts && sbatch $SCRIPT_NAME")
JOB_ID=$(echo "$JOB_OUTPUT" | grep "Submitted batch job" | awk '{print $4}')

if [ -z "$JOB_ID" ]; then
    echo "Failed to submit job"
    echo "Output: $JOB_OUTPUT"
    exit 1
fi

echo "Job submitted with ID: $JOB_ID"

# 3. Show initial status
echo ""
echo "3. Initial job status:"
ssh "$CLUSTER_USER@$CLUSTER_HOST" "squeue --job=$JOB_ID"

echo ""
echo "Temporal PathRAG Job $JOB_ID is running on the cluster"
echo ""

# 4. Enhanced monitoring with real-time updates
read -p "Monitor job with real-time updates? (y/n): " monitor_choice

if [[ $monitor_choice =~ ^[Yy]$ ]]; then
    echo ""
    echo "Monitoring job $JOB_ID with real-time updates"
    echo "(Ctrl+C to stop monitoring - job continues running)"
    
    # Function to get job status
    get_job_status() {
        ssh "$CLUSTER_USER@$CLUSTER_HOST" "squeue --job=$JOB_ID --noheader --format='%T %M %l %R' 2>/dev/null" || echo "Completed 00:00:00 4:00:00 completed"
    }
    
    # Function to get recent log output
    get_recent_output() {
        ssh "$CLUSTER_USER@$CLUSTER_HOST" "tail -n 10 /home/$CLUSTER_USER/temporal_pathrag_logs/${JOB_PREFIX}_${JOB_ID}.out 2>/dev/null" || echo "No output yet"
    }
    
    # Monitor job until completion
    while true; do
        STATUS_LINE=$(get_job_status)
        STATUS=$(echo "$STATUS_LINE" | awk '{print $1}')
        RUNTIME=$(echo "$STATUS_LINE" | awk '{print $2}')
        TIMELIMIT=$(echo "$STATUS_LINE" | awk '{print $3}')
        
        if [ "$STATUS" = "Completed" ] || [ "$STATUS" = "Failed" ] || [ "$STATUS" = "Cancelled" ] || [ -z "$STATUS" ]; then
            echo "$(date '+%H:%M:%S'): Job $JOB_ID finished with status: ${STATUS:-COMPLETED}"
            break
        else
            echo "$(date '+%H:%M:%S'): Job $JOB_ID - Status: $STATUS, Runtime: $RUNTIME, Limit: $TIMELIMIT"
            
            # Show recent output every 2 minutes
            if [ $(($(date +%s) % 120)) -eq 0 ]; then
                echo "--- Recent output ---"
                get_recent_output
                echo "--- End recent output ---"
            fi
            
            sleep 30  # Check every 30 seconds
        fi
    done
    
    # 5. Auto-copy results
    echo ""
    echo "Copying results to local logs"
    
    # Create local logs directory with job-specific sub-directory
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    JOB_LOG_DIR="$LOCAL_LOGS_DIR/${JOB_PREFIX}_${JOB_ID}_${TIMESTAMP}"
    mkdir -p "$JOB_LOG_DIR"
    
    # Copy all related files - try multiple locations
    echo "Copying job outputs"
    # Try bitbucket location first
    scp "$CLUSTER_USER@$CLUSTER_HOST:/vol/bitbucket/$CLUSTER_USER/temporal_pathrag_logs/${JOB_PREFIX}_${JOB_ID}.out" "$JOB_LOG_DIR/" 2>/dev/null || \
    # Try default SLURM location
    scp "$CLUSTER_USER@$CLUSTER_HOST:/vol/bitbucket/$CLUSTER_USER/slurm-${JOB_ID}.out" "$JOB_LOG_DIR/" 2>/dev/null || \
    # Try current directory
    scp "$CLUSTER_USER@$CLUSTER_HOST:/vol/bitbucket/$CLUSTER_USER/Temporal_PathRAG/slurm-${JOB_ID}.out" "$JOB_LOG_DIR/" 2>/dev/null || \
    echo "Warning: No output log found"
    
    scp "$CLUSTER_USER@$CLUSTER_HOST:/vol/bitbucket/$CLUSTER_USER/temporal_pathrag_logs/${JOB_PREFIX}_${JOB_ID}.err" "$JOB_LOG_DIR/" 2>/dev/null || \
    scp "$CLUSTER_USER@$CLUSTER_HOST:/vol/bitbucket/$CLUSTER_USER/slurm-${JOB_ID}.err" "$JOB_LOG_DIR/" 2>/dev/null || \
    scp "$CLUSTER_USER@$CLUSTER_HOST:/vol/bitbucket/$CLUSTER_USER/Temporal_PathRAG/slurm-${JOB_ID}.err" "$JOB_LOG_DIR/" 2>/dev/null || \
    echo "Warning: No error log found"
    
    # Copy checkpoints if they exist
    echo "Copying checkpoints"
    scp -r "$CLUSTER_USER@$CLUSTER_HOST:/vol/bitbucket/$CLUSTER_USER/temporal_pathrag_checkpoints" "$JOB_LOG_DIR/" 2>/dev/null || echo "No checkpoints found"
    
    echo ""
    echo "Results copied to: $JOB_LOG_DIR"
    echo ""
    
    # Show summary
    echo "Job Summar"
    echo "Job ID: $JOB_ID"
    echo "Script: $SCRIPT_NAME"
    echo "Final Status: ${STATUS:-COMPLETED}"
    echo "Results Directory: $JOB_LOG_DIR"
    echo ""
    
    # Show output summary
    OUTPUT_FILE="$JOB_LOG_DIR/${JOB_PREFIX}_${JOB_ID}.out"
    if [ -f "$OUTPUT_FILE" ]; then
        echo "Output Summary"
        echo "Last 20 lines of output:"
        tail -n 20 "$OUTPUT_FILE"
        echo ""
        
        # Look for key indicators
        echo "Key Indicators"
        grep -E "(Yes|No|Error|Sucess|Faol)" "$OUTPUT_FILE" | tail -10 || echo "No key indicators found"
    fi
    
    # Show error summary if exists
    ERROR_FILE="$JOB_LOG_DIR/${JOB_PREFIX}_${JOB_ID}.err"
    if [ -f "$ERROR_FILE" ] && [ -s "$ERROR_FILE" ]; then
        echo ""
        echo "Error Summary"
        echo "Errors found in error log:"
        tail -n 10 "$ERROR_FILE"
    fi
    
else
    echo ""
    echo "Job is running in background - to check later:"
    echo "ssh $CLUSTER_USER@$CLUSTER_HOST 'squeue --me'"
    echo ""
    echo "To copy results manually:"
    echo "scp $CLUSTER_USER@$CLUSTER_HOST:/home/$CLUSTER_USER/temporal_pathrag_logs/${JOB_PREFIX}_*.out ./logs/"
    echo "scp $CLUSTER_USER@$CLUSTER_HOST:/home/$CLUSTER_USER/temporal_pathrag_logs/${JOB_PREFIX}_*.err ./logs/"
fi

echo ""
echo "Next Steps"
echo "1. Review the output logs for your results"
echo "2. Check checkpoints for progress tracking"
echo "3. Use interactive_debug.sh for debugging if needed"

echo ""
echo "Done"