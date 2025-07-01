#!/bin/bash

set -e

CLUSTER_HOST="gpucluster2.doc.ic.ac.uk"
CLUSTER_USER="cih124"
PROJECT_NAME="Temporal_PathRAG"
LOCAL_PROJECT_PATH="/Users/hillcallum/Temporal_PathRAG"
LOCAL_LOGS_DIR="/Users/hillcallum/Temporal_PathRAG/logs"

echo "TimeR4 Prompt Generation"
echo "===================================================="
echo ""
echo "Target: match with original TimeR4 datasets"
echo "Expected Output:"
echo " - MultiTQ train: ~77,357 prompts (~80MB)"
echo " - MultiTQ test: ~54,584 prompts (~75MB)"
echo " - TimeQuestions train: ~9,708 prompts (~10MB)"
echo " - TimeQuestions test: ~3,237 prompts (~1MB)"
echo " - Total: ~145,000 prompts (~165MB)"
echo ""

# 1. Sync code to cluster
echo "1. Syncing code to cluster"
rsync -avz --delete \
    --exclude='venv/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='.git/' \
    --exclude='logs/' \
    --exclude='reports/' \
    "$LOCAL_PROJECT_PATH/" \
    "$CLUSTER_USER@$CLUSTER_HOST:/vol/bitbucket/$CLUSTER_USER/$PROJECT_NAME/"

# 2. Submit job
echo "2. Submitting SLURM job"
JOB_OUTPUT=$(ssh "$CLUSTER_USER@$CLUSTER_HOST" "cd /vol/bitbucket/$CLUSTER_USER/$PROJECT_NAME/slurm && sbatch generate_prompts.sh")
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
echo "TimeR4 Job $JOB_ID is running"
echo "Cluster: $CLUSTER_HOST (gpgpuB partition)"
echo "================================================"

# 4. Ask for monitoring preference
read -p "Monitor job and auto-copy results when done? (y/n): " monitor_choice

if [[ $monitor_choice =~ ^[Yy]$ ]]; then
    echo ""
    echo "Monitoring job $JOB_ID"
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
            sleep 60  # Check every minute for long-running jobs
        fi
    done
    
    # 5. Auto-copy results
    echo ""
    echo "Copying results and logs to local machine"
    
    # Create local logs directory
    mkdir -p "$LOCAL_LOGS_DIR"
    
    # Copy logs with timestamp
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    LOCAL_OUT="$LOCAL_LOGS_DIR/timeR4_prompts_${JOB_ID}_${TIMESTAMP}.out"
    LOCAL_ERR="$LOCAL_LOGS_DIR/timeR4_prompts_${JOB_ID}_${TIMESTAMP}.err"
    
    scp "$CLUSTER_USER@$CLUSTER_HOST:/vol/bitbucket/$CLUSTER_USER/temporal_pathrag_logs/timeR4_full_prompts_${JOB_ID}.out" "$LOCAL_OUT" 2>/dev/null || echo "Warning: No output log found"
    scp "$CLUSTER_USER@$CLUSTER_HOST:/vol/bitbucket/$CLUSTER_USER/temporal_pathrag_logs/timeR4_full_prompts_${JOB_ID}.err" "$LOCAL_ERR" 2>/dev/null || echo "Warning: No error log found"
    
    # Copy generated prompt files
    echo "Copying generated prompt files"
    rsync -avz "$CLUSTER_USER@$CLUSTER_HOST:/vol/bitbucket/$CLUSTER_USER/$PROJECT_NAME/datasets/*/prompt/*_prompt.json" "$LOCAL_PROJECT_PATH/datasets/" --relative || echo "Warning: Some prompt files may not be ready yet"
    
    echo ""
    echo "Results copied to:"
    echo "Output log: $LOCAL_OUT"
    echo "Error log:  $LOCAL_ERR"
    echo "Prompt files: $LOCAL_PROJECT_PATH/datasets/*/prompt/"
    
    # Show quick summary
    if [ -f "$LOCAL_OUT" ]; then
        echo ""
        echo "Job Summary:"
        echo "==============="
        grep -E "(prompts|MB|entries)" "$LOCAL_OUT" | tail -10 || echo "Check log files for details"
        
        # Show final prompt counts if available
        echo ""
        echo "Generated Prompt Counts:"
        echo "=========================="
        for dataset in MultiTQ TimeQuestions; do
            for split in train test; do
                prompt_file="$LOCAL_PROJECT_PATH/datasets/$dataset/prompt/${split}_prompt.json"
                if [ -f "$prompt_file" ]; then
                    count=$(python3 -c "import json; print(len(json.load(open('$prompt_file'))))" 2>/dev/null || echo "error")
                    size=$(ls -lh "$prompt_file" | awk '{print $5}')
                    echo "$dataset $split: $count prompts ($size)"
                fi
            done
        done
    fi
    
    # Final status
    if [ "$STATUS" = "COMPLETED" ]; then
        echo ""
        echo "Success - TimeR4 prompt generation completed"
        echo "Check datasets/*/prompt/ for the generated files"
    else
        echo ""
        echo "Job finished with status: $STATUS"
        echo "Check log files for error details"
    fi
    
else
    echo ""
    echo "Job is running in background. To check later:"
    echo "ssh $CLUSTER_USER@$CLUSTER_HOST 'squeue --me'"
    echo ""
    echo "To copy results manually when job completes:"
    echo "rsync -avz $CLUSTER_USER@$CLUSTER_HOST:/vol/bitbucket/$CLUSTER_USER/$PROJECT_NAME/datasets/*/prompt/ ./datasets/ --relative"
    echo "scp $CLUSTER_USER@$CLUSTER_HOST:/vol/bitbucket/$CLUSTER_USER/temporal_pathrag_logs/timeR4_full_prompts_*.out ./logs/"
fi

echo ""
echo "Done"