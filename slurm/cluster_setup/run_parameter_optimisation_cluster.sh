#!/bin/bash

set -e

CLUSTER_HOST="gpucluster2.doc.ic.ac.uk"
CLUSTER_USER="cih124"
PROJECT_NAME="Temporal_PathRAG"
LOCAL_PROJECT_PATH="/Users/hillcallum/Temporal_PathRAG"
LOCAL_LOGS_DIR="/Users/hillcallum/Temporal_PathRAG/logs"

echo "Temporal PathRAG Parameter Optimisation Cluster Run"
echo "=================================================="

# 1. Sync code to cluster
echo "1. Syncing code to cluster"
rsync -avz --delete \
    --exclude='venv/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='.git/' \
    --exclude='logs/' \
    --exclude='reports/' \
    --exclude='test_results/' \
    "$LOCAL_PROJECT_PATH/" \
    "$CLUSTER_USER@$CLUSTER_HOST:/vol/bitbucket/$CLUSTER_USER/$PROJECT_NAME/"

# 2. Submit parameter optimisation job
echo "2. Submitting SLURM parameter optimisation job"
JOB_OUTPUT=$(ssh "$CLUSTER_USER@$CLUSTER_HOST" "cd /vol/bitbucket/$CLUSTER_USER/$PROJECT_NAME/slurm/job_scripts && sbatch run_parameter_optimisation.sh")
JOB_ID=$(echo "$JOB_OUTPUT" | grep "Submitted batch job" | awk '{print $4}')

if [ -z "$JOB_ID" ]; then
    echo "Failed to submit job"
    exit 1
fi

echo "Parameter optimisation job submitted with ID: $JOB_ID"

# 3. Show initial status
echo "3. Initial job status:"
ssh "$CLUSTER_USER@$CLUSTER_HOST" "squeue --job=$JOB_ID"

echo ""
echo "================================================"
echo "Parameter optimisation job $JOB_ID is running on the cluster"
echo "================================================"

# 4. Ask for monitoring preference
read -p "Monitor job and auto-copy results when done? (y/n): " monitor_choice

if [[ $monitor_choice =~ ^[Yy]$ ]]; then
    echo ""
    echo "Monitoring parameter optimisation job $JOB_ID"
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
            sleep 60  # Check every minute for long-running optimisation
        fi
    done
    
    # 5. Auto-copy results
    echo ""
    echo "Copying parameter optimisation results to local directory"
    
    # Create local logs and results directories
    mkdir -p "$LOCAL_LOGS_DIR"
    mkdir -p "$LOCAL_PROJECT_PATH/test_results"
    
    # Copy logs with timestamp
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    LOCAL_OUT="$LOCAL_LOGS_DIR/param_opt_${JOB_ID}_${TIMESTAMP}.out"
    LOCAL_ERR="$LOCAL_LOGS_DIR/param_opt_${JOB_ID}_${TIMESTAMP}.err"
    
    scp "$CLUSTER_USER@$CLUSTER_HOST:/vol/bitbucket/$CLUSTER_USER/temporal_pathrag_logs/param_opt_${JOB_ID}.out" "$LOCAL_OUT" 2>/dev/null || echo "Warning: No output log found"
    scp "$CLUSTER_USER@$CLUSTER_HOST:/vol/bitbucket/$CLUSTER_USER/temporal_pathrag_logs/param_opt_${JOB_ID}.err" "$LOCAL_ERR" 2>/dev/null || echo "Warning: No error log found"
    
    # Copy JSON results
    echo "Copying optimisation results"
    scp "$CLUSTER_USER@$CLUSTER_HOST:/vol/bitbucket/$CLUSTER_USER/temporal_pathrag_logs/param_opt_${JOB_ID}_model.json" "$LOCAL_PROJECT_PATH/test_results/" 2>/dev/null && echo "Optimisation results copied"
    scp "$CLUSTER_USER@$CLUSTER_HOST:/vol/bitbucket/$CLUSTER_USER/temporal_pathrag_logs/param_opt_${JOB_ID}_temporal.json" "$LOCAL_PROJECT_PATH/test_results/" 2>/dev/null && echo "Temporal path results copied"
    
    echo ""
    echo "Results copied to:"
    echo "Output log: $LOCAL_OUT"
    echo "Error log:  $LOCAL_ERR"
    echo "Results:    $LOCAL_PROJECT_PATH/test_results/"
    
    # Show optimisation summary
    if [ -f "$LOCAL_OUT" ]; then
        echo ""
        echo "Parameter Optimisation Summary:"
        echo "==============================="
        
        # Extract key results from output log
        grep -E "(Optimal Alpha|Optimal Theta|Reliability|Success Rate|Parameter Optimisation.*Successfully|Parameter Optimisation.*Failed)" "$LOCAL_OUT" | tail -10
        
        echo ""
        echo "For detailed results, check the JSON files in test_results/"
    fi
    
    # Parse and display optimal parameters if available
    MODEL_REUSE_RESULTS="$LOCAL_PROJECT_PATH/test_results/param_opt_${JOB_ID}_model.json"
    
    if [ -f "$MODEL_REUSE_RESULTS" ]; then
        echo ""
        echo "Parameter Optimisation Results:"
        echo "==========================================="
        python3 -c "
import json
import sys
try:
    with open('$MODEL_REUSE_RESULTS', 'r') as f:
        report = json.load(f)
    
    optimal = report.get('optimal_parameters', {})
    alpha = optimal.get('alpha', 'N/A')
    theta = optimal.get('theta', 'N/A')
    
    training = report.get('performance_metrics', {}).get('training', {})
    reliability = training.get('avg_reliability', 'N/A')
    success_rate = training.get('success_rate', 'N/A')
    ci_lower = training.get('reliability_ci_lower', 'N/A')
    ci_upper = training.get('reliability_ci_upper', 'N/A')
    execution_time = training.get('execution_time', 'N/A')
    
    validation = report.get('performance_metrics', {}).get('validation', {})
    generalisation = validation.get('generalisation', 'N/A')
    val_reliability = validation.get('avg_val_reliability', 'N/A')
    
    metadata = report.get('optimisation_metadata', {})
    total_combinations = metadata.get('total_combinations_tested', 'N/A')
    successful_combinations = metadata.get('successful_combinations', 'N/A')
    
    print(f'Recommended Optimal Parameters:')
    print(f'Alpha (temporal decay): {alpha}')
    print(f'Theta (pruning threshold): {theta}')
    print(f'')
    print(f'Performance Metrics:')
    print(f'Training reliability: {reliability}')
    print(f'95% Confidence interval: [{ci_lower}, {ci_upper}]')
    print(f'Success rate: {success_rate}')
    print(f'Execution time: {execution_time}s per query')
    print(f'')
    print(f'Validation Results:')
    print(f'Validation reliability: {val_reliability}')
    print(f'Generalisation assessment: {generalisation}')
    print(f'')
    print(f'Optimisation Statistics:')
    print(f'Parameter combinations tested: {total_combinations}')
    print(f'Successful combinations: {successful_combinations}')
        
except Exception as e:
    print(f'Could not parse model results: {e}')
    sys.exit(1)
" 2>/dev/null
    fi
    
else
    echo ""
    echo "Job is running in background. To check later:"
    echo "ssh $CLUSTER_USER@$CLUSTER_HOST 'squeue --me'"
    echo ""
    echo "To copy results manually when job completes:"
    echo "scp $CLUSTER_USER@$CLUSTER_HOST:/vol/bitbucket/$CLUSTER_USER/temporal_pathrag_logs/param_opt_*.out ./logs/"
    echo "scp $CLUSTER_USER@$CLUSTER_HOST:/vol/bitbucket/$CLUSTER_USER/temporal_pathrag_logs/param_opt_*.json ./test_results/"
fi

echo ""
echo "Done"