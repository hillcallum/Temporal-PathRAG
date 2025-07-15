#!/bin/bash
# Download training results from cluster to local repository

# Configuration
CLUSTER_USER="${CLUSTER_USER:-cih124}"
CLUSTER_HOST="${CLUSTER_HOST:-gpucluster2.doc.ic.ac.uk}"
LOCAL_MODELS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/models/trained"

# Parse arguments
JOB_ID=""
DOWNLOAD_BEST=false
DOWNLOAD_ALL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --job)
            JOB_ID="$2"
            shift 2
            ;;
        --best)
            DOWNLOAD_BEST=true
            shift
            ;;
        --all)
            DOWNLOAD_ALL=true
            shift
            ;;
        *)
            echo "Usage: $0 [--job JOB_ID] [--best] [--all]"
            echo ""
            echo "Options:"
            echo "  --job JOB_ID  Download results from specific job"
            echo "  --best        Download only the global best model"
            echo "  --all         Download all checkpoints (warning: large)"
            echo ""
            echo "Examples:"
            echo "  ./download_results.sh --job 182642"
            echo "  ./download_results.sh --best"
            exit 1
            ;;
    esac
done

# Create local models directory
mkdir -p "$LOCAL_MODELS_DIR"
echo "Local models directory: $LOCAL_MODELS_DIR"

# Download global best model
if [ "$DOWNLOAD_BEST" = true ] || [ -z "$JOB_ID" ]; then
    echo ""
    echo "Downloading global best model"
    
    # Create best model directory with timestamp
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    BEST_DIR="$LOCAL_MODELS_DIR/best_model_${TIMESTAMP}"
    mkdir -p "$BEST_DIR"
    
    # Download best model and info
    scp ${CLUSTER_USER}@${CLUSTER_HOST}:/vol/bitbucket/${CLUSTER_USER}/temporal_embeddings/best_model/best_model.pt "$BEST_DIR/" 2>/dev/null
    scp ${CLUSTER_USER}@${CLUSTER_HOST}:/vol/bitbucket/${CLUSTER_USER}/temporal_embeddings/best_model/best_model_info.json "$BEST_DIR/" 2>/dev/null
    
    if [ -f "$BEST_DIR/best_model.pt" ]; then
        echo "Downloaded best model to: $BEST_DIR"
        
        # Create symlink to latest best model
        ln -sfn "$BEST_DIR" "$LOCAL_MODELS_DIR/best_model_latest"
        echo "Created symlink: best_model_latest -> $(basename $BEST_DIR)"
        
        # Show model info
        if [ -f "$BEST_DIR/best_model_info.json" ]; then
            echo ""
            echo "Best model info:"
            cat "$BEST_DIR/best_model_info.json"
        fi
    else
        echo "No global best model found on cluster"
        rmdir "$BEST_DIR" 2>/dev/null
    fi
fi

# Download specific job results
if [ -n "$JOB_ID" ]; then
    echo ""
    echo "Downloading results from job $JOB_ID"
    
    JOB_DIR="$LOCAL_MODELS_DIR/job_${JOB_ID}"
    mkdir -p "$JOB_DIR"
    
    # Download job info
    scp ${CLUSTER_USER}@${CLUSTER_HOST}:/vol/bitbucket/${CLUSTER_USER}/temporal_embeddings/runs/job_${JOB_ID}/job_info.txt "$JOB_DIR/" 2>/dev/null
    
    # Download final model
    echo "Downloading final model"
    mkdir -p "$JOB_DIR/models"
    scp ${CLUSTER_USER}@${CLUSTER_HOST}:/vol/bitbucket/${CLUSTER_USER}/temporal_embeddings/runs/job_${JOB_ID}/models/best_model.pt "$JOB_DIR/models/" 2>/dev/null
    
    # Download training logs
    echo "Downloading logs"
    mkdir -p "$JOB_DIR/logs"
    scp ${CLUSTER_USER}@${CLUSTER_HOST}:/vol/bitbucket/${CLUSTER_USER}/temporal_pathrag_logs/train_embeddings_${JOB_ID}.out "$JOB_DIR/logs/" 2>/dev/null
    scp ${CLUSTER_USER}@${CLUSTER_HOST}:/vol/bitbucket/${CLUSTER_USER}/temporal_pathrag_logs/train_embeddings_${JOB_ID}.err "$JOB_DIR/logs/" 2>/dev/null
    
    # Optionally download all checkpoints
    if [ "$DOWNLOAD_ALL" = true ]; then
        echo "Downloading all checkpoints"
        mkdir -p "$JOB_DIR/checkpoints"
        scp ${CLUSTER_USER}@${CLUSTER_HOST}:/vol/bitbucket/${CLUSTER_USER}/temporal_embeddings/runs/job_${JOB_ID}/checkpoints/*.pt "$JOB_DIR/checkpoints/" 2>/dev/null
    fi
    
    echo "Downloaded job $JOB_ID results to: $JOB_DIR"
    
    # Extract key metrics from logs
    if [ -f "$JOB_DIR/logs/train_embeddings_${JOB_ID}.err" ]; then
        echo ""
        echo "Training summary:"
        grep -E "Epoch .* completed" "$JOB_DIR/logs/train_embeddings_${JOB_ID}.err" | tail -5
    fi
fi

echo ""
echo "Download complete"
echo ""
echo "Local models are stored in: $LOCAL_MODELS_DIR"
ls -la "$LOCAL_MODELS_DIR"