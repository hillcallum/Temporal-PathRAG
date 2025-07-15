#!/bin/bash

set -e

CLUSTER_HOST="gpucluster2.doc.ic.ac.uk"
CLUSTER_USER="cih124"
PROJECT_NAME="Temporal_PathRAG"
LOCAL_PROJECT_PATH="/Users/hillcallum/Temporal_PathRAG"

echo "Starting interactive debugging session"
echo "This will sync code and then start an interactive session"

# 1. Clean up and sync code to cluster
echo ""
echo "1. Cleaning up cluster space and syncing code"

echo "Aggressive cleanup of bitbucket given lack of space currently available"
ssh "$CLUSTER_USER@$CLUSTER_HOST" "
    echo 'Current bitbucket usage:'
    df -h /vol/bitbucket
    
    echo 'Performing aggressive cleanup'
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

echo "Performing full repository sync"
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

# 2. Start interactive session
echo ""
echo "2. Starting interactive session"
echo "This gives us a GPU node for about 2 hours"
echo "Use the interactive_llm_setup.sh script once connected"
echo ""

ssh -t "$CLUSTER_USER@$CLUSTER_HOST" "cd /vol/bitbucket/$CLUSTER_USER/$PROJECT_NAME && salloc --gres=gpu:1 --time=2:00:00 --partition=gpgpuC --mem=32G"