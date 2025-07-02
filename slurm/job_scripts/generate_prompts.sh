#!/bin/bash
# SLURM job script for prompt generation
#SBATCH --job-name=timeR4_prompts
#SBATCH --partition=gpgpuB
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/vol/bitbucket/%u/temporal_pathrag_logs/timeR4_%j.out
#SBATCH --error=/vol/bitbucket/%u/temporal_pathrag_logs/timeR4_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=cih124

echo "Starting prompt generation"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Time: $(date)"

# Create log directory
mkdir -p /vol/bitbucket/${USER}/temporal_pathrag_logs

# Set up Python environment 
echo "Setting up Python environment"
echo "Using system Python with --break-system-packages for cluster compatibility"

# Load CUDA 
echo "Loading CUDA 12.0.0"
source /vol/cuda/12.0.0/setup.sh

# Check GPU assignment
echo "Checking GPU assignment"
/usr/bin/nvidia-smi

echo "Environment: $(which python)"
echo "Working directory: $(pwd)"

# Change to project directory
cd /vol/bitbucket/${USER}/Temporal_PathRAG

# Install PyTorch with CUDA support and sentence transformers
echo "Installing PyTorch with CUDA and ML dependencies"
# Use /tmp for packages to avoid disk quota issues
export TMPDIR=/tmp/pip_cache_$$
mkdir -p $TMPDIR
export PIP_CACHE_DIR=$TMPDIR
export PIP_TMPDIR=$TMPDIR

# Install to /vol/bitbucket to avoid home directory quota
export PYTHONUSERBASE=/vol/bitbucket/${USER}/python_packages
mkdir -p $PYTHONUSERBASE
export PATH=$PYTHONUSERBASE/bin:$PATH
export PYTHONPATH=$PYTHONUSERBASE/lib/python3.12/site-packages:$PYTHONPATH

# Set HuggingFace cache to avoid disk quota issues
export HF_HOME=/vol/bitbucket/${USER}/huggingface_cache
export TRANSFORMERS_CACHE=/vol/bitbucket/${USER}/huggingface_cache
export HF_HUB_CACHE=/vol/bitbucket/${USER}/huggingface_cache
mkdir -p $HF_HOME

pip install --user torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install --user sentence-transformers transformers numpy

# Clean up temp directory
rm -rf $TMPDIR

# Set CUDA environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Test GPU and PyTorch setup
echo "Testing PyTorch GPU setup"
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name()}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    print('GPU setup verified!')
else:
    print('CUDA not available, exiting')
    exit(1)
"

# Check datasets exist
echo "Verifying datasets"
if [ ! -d "datasets/MultiTQ" ] || [ ! -d "datasets/TimeQuestions" ]; then
    echo "Error: Required datasets not found"
    echo "Available directories:"
    ls -la datasets/ || echo "No datasets directory found"
    exit 1
fi

echo ""
echo "Dataset information:"
echo "MultiTQ test questions: $(python -c "import json; print(len(json.load(open('datasets/MultiTQ/questions/test.json'))))" 2>/dev/null || echo 'error')"
echo "MultiTQ dev questions: $(python -c "import json; print(len(json.load(open('datasets/MultiTQ/questions/dev.json'))))" 2>/dev/null || echo 'not found')"
echo "TimeQuestions test questions: $(python -c "import json; print(len(json.load(open('datasets/TimeQuestions/questions/test.json'))))" 2>/dev/null || echo 'error')"
echo "TimeQuestions dev questions: $(python -c "import json; print(len(json.load(open('datasets/TimeQuestions/questions/dev.json'))))" 2>/dev/null || echo 'not found')"

echo ""
echo "Starting promtp generation"
echo ""

# Start GPU monitoring in background
echo "Starting GPU monitoring"
(
    while true; do
        sleep 300  # Every 5 minutes
        echo "$(date '+%H:%M:%S'): GPU Status:"
        nvidia-smi --query-gpu=utilisation.gpu,memory.used,memory.total --format=csv,noheader,nounits | \
        awk '{printf " GPU Util: %s%%, Memory: %s/%s MB\n", $1, $2, $3}'
    done
) &
GPU_MONITOR_PID=$!

# Run the main script
python scripts/prompt_generator.py

# Capture exit code
EXIT_CODE=$?

# Stop GPU monitoring
kill $GPU_MONITOR_PID 2>/dev/null || true

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "Prompt generation completed successfully"
    echo ""
    echo "Generated files:"
    find datasets -name "*_prompt.json" -exec ls -lh {} \; 2>/dev/null || echo "No prompt files found"
    echo ""
    echo "Prompt counts:"
    find datasets -name "*_prompt.json" -exec sh -c 'echo -n "$1: "; python -c "import json; print(len(json.load(open(\"$1\"))))" 2>/dev/null || echo "Error reading file"' _ {} \;
else
    echo "Prompt generation failed with exit code $EXIT_CODE"
    echo "Check error logs for details"
fi

echo ""
echo "Job finished at: $(date)"
echo "Total runtime: $SECONDS seconds"