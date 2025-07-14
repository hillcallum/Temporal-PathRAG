#!/bin/bash
#SBATCH --job-name=train_temporal_embeddings
#SBATCH --partition=gpgpuC
#SBATCH --time=4-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/vol/bitbucket/%u/temporal_pathrag_logs/train_embeddings_%j.out
#SBATCH --error=/vol/bitbucket/%u/temporal_pathrag_logs/train_embeddings_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=cih124

echo "Training Temporal Embeddings on Imperial GPU Cluster"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Time: $(date)"
echo ""

# Create log directory
mkdir -p /vol/bitbucket/${USER}/temporal_pathrag_logs
mkdir -p /vol/bitbucket/${USER}/temporal_embeddings/checkpoints

# Load CUDA
echo "Loading CUDA"
. /vol/cuda/12.0.0/setup.sh

# Check GPU
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader,nounits
echo ""

# Setup environment
cd /vol/bitbucket/${USER}/Temporal_PathRAG

# Environment variables for Python packages
export PYTHONUSERBASE="/vol/bitbucket/${USER}/python_packages"
export PATH="$PYTHONUSERBASE/bin:$PATH"
export PYTHONPATH="$PYTHONUSERBASE/lib/python3.12/site-packages:$(pwd):$PYTHONPATH"

# HuggingFace cache settings
export HF_HOME="/vol/bitbucket/${USER}/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$HF_HOME"
export SENTENCE_TRANSFORMERS_HOME="$HF_HOME"

# GPU settings
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Create directories
mkdir -p "$HF_HOME"
mkdir -p "$PYTHONUSERBASE"

# Progress monitoring
log_progress() {
    echo "[$(date '+%H:%M:%S')] $1"
}

# Checkpoint directory
CHECKPOINT_DIR="/vol/bitbucket/${USER}/temporal_embeddings/checkpoints"
OUTPUT_DIR="/vol/bitbucket/${USER}/temporal_embeddings/models"
mkdir -p "$CHECKPOINT_DIR"
mkdir -p "$OUTPUT_DIR"

log_progress "Starting temporal embeddings training"
log_progress "Environment setup complete"

# Install required packages if not already installed
log_progress "Checking required packages"
python3 -m pip install --user --break-system-packages torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 2>/dev/null || true
python3 -m pip install --user --break-system-packages transformers bitsandbytes peft networkx tqdm accelerate 2>/dev/null || true

# Test packages
log_progress "Testing required packages"
python3 -c "
import torch
print(f'PyTorch {torch.__version__} - CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

import transformers
print(f'Transformers {transformers.__version__}')
import bitsandbytes
print(f'BitsAndBytes available')

print('All packages available')
"

if [ $? -ne 0 ]; then
    log_progress "Package test failed"
    exit 1
fi

# Install additional required packages
log_progress "Installing additional training dependencies"
python3 -m pip install --user --break-system-packages wandb psutil GPUtil 2>/dev/null || true

# Use the temporal trainer
log_progress "Using temporal trainer from src/training/"

# Run training with the new trainer
log_progress "Starting training script"
python3 src/training/train_temporal_embeddings.py \
    --dataset "${DATASET:-MultiTQ}" \
    --data_dir "/vol/bitbucket/${USER}/temporal_pathrag_data/processed_datasets" \
    --output_dir "/vol/bitbucket/${USER}/temporal_embeddings/training_data" \
    --batch_size "${BATCH_SIZE:-32}" \
    --grad_accum_steps "${GRAD_ACCUM_STEPS:-4}" \
    --num_epochs "${EPOCHS:-10}" \
    --learning_rate "${LR:-1e-4}" \
    --warmup_steps "${WARMUP_STEPS:-500}" \
    --quadruplet_margin "${QUADRUPLET_MARGIN:-0.5}" \
    --temporal_weight "${TEMPORAL_WEIGHT:-0.3}" \
    --checkpoint_dir "/vol/bitbucket/${USER}/temporal_embeddings/checkpoints" \
    --save_every "${CHECKPOINT_EVERY:-1000}" \
    --eval_every "${EVAL_EVERY:-500}" \
    --mixed_precision \
    --log_every 100 \
    ${RESUME:+--resume_from "$RESUME"} \
    ${USE_WANDB:+--use_wandb --wandb_project "temporal-pathrag"}

EXIT_CODE=$?

log_progress "Training completed with exit code: $EXIT_CODE"

# Copy results
if [ -d "$OUTPUT_DIR" ]; then
    log_progress "Model outputs saved in: $OUTPUT_DIR"
    ls -la "$OUTPUT_DIR"
fi

# Clean up temporary files
rm -rf "/tmp/hf_cache_${SLURM_JOB_ID}" 2>/dev/null || true

echo ""
echo "Job Summary"
echo "Job ID: $SLURM_JOB_ID"
echo "Exit Code: $EXIT_CODE"
echo "Completion Time: $(date)"
echo "Models saved in: $OUTPUT_DIR"
echo "Checkpoints saved in: $CHECKPOINT_DIR"
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo "Temporal embeddings training completed successfully"
else
    echo "Temporal embeddings training failed"
fi

echo ""
echo "Job Complete"