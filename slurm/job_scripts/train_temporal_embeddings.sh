#!/bin/bash
#SBATCH --job-name=train_temporal_embeddings
#SBATCH --partition=gpgpuC
#SBATCH --time=4-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
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

# Create organised directory structure
JOB_RUN_DIR="/vol/bitbucket/${USER}/temporal_embeddings/runs/job_${SLURM_JOB_ID}"
LOG_DIR="${JOB_RUN_DIR}/logs"
CHECKPOINT_DIR="${JOB_RUN_DIR}/checkpoints"
BEST_MODEL_DIR="/vol/bitbucket/${USER}/temporal_embeddings/best_model"

mkdir -p "$LOG_DIR"
mkdir -p "$CHECKPOINT_DIR"
mkdir -p "$BEST_MODEL_DIR"
mkdir -p /vol/bitbucket/${USER}/temporal_pathrag_logs

# Check if data directory exists
DATA_DIR="/vol/bitbucket/${USER}/Temporal_PathRAG/data/training"
if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: Training data directory not found: $DATA_DIR"
    echo "Please ensure the training data has been synced to the cluster"
    exit 1
fi

echo "Training data directory found: $DATA_DIR"
ls -la "$DATA_DIR"

# Load CUDA
echo "Loading CUDA"
. /vol/cuda/12.0.0/setup.sh

# Check GPU
echo "GPU Info:"
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. GPU drivers not properly installed."
    exit 1
fi

nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader,nounits
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to query GPU. No GPU available."
    exit 1
fi
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

# Triton/BitsAndBytes cache settings (to avoid home directory quota)
export TRITON_CACHE_DIR="/vol/bitbucket/${USER}/triton_cache"
export XDG_CACHE_HOME="/vol/bitbucket/${USER}/cache"
export HOME="/vol/bitbucket/${USER}/temp_home"
mkdir -p "$TRITON_CACHE_DIR"
mkdir -p "$XDG_CACHE_HOME"
mkdir -p "$HOME/.triton"
mkdir -p "$HOME/.cache"

# GPU settings
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Force CUDA initialisation
export CUDA_LAUNCH_BLOCKING=1

# Create directories
mkdir -p "$HF_HOME"
mkdir -p "$PYTHONUSERBASE"

# Progress monitoring
log_progress() {
    echo "[$(date '+%H:%M:%S')] $1"
}

# Set output directories for this job
OUTPUT_DIR="${JOB_RUN_DIR}/models"
mkdir -p "$OUTPUT_DIR"

# Create job info file
cat > "${JOB_RUN_DIR}/job_info.txt" << EOF
Job ID: $SLURM_JOB_ID
Start Time: $(date)
Node: $SLURM_NODELIST
Dataset: ${DATASET}
Epochs: ${EPOCHS}
Batch Size: ${BATCH_SIZE}
Learning Rate: ${LR}
Temporal Weight: ${TEMPORAL_WEIGHT}
Quadruplet Margin: ${QUADRUPLET_MARGIN}
EOF

log_progress "Starting temporal embeddings training"
log_progress "Environment setup complete"

# Install required packages if not already installed
log_progress "Checking required packages"
python3 -m pip install --user --break-system-packages torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 2>/dev/null || true
python3 -m pip install --user --break-system-packages transformers bitsandbytes peft networkx tqdm accelerate 2>/dev/null || true

# Test packages
log_progress "Testing required packages"
# Ensure CUDA is available before testing
export CUDA_HOME=/vol/cuda/12.0.0
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
python3 -c "
import torch
print(f'PyTorch {torch.__version__} - CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    print(f'CUDA_VISIBLE_DEVICES: {torch.cuda.device_count()} device(s)')
else:
    print('ERROR: CUDA not available')
    import sys
    sys.exit(1)

import transformers
print(f'Transformers {transformers.__version__}')

# BitsAndBytes is optional - warn but don't fail
try:
    import bitsandbytes
    print('BitsAndBytes available')
except ImportError as e:
    print(f'WARNING: BitsAndBytes not available: {e}')
    print('Will use regular model loading without quantization')

print('Core packages available - continuing')
"

if [ $? -ne 0 ]; then
    log_progress "Package test failed"
    exit 1
fi

# Install additional required packages
log_progress "Installing additional training dependencies"
python3 -m pip install --user --break-system-packages psutil GPUtil 2>/dev/null || true

# Use the temporal trainer
log_progress "Using temporal trainer from src/training/"

# Run training with the new trainer
log_progress "Starting training script"
python3 src/training/train_temporal_embeddings.py \
    --dataset "${DATASET:-MultiTQ}" \
    --data_dir "/vol/bitbucket/${USER}/Temporal_PathRAG/data/training" \
    --output_dir "${OUTPUT_DIR}" \
    --batch_size "${BATCH_SIZE:-32}" \
    --grad_accum_steps "${GRAD_ACCUM_STEPS:-4}" \
    --num_epochs "${EPOCHS:-10}" \
    --learning_rate "${LR:-1e-4}" \
    --warmup_steps "${WARMUP_STEPS:-500}" \
    --quadruplet_margin "${QUADRUPLET_MARGIN:-0.5}" \
    --temporal_weight "${TEMPORAL_WEIGHT:-0.3}" \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --best_model_dir "${BEST_MODEL_DIR}" \
    --job_id "${SLURM_JOB_ID}" \
    --save_every "${CHECKPOINT_EVERY:-1000}" \
    --eval_every "${EVAL_EVERY:-500}" \
    --use_lora \
    ${USE_MIXED_PRECISION:+--mixed_precision} \
    --log_every 100 \
    ${RESUME:+--resume_from "$RESUME"} \
    ${USE_WANDB:+--use_wandb --wandb_project "temporal-pathrag"}

EXIT_CODE=$?

log_progress "Training completed with exit code: $EXIT_CODE"

# Copy results and training logs
if [ -d "$OUTPUT_DIR" ]; then
    log_progress "Model outputs saved in: $OUTPUT_DIR"
    ls -la "$OUTPUT_DIR"
fi

# Copy training logs to job directory
cp /vol/bitbucket/${USER}/temporal_pathrag_logs/train_embeddings_${SLURM_JOB_ID}.* "$LOG_DIR/" 2>/dev/null || true

# Update job info with completion
echo "End Time: $(date)" >> "${JOB_RUN_DIR}/job_info.txt"
echo "Exit Code: $EXIT_CODE" >> "${JOB_RUN_DIR}/job_info.txt"

# Clean up temporary files
rm -rf "/tmp/hf_cache_${SLURM_JOB_ID}" 2>/dev/null || true

echo ""
echo "Job Summary"
echo "Job ID: $SLURM_JOB_ID"
echo "Exit Code: $EXIT_CODE"
echo "Completion Time: $(date)"
echo "Job outputs saved in: $JOB_RUN_DIR"
echo "Best model (if improved) saved in: $BEST_MODEL_DIR"
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo "Temporal embeddings training completed successfully"
else
    echo "Temporal embeddings training failed"
fi

echo ""
echo "Job Complete"