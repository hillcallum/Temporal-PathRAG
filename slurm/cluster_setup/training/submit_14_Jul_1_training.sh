#!/bin/bash
# Submit first training run (14th July) with initial parameters

# Source the main submission script functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Training configuration for first run
export DATASET="MultiTQ"
export EPOCHS=5                    # Start with fewer epochs for first run
export BATCH_SIZE=16               # Small batch size for stability
export GRAD_ACCUM_STEPS=8          # Effective batch size = 16 * 8 = 128
export LR=5e-5                     # Learning rate
export TEMPORAL_WEIGHT=0.3         # Default temporal weight
export QUADRUPLET_MARGIN=0.5       # Default margin
export MAX_SEQ_LENGTH=512          # Sequence length
export CHECKPOINT_STEPS=500        # Save checkpoints frequently
export EVAL_STEPS=1000             # Evaluate every 1000 steps
export WARMUP_STEPS=500            # Warmup for stability
export USE_WANDB=false             # Disable W&B for first run
export USE_MIXED_PRECISION=true    # Enable mixed precision for faster training

export CHECKPOINT_EVERY=${CHECKPOINT_STEPS}
export EVAL_EVERY=${EVAL_STEPS}

echo "First Training Run Configuration"
echo "Dataset: $DATASET"
echo "Epochs: $EPOCHS"
echo "Batch Size: $BATCH_SIZE"
echo "Gradient Accumulation Steps: $GRAD_ACCUM_STEPS"
echo "Effective Batch Size: $((BATCH_SIZE * GRAD_ACCUM_STEPS))"
echo "Learning Rate: $LR"
echo "Max Sequence Length: $MAX_SEQ_LENGTH"
echo "Mixed Precision: $USE_MIXED_PRECISION"
echo "Checkpoint Every: $CHECKPOINT_EVERY steps"
echo "Evaluate Every: $EVAL_EVERY steps"
echo ""

# Run the main submission script
"${SCRIPT_DIR}/submit_training_jobs.sh"