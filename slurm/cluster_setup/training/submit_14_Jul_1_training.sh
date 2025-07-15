#!/bin/bash
# Configuration for first training run (14th July)
# This script sets parameters and calls the master submission script

# Export training parameters
export DATASET="MultiTQ"
export EPOCHS=5                    # Start with fewer epochs for testing
export BATCH_SIZE=16               # Small batch size for stability
export GRAD_ACCUM_STEPS=8          # Effective batch size = 16 * 8 = 128
export LR=5e-5                     # Conservative learning rate
export TEMPORAL_WEIGHT=0.3         # Default temporal weight
export QUADRUPLET_MARGIN=0.5       # Default margin
export CHECKPOINT_EVERY=500        # Save checkpoints frequently
export EVAL_EVERY=1000             # Evaluate every 1000 steps
export WARMUP_STEPS=500            # Warmup for stability
export USE_WANDB=""                # Disable W&B for first run (empty = disabled)
export USE_MIXED_PRECISION=true    # Enable mixed precision

echo "First Training Run Configuration"
echo "Dataset: $DATASET"
echo "Epochs: $EPOCHS"
echo "Batch Size: $BATCH_SIZE (x$GRAD_ACCUM_STEPS = $((BATCH_SIZE * GRAD_ACCUM_STEPS)) effective)"
echo "Learning Rate: $LR"
echo ""

# Call the master submission script
cd "$(dirname "$0")/.."
./submit_training_jobs.sh