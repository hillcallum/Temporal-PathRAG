#!/bin/bash
# Configuration for second training run (15th July)
# This script sets parameters and calls the master submission script

# Export training parameters
export DATASET="MultiTQ"
export EPOCHS=10                   # Full 10 epochs
export BATCH_SIZE=32               # Larger batch size for better GPU utilisation
export GRAD_ACCUM_STEPS=4          # Effective batch size = 32 * 4 = 128
export LR=1e-4                     # Standard learning rate
export TEMPORAL_WEIGHT=0.4         # Slightly higher temporal weight
export QUADRUPLET_MARGIN=0.3       # Tighter margin for better discrimination
export CHECKPOINT_EVERY=1000       # Save checkpoints less frequently
export EVAL_EVERY=500              # Evaluate every 500 steps
export WARMUP_STEPS=1000           # Longer warmup for stability
export USE_WANDB=""                # Disable W&B (empty = disabled)
export USE_MIXED_PRECISION=true    # Enable mixed precision
export HARD_NEGATIVE_RATIO=0.9     # More hard negatives for better training

# Optional: Resume from previous checkpoint
# export RESUME="/vol/bitbucket/${USER}/temporal_embeddings/checkpoints/checkpoint_step_500.pt"

echo "Second Training Run Configuration"
echo "Dataset: $DATASET"
echo "Epochs: $EPOCHS"
echo "Batch Size: $BATCH_SIZE (x$GRAD_ACCUM_STEPS = $((BATCH_SIZE * GRAD_ACCUM_STEPS)) effective)"
echo "Learning Rate: $LR"
echo "Temporal Weight: $TEMPORAL_WEIGHT"
echo "Quadruplet Margin: $QUADRUPLET_MARGIN"
echo ""

# Call the master submission script
cd "$(dirname "$0")/.."
./submit_training_jobs.sh