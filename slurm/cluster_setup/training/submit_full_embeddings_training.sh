#!/bin/bash
# Full-scale temporal embeddings training configuration
# This trains on combined datasets 

# Export training parameters for full training
export DATASET="combined"          # Train on both MultiTQ and TimeQuestions
export EPOCHS=25                   # Full training (not test run)
export BATCH_SIZE=64              # Larger batch for A100 GPU
export GRAD_ACCUM_STEPS=2         # Effective batch size = 64 * 2 = 128
export LR=2e-4                    # Slightly higher LR for full dataset
export TEMPORAL_WEIGHT=0.3        # Balanced temporal weight
export QUADRUPLET_MARGIN=0.5      # Standard margin
export CHECKPOINT_EVERY=2000      # Save every 2000 steps
export EVAL_EVERY=1000            # Evaluate every 1000 steps
export WARMUP_STEPS=1000          # 1000 step warmup
export USE_WANDB=true             # Enable W&B for monitoring
export USE_MIXED_PRECISION=true   # Enable for A100
export HARD_NEGATIVE_RATIO=0.8    # 80% hard negatives

echo "Full-Scale Temporal Embeddings Training"
echo "Dataset: $DATASET (MultiTQ + TimeQuestions combined)"
echo "Total training samples: 702K quadruplets"
echo "Epochs: $EPOCHS"
echo "Batch Size: $BATCH_SIZE (x$GRAD_ACCUM_STEPS = $((BATCH_SIZE * GRAD_ACCUM_STEPS)) effective)"
echo "Learning Rate: $LR"
echo "GPU: $GPU_TYPE"
echo ""


# Confirm before submission
read -p "Submit full training job? This will take 24-48 hours. (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Training submission cancelled"
    exit 1
fi

# Call the master submission script
cd "$(dirname "$0")/.."
./submit_training_jobs.sh