#!/bin/bash
# Script to download the best trained model from cluster to local repo

echo "Downloading Best Temporal Embedding Model"
echo ""

# Configuration
CLUSTER_USER="cih124"
CLUSTER_HOST="gpucluster2.doc.ic.ac.uk"
REMOTE_MODEL_DIR="/vol/bitbucket/${CLUSTER_USER}/temporal_embeddings/best_model"
LOCAL_MODEL_DIR="./models/temporal_embeddings/best_model"

# Create local directory
echo "Creating local directory: ${LOCAL_MODEL_DIR}"
mkdir -p "${LOCAL_MODEL_DIR}"

# Download model files
echo "Downloading model files from cluster"
echo ""

# Download the model checkpoint
echo "1. Downloading best_model.pt"
scp "${CLUSTER_USER}@${CLUSTER_HOST}:${REMOTE_MODEL_DIR}/best_model.pt" "${LOCAL_MODEL_DIR}/"

# Download the model info
echo "2. Downloading best_model_info.json"
scp "${CLUSTER_USER}@${CLUSTER_HOST}:${REMOTE_MODEL_DIR}/best_model_info.json" "${LOCAL_MODEL_DIR}/"

# Also download the actual checkpoint file for backup
echo "3. Downloading checkpoint_step_34000.pt"
CHECKPOINT_PATH="/vol/bitbucket/${CLUSTER_USER}/temporal_embeddings/runs/job_183972/checkpoints/checkpoint_step_34000.pt"
scp "${CLUSTER_USER}@${CLUSTER_HOST}:${CHECKPOINT_PATH}" "${LOCAL_MODEL_DIR}/"

# Create a README for the model
cat > "${LOCAL_MODEL_DIR}/README.md" << EOF
# Temporal Embedding Model - Best Checkpoint

## Model Information
- **Validation Loss**: 0.1204
- **Training Step**: 34,000 
- **Epochs Completed**: ~8/30
- **Job ID**: 183972
- **Date Trained**: 2025-07-20 18:03:33

## Model Architecture
- **Base Model**: LLaMA-3.2-1B-Instruct
- **LoRA Parameters**: r=16, alpha=32
- **Trainable Parameters**: 3,407,872 (0.275%)
- **Total Parameters**: 1,239,222,272

## Training Configuration
- **Dataset**: Combined (MultiTQ + TimeQuestions)
- **Training Examples**: 552,000
- **Batch Size**: 32 (effective 128 with gradient accumulation)
- **Learning Rate**: 1e-4
- **Loss Type**: Quadruplet loss with temporal attention

## Files
- \`best_model.pt\` - The model checkpoint with best validation performance
- \`best_model_info.json\` - Metadata about the checkpoint
- \`checkpoint_step_34000.pt\` - Original checkpoint file (backup)

## Usage
\`\`\`python
from src.kg.retrieval.temporal_embedding_retriever import TemporalEmbeddingRetriever

# Load the model
retriever = TemporalEmbeddingRetriever(
    model_path="./models/temporal_embeddings/best_model",
    device="cuda"
)
\`\`\`
EOF

echo ""
echo "Download complete - model saved to: ${LOCAL_MODEL_DIR}"
echo ""
ls -lah "${LOCAL_MODEL_DIR}"