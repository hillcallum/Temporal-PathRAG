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
- `best_model.pt` - The model checkpoint with best validation performance
- `best_model_info.json` - Metadata about the checkpoint
- `checkpoint_step_34000.pt` - Original checkpoint file (backup)

## Usage
```python
from src.kg.temporal_embedding_retriever import TemporalEmbeddingRetriever

# Load the model
retriever = TemporalEmbeddingRetriever(
    model_path="./models/temporal_embeddings/best_model",
    device="cuda"
)
```
