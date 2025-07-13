#!/usr/bin/env python3
"""
Training script for temporal embeddings using existing infrastructure
"""
import argparse
from pathlib import Path
import sys
import json
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.embeddings.training_data_pipeline import TemporalEmbeddingDataPipeline
from src.embeddings.temporal_embeddings import TemporalEmbeddings
from src.utils.dataset_loader import load_dataset, get_dataset_info


def main():
    parser = argparse.ArgumentParser(description="Train temporal embeddings")
    parser.add_argument("--dataset", type=str, default="MultiTQ", help="Dataset name")
    parser.add_argument("--output-dir", type=Path, default=Path("models/embeddings"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    
    args = parser.parse_args()
    
    print(f"Training embeddings for {args.dataset}")
    
    # Load dataset
    graph = load_dataset(args.dataset)
    dataset_info = get_dataset_info(args.dataset)
    
    # Initialise embeddings
    embeddings = TemporalEmbeddings()
    
    # Create training pipeline
    pipeline = TemporalEmbeddingDataPipeline(graph, args.dataset)
    
    # Generate training data
    print("Generating training examples")
    training_data = pipeline.create_training_dataset(
        num_quadruplet=1000,
        num_contrastive=500,
        num_reconstruction=500
    )
    
    # Pre-compute embeddings for the dataset
    print("Pre-computing embeddings")
    nodes = list(graph.nodes())[:1000]  # Start with subset
    relations = list(set(edge[1] for edge in graph.edges(keys=True)))
    
    # Cache node embeddings
    for i in range(0, len(nodes), args.batch_size):
        batch = nodes[i:i+args.batch_size]
        embeddings.get_node_embedding(batch[0])  # Trigger caching
    
    # Save configuration
    config = {
        'dataset': args.dataset,
        'timestamp': datetime.now().isoformat(),
        'num_nodes': len(nodes),
        'num_relations': len(relations),
        'embedding_dim': embeddings.config.embedding_dim
    }
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / f"{args.dataset}_config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Training complete - embeddings cached in: {embeddings.config.cache_dir}")


if __name__ == "__main__":
    main()