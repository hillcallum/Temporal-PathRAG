#!/usr/bin/env python3
"""
Generate training data for temporal embeddings from a dataset
"""

import sys
from pathlib import Path
import argparse
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.embeddings.training_data_pipeline import create_training_pipeline
from src.utils.dataset_loader import load_dataset


def main():
    parser = argparse.ArgumentParser(description="Generate training data for temporal embeddings")
    parser.add_argument("--dataset", type=str, default="MultiTQ", help="Dataset name")
    parser.add_argument("--output-dir", type=str, default="./data/training", help="Output directory")
    parser.add_argument("--num-quadruplet", type=int, default=10000, help="Number of quadruplet examples")
    parser.add_argument("--num-contrastive", type=int, default=5000, help="Number of contrastive examples")
    parser.add_argument("--num-reconstruction", type=int, default=5000, help="Number of reconstruction examples")
    
    args = parser.parse_args()
    
    print(f"Generating training data for {args.dataset}")
    
    # Load dataset
    try:
        graph = load_dataset(args.dataset)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return
    
    print(f"Loaded graph with {len(graph.nodes())} nodes and {len(graph.edges())} edges")
    
    # Create training pipeline
    pipeline = create_training_pipeline(graph, args.dataset)
    
    # Generate training data
    output_path = Path(args.output_dir) / args.dataset
    dataset = pipeline.create_training_dataset(
        num_quadruplet=args.num_quadruplet,
        num_contrastive=args.num_contrastive,
        num_reconstruction=args.num_reconstruction,
        output_dir=output_path
    )
    
    print(f"\nTraining data saved to {output_path}")
    print(f"Generated:")
    print(f" - {len(dataset['train'])} training examples")
    print(f" - {len(dataset['validation'])} validation examples")
    print(f" - {len(dataset['test'])} test examples")


if __name__ == "__main__":
    main()