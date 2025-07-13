#!/usr/bin/env python3
"""
Generate training data for temporal embeddings from KGs
"""

import argparse
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.dataset_loader import load_dataset
from src.embeddings.training_data_pipeline import create_training_pipeline, TemporalEmbeddingDataPipeline


def main():
    parser = argparse.ArgumentParser(
        description="Generate training data for temporal embeddings"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="MultiTQ",
        choices=["MultiTQ", "TimeQuestions"],
        help="Dataset to use"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("training_data/embeddings"),
        help="Output directory for training data"
    )
    parser.add_argument(
        "--num-quadruplet",
        type=int,
        default=10000,
        help="Number of quadruplet examples to generate"
    )
    parser.add_argument(
        "--num-contrastive",
        type=int,
        default=5000,
        help="Number of contrastive examples to generate"
    )
    parser.add_argument(
        "--num-reconstruction",
        type=int,
        default=5000,
        help="Number of reconstruction examples to generate"
    )
    
    args = parser.parse_args()
    
    print(f"Generating training data for {args.dataset}")
    
    # Load the dataset
    print(f"\n1. Loading {args.dataset} dataset")
    graph = load_dataset(args.dataset)
    
    # Create training pipeline
    print("\n2. Creating training data pipeline")
    pipeline = create_training_pipeline(graph, args.dataset)
    
    # Generate training data
    print("\n3. Generating training examples")
    output_dir = args.output_dir / args.dataset
    
    dataset = pipeline.create_training_dataset(
        num_quadruplet=args.num_quadruplet,
        num_contrastive=args.num_contrastive,
        num_reconstruction=args.num_reconstruction,
        output_dir=output_dir
    )
    
    # Print statistics
    print("\n4. Dataset Statistics:")
    print(f"Total examples: {sum(len(split) for split in dataset.values())}")
    for split, examples in dataset.items():
        print(f"{split}: {len(examples)} examples")
        
        # Count by type
        type_counts = {}
        for ex in examples:
            type_counts[ex.example_type] = type_counts.get(ex.example_type, 0) + 1
        
        print(f"Types: {type_counts}")
    
    print(f"\nTraining data saved to: {output_dir}")


if __name__ == "__main__":
    main()