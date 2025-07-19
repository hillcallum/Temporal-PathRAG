#!/usr/bin/env python3
"""
Combine multiple training datasets into a single combined dataset
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict
import random
from collections import defaultdict


def load_dataset(dataset_dir: Path) -> Dict[str, List[dict]]:
    """Load a dataset from directory"""
    dataset = {}
    
    for split in ['train', 'validation', 'test']:
        split_file = dataset_dir / f"{split}.json"
        if split_file.exists():
            with open(split_file, 'r') as f:
                dataset[split] = json.load(f)
            print(f"Loaded {len(dataset[split])} {split} examples from {dataset_dir.name}")
        else:
            print(f"Warning: {split_file} not found")
            dataset[split] = []
    
    return dataset


def combine_datasets(datasets: List[Dict[str, List[dict]]], 
                    dataset_names: List[str],
                    shuffle: bool = True) -> Dict[str, List[dict]]:
    """Combine multiple datasets into one"""
    combined = defaultdict(list)
    
    # Track statistics
    stats = defaultdict(lambda: defaultdict(int))
    
    for dataset, name in zip(datasets, dataset_names):
        for split, examples in dataset.items():
            # Add dataset source to each example
            for ex in examples:
                ex['source_dataset'] = name
                stats[split][name] += 1
                stats[split]['total'] += 1
                stats[split][f"{name}_{ex['example_type']}"] += 1
            
            combined[split].extend(examples)
    
    # Shuffle if requested
    if shuffle:
        for split in combined:
            random.shuffle(combined[split])
    
    # Print statistics
    print("\nCombined dataset statistics:")
    for split, split_stats in stats.items():
        print(f"\n{split}:")
        print(f"Total: {split_stats['total']}")
        for name in dataset_names:
            print(f"{name}: {split_stats[name]}")
            for ex_type in ['quadruplet', 'contrastive', 'reconstruction']:
                count = split_stats.get(f"{name}_{ex_type}", 0)
                if count > 0:
                    print(f" - {ex_type}: {count}")
    
    return dict(combined)


def save_combined_dataset(dataset: Dict[str, List[dict]], 
                         output_dir: Path,
                         dataset_names: List[str]):
    """Save combined dataset to disk"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save each split
    for split, examples in dataset.items():
        output_file = output_dir / f"{split}.json"
        with open(output_file, 'w') as f:
            json.dump(examples, f, indent=2)
        print(f"Saved {len(examples)} examples to {output_file}")
    
    # Save metadata
    metadata = {
        'combined_from': dataset_names,
        'splits': {split: len(examples) for split, examples in dataset.items()},
        'total_examples': sum(len(examples) for examples in dataset.values())
    }
    
    # Count example types per dataset
    type_counts = defaultdict(lambda: defaultdict(int))
    for split, examples in dataset.items():
        for ex in examples:
            source = ex.get('source_dataset', 'unknown')
            ex_type = ex.get('example_type', 'unknown')
            type_counts[source][ex_type] += 1
    
    metadata['example_types_by_dataset'] = dict(type_counts)
    
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nCombined dataset saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Combine multiple training datasets"
    )
    parser.add_argument(
        "--input-dirs",
        type=Path,
        nargs='+',
        required=True,
        help="Input directories containing datasets to combine"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for combined dataset"
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Don't shuffle the combined dataset"
    )
    
    args = parser.parse_args()
    
    print("Combining datasets")
    print(f"Input directories: {[str(d) for d in args.input_dirs]}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    # Load all datasets
    datasets = []
    dataset_names = []
    
    for input_dir in args.input_dirs:
        if not input_dir.exists():
            print(f"Warning: {input_dir} does not exist, skipping")
            continue
        
        print(f"Loading dataset from {input_dir}")
        dataset = load_dataset(input_dir)
        datasets.append(dataset)
        dataset_names.append(input_dir.name)
    
    if not datasets:
        print("Error: No valid datasets found!")
        return 1
    
    # Combine datasets
    print("\nCombining datasets")
    combined = combine_datasets(datasets, dataset_names, shuffle=not args.no_shuffle)
    
    # Save combined dataset
    save_combined_dataset(combined, args.output_dir, dataset_names)
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())