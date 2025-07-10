#!/usr/bin/env python3
"""
Run evaluation of Temporal PathRAG against all baselines

This script evaluates Temporal PathRAG against:
1. Primary Temporal QA Baseline: TimeR4
2. Graph-based RAG Baselines: PathRAG, KG-IRAG (if available)
3. Standard RAGs: Vanilla RAG, HyDE
4. Direct LLM Baselines: LLaMA2, GPT-3.5, GPT-4
"""

import argparse
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from evaluation import run_baseline_comparison
from evaluation.ablation_framework import run_ablation


def main():
    parser = argparse.ArgumentParser(
        description="Run evaluation of Temporal PathRAG"
    )
    parser.add_argument(
        "--dataset",
        choices=["MultiTQ", "TimeQuestions", "both"],
        default="both",
        help="Dataset to evaluate on"
    )
    parser.add_argument(
        "--max_questions",
        type=int,
        default=None,
        help="Maximum number of questions to evaluate (for testing)"
    )
    parser.add_argument(
        "--baselines",
        nargs="+",
        help="Specific baselines to run (default: all)"
    )
    parser.add_argument(
        "--skip_ablation",
        action="store_true",
        help="Skip ablation study"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    # Define all baselines
    all_baselines = [
        # Our system
        "temporal_pathrag",
        
        # Primary Temporal QA Baseline
        "timer4",
        
        # Graph-based RAG Baselines
        "pathrag",
        # "kg_irag",  # Need implementation
        
        # Standard RAGs
        "vanilla_rag",
        "hyde",
        
        # Direct LLM Baselines
        "vanilla_llm",  # Default LLM
        "llama2",       # LLaMA2-7B
        "gpt3.5",       # GPT-3.5
        # "gpt4",       # GPT-4 (expensive, so commented out - might include if I have $$$)
    ]
    
    # Use specified baselines or all
    baselines = args.baselines if args.baselines else all_baselines
    
    # Determine datasets to evaluate
    if args.dataset == "both":
        datasets = ["MultiTQ", "TimeQuestions"]
    else:
        datasets = [args.dataset]
    
    print("=" * 70)
    print("Temporal PathRAG Evaluation")
    print("=" * 70)
    print(f"Datasets: {', '.join(datasets)}")
    print(f"Baselines: {', '.join(baselines)}")
    print(f"Max Questions: {args.max_questions or 'All'}")
    print("=" * 70)
    
    # Run baseline comparison for each dataset
    for dataset in datasets:
        print(f"\n{'='*70}")
        print(f"Evaluating on {dataset.upper()}")
        print(f"{'='*70}")
        
        try:
            results = run_baseline_comparison(
                dataset_name=dataset,
                baselines=baselines,
                max_questions=args.max_questions,
                output_dir=args.output_dir
            )
            
            # Print summary
            print(f"\n{'-'*50}")
            print(f"Summary for {dataset}")
            print(f"{'-'*50}")
            
            # Sort baselines by exact match score
            baseline_scores = []
            for baseline, result in results.items():
                if 'metrics' in result:
                    baseline_scores.append((
                        baseline,
                        result['metrics'].exact_match,
                        result['metrics'].f1_score,
                        result['metrics'].temporal_accuracy
                    ))
            
            baseline_scores.sort(key=lambda x: x[1], reverse=True)
            
            print(f"{'Baseline':<20} {'Exact Match':<12} {'F1 Score':<12} {'Temporal Acc':<12}")
            print("-" * 60)
            for baseline, em, f1, temp in baseline_scores:
                print(f"{baseline:<20} {em:<12.3f} {f1:<12.3f} {temp:<12.3f}")
                
        except Exception as e:
            print(f"Error evaluating on {dataset}: {e}")
            
    # Run ablation study if not skipped
    if not args.skip_ablation:
        print(f"\n{'='*70}")
        print("Running Ablation Study")
        print(f"{'='*70}")
        
        try:
            run_ablation(
                dataset_names=datasets,
                ablation_types=["leave_one_out", "category"],
                max_questions=args.max_questions
            )
        except Exception as e:
            print(f"Error running ablation study: {e}")
    
    print(f"\n{'='*70}")
    print("Evaluation Complete")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()