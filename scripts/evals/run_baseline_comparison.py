#!/usr/bin/env python3
"""
Script to run baseline comparison tests for Temporal PathRAG evaluation
"""

import argparse
from pathlib import Path
import sys
import json
from datetime import datetime

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent # fixed the import issues I was having
sys.path.insert(0, str(project_root))

from evaluation.baseline_runners import run_baseline_comparison
from src.utils.dataset_loader import get_cache_info, clear_dataset_cache


def main():
    parser = argparse.ArgumentParser(
        description="Run baseline comparison for Temporal PathRAG"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="MultiTQ",
        choices=["MultiTQ", "TimeQuestions"],
        help="Dataset to evaluate on"
    )
    parser.add_argument(
        "--baselines",
        nargs="+",
        default=["vanilla_llm", "temporal_pathrag"],
        help="Baselines to run (available: vanilla_llm, direct_llm, vanilla_rag, hyde, temporal_pathrag)"
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=100,
        help="Maximum number of questions to evaluate"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for results"
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear dataset cache before running"
    )
    
    args = parser.parse_args()
    
    print(f"Running Baseline Comparison")
    print(f"Dataset: {args.dataset}")
    print(f"Baselines: {', '.join(args.baselines)}")
    print(f"Max questions: {args.max_questions}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Clear cache if requested
    if args.clear_cache:
        print("\nClearing dataset cache")
        clear_dataset_cache()
    else:
        # Show cache info
        print("\nCache information:")
        cache_info = get_cache_info()
        print(f"Memory cache: {cache_info['memory_cache']['count']} datasets")
        print(f"Disk cache: {cache_info['disk_cache'].get('count', 0)} files")
    
    # Run comparison
    print(f"\nRunning baseline comparison")
    try:
        results = run_baseline_comparison(
            dataset_name=args.dataset,
            baselines=args.baselines,
            max_questions=args.max_questions,
            output_dir=args.output_dir
        )
        
        # Print summary
        print("Summary")
        
        for baseline_name, baseline_results in results.items():
            print(f"\n{baseline_name}:")
            
            if 'error' in baseline_results:
                print(f"Error: {baseline_results['error']}")
            elif 'metrics' in baseline_results:
                metrics = baseline_results['metrics']
                print(f"Exact Match: {metrics.exact_match:.3f}")
                print(f"F1 Score: {metrics.f1_score:.3f}")
                print(f"Temporal Accuracy: {metrics.temporal_accuracy:.3f}")
                print(f"Avg Total Time: {metrics.avg_retrieval_time + metrics.avg_reasoning_time:.3f}s")
                
                # Show breakdown by question type if available
                if hasattr(metrics, 'type_metrics') and metrics.type_metrics:
                    print(f"By question type:")
                    for qtype, type_metrics in metrics.type_metrics.items():
                        if type_metrics['count'] > 0:
                            print(f"{qtype}: EM={type_metrics['exact_match']:.3f}, "
                                  f"F1={type_metrics['f1_score']:.3f} (n={type_metrics['count']})")
        
        print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Show where results were saved
        if args.output_dir:
            output_dir = args.output_dir
        else:
            from src.utils.config import get_config
            config = get_config()
            output_dir = config.get_output_dir("baseline_comparison")
        
        print(f"\nResults saved to: {output_dir}")
        
        # List saved files
        if output_dir.exists():
            print("\nSaved files:")
            for file in sorted(output_dir.glob(f"*{args.dataset}*")):
                if file.is_file():
                    print(f" - {file.name}")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())