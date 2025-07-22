#!/usr/bin/env python3
"""
Script to evaluate trained temporal embeddings against baselines
"""

import argparse
import logging
import sys
from pathlib import Path
import torch
import json
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from evaluation.temporal_embedding_evaluator import TemporalEmbeddingEvaluator
from evaluation.temporal_qa_benchmarks import MultiTQBenchmark, TimeQuestionsBenchmark

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained temporal embeddings"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained temporal embedding model"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./evaluation_results",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["MultiTQ", "TimeQuestions", "both"],
        default="both",
        help="Dataset(s) to evaluate on"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1000,
        help="Number of samples to evaluate"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run evaluation on"
    )
    parser.add_argument(
        "--eval_types",
        nargs="+",
        choices=["retrieval", "ranking", "end_to_end"],
        default=["retrieval", "ranking"],
        help="Types of evaluation to run"
    )
    parser.add_argument(
        "--create_plots",
        action="store_true",
        help="Create visualisation plots"
    )
    parser.add_argument(
        "--compare_to_baseline",
        action="store_true",
        help="Run baseline comparison"
    )
    
    args = parser.parse_args()
    
    # Check if model exists
    model_path = Path(args.model_path)
    if not model_path.exists():
        logger.error(f"Model path does not exist: {model_path}")
        return 1
        
    # Initialise evaluator
    logger.info(f"Initialising evaluator with model from {model_path}")
    evaluator = TemporalEmbeddingEvaluator(
        model_path=str(model_path),
        output_dir=args.output_dir,
        device=args.device
    )
    
    # Determine datasets to evaluate
    datasets = []
    if args.dataset == "both":
        datasets = ["MultiTQ", "TimeQuestions"]
    else:
        datasets = [args.dataset]
        
    # Run evaluations
    all_results = {}
    
    for dataset in datasets:
        logger.info(f"\nEvaluating on {dataset}")
        dataset_results = {}
        
        # Retrieval quality evaluation
        if "retrieval" in args.eval_types:
            logger.info("Running retrieval quality evaluation")
            retrieval_results = evaluator.evaluate_retrieval_quality(
                dataset=dataset,
                num_samples=args.num_samples,
                save_results=True
            )
            dataset_results["retrieval"] = retrieval_results
            
            # Print summary
            if "metrics" in retrieval_results:
                print(f"\nRetrieval Quality Results for {dataset}:")
                for config, metrics in retrieval_results["metrics"].items():
                    print(f"\n{config}:")
                    for metric, value in metrics.items():
                        print(f"{metric}: {value:.4f}")
                        
                if "improvements" in retrieval_results:
                    print(f"\nImprovements over baseline:")
                    for metric, improvement in retrieval_results["improvements"].items():
                        print(f"{metric}: {improvement:.2f}%")
                        
        # Path ranking evaluation
        if "ranking" in args.eval_types:
            logger.info("Running path ranking evaluation")
            ranking_results = evaluator.evaluate_path_ranking(
                dataset=dataset,
                num_samples=min(args.num_samples, 500)  # Ranking eval is slower
            )
            dataset_results["ranking"] = ranking_results
            
            # Print summary
            if "ranking_metrics" in ranking_results:
                print(f"\nPath Ranking Results for {dataset}:")
                for config, metrics in ranking_results["ranking_metrics"].items():
                    print(f"\n{config}:")
                    for metric, value in metrics.items():
                        print(f"{metric}: {value:.4f}")
                        
        # End-to-end evaluation (requires LLM)
        if "end_to_end" in args.eval_types:
            logger.warning("End-to-end evaluation requires LLM client setup")
            logger.info("Skipping end-to-end evaluation for now")
            # TODO: Add LLM client initialisation and run end-to-end eval
            
        all_results[dataset] = dataset_results
        
    # Create visualisations
    if args.create_plots:
        logger.info("Creating visualisation plots")
        for dataset, results in all_results.items():
            if "retrieval" in results:
                evaluator.create_visualisations(
                    results["retrieval"],
                    save_dir=Path(args.output_dir) / "plots" / dataset
                )
                
    # Save combined results
    combined_results = {
        "timestamp": datetime.now().isoformat(),
        "model_path": str(model_path),
        "evaluation_config": vars(args),
        "results": all_results
    }
    
    output_path = Path(args.output_dir) / f"combined_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_path, 'w') as f:
        json.dump(combined_results, f, indent=2, default=str)
        
    logger.info(f"\nEvaluation complete - results saved to {output_path}")
    
    # Print final summary
    print("Evaluation Summary")
    
    for dataset in datasets:
        print(f"\n{dataset}:")
        if dataset in all_results and "retrieval" in all_results[dataset]:
            retrieval = all_results[dataset]["retrieval"]
            if "improvements" in retrieval:
                avg_improvement = sum(retrieval["improvements"].values()) / len(retrieval["improvements"])
                print(f"Average improvement: {avg_improvement:.2f}%")
                
    return 0


if __name__ == "__main__":
    sys.exit(main())