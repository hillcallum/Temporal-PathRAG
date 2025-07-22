"""
Evaluation pipeline for temporal embeddings
Measures the improvement of trained embeddings over baseline methods
"""

import torch
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path
import logging
import json
from datetime import datetime
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve, auc

from src.kg.retrieval.temporal_embedding_retriever import TemporalEmbeddingRetriever
from src.kg.retrieval.enhanced_temporal_pathrag import EnhancedTemporalPathRAG
from src.kg.scoring.updated_temporal_scoring import UpdatedTemporalScorer
from .baseline_runners import run_temporal_pathrag_baseline
from .temporal_qa_benchmarks import MultiTQBenchmark, TimeQuestionsBenchmark

logger = logging.getLogger(__name__)


class TemporalEmbeddingEvaluator:
    """Evaluator for temporal embedding performance"""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        output_dir: str = "./evaluation_results",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialise the evaluator
        """
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        
        # Initialise components
        if model_path:
            self.embedding_retriever = TemporalEmbeddingRetriever(
                model_path, device=device
            )
            self.enhanced_pathrag = EnhancedTemporalPathRAG(
                model_path=model_path,
                use_embeddings=True
            )
        else:
            self.embedding_retriever = None
            self.enhanced_pathrag = None
            
        self.baseline_scorer = UpdatedTemporalScorer()
        
        # Initialise benchmarks
        self.benchmarks = {
            'MultiTQ': MultiTQBenchmark(),
            'TimeQuestions': TimeQuestionsBenchmark()
        }
        
    def evaluate_retrieval_quality(
        self,
        dataset: str = "MultiTQ",
        num_samples: int = 1000,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate retrieval quality with and without embeddings
        """
        logger.info(f"Evaluating retrieval quality on {dataset}")
        
        benchmark = self.benchmarks.get(dataset)
        if not benchmark:
            raise ValueError(f"Unknown dataset: {dataset}")
        
        # Load test data
        test_data = benchmark.load_test_data()[:num_samples]
        
        results = {
            'dataset': dataset,
            'num_samples': num_samples,
            'timestamp': datetime.now().isoformat(),
            'metrics': {},
            'detailed_results': []
        }
        
        # Evaluate with different configurations
        configs = [
            ('baseline', False),
            ('with_embeddings', True)
        ]
        
        for config_name, use_embeddings in configs:
            logger.info(f"Evaluating {config_name}")
            
            if use_embeddings and not self.embedding_retriever:
                logger.warning("No embedding model loaded, skipping embedding evaluation")
                continue
                
            config_results = self.evaluate_config(
                test_data,
                benchmark,
                use_embeddings
            )
            
            results['metrics'][config_name] = config_results
            
        # Calculate improvements
        if 'baseline' in results['metrics'] and 'with_embeddings' in results['metrics']:
            improvements = {}
            for metric in results['metrics']['baseline']:
                baseline_val = results['metrics']['baseline'][metric]
                embed_val = results['metrics']['with_embeddings'][metric]
                if baseline_val > 0:
                    improvements[f'{metric}_improvement'] = (
                        (embed_val - baseline_val) / baseline_val * 100
                    )
            results['improvements'] = improvements
            
        # Save results
        if save_results:
            self.save_results(results, f"retrieval_eval_{dataset}")
            
        return results
    
    def evaluate_path_ranking(
        self,
        dataset: str = "MultiTQ",
        num_samples: int = 500
    ) -> Dict[str, Any]:
        """
        Evaluate path ranking quality
        """
        logger.info(f"Evaluating path ranking on {dataset}")
        
        benchmark = self.benchmarks.get(dataset)
        if not benchmark:
            raise ValueError(f"Unknown dataset: {dataset}")
            
        # Load test data
        test_data = benchmark.load_test_data()[:num_samples]
        
        results = {
            'dataset': dataset,
            'num_samples': num_samples,
            'timestamp': datetime.now().isoformat(),
            'ranking_metrics': {},
            'score_distributions': {}
        }
        
        # Collect ranking data
        baseline_rankings = []
        embedding_rankings = []
        
        for sample in tqdm(test_data, desc="Evaluating rankings"):
            query = sample['question']
            graph = sample.get('graph')
            
            if not graph:
                continue
                
            # Get paths (simplified for evaluation)
            paths = self.get_sample_paths(query, graph)
            
            if not paths:
                continue
                
            # Score with baseline
            baseline_scores = []
            for path in paths:
                score = self.baseline_scorer.score_path(path, query, graph)
                baseline_scores.append(score)
            baseline_rankings.append(baseline_scores)
            
            # Score with embeddings
            if self.embedding_retriever:
                embedding_scores = self.embedding_retriever.score_paths(
                    query, paths, graph
                )
                embedding_rankings.append(embedding_scores)
                
        # Calculate ranking metrics
        results['ranking_metrics'] = {
            'baseline': self.calculate_ranking_metrics(baseline_rankings),
            'with_embeddings': self.calculate_ranking_metrics(embedding_rankings)
        }
        
        # Save score distributions
        results['score_distributions'] = {
            'baseline': baseline_rankings,
            'with_embeddings': embedding_rankings
        }
        
        return results
    
    def evaluate_end_to_end(
        self,
        dataset: str = "MultiTQ",
        num_samples: int = 100,
        llm_client: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Evaluate end-to-end QA performance
        """
        logger.info(f"Evaluating end-to-end QA on {dataset}")
        
        if not llm_client:
            logger.warning("No LLM client provided, skipping end-to-end evaluation")
            return {}
            
        benchmark = self.benchmarks.get(dataset)
        if not benchmark:
            raise ValueError(f"Unknown dataset: {dataset}")
            
        # Run evaluation
        results = {
            'dataset': dataset,
            'num_samples': num_samples,
            'timestamp': datetime.now().isoformat(),
            'qa_metrics': {}
        }
        
        # Evaluate baseline
        baseline_results = run_temporal_pathrag_baseline(
            benchmark=benchmark,
            num_samples=num_samples,
            llm_client=llm_client
        )
        results['qa_metrics']['baseline'] = baseline_results
        
        # Evaluate with embeddings
        if self.enhanced_pathrag:
            enhanced_results = self.run_enhanced_evaluation(
                benchmark,
                num_samples,
                llm_client
            )
            results['qa_metrics']['with_embeddings'] = enhanced_results
            
        return results
    
    def evaluate_config(
        self,
        test_data: List[Dict],
        benchmark: Any,
        use_embeddings: bool
    ) -> Dict[str, float]:
        """
        Evaluate a specific configuration
        """
        predictions = []
        ground_truths = []
        retrieval_scores = []
        
        for sample in tqdm(test_data, desc=f"Evaluating (embeddings={use_embeddings})"):
            query = sample['question']
            graph = sample.get('graph')
            answer = sample['answer']
            
            if not graph:
                continue
                
            # Retrieve paths
            if use_embeddings and self.enhanced_pathrag:
                path_scores = self.enhanced_pathrag.retrieve_paths(
                    query, graph, top_k=10
                )
            else:
                # Baseline retrieval
                paths = self.get_sample_paths(query, graph)
                path_scores = []
                for path in paths:
                    score = self.baseline_scorer.score_path(path, query, graph)
                    path_scores.append((path, score))
                path_scores.sort(key=lambda x: x[1], reverse=True)
                path_scores = path_scores[:10]
                
            # Extract predicted answer from top paths
            if path_scores:
                predicted = self.extract_answer_from_paths(
                    path_scores, query, graph
                )
                retrieval_scores.append(path_scores[0][1])
            else:
                predicted = "No answer found"
                retrieval_scores.append(0.0)
                
            predictions.append(predicted)
            ground_truths.append(answer)
            
        # Calculate metrics
        metrics = benchmark.evaluate(predictions, ground_truths)
        metrics['avg_retrieval_score'] = np.mean(retrieval_scores)
        metrics['retrieval_coverage'] = sum(s > 0 for s in retrieval_scores) / len(retrieval_scores)
        
        return metrics
    
    def get_sample_paths(
        self,
        query: str,
        graph: Any,
        max_paths: int = 20
    ) -> List[List[Tuple[str, str, str]]]:
        """
        Get sample paths for evaluation
        """
        # This is a simplified version for evaluation
        # In future, will use the full path retrieval logic
        paths = []
        
        # Extract entities from query
        entities = []
        for node in graph.nodes():
            if str(node).lower() in query.lower():
                entities.append(node)
                
        # Find paths between entities
        for i, start in enumerate(entities):
            for end in entities[i+1:]:
                try:
                    # Simple path finding
                    if hasattr(graph, 'edges'):
                        # Create simple paths
                        if graph.has_edge(start, end):
                            edge_data = graph[start][end]
                            rel = edge_data.get('relation', 'connected_to')
                            paths.append([(start, rel, end)])
                except:
                    pass
                    
                if len(paths) >= max_paths:
                    break
                    
        return paths[:max_paths]
    
    def extract_answer_from_paths(
        self,
        path_scores: List[Tuple[List[Tuple[str, str, str]], float]],
        query: str,
        graph: Any
    ) -> str:
        """
        Extract answer from top paths
        """
        # Simple extraction logic for evaluation
        if not path_scores:
            return "No answer found"
            
        # Use the top path
        top_path, _ = path_scores[0]
        
        # Extract entities from path
        entities = set()
        for source, _, target in top_path:
            entities.add(source)
            entities.add(target)
            
        # Return the most relevant entity
        # This is simplified - real implementation would be more sophisticated
        for entity in entities:
            if entity.lower() not in query.lower():
                return str(entity)
                
        return str(list(entities)[0]) if entities else "No answer found"
    
    def calculate_ranking_metrics(
        self,
        rankings: List[List[float]]
    ) -> Dict[str, float]:
        """
        Calculate ranking quality metrics
        """
        if not rankings:
            return {}
            
        metrics = {}
        
        # Calculate average scores
        all_scores = [score for ranking in rankings for score in ranking]
        metrics['avg_score'] = np.mean(all_scores)
        metrics['std_score'] = np.std(all_scores)
        
        # Calculate score separability
        top_scores = [max(ranking) if ranking else 0 for ranking in rankings]
        other_scores = [s for ranking in rankings for s in ranking[1:]]
        
        if top_scores and other_scores:
            metrics['top_score_avg'] = np.mean(top_scores)
            metrics['other_score_avg'] = np.mean(other_scores)
            metrics['score_separation'] = metrics['top_score_avg'] - metrics['other_score_avg']
            
        # Calculate ranking consistency
        rank_correlations = []
        for i in range(len(rankings) - 1):
            if len(rankings[i]) > 1 and len(rankings[i+1]) > 1:
                corr = np.corrcoef(rankings[i], rankings[i+1][:len(rankings[i])])[0, 1]
                if not np.isnan(corr):
                    rank_correlations.append(corr)
                    
        if rank_correlations:
            metrics['ranking_consistency'] = np.mean(rank_correlations)
            
        return metrics
    
    def run_enhanced_evaluation(
        self,
        benchmark: Any,
        num_samples: int,
        llm_client: Any
    ) -> Dict[str, float]:
        """
        Run evaluation with enhanced PathRAG
        """
        test_data = benchmark.load_test_data()[:num_samples]
        predictions = []
        ground_truths = []
        
        for sample in tqdm(test_data, desc="Enhanced evaluation"):
            query = sample['question']
            answer = sample['answer']
            graph = sample.get('graph')
            
            if not graph:
                predictions.append("No graph available")
                ground_truths.append(answer)
                continue
                
            # Get answer using enhanced PathRAG
            try:
                predicted = self.enhanced_pathrag.answer_query(
                    query,
                    graph,
                    llm_client
                )
            except Exception as e:
                logger.error(f"Error in enhanced evaluation: {e}")
                predicted = "Error generating answer"
                
            predictions.append(predicted)
            ground_truths.append(answer)
            
        # Calculate metrics
        return benchmark.evaluate(predictions, ground_truths)
    
    def save_results(self, results: Dict, prefix: str):
        """
        Save evaluation results
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.output_dir / f"{prefix}_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
            
        logger.info(f"Saved results to {filename}")
        
    def create_visualisations(self, results: Dict, save_dir: Optional[str] = None):
        """
        Create visualisations of evaluation results
        """
        if save_dir is None:
            save_dir = self.output_dir / "plots"
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Plot retrieval quality metrics
        if 'metrics' in results:
            self.plot_metric_comparison(
                results['metrics'],
                save_dir / "retrieval_metrics.png"
            )
            
        # Plot score distributions
        if 'score_distributions' in results:
            self.plot_score_distributions(
                results['score_distributions'],
                save_dir / "score_distributions.png"
            )
            
        # Plot improvements
        if 'improvements' in results:
            self.plot_improvements(
                results['improvements'],
                save_dir / "improvements.png"
            )
            
    def plot_metric_comparison(self, metrics: Dict, save_path: Path):
        """
        Plot comparison of metrics
        """
        plt.figure(figsize=(10, 6))
        
        configs = list(metrics.keys())
        metric_names = list(metrics[configs[0]].keys())
        
        x = np.arange(len(metric_names))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for i, config in enumerate(configs):
            values = [metrics[config].get(m, 0) for m in metric_names]
            ax.bar(x + i * width, values, width, label=config)
            
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Value')
        ax.set_title('Retrieval Quality Metrics Comparison')
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(metric_names, rotation=45, ha='right')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
    def plot_score_distributions(self, distributions: Dict, save_path: Path):
        """
        Plot score distributions
        """
        plt.figure(figsize=(10, 6))
        
        for config, rankings in distributions.items():
            if rankings:
                all_scores = [score for ranking in rankings for score in ranking]
                plt.hist(all_scores, bins=50, alpha=0.5, label=config, density=True)
                
        plt.xlabel('Score')
        plt.ylabel('Density')
        plt.title('Score Distributions')
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
    def plot_improvements(self, improvements: Dict, save_path: Path):
        """
        Plot improvement percentages
        """
        plt.figure(figsize=(10, 6))
        
        metrics = list(improvements.keys())
        values = list(improvements.values())
        
        colours = ['green' if v > 0 else 'red' for v in values]
        
        plt.bar(metrics, values, colour=colours)
        plt.axhline(y=0, colour='black', linestyle='-', linewidth=0.5)
        plt.xlabel('Metric')
        plt.ylabel('Improvement (%)')
        plt.title('Performance Improvements with Temporal Embeddings')
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels
        for i, v in enumerate(values):
            plt.text(i, v + (1 if v > 0 else -1), f'{v:.1f}%', ha='center')
            
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()