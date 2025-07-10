"""
Ablation Study Framework for Temporal PathRAG
"""

import json
import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Tuple
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from .temporal_qa_benchmarks import (
    TemporalQABenchmark,
    MultiTQBenchmark,
    TimeQuestionsBenchmark,
    TemporalMetrics,
    create_benchmark
)
from .baseline_runners import TemporalPathRAGBaseline
from src.utils.config import get_config


@dataclass
class AblationComponent:
    """Represents a component that can be ablated"""
    name: str
    description: str
    config_key: str
    default_value: Any
    ablation_value: Any
    category: str = "general"


@dataclass
class AblationResult:
    """Result of an ablation experiment"""
    configuration: Dict[str, Any]
    metrics: TemporalMetrics
    component_states: Dict[str, bool]  # True if component is active
    runtime: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class AblationStudy:
    """
    Ablation study framework for Temporal PathRAG
    """
    
    def __init__(self, dataset_name: str, output_dir: Optional[Path] = None):
        """
        Initialise ablation study
        """
        self.dataset_name = dataset_name
        self.benchmark = create_benchmark(dataset_name)
        self.config = get_config()
        
        if output_dir is None:
            self.output_dir = self.config.get_output_dir(f"ablation_study/{dataset_name}")
        else:
            self.output_dir = output_dir
            
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define default components for ablation
        self.components = self.define_default_components()
        self.results: List[AblationResult] = []
        
    def define_default_components(self) -> List[AblationComponent]:
        """Define the default components for ablation study"""
        return [
            # Temporal Components
            AblationComponent(
                name="temporal_weighting",
                description="Temporal weighting in path scoring",
                config_key="use_temporal_weighting",
                default_value=True,
                ablation_value=False,
                category="temporal"
            ),
            AblationComponent(
                name="temporal_flow_pruning",
                description="Temporal-aware flow-based pruning",
                config_key="use_temporal_pruning",
                default_value=True,
                ablation_value=False,
                category="temporal"
            ),
            AblationComponent(
                name="temporal_stopping",
                description="Temporal-aware stopping controller",
                config_key="use_temporal_stopping",
                default_value=True,
                ablation_value=False,
                category="temporal"
            ),
            
            # Iterative Components
            AblationComponent(
                name="iterative_reasoning",
                description="Multi-step iterative reasoning",
                config_key="use_iterative_reasoning",
                default_value=True,
                ablation_value=False,
                category="iterative"
            ),
            AblationComponent(
                name="dynamic_depth",
                description="Dynamic retrieval depth adjustment",
                config_key="use_dynamic_depth",
                default_value=True,
                ablation_value=False,
                category="iterative"
            ),
            
            # Path-based Components
            AblationComponent(
                name="path_reliability",
                description="Path reliability scoring",
                config_key="use_reliability_scoring",
                default_value=True,
                ablation_value=False,
                category="path"
            ),
            AblationComponent(
                name="path_ordering",
                description="Strategic path ordering in prompts",
                config_key="use_path_ordering",
                default_value=True,
                ablation_value=False,
                category="path"
            ),
            
            # Context Components
            AblationComponent(
                name="time_ordered_context",
                description="Chronological ordering of context",
                config_key="use_time_ordering",
                default_value=True,
                ablation_value=False,
                category="context"
            ),
            AblationComponent(
                name="context_compression",
                description="Context compression and summarisation",
                config_key="use_context_compression",
                default_value=True,
                ablation_value=False,
                category="context"
            ),
        ]
        
    def add_component(self, component: AblationComponent) -> None:
        """Add a component to the ablation study"""
        self.components.append(component)
        
    def create_configuration(self, active_components: List[str]) -> Dict[str, Any]:
        """
        Create a configuration with specified components active
        """
        config = {}
        component_states = {}
        
        for component in self.components:
            if component.name in active_components:
                config[component.config_key] = component.default_value
                component_states[component.name] = True
            else:
                config[component.config_key] = component.ablation_value
                component_states[component.name] = False
                
        return config, component_states
        
    def run_single_ablation(self, active_components: List[str],
                           max_questions: Optional[int] = None) -> AblationResult:
        """
        Run a single ablation experiment
        """
        # Create configuration
        config, component_states = self.create_configuration(active_components)
        
        print(f"\nRunning ablation with components: {active_components}")
        
        # Create system with configuration
        system = TemporalPathRAGBaseline(config_override=config)
        
        # Run predictions
        start_time = datetime.now()
        predictions = system.run_benchmark(
            self.benchmark,
            max_questions=max_questions,
            verbose=False
        )
        runtime = (datetime.now() - start_time).total_seconds()
        
        # Evaluate
        metrics = self.benchmark.evaluate(predictions, verbose=False)
        
        # Create result
        result = AblationResult(
            configuration=config,
            metrics=metrics,
            component_states=component_states,
            runtime=runtime,
            metadata={
                'num_questions': len(predictions),
                'timestamp': datetime.now().isoformat()
            }
        )
        
        return result
        
    def run_full_ablation_study(self, 
                               ablation_type: str = "leave_one_out",
                               max_questions: Optional[int] = None) -> None:
        """
        Run complete ablation study
        """

        print(f"Starting {ablation_type} ablation study on {self.dataset_name}")
        
        component_names = [c.name for c in self.components]
        
        if ablation_type == "leave_one_out":
            # Full system baseline
            print("\n1. Running full system baseline")
            baseline_result = self.run_single_ablation(component_names, max_questions)
            self.results.append(baseline_result)
            
            # Remove one component at a time
            for i, component in enumerate(self.components):
                print(f"\n{i+2}. Ablating {component.name}")
                active = [c for c in component_names if c != component.name]
                result = self.run_single_ablation(active, max_questions)
                self.results.append(result)
                
        elif ablation_type == "incremental":
            # Start with no components
            print("\n1. Running with no components")
            result = self.run_single_ablation([], max_questions)
            self.results.append(result)
            
            # Add components one by one
            active = []
            for i, component in enumerate(self.components):
                active.append(component.name)
                print(f"\n{i+2}. Adding {component.name}")
                result = self.run_single_ablation(active.copy(), max_questions)
                self.results.append(result)
                
        elif ablation_type == "category":
            # Get unique categories
            categories = list(set(c.category for c in self.components))
            
            # Full system baseline
            print("\n1. Running full system baseline")
            baseline_result = self.run_single_ablation(component_names, max_questions)
            self.results.append(baseline_result)
            
            # Remove one category at a time
            for i, category in enumerate(categories):
                print(f"\n{i+2}. Ablating category: {category}")
                active = [c.name for c in self.components if c.category != category]
                result = self.run_single_ablation(active, max_questions)
                self.results.append(result)
                
        elif ablation_type == "all_combinations":
            # Warning for expensive operation
            num_combinations = 2 ** len(self.components)
            if num_combinations > 32:
                print(f"Warning: Testing {num_combinations} combinations")
                response = input("Continue? (y/n): ")
                if response.lower() != 'y':
                    return
                    
            # Test all combinations
            for i in range(num_combinations):
                active = []
                for j, component in enumerate(self.components):
                    if i & (1 << j):
                        active.append(component.name)
                        
                print(f"\nCombination {i+1}/{num_combinations}: {active}")
                result = self.run_single_ablation(active, max_questions)
                self.results.append(result)
                
        else:
            raise ValueError(f"Unknown ablation type: {ablation_type}")
            
        # Save results
        self.save_results()
        
        # Generate analysis
        self.analyse_results()
        
    def analyse_results(self) -> Dict[str, Any]:
        """Analyse ablation results and compute component importance"""
        if not self.results:
            print("No results to analyse")
            return {}
            
        analysis = {
            'dataset': self.dataset_name,
            'num_experiments': len(self.results),
            'timestamp': datetime.now().isoformat(),
            'component_importance': {},
            'category_importance': {},
            'interaction_effects': {},
            'performance_summary': {}
        }
        
        # Find baseline (all components active)
        baseline_result = None
        for result in self.results:
            if all(result.component_states.values()):
                baseline_result = result
                break
                
        if baseline_result:
            baseline_em = baseline_result.metrics.exact_match
            baseline_f1 = baseline_result.metrics.f1_score
            baseline_temp = baseline_result.metrics.temporal_accuracy
            
            # Compute component importance (performance drop when removed)
            for component in self.components:
                # Find result with this component removed
                for result in self.results:
                    if (not result.component_states.get(component.name, True) and
                        sum(result.component_states.values()) == len(self.components) - 1):
                        
                        importance = {
                            'exact_match_drop': baseline_em - result.metrics.exact_match,
                            'f1_drop': baseline_f1 - result.metrics.f1_score,
                            'temporal_accuracy_drop': baseline_temp - result.metrics.temporal_accuracy,
                            'relative_importance': (baseline_em - result.metrics.exact_match) / baseline_em if baseline_em > 0 else 0
                        }
                        analysis['component_importance'][component.name] = importance
                        break
                        
            # Compute category importance
            categories = list(set(c.category for c in self.components))
            for category in categories:
                category_components = [c.name for c in self.components if c.category == category]
                total_importance = sum(
                    analysis['component_importance'].get(c, {}).get('exact_match_drop', 0)
                    for c in category_components
                )
                analysis['category_importance'][category] = {
                    'total_importance': total_importance,
                    'avg_importance': total_importance / len(category_components) if category_components else 0,
                    'components': category_components
                }
                
        # Performance summary statistics
        all_exact_match = [r.metrics.exact_match for r in self.results]
        all_f1 = [r.metrics.f1_score for r in self.results]
        all_temporal = [r.metrics.temporal_accuracy for r in self.results]
        
        analysis['performance_summary'] = {
            'exact_match': {
                'mean': np.mean(all_exact_match),
                'std': np.std(all_exact_match),
                'min': np.min(all_exact_match),
                'max': np.max(all_exact_match)
            },
            'f1_score': {
                'mean': np.mean(all_f1),
                'std': np.std(all_f1),
                'min': np.min(all_f1),
                'max': np.max(all_f1)
            },
            'temporal_accuracy': {
                'mean': np.mean(all_temporal),
                'std': np.std(all_temporal),
                'min': np.min(all_temporal),
                'max': np.max(all_temporal)
            }
        }
        
        # Save analysis
        analysis_path = self.output_dir / "ablation_analysis.json"
        with open(analysis_path, 'w') as f:
            json.dump(analysis, f, indent=2)
            
        print(f"\nAnalysis saved to: {analysis_path}")
        
        # Generate visualisations
        self.visualise_results(analysis)
        
        return analysis
        
    def visualise_results(self, analysis: Dict[str, Any]) -> None:
        """Generate visualisations of ablation results."""
        # Component importance bar chart
        if analysis.get('component_importance'):
            plt.figure(figsize=(10, 6))
            components = list(analysis['component_importance'].keys())
            importance_values = [
                analysis['component_importance'][c]['exact_match_drop'] 
                for c in components
            ]
            
            # Sort by importance
            sorted_indices = np.argsort(importance_values)[::-1]
            components = [components[i] for i in sorted_indices]
            importance_values = [importance_values[i] for i in sorted_indices]
            
            plt.bar(range(len(components)), importance_values)
            plt.xticks(range(len(components)), components, rotation=45, ha='right')
            plt.ylabel('Exact Match Performance Drop')
            plt.title(f'Component Importance - {self.dataset_name}')
            plt.tight_layout()
            
            importance_path = self.output_dir / "component_importance.png"
            plt.savefig(importance_path)
            plt.close()
            
        # Performance distribution
        if self.results:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            metrics_data = {
                'Exact Match': [r.metrics.exact_match for r in self.results],
                'F1 Score': [r.metrics.f1_score for r in self.results],
                'Temporal Accuracy': [r.metrics.temporal_accuracy for r in self.results]
            }
            
            for ax, (metric_name, values) in zip(axes, metrics_data.items()):
                ax.hist(values, bins=20, alpha=0.7, edgecolor='black')
                ax.axvline(np.mean(values), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(values):.3f}')
                ax.set_xlabel(metric_name)
                ax.set_ylabel('Frequency')
                ax.set_title(f'{metric_name} Distribution')
                ax.legend()
                
            plt.suptitle(f'Performance Distributions - {self.dataset_name}')
            plt.tight_layout()
            
            dist_path = self.output_dir / "performance_distributions.png"
            plt.savefig(dist_path)
            plt.close()
            
        print(f"Visualisations saved to: {self.output_dir}")
        
    def save_results(self) -> None:
        """Save all ablation results"""
        results_data = []
        for result in self.results:
            results_data.append({
                'configuration': result.configuration,
                'component_states': result.component_states,
                'metrics': {
                    'exact_match': result.metrics.exact_match,
                    'f1_score': result.metrics.f1_score,
                    'temporal_accuracy': result.metrics.temporal_accuracy,
                    'entity_accuracy': result.metrics.entity_accuracy,
                    'avg_retrieval_time': result.metrics.avg_retrieval_time,
                    'avg_reasoning_time': result.metrics.avg_reasoning_time,
                },
                'runtime': result.runtime,
                'metadata': result.metadata
            })
            
        results_path = self.output_dir / "ablation_results.json"
        with open(results_path, 'w') as f:
            json.dump(results_data, f, indent=2)
            
        print(f"Results saved to: {results_path}")
        
    def get_statistical_significance(self, 
                                   component_name: str,
                                   metric: str = "exact_match") -> Dict[str, float]:
        """
        Compute statistical significance of a component's contribution
        """
        # Collect results with and without the component
        with_component = []
        without_component = []
        
        for result in self.results:
            metric_value = getattr(result.metrics, metric)
            if result.component_states.get(component_name, False):
                with_component.append(metric_value)
            else:
                without_component.append(metric_value)
                
        if not with_component or not without_component:
            return {'error': 'Insufficient data for statistical test'}
            
        # Perform t-test
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(with_component, without_component)
        
        return {
            'component': component_name,
            'metric': metric,
            'mean_with': np.mean(with_component),
            'mean_without': np.mean(without_component),
            'difference': np.mean(with_component) - np.mean(without_component),
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }


def run_ablation(dataset_names: List[str] = ["MultiTQ", "TimeQuestions"],
                              ablation_types: List[str] = ["leave_one_out", "category"],
                              max_questions: Optional[int] = None) -> None:
    """
    Run ablation study across multiple datasets and types
    """
    for dataset in dataset_names:
        for ablation_type in ablation_types:
            print(f"\n{'='*60}")
            print(f"Running {ablation_type} ablation on {dataset}")
            print(f"{'='*60}")
            
            study = AblationStudy(dataset)
            study.run_full_ablation_study(
                ablation_type=ablation_type,
                max_questions=max_questions
            )


if __name__ == "__main__":
    # Example usage
    print("Running ablation study example")
    
    # Create ablation study
    study = AblationStudy("MultiTQ")
    
    # Run leave-one-out ablation on small subset
    study.run_full_ablation_study(
        ablation_type="leave_one_out",
        max_questions=20  # Small subset for testing
    )
    
    # Test statistical significance
    if study.results:
        sig_test = study.get_statistical_significance(
            "temporal_weighting",
            metric="exact_match"
        )
        print(f"\nStatistical significance test:")
        print(json.dumps(sig_test, indent=2))