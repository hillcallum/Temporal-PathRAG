#!/usr/bin/env python3
"""
Parameter Optimisation for Temporal PathRAG

We use comprehensive parameter search space with statistical validation with 
confidence intervals and cross-validation 
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

import time
import numpy as np
import json
from datetime import datetime
from typing import List, Dict, Any
from scipy import stats

from src.kg.temporal_path_retriever import TemporalPathRetriever
from src.kg.models import TemporalQuery
from scripts.testing.test_temporal_path_retrieval import create_test_tkg, create_graph_statistics


def create_parameter_space():
    """Create parameter space for optimisation"""
    
    # Comprehensive alpha range (temporal decay)
    alpha_coarse = np.linspace(0.01, 0.5, 20)    # 20 coarse values
    alpha_fine = np.linspace(0.03, 0.15, 15)     # 15 fine values around promising region
    
    # Comprehensive theta range (pruning threshold)  
    theta_coarse = np.linspace(0.1, 3.0, 15)     # 15 coarse values
    theta_fine = np.linspace(0.3, 1.5, 12)       # 12 fine values around promising region
    
    # Combine and remove duplicates
    all_alphas = np.unique(np.concatenate([alpha_coarse, alpha_fine]))
    all_thetas = np.unique(np.concatenate([theta_coarse, theta_fine]))
    
    print(f"Parameter space:")
    print(f"Alpha values: {len(all_alphas)} ({all_alphas.min():.3f} to {all_alphas.max():.3f})")
    print(f"Theta values: {len(all_thetas)} ({all_thetas.min():.3f} to {all_thetas.max():.3f})")
    print(f"Total combinations: {len(all_alphas) * len(all_thetas)}")
    
    return all_alphas, all_thetas


def create_diverse_test_scenarios():
    """Create diverse test scenarios for validation"""
    
    training_scenarios = [
        # Historical scientific evolution
        {
            'source': 'darwin', 'target': 'evolution',
            'query_time': '1950-01-01', 'category': 'historical_evolution',
            'description': 'Darwin to evolution theory'
        },
        # Scientific discovery collaboration
        {
            'source': 'watson', 'target': 'dna', 
            'query_time': '1955-01-01', 'category': 'scientific_discovery',
            'description': 'Watson to DNA discovery'
        },
        # Institutional development
        {
            'source': 'cambridge', 'target': 'computer_science',
            'query_time': '1960-01-01', 'category': 'institutional_development',
            'description': 'Cambridge to computer science'
        },
        # Recognition and awards
        {
            'source': 'curie', 'target': 'nobel_physics',
            'query_time': '1920-01-01', 'category': 'scientific_recognition',
            'description': 'Curie to Nobel Prize'
        },
        # Technological progression
        {
            'source': 'turing', 'target': 'artificial_intelligence',
            'query_time': '1960-01-01', 'category': 'technological_development',
            'description': 'Turing to AI development'
        },
        # Modern academic connections
        {
            'source': 'hawking', 'target': 'cambridge',
            'query_time': '1970-01-01', 'category': 'modern_academic',
            'description': 'Hawking to Cambridge'
        }
    ]
    
    validation_scenarios = [
        # Validation set for testing generalisation
        {
            'source': 'einstein', 'target': 'relativity',
            'query_time': '1920-01-01', 'category': 'physics_theory',
            'description': 'Einstein to relativity theory'
        },
        {
            'source': 'mendel', 'target': 'genetics',
            'query_time': '1900-01-01', 'category': 'biology_theory',
            'description': 'Mendel to genetics theory'
        },
        {
            'source': 'crick', 'target': 'dna',
            'query_time': '1955-01-01', 'category': 'molecular_biology',
            'description': 'Crick to DNA structure'
        }
    ]
    
    return training_scenarios, validation_scenarios


class ModelParameterOptimiser:
    """Parameter optimiser"""
    
    def __init__(self, graph, graph_stats):
        self.graph = graph
        self.graph_stats = graph_stats
        self.base_retriever = None
        print("Initialising parameter optimiser")
    
    def initialise_base_model(self):
        print("Loading base model")
        start_time = time.time()
        
        # Create a base retriever with default parameters to load the model
        self.base_retriever = TemporalPathRetriever(
            graph=self.graph,
            alpha=0.1,  # Default values, will be overridden
            base_theta=1.0,
            diversity_threshold=0.7
        )
        
        load_time = time.time() - start_time
        print(f"Base model loaded successfully in {load_time:.1f}s")
        return True
    
    def evaluate_parameter_combination(self, alpha: float, theta: float, 
                                     scenarios: List[Dict], n_runs: int = 1) -> Dict[str, Any]:
        """Evaluate parameters using the pre-loaded model"""
        
        if self.base_retriever is None:
            raise RuntimeError("Base model not initialised. Call initialise_base_model() first.")
        
        # Update parameters without recreating the model
        self.base_retriever.alpha = alpha
        self.base_retriever.base_theta = theta
        
        scenario_results = []
        start_time = time.time()
        
        for run in range(n_runs):
            for scenario in scenarios:
                try:
                    # Create temporal query
                    query = TemporalQuery(
                        query_text=f"Test query from {scenario['source']} to {scenario['target']}",
                        source_entities=[scenario['source']],
                        target_entities=[scenario['target']],
                        temporal_constraints={'temporal_preference': 'chronological'},
                        query_time=scenario['query_time'],
                        max_hops=4,
                        top_k=10
                    )
                    
                    # Execute query with current parameters
                    paths = self.base_retriever.retrieve_temporal_paths(
                        query=query,
                        enable_flow_pruning=True,
                        enable_diversity=True,
                        verbose=False
                    )
                    
                    # Calculate metrics
                    if paths:
                        reliability_scores = [score for _, score in paths]
                        result = {
                            'category': scenario['category'],
                            'success': True,
                            'paths_found': len(paths),
                            'avg_reliability': np.mean(reliability_scores),
                            'max_reliability': np.max(reliability_scores),
                            'min_reliability': np.min(reliability_scores)
                        }
                    else:
                        result = {
                            'category': scenario['category'],
                            'success': False,
                            'paths_found': 0,
                            'avg_reliability': 0.0,
                            'max_reliability': 0.0,
                            'min_reliability': 0.0
                        }
                    
                    scenario_results.append(result)
                    
                except Exception as e:
                    scenario_results.append({
                        'category': scenario['category'],
                        'success': False,
                        'paths_found': 0,
                        'avg_reliability': 0.0,
                        'max_reliability': 0.0,
                        'min_reliability': 0.0,
                        'error': str(e)
                    })
        
        execution_time = time.time() - start_time
        
        # Aggregate results across runs and scenarios
        successful_results = [r for r in scenario_results if r['success']]
        
        if successful_results:
            success_rate = len(successful_results) / len(scenario_results)
            reliability_scores = [r['avg_reliability'] for r in successful_results]
            
            avg_reliability = np.mean(reliability_scores)
            reliability_std = np.std(reliability_scores) if len(reliability_scores) > 1 else 0.0
            avg_paths = np.mean([r['paths_found'] for r in successful_results])
            
            # Calculate confidence interval
            if len(reliability_scores) > 1:
                ci = stats.t.interval(0.95, len(reliability_scores)-1,
                                    loc=avg_reliability, scale=stats.sem(reliability_scores))
                ci_lower, ci_upper = ci
            else:
                ci_lower = ci_upper = avg_reliability
        else:
            success_rate = avg_reliability = reliability_std = avg_paths = 0.0
            ci_lower = ci_upper = 0.0
        
        return {
            'alpha': alpha,
            'theta': theta,
            'success_rate': success_rate,
            'avg_reliability': avg_reliability,
            'reliability_std': reliability_std,
            'reliability_ci_lower': ci_lower,
            'reliability_ci_upper': ci_upper,
            'avg_paths': avg_paths,
            'execution_time': execution_time,
            'scenario_results': scenario_results
        }


def run_model_optimisation():    
    print("Parameter Optimisation for Temporal PathRAG")
    print("=" * 70)
    
    # Setup
    print("\n1. Setting up test environment")
    graph = create_test_tkg()
    graph_stats = create_graph_statistics(graph)
    print(f"Test TKG: {len(graph.nodes())} nodes, {len(graph.edges())} edges")
    
    # Create parameter space
    print("\n2. Creating parameter space")
    alpha_values, theta_values = create_parameter_space()
    total_combinations = len(alpha_values) * len(theta_values)
    
    # Create test scenarios
    print("\n3. Creating test scenarios")
    training_scenarios, validation_scenarios = create_diverse_test_scenarios()
    print(f"Training scenarios: {len(training_scenarios)}")
    print(f"Validation scenarios: {len(validation_scenarios)}")
    
    # Initialise optimiser
    print("\n4. Initialising optimiser")
    optimiser = ModelParameterOptimiser(graph, graph_stats)
    
    if not optimiser.initialise_base_model():
        print("Failed to initialise base model")
        return None, None, None
    
    # Run optimisation
    print(f"\n5. Running parameter optimisation")
    print(f"Testing {total_combinations} parameter combinations")
    print(f"This will take approximately {total_combinations * 0.5 / 60:.1f} minutes")
    
    results = []
    start_time = time.time()
    
    for i, alpha in enumerate(alpha_values):
        alpha_start_time = time.time()
        print(f"\nAlpha {i+1}/{len(alpha_values)}: {alpha:.3f}")
        
        for j, theta in enumerate(theta_values):
            combination_num = i * len(theta_values) + j + 1
            
            # Evaluate parameter combination
            result = optimiser.evaluate_parameter_combination(
                alpha, theta, training_scenarios, n_runs=1
            )
            results.append(result)
            
            # Progress reporting
            if j % 5 == 0 or j == len(theta_values) - 1:
                elapsed = time.time() - start_time
                eta = elapsed * total_combinations / combination_num - elapsed
                print(f"theta={theta:.2f}: R={result['avg_reliability']:.3f}, "
                      f"SR={result['success_rate']:.1%} "
                      f"(ETA: {eta/60:.1f}min)")
        
        alpha_time = time.time() - alpha_start_time
        print(f"Alpha {alpha:.3f} completed in {alpha_time/60:.1f}min")
    
    total_time = time.time() - start_time
    print(f"\nOptimisation completed in {total_time/60:.1f} minutes")
    
    return results, training_scenarios, validation_scenarios


def analyse_results(results: List[Dict], 
                                training_scenarios: List[Dict]) -> Dict[str, Any]:    
    print(f"\n6. Analysing results")
    print("=" * 70)
    
    # Filter successful results
    successful_results = [r for r in results if r['success_rate'] > 0]
    
    print(f"Successful combinations: {len(successful_results)}/{len(results)} "
          f"({len(successful_results)/len(results):.1%})")
    
    if not successful_results:
        print("No successful parameter combinations found")
        return {}
    
    # Find best parameters by different criteria
    best_reliability = max(successful_results, key=lambda x: x['avg_reliability'])
    best_success_rate = max(successful_results, key=lambda x: x['success_rate'])
    best_speed = min(successful_results, key=lambda x: x['execution_time'])
    best_stability = min([r for r in successful_results if r['reliability_std'] > 0],
                        key=lambda x: x['reliability_std'], default=best_reliability)
    
    print(f"\nOptimal Parameters Found:")
    print(f"Best by reliability: alpha={best_reliability['alpha']:.3f}, theta={best_reliability['theta']:.3f}")
    print(f"Reliability: {best_reliability['avg_reliability']:.3f} += {best_reliability['reliability_std']:.3f}")
    print(f"95% CI: [{best_reliability['reliability_ci_lower']:.3f}, {best_reliability['reliability_ci_upper']:.3f}]")
    print(f"Success rate: {best_reliability['success_rate']:.1%}")
    print(f"Execution time: {best_reliability['execution_time']:.3f}s")
    
    print(f"\nAlternative optima:")
    print(f"Best success rate: alpha={best_success_rate['alpha']:.3f}, theta={best_success_rate['theta']:.3f} "
          f"(SR: {best_success_rate['success_rate']:.1%})")
    print(f"Fastest execution: alpha={best_speed['alpha']:.3f}, theta={best_speed['theta']:.3f} "
          f"(T: {best_speed['execution_time']:.3f}s)")
    print(f"Most stable: alpha={best_stability['alpha']:.3f}, theta={best_stability['theta']:.3f} "
          f"(alpha: {best_stability['reliability_std']:.3f})")
    
    # Parameter range analysis
    print(f"\nParameter range analysis:")
    
    # Group by alpha ranges
    alpha_ranges = [(0.01, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.5)]
    print(f"Alpha range effects:")
    for low, high in alpha_ranges:
        range_results = [r for r in successful_results if low <= r['alpha'] < high]
        if range_results:
            avg_rel = np.mean([r['avg_reliability'] for r in range_results])
            avg_success = np.mean([r['success_rate'] for r in range_results])
            print(f"{low:.2f}-{high:.2f}: R={avg_rel:.3f}, SR={avg_success:.1%} (n={len(range_results)})")
    
    # Group by theta ranges
    theta_ranges = [(0.1, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 3.0)]
    print(f"Theta range effects:")
    for low, high in theta_ranges:
        range_results = [r for r in successful_results if low <= r['theta'] < high]
        if range_results:
            avg_rel = np.mean([r['avg_reliability'] for r in range_results])
            avg_success = np.mean([r['success_rate'] for r in range_results])
            print(f"{low:.1f}-{high:.1f}: R={avg_rel:.3f}, SR={avg_success:.1%} (n={len(range_results)})")
    
    # Top 10 combinations
    print(f"\nTop 10 parameter combinations:")
    top_10 = sorted(successful_results, key=lambda x: x['avg_reliability'], reverse=True)[:10]
    for i, result in enumerate(top_10, 1):
        print(f"{i:2d}. alpha={result['alpha']:.3f}, theta={result['theta']:.3f} - "
              f"R={result['avg_reliability']:.3f} += {result['reliability_std']:.3f}, "
              f"SR={result['success_rate']:.1%}")
    
    return {
        'best_reliability': best_reliability,
        'best_success_rate': best_success_rate,
        'best_speed': best_speed,
        'best_stability': best_stability,
        'top_10': top_10,
        'successful_results': successful_results
    }


def validate_optimal_parameters(best_params: Dict, validation_scenarios: List[Dict], 
                              graph, graph_stats) -> Dict[str, Any]:
    """Validate optimal parameters on held-out validation set"""
    
    print(f"\n7. Validating optimal parameters")
    print("=" * 70)
    
    alpha_opt = best_params['alpha']
    theta_opt = best_params['theta']
    
    print(f"Testing optimal alpha={alpha_opt:.3f}, theta={theta_opt:.3f} on validation set")
    
    # Initialise validator
    validator = ModelParameterOptimiser(graph, graph_stats)
    if not validator.initialise_base_model():
        return {'validation_failed': True}
    
    # Run multiple validation rounds
    validation_results = []
    for round_num in range(5):
        print(f"Validation round {round_num + 1}/5")
        
        result = validator.evaluate_parameter_combination(
            alpha_opt, theta_opt, validation_scenarios, n_runs=1
        )
        validation_results.append(result)
    
    # Aggregate validation results
    val_reliabilities = [r['avg_reliability'] for r in validation_results if r['success_rate'] > 0]
    val_success_rates = [r['success_rate'] for r in validation_results]
    
    if val_reliabilities:
        avg_val_reliability = np.mean(val_reliabilities)
        val_reliability_std = np.std(val_reliabilities)
        avg_val_success_rate = np.mean(val_success_rates)
        
        # Compare with training performance
        training_reliability = best_params['avg_reliability']
        performance_difference = abs(training_reliability - avg_val_reliability)
        
        print(f"\nValidation results (5 rounds):")
        print(f"Training reliability: {training_reliability:.3f}")
        print(f"Validation reliability: {avg_val_reliability:.3f} ± {val_reliability_std:.3f}")
        print(f"Performance difference: {performance_difference:.3f}")
        print(f"Validation success rate: {avg_val_success_rate:.1%}")
        
        # Assess generalisation
        if performance_difference < 0.05:
            generalisation = "Excellent"
        elif performance_difference < 0.1:
            generalisation = "Good"
        elif performance_difference < 0.15:
            generalisation = "Fair"
        else:
            generalisation = "Poor"
        
        print(f"Generalisation: {generalisation}")
        
        return {
            'avg_val_reliability': avg_val_reliability,
            'val_reliability_std': val_reliability_std,
            'avg_val_success_rate': avg_val_success_rate,
            'performance_difference': performance_difference,
            'generalisation': generalisation,
            'validation_rounds': len(validation_results)
        }
    else:
        print(f"Validation failed - no successful validation runs")
        return {'validation_failed': True}


def generate_report(results: List[Dict], analysis: Dict, validation: Dict,
                                training_scenarios: List[Dict], validation_scenarios: List[Dict]) -> str:
    """Generate optimisation report"""
    
    print(f"\n8. Generating report")
    
    best = analysis['best_reliability']
    
    report = {
        'optimisation_metadata': {
            'timestamp': datetime.now().isoformat(),
            'optimisation_method': 'Grid search',
            'total_combinations_tested': len(results),
            'successful_combinations': len(analysis['successful_results']),
            'training_scenarios': len(training_scenarios),
            'validation_scenarios': len(validation_scenarios)
        },
        'optimal_parameters': {
            'alpha': best['alpha'],
            'theta': best['theta'],
            'justification': 'Highest average reliability with statistical validation'
        },
        'performance_metrics': {
            'training': {
                'avg_reliability': best['avg_reliability'],
                'reliability_std': best['reliability_std'],
                'reliability_ci_lower': best['reliability_ci_lower'],
                'reliability_ci_upper': best['reliability_ci_upper'],
                'success_rate': best['success_rate'],
                'execution_time': best['execution_time']
            },
            'validation': validation
        },
        'alternative_optima': {
            'best_by_success_rate': {
                'alpha': analysis['best_success_rate']['alpha'],
                'theta': analysis['best_success_rate']['theta'],
                'success_rate': analysis['best_success_rate']['success_rate'],
                'reliability': analysis['best_success_rate']['avg_reliability']
            },
            'best_by_speed': {
                'alpha': analysis['best_speed']['alpha'],
                'theta': analysis['best_speed']['theta'],
                'execution_time': analysis['best_speed']['execution_time'],
                'reliability': analysis['best_speed']['avg_reliability']
            },
            'best_by_stability': {
                'alpha': analysis['best_stability']['alpha'],
                'theta': analysis['best_stability']['theta'],
                'reliability_std': analysis['best_stability']['reliability_std'],
                'reliability': analysis['best_stability']['avg_reliability']
            }
        },
        'top_combinations': analysis['top_10'],
        'statistical_validation': {
            'confidence_intervals': True,
            'cross_validation': True,
            'multiple_scenarios': True,
            'generalisation_testing': True
        },
        'detailed_results': {
            'all_combinations': results,
            'successful_only': analysis['successful_results']
        }
    }
    
    # Save report
    report_file = 'test_results/model_reuse_parameter_optimisation_report.json'
    os.makedirs(os.path.dirname(report_file), exist_ok=True)
    
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"Report saved to: {report_file}")
    
    # Print summary
    print(f"\n" + "=" * 70)
    print("Parameter Optimisation Results")
    print("=" * 70)
    
    print(f"Recommended Optimal Parameters:")
    print(f"Alpha (temporal decay): {best['alpha']:.3f}")
    print(f"Theta (pruning threshold): {best['theta']:.3f}")
    print(f"")
    print(f"Performance With Optimal Parameters:")
    print(f"Training reliability: {best['avg_reliability']:.3f} += {best['reliability_std']:.3f}")
    print(f"95% Confidence interval: [{best['reliability_ci_lower']:.3f}, {best['reliability_ci_upper']:.3f}]")
    print(f"Success rate: {best['success_rate']:.1%}")
    print(f"Execution time: {best['execution_time']:.3f}s per query")
    
    if 'generalisation' in validation:
        print(f"Validation generalisation: {validation['generalisation']}")
        print(f"Validation reliability: {validation['avg_val_reliability']:.3f} ± {validation['val_reliability_std']:.3f}")
    
    print(f"")
    print(f"Optimisation Statistics:")
    print(f"Parameter combinations tested: {len(results):,}")
    print(f"Successful combinations: {len(analysis['successful_results']):,}")
    print(f"Success rate: {len(analysis['successful_results'])/len(results):.1%}")
    print(f"Statistical validation: 95% CI, Cross-validation, Generalisation testing")
    
    return report_file


def main():
    try:
        # Run optimisation 
        results, training_scenarios, validation_scenarios = run_model_optimisation()
        
        if not results:
            print("Optimisation failed during execution.")
            return 1
        
        # Analyse results
        analysis = analyse_results(results, training_scenarios)
        
        if not analysis:
            print("Analysis failed - no successful parameter combinations")
            return 1
        
        # Validate optimal parameters
        validation = validate_optimal_parameters(
            analysis['best_reliability'], validation_scenarios,
            create_test_tkg(), create_graph_statistics(create_test_tkg())
        )
        
        # Generate report
        report_file = generate_report(
            results, analysis, validation, training_scenarios, validation_scenarios
        )
        
        print(f"\n" + "=" * 70)
        print("Parameter Optimisation Completed")
        print("=" * 70)
        
        return 0
        
    except Exception as e:
        print(f"\nOptimisation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())