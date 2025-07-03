#!/usr/bin/env python3
"""
Alpha-Theta Parameter Interaction in Temporal Flow Pruning

This script demonstrates:
1. Modified resource propagation in PathRAG's flow-based pruning with temporal weighting
2. Effective interaction between decay rate (alpha) and pruning threshold (theta) parameters
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from src.kg.temporal_flow_pruning import TemporalFlowPruning
from src.kg.temporal_scoring import TemporalWeightingFunction, TemporalRelevanceMode
from src.kg.models import Path, TemporalPathRAGNode, TemporalPathRAGEdge
import numpy as np
import matplotlib.pyplot as plt


def create_test_paths_with_temporal_variation():
    """Create test paths with varying temporal characteristics"""
    paths = []
    
    # Path 1: Recent, well-ordered temporal events
    path1 = Path()
    path1.nodes = [
        TemporalPathRAGNode(id="researcher_1", entity_type="Person", name="Modern_AI_Researcher"),
        TemporalPathRAGNode(id="paper_1", entity_type="Publication", name="Breakthrough_Paper"),
        TemporalPathRAGNode(id="award_1", entity_type="Award", name="Nobel_Prize")
    ]
    path1.edges = [
        TemporalPathRAGEdge(
            source_id="researcher_1", target_id="paper_1",
            relation_type="published", timestamp="2023-01-15",
            flow_capacity=1.0
        ),
        TemporalPathRAGEdge(
            source_id="paper_1", target_id="award_1",
            relation_type="led_to", timestamp="2023-06-10",
            flow_capacity=1.0
        )
    ]
    path1.score = 0.85
    paths.append(path1)
    
    # Path 2: Historical events with poor chronological order
    path2 = Path()
    path2.nodes = [
        TemporalPathRAGNode(id="scientist_1", entity_type="Person", name="Historical_Scientist"),
        TemporalPathRAGNode(id="discovery_1", entity_type="Discovery", name="Important_Discovery"),
        TemporalPathRAGNode(id="institution_1", entity_type="Institution", name="University")
    ]
    path2.edges = [
        TemporalPathRAGEdge(
            source_id="scientist_1", target_id="discovery_1",
            relation_type="made", timestamp="1950-03-20",
            flow_capacity=1.0
        ),
        TemporalPathRAGEdge(
            source_id="discovery_1", target_id="institution_1",
            relation_type="recognised_by", timestamp="1945-08-15",  # Out of order
            flow_capacity=1.0
        )
    ]
    path2.score = 0.90
    paths.append(path2)
    
    # Path 3: Mixed temporal span (very long)
    path3 = Path()
    path3.nodes = [
        TemporalPathRAGNode(id="inventor_1", entity_type="Person", name="Inventor"),
        TemporalPathRAGNode(id="patent_1", entity_type="Patent", name="Patent_New"),
        TemporalPathRAGNode(id="company_1", entity_type="Company", name="Tech_Company")
    ]
    path3.edges = [
        TemporalPathRAGEdge(
            source_id="inventor_1", target_id="patent_1",
            relation_type="filed", timestamp="1980-12-05",
            flow_capacity=1.0
        ),
        TemporalPathRAGEdge(
            source_id="patent_1", target_id="company_1",
            relation_type="licensed_to", timestamp="2020-07-22",  # 40-year span
            flow_capacity=1.0
        )
    ]
    path3.score = 0.75
    paths.append(path3)
    
    # Path 4: No temporal information
    path4 = Path()
    path4.nodes = [
        TemporalPathRAGNode(id="entity_1", entity_type="Entity", name="Generic_Entity"),
        TemporalPathRAGNode(id="entity_2", entity_type="Entity", name="Related_Entity")
    ]
    path4.edges = [
        TemporalPathRAGEdge(
            source_id="entity_1", target_id="entity_2",
            relation_type="related_to", timestamp=None,  # No timestamp
            flow_capacity=1.0
        )
    ]
    path4.score = 0.60
    paths.append(path4)
    
    return paths


def resource_propagation():
    """Demonstrate that resource propagation is modified with temporal weighting"""
    print("=== Resource Propagation Modification ===\n")
    
    weighting_func = TemporalWeightingFunction()
    flow_pruning = TemporalFlowPruning(
        temporal_weighting=weighting_func,
        alpha=0.15,  # Moderate decay rate
        base_theta=1.2
    )
    
    paths = create_test_paths_with_temporal_variation()
    query_time = "2023-07-01"
    
    print("1. Original Edge Capacities:")
    original_capacities = []
    for i, path in enumerate(paths):
        print(f"Path {i+1}:")
        for j, edge in enumerate(path.edges):
            orig_cap = getattr(edge, 'flow_capacity', 1.0)
            original_capacities.append(orig_cap)
            print(f"Edge {j+1}: {orig_cap:.3f} (timestamp: {edge.timestamp})")
    
    print("\n2. Applying Temporal Edge Capacity Modification")
    flow_pruning._update_temporal_edge_capacities(paths, query_time)
    
    print("\n3. Modified Edge Capacities (with temporal weighting):")
    modifications = 0
    for i, path in enumerate(paths):
        print(f"Path {i+1}:")
        for j, edge in enumerate(path.edges):
            new_cap = edge.flow_capacity
            orig_cap = getattr(edge, '_original_capacity', 1.0)
            change = ((new_cap - orig_cap) / orig_cap) * 100 if orig_cap > 0 else 0
            print(f"Edge {j+1}: {new_cap:.3f} (change: {change:+.1f}%)")
            if abs(change) > 1:
                modifications += 1
    
    print(f"\n4. Resource Propagation Impact:")
    print(f"- Edges modified: {modifications}/{sum(len(p.edges) for p in paths)}")
    print(f"- Modification rate: {modifications / sum(len(p.edges) for p in paths) * 100:.1f}%")
    
    # Calculate temporal flows
    print(f"\n5. Temporal Flow Calculations:")
    temporal_flows = []
    for i, path in enumerate(paths):
        flow = flow_pruning._calculate_temporal_path_flow(path, query_time)
        temporal_flows.append(flow)
        print(f"Path {i+1}: {flow:.3f}")
    
    flow_variance = np.var(temporal_flows)
    print(f"Flow variance: {flow_variance:.3f} (higher = more temporal discrimination)")
        
    return modifications > 0, flow_variance > 0.01


def alpha_theta_interaction():
    """Demonstrate that alpha and theta parameters interact effectively with temporal scores"""
    print("=== Alpha-Theta Parameter Interaction ===\n")
    
    weighting_func = TemporalWeightingFunction()
    paths = create_test_paths_with_temporal_variation()
    query_time = "2023-07-01"
    
    # Test different alpha values
    alpha_values = [0.05, 0.1, 0.2, 0.4]
    theta_values = [0.5, 1.0, 1.5, 2.0]
    
    print("1. Testing Alpha (Decay Rate) Sensitivity:")
    alpha_results = []
    for alpha in alpha_values:
        flow_pruning = TemporalFlowPruning(
            temporal_weighting=weighting_func,
            alpha=alpha,
            base_theta=1.0
        )
        pruned = flow_pruning.flow_based_pruning_with_temporal_weighting(paths, 3, query_time)
        avg_score = sum(p.score for p in pruned) / len(pruned) if pruned else 0.0
        alpha_results.append((alpha, len(pruned), avg_score))
        print(f"alpha={alpha:.2f}: {len(pruned)} paths selected, avg_score={avg_score:.3f}")
    
    print("\n2. Testing Theta (Threshold) Sensitivity:")
    theta_results = []
    for theta in theta_values:
        flow_pruning = TemporalFlowPruning(
            temporal_weighting=weighting_func,
            alpha=0.1,
            base_theta=theta
        )
        pruned = flow_pruning.flow_based_pruning_with_temporal_weighting(paths, 3, query_time)
        avg_score = sum(p.score for p in pruned) / len(pruned) if pruned else 0.0
        theta_results.append((theta, len(pruned), avg_score))
        print(f"Theta={theta:.1f}: {len(pruned)} paths selected, avg_score={avg_score:.3f}")
    
    print("\n3. Testing Alpha-Theta Interaction Strength:")
    
    # Low alpha, low theta configuration
    flow_pruning_low = TemporalFlowPruning(
        temporal_weighting=weighting_func,
        alpha=0.05,
        base_theta=0.5
    )
    low_result = flow_pruning_low.flow_based_pruning_with_temporal_weighting(paths, 4, query_time)
    low_score = sum(p.score for p in low_result) / len(low_result) if low_result else 0.0
    
    # High alpha, high theta configuration
    flow_pruning_high = TemporalFlowPruning(
        temporal_weighting=weighting_func,
        alpha=0.3,
        base_theta=2.0
    )
    high_result = flow_pruning_high.flow_based_pruning_with_temporal_weighting(paths, 4, query_time)
    high_score = sum(p.score for p in high_result) / len(high_result) if high_result else 0.0
    
    interaction_strength = abs(high_score - low_score)
    print(f"Low params (alpha=0.05, theta=0.5): {len(low_result)} paths, avg_score={low_score:.3f}")
    print(f"High params (alpha=0.3, theta=2.0): {len(high_result)} paths, avg_score={high_score:.3f}")
    print(f"Interaction strength: {interaction_strength:.3f}")
    
    print("\n4. Validation with Built-in Method:")
    flow_pruning = TemporalFlowPruning(
        temporal_weighting=weighting_func,
        alpha=0.15,
        base_theta=1.2
    )
    
    validation = flow_pruning.validate_alpha_theta_interaction(paths, query_time)
    print(f"Resource modification rate: {validation['resource_modification_rate']:.3f}")
    print(f"Temporal flow variance: {validation['temporal_flow_variance']:.3f}")
    print(f"Interaction strength: {validation['interaction_strength']:.3f}")
    print(f"Overall validation score: {validation['validation_score']:.3f}")
    
    # Check for meaningful parameter sensitivity
    alpha_sensitivity = len(set(r[1] for r in alpha_results)) > 1  # Different path counts
    theta_sensitivity = len(set(r[1] for r in theta_results)) > 1  # Different path counts
    meaningful_interaction = interaction_strength > 0.1
    
    print(f"\n5. Interaction Analysis:")
    print(f"alpha parameter sensitivity: {'yes' if alpha_sensitivity else 'no'}")
    print(f"theta parameter sensitivity: {'yes' if theta_sensitivity else 'no'}")
    print(f"Meaningful interaction: {'yes' if meaningful_interaction else 'no'}")
    
    return alpha_sensitivity and theta_sensitivity and meaningful_interaction

def run_validation():
    """
    Temporal flow pruning validation
    """
    print("Testing temporal flow pruning implementation\n")
    
    results = {
        'resource_propagation_test': {'passed': False, 'details': {}},
        'parameter_interaction_test': {'passed': False, 'details': {}},
        'robustness_test': {'passed': False, 'details': {}},
        'performance_test': {'passed': False, 'details': {}}
    }
    
    # Test 1: Resource Propagation Validation
    print("Test 1: Resource Propagation Modification")
    print("-" * 50)
    try:
        resource_success, flow_variance = resource_propagation()
        results['resource_propagation_test']['passed'] = resource_success and flow_variance > 0.01
        results['resource_propagation_test']['details'] = {
            'modification_success': resource_success,
            'flow_variance': flow_variance,
            'threshold_met': flow_variance > 0.01
        }
        status = "Pass" if results['resource_propagation_test']['passed'] else "Fail"
        print(f"Result: {status}\n")
    except Exception as e:
        print(f"FAIL - Error: {e}\n")
        results['resource_propagation_test']['details']['error'] = str(e)
    
    # Test 2: Parameter Interaction Validation
    print("Test 2: Alpha-Theta Parameter Interaction")
    print("-" * 50)
    try:
        interaction_success = alpha_theta_interaction()
        results['parameter_interaction_test']['passed'] = interaction_success
        results['parameter_interaction_test']['details'] = {
            'interaction_detected': interaction_success
        }
        status = "Pass" if interaction_success else "Fail"
        print(f"Result: {status}\n")
    except Exception as e:
        print(f"Fail - Error: {e}\n")
        results['parameter_interaction_test']['details']['error'] = str(e)
    
    # Test 3: Robustness Testing
    print("Test 3: Implementation Robustness")
    print("-" * 50)
    try:
        robustness_results = test_implementation_robustness()
        results['robustness_test']['passed'] = robustness_results['overall_robust']
        results['robustness_test']['details'] = robustness_results
        status = "Pass" if robustness_results['overall_robust'] else "Fail"
        print(f"Result: {status}\n")
    except Exception as e:
        print(f"Fail - Error: {e}\n")
        results['robustness_test']['details']['error'] = str(e)
    
    # Test 4: Performance Validation
    print("Test 4: Performance Characteristics")
    print("-" * 50)
    try:
        performance_results = test_performance_characteristics()
        results['performance_test']['passed'] = performance_results['acceptable_performance']
        results['performance_test']['details'] = performance_results
        status = "Pass" if performance_results['acceptable_performance'] else "Fail"
        print(f"Result: {status}\n")
    except Exception as e:
        print(f"Fail - Error: {e}\n")
        results['performance_test']['details']['error'] = str(e)
    
    return results

def test_implementation_robustness():
    """
    Test robustness of the temporal flow pruning implementation
    """
    weighting_func = TemporalWeightingFunction()
    paths = create_test_paths_with_temporal_variation()
    query_time = "2023-07-01"
    
    robustness_metrics = {
        'parameter_validation': False,
        'edge_case_handling': False,
        'mathematical_consistency': False,
        'overall_robust': False
    }
    
    # Test parameter validation
    try:
        # Should raise ValueError for invalid alpha
        try:
            TemporalFlowPruning(weighting_func, alpha=2.0)  # Invalid alpha > 1.0
            robustness_metrics['parameter_validation'] = False
        except ValueError:
            robustness_metrics['parameter_validation'] = True
    except Exception:
        pass
    
    # Test edge case handling
    try:
        flow_pruning = TemporalFlowPruning(weighting_func, alpha=0.1, base_theta=1.0)
        empty_result = flow_pruning.flow_based_pruning_with_temporal_weighting([], 5, query_time)
        robustness_metrics['edge_case_handling'] = isinstance(empty_result, list)
    except Exception:
        pass
    
    # Test mathematical consistency
    try:
        flow_pruning = TemporalFlowPruning(weighting_func, alpha=0.1, base_theta=1.0)
        validation = flow_pruning.validate_alpha_theta_interaction(paths, query_time)
        robustness_metrics['mathematical_consistency'] = validation.get('validation_score', 0) > 0.2
    except Exception:
        pass
    
    # Overall assessment
    robustness_metrics['overall_robust'] = (
        robustness_metrics['parameter_validation'] and
        robustness_metrics['edge_case_handling'] and
        robustness_metrics['mathematical_consistency']
    )
    
    print(f"Parameter validation: {'Pass' if robustness_metrics['parameter_validation'] else 'Fail'}")
    print(f"Edge case handling: {'Pass' if robustness_metrics['edge_case_handling'] else 'Fail'}")
    print(f"Mathematical consistency: {'Pass' if robustness_metrics['mathematical_consistency'] else 'Fail'}")
    
    return robustness_metrics

def test_performance_characteristics():
    """
    Test performance characteristics of the implementation
    """
    import time
    
    weighting_func = TemporalWeightingFunction()
    paths = create_test_paths_with_temporal_variation() * 10  # Scale up for performance testing
    query_time = "2023-07-01"
    
    performance_metrics = {
        'execution_time': 0.0,
        'memory_efficiency': True,
        'scalability': True,
        'acceptable_performance': False
    }
    
    # Test execution time
    flow_pruning = TemporalFlowPruning(weighting_func, alpha=0.1, base_theta=1.0)
    
    start_time = time.time()
    result = flow_pruning.flow_based_pruning_with_temporal_weighting(paths, 10, query_time)
    execution_time = time.time() - start_time
    
    performance_metrics['execution_time'] = execution_time
    performance_metrics['scalability'] = len(result) > 0 and execution_time < 5.0  # Should complete in < 5 seconds
    performance_metrics['memory_efficiency'] = len(result) <= len(paths)  # Should not expand path count
    
    performance_metrics['acceptable_performance'] = (
        performance_metrics['scalability'] and
        performance_metrics['memory_efficiency']
    )
    
    print(f"Execution time: {execution_time:.3f}s")
    print(f"Scalability: {'Pass' if performance_metrics['scalability'] else 'Fail'}")
    print(f"Memory efficiency: {'Pass' if performance_metrics['memory_efficiency'] else 'Fail'}")
    
    return performance_metrics

def main():
    print("Temporal PathRAG Alpha-Theta Parameter Interaction Validation")
    
    try:
        # Run validation
        validation_results = run_validation()
        
        # Return exit code
        total_tests = len(validation_results)
        passed_tests = sum(1 for test in validation_results.values() if test['passed'])
        
        if passed_tests == total_tests:
            print("\nAll validations passed successfully")
            return 0
        else:
            print(f"\n{total_tests - passed_tests} validation(s) failed")
            return 1
            
    except Exception as e:
        print(f"\nCritical error during validation: {e}")
        import traceback
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    main()