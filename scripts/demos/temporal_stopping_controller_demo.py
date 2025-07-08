#!/usr/bin/env python3
"""
Demo script for Temporal Stopping Controller
Tests the new TemporalStoppingController with various temporal scenarios
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import time
import json
from datetime import datetime
from typing import List, Tuple

from src.kg.models import TemporalQuery, Path, TemporalReliabilityMetrics, IterativeStep
from src.kg.temporal_stopping_controller import TemporalStoppingController
from src.llm.llm_manager import LLMManager


def create_mock_temporal_paths() -> List[Tuple[Path, TemporalReliabilityMetrics]]:
    """Create mock temporal paths for testing"""
    
    paths = []
    
    # Path 1: Recent temporal events
    path1 = Path(
        path=["Barack Obama", "became", "President", "of", "United States"],
        path_text="Barack Obama became President of United States in 2009",
        reliability_score=0.9
    )
    
    # Mock temporal info for path1
    path1.get_temporal_info = lambda: {
        "timestamps": ["2009-01-20T12:00:00", "2009-01-20T15:30:00"],
        "temporal_entities": ["2009", "January 20"],
        "temporal_constraints": ["sequence"]
    }
    
    metrics1 = TemporalReliabilityMetrics(
        overall_reliability=0.9,
        timestamp_reliability=0.95,
        entity_reliability=0.88,
        relation_reliability=0.92,
        path_length_score=0.85,
        diversity_score=0.80,
        temporal_consistency_score=0.93
    )
    
    paths.append((path1, metrics1))
    
    # Path 2: Historical sequence
    path2 = Path(
        path=["Barack Obama", "served", "as", "Senator", "before", "presidency"],
        path_text="Barack Obama served as Senator from Illinois before becoming President",
        reliability_score=0.85
    )
    
    path2.get_temporal_info = lambda: {
        "timestamps": ["2005-01-04T10:00:00", "2008-11-16T14:00:00"],
        "temporal_entities": ["2005", "2008", "Senator"],
        "temporal_constraints": ["sequence", "causation"]
    }
    
    metrics2 = TemporalReliabilityMetrics(
        overall_reliability=0.85,
        timestamp_reliability=0.90,
        entity_reliability=0.82,
        relation_reliability=0.87,
        path_length_score=0.88,
        diversity_score=0.75,
        temporal_consistency_score=0.89
    )
    
    paths.append((path2, metrics2))
    
    # Path 3: Extended temporal coverage
    path3 = Path(
        path=["Barack Obama", "born", "in", "Hawaii"],
        path_text="Barack Obama was born in Hawaii in 1961",
        reliability_score=0.92
    )
    
    path3.get_temporal_info = lambda: {
        "timestamps": ["1961-08-04T06:24:00"],
        "temporal_entities": ["1961", "August", "Hawaii"],
        "temporal_constraints": ["duration"]
    }
    
    metrics3 = TemporalReliabilityMetrics(
        overall_reliability=0.92,
        timestamp_reliability=0.88,
        entity_reliability=0.95,
        relation_reliability=0.90,
        path_length_score=0.80,
        diversity_score=0.85,
        temporal_consistency_score=0.95
    )
    
    paths.append((path3, metrics3))
    
    # Path 4: Additional context (for testing overload)
    path4 = Path(
        path=["Barack Obama", "graduated", "Harvard Law School"],
        path_text="Barack Obama graduated from Harvard Law School in 1991",
        reliability_score=0.88
    )
    
    path4.get_temporal_info = lambda: {
        "timestamps": ["1991-06-15T14:00:00"],
        "temporal_entities": ["1991", "Harvard", "graduation"],
        "temporal_constraints": ["sequence"]
    }
    
    metrics4 = TemporalReliabilityMetrics(
        overall_reliability=0.88,
        timestamp_reliability=0.85,
        entity_reliability=0.90,
        relation_reliability=0.89,
        path_length_score=0.82,
        diversity_score=0.78,
        temporal_consistency_score=0.91
    )
    
    paths.append((path4, metrics4))
    
    return paths


def create_mock_low_quality_paths() -> List[Tuple[Path, TemporalReliabilityMetrics]]:
    """Create mock low-quality paths for testing overload prevention"""
    
    paths = []
    
    for i in range(3):
        path = Path(
            path=[f"Entity{i}", "relates", "to", f"Entity{i+1}"],
            path_text=f"Low quality temporal relation {i}",
            reliability_score=0.3 - i * 0.1  # Declining quality
        )
        
        path.get_temporal_info = lambda: {
            "timestamps": [f"2020-0{i+1}-01T10:00:00"],
            "temporal_entities": [f"2020", f"Entity{i}"],
            "temporal_constraints": []
        }
        
        metrics = TemporalReliabilityMetrics(
            overall_reliability=0.3 - i * 0.1,
            timestamp_reliability=0.4 - i * 0.1,
            entity_reliability=0.35 - i * 0.1,
            relation_reliability=0.25 - i * 0.1,
            path_length_score=0.3,
            diversity_score=0.2,
            temporal_consistency_score=0.4 - i * 0.1
        )
        
        paths.append((path, metrics))
    
    return paths


def test_temporal_coverage_assessment():
    """Test temporal coverage assessment functionality"""
    
    print("\\nTesting Temporal Coverage Assessment")
    
    # Initialise stopping controller with mock LLM
    class MockLLMManager:
        def generate_response(self, prompt):
            return '{"is_sufficient": false, "confidence": 0.6, "missing_information": ["more temporal context"], "exploration_hints": ["explore earlier periods"]}'
    
    stopping_controller = TemporalStoppingController(
        llm_manager=MockLLMManager(),
        temporal_coverage_threshold=0.7,
        chain_satisfaction_threshold=0.8
    )
    
    # Test with comprehensive temporal coverage
    query = TemporalQuery(
        query_text="What is the timeline of Barack Obama's political career?"
    )
    
    mock_paths = create_mock_temporal_paths()
    
    decision = stopping_controller.should_stop(
        query, mock_paths, "Context about Obama's career timeline", iteration=2
    )
    
    print(f"Query: {query.query_text}")
    print(f"Should stop: {decision.should_stop}")
    print(f"Stopping criterion: {decision.stopping_criterion}")
    print(f"Confidence: {decision.confidence:.2f}")
    print(f"Reasoning: {decision.reasoning}")
    print(f"Temporal coverage score: {decision.temporal_coverage.coverage_score:.2f}")
    print(f"Temporal span: {decision.temporal_coverage.temporal_span_days} days")
    print(f"Timestamp count: {decision.temporal_coverage.timestamp_count}")
    print(f"Chronological continuity: {decision.temporal_coverage.chronological_continuity:.2f}")


def test_overload_prevention():
    """Test temporal information overload prevention"""
    
    print("\\nTesting Overload Prevention")
    
    class MockLLMManager:
        def generate_response(self, prompt):
            return '{"is_sufficient": false, "confidence": 0.3, "missing_information": ["unclear"], "exploration_hints": ["continue searching"]}'
    
    stopping_controller = TemporalStoppingController(
        llm_manager=MockLLMManager(),
        overload_prevention_threshold=0.7  # Lower threshold for testing
    )
    
    query = TemporalQuery(
        query_text="Find any information about random entities"
    )
    
    # Start with good quality paths
    good_paths = create_mock_temporal_paths()[:2]
    # Add declining quality paths
    bad_paths = create_mock_low_quality_paths()
    
    all_paths = good_paths + bad_paths
    
    decision = stopping_controller.should_stop(
        query, all_paths, "Mixed quality temporal information", iteration=3
    )
    
    print(f"Query: {query.query_text}")
    print(f"Should stop: {decision.should_stop}")
    print(f"Stopping criterion: {decision.stopping_criterion}")
    print(f"Confidence: {decision.confidence:.2f}")
    print(f"Information quality score: {decision.information_quality_score:.2f}")
    print(f"Retrieval efficiency score: {decision.retrieval_efficiency_score:.2f}")
    
    if "overload" in decision.stopping_criterion:
        print("Successfully detected information overload!")
    else:
        print("Overload prevention not triggered")


def test_constraint_fulfillment():
    """Test temporal constraint fulfillment"""
    
    print("\\nTesting Temporal Constraint Fulfillment")
    
    class MockLLMManager:
        def generate_response(self, prompt):
            return '{"is_sufficient": true, "confidence": 0.85, "missing_information": [], "exploration_hints": []}'
    
    stopping_controller = TemporalStoppingController(
        llm_manager=MockLLMManager(),
        constraint_fulfillment_threshold=0.6
    )
    
    # Query with specific temporal constraints
    query = TemporalQuery(
        query_text="What happened after Barack Obama became President and before he left office?"
    )
    
    mock_paths = create_mock_temporal_paths()
    
    decision = stopping_controller.should_stop(
        query, mock_paths, "Comprehensive Obama presidency timeline", iteration=1
    )
    
    print(f"Query: {query.query_text}")
    print(f"Should stop: {decision.should_stop}")
    print(f"Stopping criterion: {decision.stopping_criterion}")
    print(f"Constraint satisfaction: {decision.temporal_coverage.constraint_satisfaction_score:.2f}")
    print(f"Next exploration hints: {decision.next_exploration_hints}")


def test_progressive_stopping_decisions():
    """Test stopping decisions across multiple iterations"""
    
    print("\\nTesting Progressive Stopping Decisions")
    
    class MockLLMManager:
        def generate_response(self, prompt):
            return '{"is_sufficient": false, "confidence": 0.5, "missing_information": ["more context"], "exploration_hints": ["explore related events"]}'
    
    stopping_controller = TemporalStoppingController(
        llm_manager=MockLLMManager(),
        temporal_coverage_threshold=0.6,
        chain_satisfaction_threshold=0.7
    )
    
    query = TemporalQuery(
        query_text="Trace the evolution of Barack Obama's political career"
    )
    
    all_paths = create_mock_temporal_paths()
    
    print(f"Query: {query.query_text}")
    print("\\nProgressive evaluation:")
    
    # Test with increasing number of paths
    for i in range(1, len(all_paths) + 1):
        current_paths = all_paths[:i]
        
        decision = stopping_controller.should_stop(
            query, current_paths, f"Context with {i} paths", iteration=i-1
        )
        
        print(f"Iteration {i}: {len(current_paths)} paths")
        print(f"Should stop: {decision.should_stop}")
        print(f"Coverage score: {decision.temporal_coverage.coverage_score:.2f}")
        print(f"Confidence: {decision.confidence:.2f}")
        print(f"Criterion: {decision.stopping_criterion if decision.should_stop else 'continue'}")
        
        if decision.should_stop:
            print(f"Stopped at iteration {i}")
            break


def test_stopping_statistics():
    """Test stopping statistics tracking"""
    
    print("\\nTesting Stopping Statistics")
    
    class MockLLMManager:
        def generate_response(self, prompt):
            return '{"is_sufficient": true, "confidence": 0.8, "missing_information": [], "exploration_hints": []}'
    
    stopping_controller = TemporalStoppingController(
        llm_manager=MockLLMManager()
    )
    
    # Run multiple stopping decisions
    for i in range(5):
        query = TemporalQuery(
            query_text=f"Test query {i}"
        )
        
        paths = create_mock_temporal_paths()[:i+1]
        
        decision = stopping_controller.should_stop(
            query, paths, f"Test context {i}", iteration=i
        )
    
    # Get statistics
    stats = stopping_controller.get_stopping_statistics()
    
    print("Stopping Statistics:")
    print(f"Total decisions: {stats.get('total_decisions', 0)}")
    print(f"Stop decisions: {stats.get('stop_decisions', 0)}")
    print(f"Stop rate: {stats.get('stop_rate', 0):.2f}")
    print(f"Average confidence: {stats.get('average_confidence', 0):.2f}")
    print(f"Average quality: {stats.get('average_quality', 0):.2f}")
    print(f"Stopping criteria: {stats.get('stopping_criteria', {})}")


def main():
    """Run all temporal stopping controller tests"""
    
    print("Temporal Stopping Controller Demo")
    print("="*50)
    
    start_time = time.time()
    
    try:
        # Run all tests
        test_temporal_coverage_assessment()
        test_overload_prevention()
        test_constraint_fulfillment()
        test_progressive_stopping_decisions()
        test_stopping_statistics()
        
        execution_time = time.time() - start_time
        
        print(f"\\nAll tests completed")
        print(f"Total execution time: {execution_time:.2f} seconds")
                
    except Exception as e:
        print(f"\\nError during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()