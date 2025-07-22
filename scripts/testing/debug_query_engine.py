#!/usr/bin/env python3
"""
Debug the TKGQueryEngine to understand why paths aren't being returned
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from src.kg.core.tkg_query_engine import TKGQueryEngine
from src.kg.models import TemporalQuery
from scripts.testing.test_temporal_path_retrieval import create_test_tkg, create_graph_statistics


def debug_query_engine():
    """Debug the query engine step by step"""
    
    # Create test graph
    graph = create_test_tkg()
    graph_stats = create_graph_statistics(graph)
    print(f"Created graph with {len(graph.nodes())} nodes and {len(graph.edges())} edges")
    
    # Create query engine with lower reliability threshold
    engine = TKGQueryEngine(
        graph=graph,
        graph_statistics=graph_stats,
        alpha=0.01,
        base_theta=0.1,
        reliability_threshold=0.3,  # Lower threshold
        use_updated_scoring=False  # Disable updated scoring for now
    )
    
    # Test simple query
    print("\nTesting Darwin to DNA query")
    
    # First, test the retriever directly
    temporal_query = TemporalQuery(
        query_text='Find path from darwin to dna',
        source_entities=['darwin'],
        target_entities=['dna'],
        temporal_constraints={},
        query_time='1950-01-01',
        max_hops=4,
        top_k=10
    )
    
    print("\nStep 1: Testing retriever directly")
    retrieved_paths = engine.path_retriever.retrieve_temporal_paths(
        query=temporal_query,
        enable_flow_pruning=True,
        enable_diversity=True,
        verbose=True
    )
    
    print(f"\nRetrieved {len(retrieved_paths)} paths from retriever")
    if retrieved_paths:
        for i, (path, score) in enumerate(retrieved_paths[:3]):
            print(f"Path {i+1} (score {score:.3f}): {' -> '.join([n.id for n in path.nodes])}")
    
    # Now test full query
    print("\n\nStep 2: Testing full query engine")
    result = engine.query(
        query_text='Find path from darwin to dna',
        source_entities=['darwin'],
        target_entities=['dna'],
        query_time='1950-01-01',
        max_hops=4,
        top_k=10,
        enable_reliability_filtering=False,  # Disable filtering
        verbose=True
    )
    
    print(f"\nFinal result:")
    print(f"Paths in result: {len(result.paths)}")
    print(f"Total discovered: {result.total_paths_discovered}")
    print(f"After pruning: {result.total_paths_after_pruning}")
    
    if result.paths:
        print("\nPaths found:")
        for i, (path, metrics) in enumerate(result.paths[:3]):
            print(f"Path {i+1}: {' -> '.join([n.id for n in path.nodes])}")
            if hasattr(metrics, 'overall_reliability'):
                print(f"Reliability: {metrics.overall_reliability:.3f}")
            else:
                print(f"Score: {metrics:.3f}")
    
    # Test with multiple source/target entities
    print("\n\nTesting Cambridge to CS/AI query")
    
    result2 = engine.query(
        query_text='Cambridge to computer science',
        source_entities=['cambridge'],
        target_entities=['computer_science', 'artificial_intelligence'],
        query_time='1960-01-01',
        max_hops=3,
        top_k=5,
        enable_reliability_filtering=False,
        verbose=True
    )
    
    print(f"\nFinal result:")
    print(f"Paths in result: {len(result2.paths)}")
    print(f"Total discovered: {result2.total_paths_discovered}")
    
    # Check retriever stats
    print("\n\nRetriever performance stats:")
    stats = engine.path_retriever.get_performance_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    debug_query_engine()