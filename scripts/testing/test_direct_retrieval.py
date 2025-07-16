#!/usr/bin/env python3
"""
Direct test of path retrieval to debug issues
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from src.kg.temporal_path_retriever import TemporalPathRetriever
from src.kg.models import TemporalQuery
from src.kg.tkg_query_engine import TKGQueryEngine
from scripts.testing.test_temporal_path_retrieval import create_test_tkg, create_graph_statistics


def test_retrieval_debug():
    """Debug the retrieval process step by step"""
    
    # Create test graph
    graph = create_test_tkg()
    graph_stats = create_graph_statistics(graph)
    print(f"Created graph with {len(graph.nodes())} nodes and {len(graph.edges())} edges")
    
    # Check if key entities exist
    test_entities = ['darwin', 'dna', 'genetics', 'cambridge', 'computer_science']
    print("\nChecking entities:")
    for entity in test_entities:
        if graph.has_node(entity):
            print(f"{entity} exists")
            # Show connections
            neighbors = list(graph.neighbors(entity))
            print(f"Connections: {neighbors[:5]}")
        else:
            print(f"{entity} not found")
    
    # Test 1: Direct retriever test
    print("\nTest 1: Direct Retriever")
    retriever = TemporalPathRetriever(
        graph=graph,
        alpha=0.01,
        base_theta=0.1,
        diversity_threshold=0.7
    )
    
    query = TemporalQuery(
        query_text='Test query',
        source_entities=['darwin'],
        target_entities=['dna'],
        temporal_constraints={},
        query_time='1950-01-01',
        max_hops=4,
        top_k=10
    )
    
    # Test path discovery directly
    print("\nDiscovering paths")
    paths = retriever.discover_temporal_paths(query, verbose=True)
    print(f"Discovered {len(paths)} paths")
    
    if paths:
        print("\nFirst 3 paths:")
        for i, path in enumerate(paths[:3]):
            print(f"Path {i+1}: {' -> '.join([n.id for n in path.nodes])}")
    
    # Test 2: TKG Query Engine test
    print("\nTest 2: TKG Query Engine")
    engine = TKGQueryEngine(
        graph=graph,
        graph_statistics=graph_stats,
        alpha=0.01,
        base_theta=0.1
    )
    
    # Test with explicit entities
    result = engine.query(
        query_text='Find path from darwin to dna',
        source_entities=['darwin'],
        target_entities=['dna'],
        query_time='1950-01-01',
        max_hops=4,
        top_k=10,
        verbose=True
    )
    
    print(f"\nQuery result:")
    print(f"Paths found: {len(result.paths)}")
    print(f"Total discovered: {result.total_paths_discovered}")
    print(f"After pruning: {result.total_paths_after_pruning}")
    
    if result.paths:
        print("\nTop 3 paths:")
        for i, (path, metrics) in enumerate(result.paths[:3]):
            print(f"Path {i+1}: {' -> '.join([n.id for n in path.nodes])}")
            print(f"Reliability: {metrics.overall_reliability:.3f}")


if __name__ == "__main__":
    test_retrieval_debug()