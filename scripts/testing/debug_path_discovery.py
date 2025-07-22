#!/usr/bin/env python3
"""
Debug path discovery to see what paths are actually being found
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from src.kg.retrieval.temporal_path_retriever import TemporalPathRetriever
from src.kg.models import TemporalQuery
from scripts.testing.test_temporal_path_retrieval import create_test_tkg


def debug_path_discovery():
    """Debug what paths are discovered"""
    
    # Create test graph
    graph = create_test_tkg()
    print(f"Created graph with {len(graph.nodes())} nodes and {len(graph.edges())} edges")
    
    # Create retriever
    retriever = TemporalPathRetriever(
        graph=graph,
        alpha=0.01,
        base_theta=0.1,
        diversity_threshold=0.7
    )
    
    # Test query
    query = TemporalQuery(
        query_text='Find path from darwin to dna',
        source_entities=['darwin'],
        target_entities=['dna'],
        temporal_constraints={},
        query_time='1950-01-01',
        max_hops=4,
        top_k=10
    )
    
    print(f"\nQuery: {query.source_entities} -> {query.target_entities}")
    
    # Test path discovery directly
    print("\nDiscovering paths")
    paths = retriever.discover_temporal_paths(query, verbose=True)
    
    print(f"\nDiscovered {len(paths)} paths:")
    for i, path in enumerate(paths):
        nodes = [n.id for n in path.nodes]
        print(f"\nPath {i+1}: {' -> '.join(nodes)}")
        print(f"Start: {nodes[0]}, End: {nodes[-1]}")
        print(f"Length: {len(nodes)} nodes")
        
        # Check if this is actually a path from source to target
        if nodes[0] in query.source_entities and nodes[-1] in query.target_entities:
            print(f"Valid source->target path")
        else:
            print(f"Not a valid source->target path")
        
        # Show timestamps
        if path.edges:
            print(f"Timestamps:")
            for edge in path.edges[:3]:
                print(f"{edge.source_id} -> {edge.target_id}: {edge.timestamp}")
    
    # Now test with pruning and scoring
    print("\n\nFull retrieval pipeline")
    results = retriever.retrieve_temporal_paths(query, verbose=True)
    
    print(f"\nFinal results: {len(results)} paths")
    for i, (path, score) in enumerate(results[:5]):
        nodes = [n.id for n in path.nodes]
        print(f"\nPath {i+1} (score {score:.3f}): {' -> '.join(nodes)}")
        if nodes[0] in query.source_entities and nodes[-1] in query.target_entities:
            print(f"Valid source->target path")
        else:
            print(f"Not a valid source->target path")


if __name__ == "__main__":
    debug_path_discovery()