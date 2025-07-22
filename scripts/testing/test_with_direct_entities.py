#!/usr/bin/env python3
"""
Test temporal path retrieval with direct entity specification
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from src.kg.retrieval.temporal_path_retriever import TemporalPathRetriever
from src.kg.models import TemporalQuery
from scripts.testing.test_temporal_path_retrieval import create_test_tkg


def test_direct_query():
    """Test using TemporalQuery directly instead of TKGQueryEngine"""
    
    # Create test graph
    graph = create_test_tkg()
    print(f"Created graph with {len(graph.nodes())} nodes and {len(graph.edges())} edges")
    
    # Create retriever directly
    retriever = TemporalPathRetriever(
        graph=graph,
        alpha=0.01,
        base_theta=0.1,
        diversity_threshold=0.7
    )
    
    # Test query 1: Darwin to DNA
    query1 = TemporalQuery(
        query_text='Trace the evolution from "darwin" to "dna" and "genetics"',
        source_entities=['darwin'],
        target_entities=['dna', 'genetics'],
        temporal_constraints={'temporal_preference': 'chronological'},
        query_time='1950-01-01',
        max_hops=4,
        top_k=10
    )
    
    print("\nTest 1: Darwin to DNA/genetics")
    print(f"Source: {query1.source_entities}")
    print(f"Target: {query1.target_entities}")
    
    # Check if entities exist in graph
    for entity in query1.source_entities + query1.target_entities:
        if graph.has_node(entity):
            print(f"{entity} found in graph")
        else:
            print(f"{entity} not found in graph")
    
    # Retrieve paths
    results = retriever.retrieve_temporal_paths(query1, verbose=True)
    
    print(f"\nFound {len(results)} paths")
    for i, (path, score) in enumerate(results[:3]):
        print(f"\nPath {i+1} (score: {score:.3f}):")
        print(f" {' -> '.join([node.id for node in path.nodes])}")
        
    # Test query 2: Cambridge to CS/AI
    query2 = TemporalQuery(
        query_text='How did "cambridge" contribute to "computer_science" and "artificial_intelligence"?',
        source_entities=['cambridge'],
        target_entities=['computer_science', 'artificial_intelligence'],
        temporal_constraints={'temporal_preference': 'institutional'},
        query_time='1960-01-01',
        max_hops=3,
        top_k=8
    )
    
    print("\n\nTest 2: Cambridge to CS/AI")
    print(f"Source: {query2.source_entities}")
    print(f"Target: {query2.target_entities}")
    
    # Check if entities exist
    for entity in query2.source_entities + query2.target_entities:
        if graph.has_node(entity):
            print(f"{entity} found in graph")
        else:
            print(f"{entity} not found in graph")
    
    # Retrieve paths
    results2 = retriever.retrieve_temporal_paths(query2, verbose=True)
    
    print(f"\nFound {len(results2)} paths")
    for i, (path, score) in enumerate(results2[:3]):
        print(f"\nPath {i+1} (score: {score:.3f}):")
        print(f" {' -> '.join([node.id for node in path.nodes])}")


if __name__ == "__main__":
    test_direct_query()