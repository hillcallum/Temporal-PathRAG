#!/usr/bin/env python3
"""
Debug script for temporal validity issues in path retrieval
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import networkx as nx
from datetime import datetime
import json

from src.kg.temporal_path_retriever import TemporalPathRetriever
from src.kg.models import TemporalQuery
from src.kg.temporal_scoring import TemporalWeightingFunction


def create_simple_test_graph():
    """Create a simple test graph with known temporal relationships"""
    graph = nx.MultiDiGraph()
    
    # Add nodes
    nodes = [
        ('A', {'entity_type': 'Person', 'name': 'Node A'}),
        ('B', {'entity_type': 'Person', 'name': 'Node B'}),
        ('C', {'entity_type': 'Person', 'name': 'Node C'}),
        ('D', {'entity_type': 'Person', 'name': 'Node D'}),
    ]
    
    for node_id, attrs in nodes:
        graph.add_node(node_id, **attrs)
    
    # Add temporal edges with different timestamps
    edges = [
        ('A', 'B', {'relation_type': 'connected_to', 'timestamp': '1950-01-01', 'weight': 0.9}),
        ('B', 'C', {'relation_type': 'connected_to', 'timestamp': '1960-01-01', 'weight': 0.8}),
        ('C', 'D', {'relation_type': 'connected_to', 'timestamp': '1970-01-01', 'weight': 0.7}),
        ('A', 'C', {'relation_type': 'related_to', 'timestamp': '1955-01-01', 'weight': 0.6}),
    ]
    
    for source, target, attrs in edges:
        graph.add_edge(source, target, **attrs)
    
    return graph


def debug_path_discovery(retriever, query):
    """Debug path discovery process with detailed logging"""
    print("\nDebug: Path Discovery")
    
    # Get paths using the discovery method directly
    paths = retriever.discover_temporal_paths(query, verbose=True)
    
    print(f"\nDiscovered {len(paths)} paths")
    for i, path in enumerate(paths):
        print(f"\nPath {i+1}:")
        print(f"Nodes: {[node.id for node in path.nodes]}")
        print(f"Edges: {len(path.edges)}")
        for edge in path.edges:
            print(f"{edge.source_id} -> {edge.target_id}")
            print(f"Relation: {edge.relation_type}")
            print(f"Timestamp: {edge.timestamp}")
            print(f"Weight: {edge.weight}")
    
    return paths


def debug_temporal_validation(paths, query_time):
    """Debug temporal validation logic"""
    print("\nDebug: Temporal Validation")
    
    for i, path in enumerate(paths):
        print(f"\nValidating Path {i+1}:")
        temporal_info = path.get_temporal_info()
        print(f"Temporal info: {temporal_info}")
        
        # Check temporal validity using the test script's logic
        is_valid, reason = validate_temporal_relevance_debug(path, query_time)
        print(f"Valid: {is_valid}")
        print(f"Reason: {reason}")


def validate_temporal_relevance_debug(path, query_time, temporal_window_days=365*50):
    """Debug version of temporal validation with detailed logging"""
    temporal_info = path.get_temporal_info()
    
    print(f"Timestamps in path: {temporal_info['timestamps']}")
    
    if not temporal_info['timestamps']:
        return False, "No temporal information found"
    
    query_date = datetime.strptime(query_time, '%Y-%m-%d')
    print(f"Query date: {query_date}")
    
    path_timestamps = []
    for ts in temporal_info['timestamps']:
        try:
            parsed_ts = datetime.strptime(ts, '%Y-%m-%d')
            path_timestamps.append(parsed_ts)
            print(f"Parsed timestamp: {ts} -> {parsed_ts}")
        except Exception as e:
            print(f"Failed to parse timestamp {ts}: {e}")
    
    # Check if timestamps are within temporal window
    relevant_timestamps = []
    for ts in path_timestamps:
        days_diff = abs((query_date - ts).days)
        print(f"Timestamp {ts}: {days_diff} days from query")
        if days_diff <= temporal_window_days:
            relevant_timestamps.append(ts)
            print(f" -> Within window ({temporal_window_days} days)")
        else:
            print(f"-> Outside window")
    
    if not relevant_timestamps:
        return False, f"No timestamps within {temporal_window_days} days of query time"
    
    # Check chronological coherence
    if len(path_timestamps) > 1:
        chronological = all(path_timestamps[i] <= path_timestamps[i+1] for i in range(len(path_timestamps)-1))
        print(f"Chronological order check: {chronological}")
        if not chronological:
            return False, "Path not chronologically coherent"
    
    return True, f"Temporally relevant with {len(relevant_timestamps)} relevant timestamps"


def debug_full_retrieval(retriever, query):
    """Debug the full retrieval process"""
    print("\nDebug: Full Retrieval Process")
    
    # Run retrieval with verbose output
    results = retriever.retrieve_temporal_paths(query, verbose=True)
    
    print(f"\nFinal results: {len(results)} paths")
    for i, (path, score) in enumerate(results):
        print(f"\nResult {i+1} (score: {score:.3f}):")
        print(f"Path: {' -> '.join([node.id for node in path.nodes])}")
        is_valid, reason = validate_temporal_relevance_debug(path, query.query_time)
        print(f"Temporal validity: {is_valid} ({reason})")


def main():
    """Main debug function"""
    print("Debugging Temporal Validity Issues")
    
    # Create test graph
    graph = create_simple_test_graph()
    print(f"Created test graph with {len(graph.nodes())} nodes and {len(graph.edges())} edges")
    
    # Create retriever
    retriever = TemporalPathRetriever(
        graph=graph,
        alpha=0.01,
        base_theta=0.1,
        diversity_threshold=0.7
    )
    
    # Create test query
    query = TemporalQuery(
        query_text="Find paths from A to D",
        source_entities=['A'],
        target_entities=['D'],
        temporal_constraints={'temporal_preference': 'chronological'},
        query_time='1965-01-01',
        max_hops=3,
        top_k=5
    )
    
    print(f"\nTest query:")
    print(f"Source: {query.source_entities}")
    print(f"Target: {query.target_entities}")
    print(f"Query time: {query.query_time}")
    print(f"Max hops: {query.max_hops}")
    
    # Debug path discovery
    paths = debug_path_discovery(retriever, query)
    
    # Debug temporal validation
    debug_temporal_validation(paths, query.query_time)
    
    # Debug full retrieval
    debug_full_retrieval(retriever, query)
    
    # Now test with a real example from the test script
    print("Testing with real TKG data")
    
    # Load the test graph creation function
    from scripts.testing.test_temporal_path_retrieval import create_test_tkg
    
    real_graph = create_test_tkg()
    print(f"\nCreated real test graph with {len(real_graph.nodes())} nodes and {len(real_graph.edges())} edges")
    
    # Create retriever with real graph
    real_retriever = TemporalPathRetriever(
        graph=real_graph,
        alpha=0.01,
        base_theta=0.1,
        diversity_threshold=0.7
    )
    
    # Test with Darwin to DNA query
    real_query = TemporalQuery(
        query_text='Trace the evolution from "darwin" to "dna" and "genetics"',
        source_entities=['darwin'],
        target_entities=['dna', 'genetics'],
        temporal_constraints={'temporal_preference': 'chronological'},
        query_time='1950-01-01',
        max_hops=4,
        top_k=10
    )
    
    print(f"\nReal query:")
    print(f"Source: {real_query.source_entities}")
    print(f"Target: {real_query.target_entities}")
    print(f"Query time: {real_query.query_time}")
    
    # Debug path discovery for real query
    real_paths = debug_path_discovery(real_retriever, real_query)
    
    # Debug temporal validation for real paths
    debug_temporal_validation(real_paths, real_query.query_time)


if __name__ == "__main__":
    main()