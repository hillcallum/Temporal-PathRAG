#!/usr/bin/env python3
"""
TKG Query Engine Demo
Demonstrates the complete temporal knowledge graph query pipeline
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from src.kg.updated_tkg_loader import load_enhanced_dataset
from src.kg.tkg_query_engine import TKGQueryEngine
from src.utils.device import setup_device_and_logging
from datetime import datetime
import json


def demonstrate_query_engine():
    """Demonstrate the TKG query engine with various queries"""
    print("Temporal Knowledge Graph Query Engine Demo\n")
    
    # Setup device and logging
    device = setup_device_and_logging()
    print(f"Using device: {device}\n")
    
    # Load graph
    print("1. Loading MultiTQ graph")
    graph = load_enhanced_dataset('MultiTQ')
    print(f"Loaded graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges\n")
    
    # Create query engine
    print("2. Creating TKG query engine")
    engine = TKGQueryEngine(graph)
    print("Query engine ready\n")
    
    # Test queries
    test_queries = [
        "What did Obama do in 2015?",
        "Who met with China in 2014?",
        "What agreements were signed with Iran?",
        "What happened between Russia and Ukraine in 2014?",
        "Who visited Japan in 2015?"
    ]
    
    print("3. Running queries:")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\nQuery {i}: {query}")
        
        try:
            # Process query
            result = engine.query(query)
            
            # Display results
            print(f"Execution time: {result.execution_time:.2f}s")
            print(f"Paths discovered: {result.total_paths_discovered}")
            print(f"Paths after pruning: {result.total_paths_after_pruning}")
            
            # Show some retrieved paths
            if result.paths:
                print(f"Found {len(result.paths)} relevant paths")
                # Show first path
                path, metrics = result.paths[0]
                print("Top path:")
                for j, edge in enumerate(path.edges[:3]):
                    print(f"{edge.timestamp}: {edge.source_id} -> {edge.relation_type} -> {edge.target_id}")
                if len(path.edges) > 3:
                    print(f"and {len(path.edges) - 3} more edges")
                print(f"Reliability score: {metrics.reliability_score:.3f}")
            else:
                print("No paths found")
        
        except Exception as e:
            print(f"Error processing query: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\n4. Demo Complete")


if __name__ == "__main__":
    demonstrate_query_engine()