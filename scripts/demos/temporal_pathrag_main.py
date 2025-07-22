#!/usr/bin/env python3
"""
Demonstrates the current (2nd July) Temporal PathRAG pipeline using real temporal KGs
and the new temporal scoring mechanisms.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from datetime import datetime
import networkx as nx
from typing import List, Dict

from src.kg.storage.temporal_graph_storage import TemporalGraphDatabase
from scripts.version_history.basic_path_traversal import BasicPathTraversal
from src.kg.scoring.temporal_scoring import TemporalWeightingFunction, TemporalPathRanker, TemporalRelevanceMode
from src.utils.device import setup_device_and_logging, optimise_for_pathrag


def load_real_temporal_knowledge_graph() -> nx.DiGraph:
    """Load the real temporal knowledge graph from our processed datasets"""
    print("Loading temporal knowledge graph from processed datasets")
    
    # Initialise temporal graph database
    tkg_db = TemporalGraphDatabase()
    
    # Load MultiTQ dataset
    tkg_db.load_dataset_from_structure('/Users/hillcallum/Temporal_PathRAG/datasets/MultiTQ')
    
    # Get the main graph
    graph = tkg_db.get_main_graph()
    stats = tkg_db.get_statistics()
    
    print(f"Loaded TKG: {stats['total_quadruplets']:,} quadruplets")
    print(f"Entities: {stats['unique_entities']:,}")
    print(f"Relations: {stats['unique_relations']:,}")
    print(f"Timestamps: {stats['unique_timestamps']:,}")
    print(f"Time span: {stats['time_span_years']} years")
    
    return graph


def demonstrate_temporal_queries(traversal: BasicPathTraversal):
    """Demonstrate temporal queries with non-toy TKG data"""
    print("\n" + "="*60)
    print("Temporal Query Demonstrations")
    print("="*60)
    
    # Sample temporal queries based on real TKG data
    temporal_queries = [
        {
            "description": "Abdul Kalam's career progression",
            "source": "Abdul_Kalam",
            "target": "ISRO", 
            "query_time": "2010-01-01",
            "temporal_mode": TemporalRelevanceMode.EXPONENTIAL_DECAY
        },
        {
            "description": "Recent diplomatic events (Mahmoud Abbas)",
            "source": "Mahmoud_Abbas",
            "target": "Associated_Press",
            "query_time": "2023-01-01", 
            "temporal_mode": TemporalRelevanceMode.LINEAR_DECAY
        },
        {
            "description": "Head of Government connections",
            "source": "Head_of_Government_(Togo)",
            "target": "Togo",
            "query_time": "2007-01-01",
            "temporal_mode": TemporalRelevanceMode.GAUSSIAN_PROXIMITY
        }
    ]
    
    for i, query in enumerate(temporal_queries, 1):
        print(f"\n{i}. {query['description']}")
        print("-" * 50)
        print(f"Source: {query['source']}")
        print(f"Target: {query['target']}")
        print(f"Query Time: {query['query_time']}")
        print(f"Temporal Mode: {query['temporal_mode'].value}")
        
        # Find paths with temporal scoring
        paths = traversal.find_paths(
            source_node_id=query['source'],
            target_node_id=query['target'],
            max_hops=3,
            top_k=5,
            query_time=query['query_time']
        )
        
        if paths:
            print(f"\nFound {len(paths)} temporally-scored paths:")
            for j, path in enumerate(paths[:3], 1):
                # Extract temporal info
                temp_info = path.get_temporal_info()
                
                print(f"\nPath {j}: Score {path.score:.3f}")
                print(f"Route: {' → '.join([node.name for node in path.nodes])}")
                print(f"Relations: {' → '.join([edge.relation_type for edge in path.edges])}")
                
                if temp_info['timestamps']:
                    print(f"Timestamps: {temp_info['timestamps']}")
                    print(f"Temporal Density: {temp_info['temporal_density']:.2f}")
                
                # Show PathRAG textual representation
                if path.path_text:
                    print(f"PathRAG Text: {path.path_text[:100]}")
        else:
            print("No paths found")


def demonstrate_temporal_scoring_components():
    """Demonstrate individual temporal scoring components"""
    print("\n" + "="*60)
    print("Temporal Scoring Components Analysis")
    print("="*60)
    
    # Initialise temporal weighting function
    temporal_func = TemporalWeightingFunction()
    query_time = "2023-01-01"
    
    # Test different temporal decay modes
    print("\n1. Temporal Decay Factor Analysis")
    print("-" * 40)
    
    test_timestamps = [
        "2023-01-01",  # Same day
        "2022-01-01",  # 1 year ago  
        "2015-01-01",  # 8 years ago
        "2005-01-01",  # 18 years ago
    ]
    
    modes = [
        TemporalRelevanceMode.EXPONENTIAL_DECAY,
        TemporalRelevanceMode.LINEAR_DECAY,
        TemporalRelevanceMode.GAUSSIAN_PROXIMITY,
        TemporalRelevanceMode.SIGMOID_TRANSITION
    ]
    
    print(f"{'Timestamp':<12} {'Exp':<6} {'Lin':<6} {'Gauss':<6} {'Sig':<6}")
    print("-" * 40)
    
    for timestamp in test_timestamps:
        scores = []
        for mode in modes:
            score = temporal_func.temporal_decay_factor(timestamp, query_time, mode)
            scores.append(score)
        
        print(f"{timestamp:<12} {scores[0]:<6.3f} {scores[1]:<6.3f} {scores[2]:<6.3f} {scores[3]:<6.3f}")


def demonstrate_chronological_validation(traversal: BasicPathTraversal):
    """Demonstrate chronological path validation"""
    print("\n" + "="*60)
    print("Chronological Path Validation")
    print("="*60)
    
    # Find some paths and analyse their chronological consistency
    sample_entities = ["Abdul_Kalam", "Mahmoud_Abbas", "Head_of_Government_(Togo)"]
    
    for entity in sample_entities:
        print(f"\nAnalysing paths from: {entity}")
        print("-" * 40)
        
        # Explore neighbourhood to find temporal paths
        paths = traversal.explore_neighbourhood(entity, max_hops=2, top_k=5)
        
        if paths:
            temporal_func = TemporalWeightingFunction()
            
            for i, path in enumerate(paths[:3], 1):
                temp_info = path.get_temporal_info()
                
                if temp_info['timestamps']:
                    # Convert to TemporalPath for scoring
                    from src.kg.scoring.temporal_scoring import TemporalPath
                    temporal_path = TemporalPath(
                        nodes=[node.id for node in path.nodes],
                        edges=[(edge.source_id, edge.relation_type, edge.target_id, 
                               getattr(edge, 'timestamp', '2023-01-01')) for edge in path.edges],
                        timestamps=temp_info['timestamps'],
                        original_score=path.score
                    )
                    
                    # Calculate chronological alignment
                    chrono_score = temporal_func.chronological_alignment_score(temporal_path)
                    consistency_score = temporal_func.temporal_consistency_score(temporal_path)
                    
                    print(f"Path {i}: {' → '.join([node.name for node in path.nodes])}")
                    print(f"Timestamps: {temp_info['timestamps']}")
                    print(f"Chronological Score: {chrono_score:.3f}")
                    print(f"Consistency Score: {consistency_score:.3f}")
                    
                    # Determine if chronologically ordered
                    if chrono_score >= 0.8:
                        print("Well-ordered chronologically")
                    elif chrono_score >= 0.5:
                        print("Partially ordered")
                    else:
                        print("Poor chronological ordering")
                else:
                    print(f"Path {i}: No temporal information available")


def run_temporal_pathrag_demo():
    """Run the Temporal PathRAG pipeline"""
    print("Temporal PathRAG - Path Retrieval with Temporal Reasoning")
    print("="*70)
    
    # Setup GPU acceleration
    print("\n📋 Setting up GPU acceleration")
    device = setup_device_and_logging()
    device = optimise_for_pathrag()
    print(f"Using device: {device}")
    
    # Load real temporal knowledge graph
    graph = load_real_temporal_knowledge_graph()
    
    # Initialise temporal-aware path traversal
    print("\nInitialising temporal-aware path traversal")
    traversal = BasicPathTraversal(
        graph, 
        device=device,
        temporal_mode=TemporalRelevanceMode.EXPONENTIAL_DECAY
    )
    print("Temporal PathRAG traversal initialised")
    
    # Show GPU memory usage
    gpu_info = traversal.get_gpu_memory_usage()
    if 'allocated_gb' in gpu_info:
        print(f"GPU Memory: {gpu_info['allocated_gb']:.2f}GB allocated")
    
    # Run demonstrations
    demonstrate_temporal_queries(traversal)
    demonstrate_temporal_scoring_components()
    demonstrate_chronological_validation(traversal)
    
    # Performance summary
    print("\n" + "="*60)
    print("Performance Summary")
    print("="*60)
    
    final_gpu_info = traversal.get_gpu_memory_usage()
    if 'allocated_gb' in final_gpu_info:
        print(f"Final GPU Memory: {final_gpu_info['allocated_gb']:.2f}GB")
        print("Cleaning up GPU memory")
        traversal.cleanup_gpu_memory()
    
    print("\nTemporal PathRAG demonstration completed!")

if __name__ == "__main__":
    run_temporal_pathrag_demo()