#!/usr/bin/env python3
"""
Temporal PathRAG - Main Implementation

The current implementation (as of 3rd July) of Temporal PathRAG that integrates:
- Real temporal KGs (MultiTQ/TimeQuestions)
- Temporal-aware path scoring with multiple decay modes
- Enhanced reliability scores S'(P) combining structural & temporal components
- GPU-accelerated semantic similarity
- Chronological consistency validation
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import networkx as nx

from src.kg.path_traversal import TemporalPathTraversal
from src.kg.temporal_scoring import TemporalWeightingFunction, TemporalRelevanceMode
from src.utils.device import setup_device_and_logging, optimise_for_pathrag
from src.utils.dataset_loader import load_dataset


def load_temporal_knowledge_graph(dataset_name: str = "MultiTQ") -> nx.DiGraph:
    """Load the temporal knowledge graph using dataset loader"""
    print("Loading temporal knowledge graph")
    return load_dataset(dataset_name)


def run_temporal_queries(traversal: TemporalPathTraversal):
    """Run sample temporal queries demonstrating the system capabilities"""
    print("\n" + "="*60)
    print("Temporal Query Demonstrations")
    print("="*60)
    
    # High-impact temporal queries using real TKG entities
    queries = [
        {
            "name": "Abdul Kalam Career Path",
            "source": "Abdul_Kalam",
            "target": "ISRO",
            "query_time": "2010-01-01",
            "expected": "Should show temporal progression of Kalam's career"
        },
        {
            "name": "Diplomatic Relations",
            "source": "Mahmoud_Abbas", 
            "target": "Associated_Press",
            "query_time": "2023-01-01",
            "expected": "Recent diplomatic communications"
        },
        {
            "name": "Government Connections",
            "source": "Head_of_Government_(Togo)",
            "target": "Togo",
            "query_time": "2007-01-01", 
            "expected": "Government administrative paths"
        }
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n{i}. {query['name']}")
        print("-" * 50)
        print(f"{query['source']} → {query['target']}")
        print(f"Query Time: {query['query_time']}")
        print(f"Expected: {query['expected']}")
        
        # Find temporally-scored paths
        paths = traversal.find_paths(
            source_id=query['source'],
            target_id=query['target'],
            max_depth=3,
            max_paths=3,
            query_time=query['query_time']
        )
        
        if paths:
            print(f"\nFound {len(paths)} paths:")
            for j, path in enumerate(paths, 1):
                temp_info = path.get_temporal_info()
                
                print(f"\nPath {j}: Score {path.score:.3f}")
                print(f"{' → '.join([node.name for node in path.nodes])}")
                print(f"{' → '.join([edge.relation_type for edge in path.edges])}")
                
                if temp_info['timestamps']:
                    print(f"Times: {temp_info['timestamps']}")
                    print(f"Temporal Density: {temp_info['temporal_density']:.2f}")
        else:
            print("No paths found")


def analyse_temporal_scoring():
    """Analyse temporal scoring components"""
    print("\n" + "="*60)
    print("Temporal Scoring Analysis")
    print("="*60)
    
    temporal_func = TemporalWeightingFunction()
    query_time = "2023-01-01"
    
    print("\n1. Temporal Decay Comparison")
    print("-" * 40)
    
    test_times = ["2023-01-01", "2020-01-01", "2010-01-01", "2005-01-01"]
    modes = [
        ("Exponential", TemporalRelevanceMode.EXPONENTIAL_DECAY),
        ("Linear", TemporalRelevanceMode.LINEAR_DECAY),
        ("Gaussian", TemporalRelevanceMode.GAUSSIAN_PROXIMITY),
        ("Sigmoid", TemporalRelevanceMode.SIGMOID_TRANSITION)
    ]
    
    print(f"{'Event Time':<12} {'Exp':<6} {'Lin':<6} {'Gauss':<6} {'Sig':<6}")
    print("-" * 40)
    
    for timestamp in test_times:
        scores = []
        for _, mode in modes:
            score = temporal_func.temporal_decay_factor(timestamp, query_time, mode)
            scores.append(score)
        
        print(f"{timestamp:<12} {scores[0]:<6.3f} {scores[1]:<6.3f} {scores[2]:<6.3f} {scores[3]:<6.3f}")
    
    print(f"\nAnalysis for query time {query_time}:")


def validate_chronological_consistency(traversal: TemporalPathTraversal):
    """Validate chronological consistency of paths"""
    print("\n" + "="*60) 
    print("Chronological Consistency Validation")
    print("="*60)
    
    # Test entities known to have temporal sequences
    test_entities = ["Abdul_Kalam", "Mahmoud_Abbas"]
    
    temporal_func = TemporalWeightingFunction()
    
    for entity in test_entities:
        print(f"\nAnalysing temporal paths from: {entity}")
        print("-" * 40)
        
        paths = traversal.explore_neighbourhood(entity, max_hops=2, top_k=3)
        
        temporal_paths_found = 0
        for i, path in enumerate(paths, 1):
            temp_info = path.get_temporal_info()
            
            if temp_info['timestamps'] and len(temp_info['timestamps']) > 1:
                temporal_paths_found += 1
                
                # Convert to TemporalPath for analysis
                from src.kg.temporal_scoring import TemporalPath
                temporal_path = TemporalPath(
                    nodes=[node.id for node in path.nodes],
                    edges=[(edge.source_id, edge.relation_type, edge.target_id,
                           getattr(edge, 'timestamp', '2023-01-01')) for edge in path.edges],
                    timestamps=temp_info['timestamps'],
                    original_score=path.score
                )
                
                # Calculate scores
                chrono_score = temporal_func.chronological_alignment_score(temporal_path)
                consistency_score = temporal_func.temporal_consistency_score(temporal_path)
                
                print(f"\nPath {i}: {' → '.join([node.name for node in path.nodes])}")
                print(f"Timestamps: {temp_info['timestamps']}")
                print(f"Chronological: {chrono_score:.3f}")
                print(f"Consistency: {consistency_score:.3f}")
                
                # Evaluation
                if chrono_score >= 0.8:
                    print("Excellent chronological ordering")
                elif chrono_score >= 0.5:
                    print("Moderate chronological ordering")
                else:
                    print("Poor chronological ordering")
        
        if temporal_paths_found == 0:
            print("No multi-temporal paths found")


def main(dataset_name: str = "MultiTQ", temporal_mode: str = "exponential_decay"):
    """Main Temporal PathRAG demonstration"""
    from src.utils import get_config
    
    print("Temporal PathRAG - Temporal-Aware Path Retrieval")
    print("="*70)
    print("Integrating temporal dimensions into PathRAG's structural flow")
    print("="*70)
    
    # Get configuration
    config = get_config()
    print(f"Configuration loaded from: {config.base_dir}")
    
    # Validate paths
    validation = config.validate_paths()
    missing_paths = [k for k, v in validation.items() if not v]
    if missing_paths:
        print(f"Warning: Missing paths: {missing_paths}")
    
    # Setup GPU acceleration
    print("\nSetting up GPU acceleration")
    device = setup_device_and_logging()
    device = optimise_for_pathrag()
    print(f"Device: {device}")
    
    # Load temporal knowledge graph
    graph = load_temporal_knowledge_graph(dataset_name)
    
    # Initialise temporal-aware traversal
    print("\nInitialising Temporal PathRAG")
    
    # Map temporal mode string to enum
    mode_map = {
        "exponential_decay": TemporalRelevanceMode.EXPONENTIAL_DECAY,
        "linear_decay": TemporalRelevanceMode.LINEAR_DECAY,
        "gaussian_proximity": TemporalRelevanceMode.GAUSSIAN_PROXIMITY,
        "sigmoid_transition": TemporalRelevanceMode.SIGMOID_TRANSITION
    }
    
    temporal_mode_enum = mode_map.get(temporal_mode, TemporalRelevanceMode.EXPONENTIAL_DECAY)
    
    traversal = TemporalPathTraversal(
        graph,
        device=device,
        temporal_mode=temporal_mode_enum
    )
    print(f"Temporal PathRAG initialised with {temporal_mode} mode")
    
    # Show system capabilities
    gpu_info = traversal.get_gpu_memory_usage()
    if 'allocated_gb' in gpu_info:
        print(f"GPU Memory: {gpu_info['allocated_gb']:.2f}GB allocated")
    
    # Run demonstrations
    run_temporal_queries(traversal)
    analyse_temporal_scoring()
    validate_chronological_consistency(traversal)
    
    # Summary
    print("\n" + "="*60)
    print("System Summary")
    print("="*60)
    
    final_gpu_info = traversal.get_gpu_memory_usage()
    if 'allocated_gb' in final_gpu_info:
        print(f"Final GPU Memory: {final_gpu_info['allocated_gb']:.2f}GB")
        print("Cleaning up")
        traversal.cleanup_gpu_memory()
        
    print("\nTemporal PathRAG demonstration completed")

if __name__ == "__main__":
    main()