#!/usr/bin/env python3
"""
Demonstration script for Temporal PathRAG weighting functions
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from datetime import datetime, timedelta
import networkx as nx
from typing import List
import matplotlib.pyplot as plt
import numpy as np

from src.kg.temporal_scoring import (
    TemporalWeightingFunction, 
    TemporalPathRanker, 
    TemporalPath, 
    TemporalRelevanceMode
)
from src.kg.models import TemporalPathRAGNode, TemporalPathRAGEdge, Path
from src.kg.path_traversal import TemporalPathTraversal


def create_sample_temporal_graph() -> nx.DiGraph:
    """
    Create a sample temporal knowledge graph with realistic temporal relationships
    """
    graph = nx.DiGraph()
    
    # Sample temporal facts (entity, relation, entity, timestamp)
    temporal_facts = [
        # Abdul Kalam career events
        ("Abdul_Kalam", "born_in", "Rameswaram", "1931-10-15"),
        ("Abdul_Kalam", "worked_at", "ISRO", "1969-01-01"),
        ("Abdul_Kalam", "led_project", "SLV-3", "1980-07-18"),
        ("Abdul_Kalam", "became_president", "India", "2002-07-25"),
        ("Abdul_Kalam", "wrote_book", "Wings_of_Fire", "1999-01-01"),
        
        # Related entities and events
        ("ISRO", "launched", "SLV-3", "1980-07-18"),
        ("India", "gained_independence", "British_Empire", "1947-08-15"),
        ("Wings_of_Fire", "published_by", "Universities_Press", "1999-01-01"),
        
        # More recent events for temporal comparison
        ("SpaceX", "launched", "Falcon_Heavy", "2018-02-06"),
        ("NASA", "landed_rover", "Mars", "2021-02-18"),
        
        # Historical context
        ("Wright_Brothers", "first_flight", "Kitty_Hawk", "1903-12-17"),
        ("Sputnik", "launched_by", "Soviet_Union", "1957-10-04"),
    ]
    
    # Add edges to graph with temporal information
    for subj, pred, obj, timestamp in temporal_facts:
        # Add nodes if they don't exist
        if not graph.has_node(subj):
            graph.add_node(subj, entity_type="Entity", name=subj)
        if not graph.has_node(obj):
            graph.add_node(obj, entity_type="Entity", name=obj)
        
        # Add edge with temporal information
        graph.add_edge(subj, obj, 
                      relation_type=pred, 
                      timestamp=timestamp,
                      weight=1.0,
                      description=f"{subj} {pred} {obj} on {timestamp}")
    
    return graph


def create_sample_temporal_paths() -> List[TemporalPath]:
    """Create sample temporal paths for demonstration"""
    
    # Path 1: Abdul Kalam's early career (chronologically ordered)
    path1 = TemporalPath(
        nodes=["Abdul_Kalam", "ISRO", "SLV-3"],
        edges=[
            ("Abdul_Kalam", "worked_at", "ISRO", "1969-01-01"),
            ("ISRO", "launched", "SLV-3", "1980-07-18")
        ],
        timestamps=["1969-01-01", "1980-07-18"],
        original_score=0.85
    )
    
    # Path 2: Space exploration timeline (well-ordered, older events)
    path2 = TemporalPath(
        nodes=["Wright_Brothers", "Soviet_Union", "NASA"],
        edges=[
            ("Wright_Brothers", "first_flight", "Kitty_Hawk", "1903-12-17"),
            ("Soviet_Union", "launched", "Sputnik", "1957-10-04")
        ],
        timestamps=["1903-12-17", "1957-10-04"],
        original_score=0.75
    )
    
    # Path 3: Modern space achievements (recent, high relevance)
    path3 = TemporalPath(
        nodes=["SpaceX", "NASA", "Mars"],
        edges=[
            ("SpaceX", "launched", "Falcon_Heavy", "2018-02-06"),
            ("NASA", "landed_rover", "Mars", "2021-02-18")
        ],
        timestamps=["2018-02-06", "2021-02-18"],
        original_score=0.70
    )
    
    # Path 4: Mixed timeline (poor chronological ordering)
    path4 = TemporalPath(
        nodes=["Abdul_Kalam", "India", "British_Empire"],
        edges=[
            ("Abdul_Kalam", "became_president", "India", "2002-07-25"),
            ("India", "gained_independence", "British_Empire", "1947-08-15")  # Out of chronological order
        ],
        timestamps=["2002-07-25", "1947-08-15"],
        original_score=0.80
    )
    
    return [path1, path2, path3, path4]


def demonstrate_temporal_decay_functions():
    """Demonstrate different temporal decay function modes"""
    print("=== Temporal Decay Function Demonstration ===\n")
    
    weighting_func = TemporalWeightingFunction()
    query_time = "2023-01-01"  # Reference time
    
    # Test timestamps at different distances from query time
    test_timestamps = [
        "2023-01-01",  # Same day
        "2022-07-01",  # 6 months ago
        "2022-01-01",  # 1 year ago
        "2020-01-01",  # 3 years ago
        "2010-01-01",  # 13 years ago
        "1990-01-01",  # 33 years ago
    ]
    
    modes = [
        TemporalRelevanceMode.EXPONENTIAL_DECAY,
        TemporalRelevanceMode.LINEAR_DECAY,
        TemporalRelevanceMode.GAUSSIAN_PROXIMITY,
        TemporalRelevanceMode.SIGMOID_TRANSITION
    ]
    
    print(f"Query Time: {query_time}")
    print(f"{'Timestamp':<12} {'Exp_Decay':<10} {'Linear':<8} {'Gaussian':<10} {'Sigmoid':<8}")
    print("-" * 55)
    
    for timestamp in test_timestamps:
        scores = []
        for mode in modes:
            score = weighting_func.temporal_decay_factor(timestamp, query_time, mode)
            scores.append(score)
        
        print(f"{timestamp:<12} {scores[0]:<10.3f} {scores[1]:<8.3f} "
              f"{scores[2]:<10.3f} {scores[3]:<8.3f}")
    
    print()


def demonstrate_path_scoring():
    """Demonstrate enhanced path scoring with temporal components"""
    print("=== Enhanced Path Scoring Demonstration ===\n")
    
    weighting_func = TemporalWeightingFunction()
    query_time = "2023-01-01"
    
    paths = create_sample_temporal_paths()
    
    print(f"Query Time: {query_time}")
    print(f"{'Path':<6} {'Original':<10} {'Chrono':<8} {'Proximity':<10} {'Consistency':<12} {'Enhanced':<10}")
    print("-" * 70)
    
    for i, path in enumerate(paths, 1):
        # Calculate individual temporal components
        chrono_score = weighting_func.chronological_alignment_score(path)
        proximity_score = weighting_func.temporal_proximity_score(path, query_time)
        consistency_score = weighting_func.temporal_consistency_score(path)
        
        # Calculate enhanced score
        enhanced_score = weighting_func.enhanced_reliability_score(
            path, query_time, path.original_score
        )
        
        print(f"Path{i:<2} {path.original_score:<10.3f} {chrono_score:<8.3f} "
              f"{proximity_score:<10.3f} {consistency_score:<12.3f} {enhanced_score:<10.3f}")
    
    print()


def demonstrate_temporal_ranking():
    """Demonstrate temporal path ranking system"""
    print("=== Temporal Path Ranking Demonstration ===\n")
    
    weighting_func = TemporalWeightingFunction()
    ranker = TemporalPathRanker(weighting_func)
    
    paths = create_sample_temporal_paths()
    query_time = "2023-01-01"
    
    # Rank paths using enhanced temporal scoring
    ranked_paths = ranker.rank_paths(paths, query_time, top_k=4)
    
    print(f"Paths ranked by enhanced temporal scoring (Query Time: {query_time}):")
    print(f"{'Rank':<6} {'Path Description':<50} {'Enhanced Score':<15}")
    print("-" * 75)
    
    for rank, (path, score) in enumerate(ranked_paths, 1):
        description = f"{' -> '.join(path.nodes[:2])}" if len(path.nodes) > 2 else ' -> '.join(path.nodes)
        time_range = f"({path.timestamps[0]} to {path.timestamps[-1]})" if len(path.timestamps) > 1 else f"({path.timestamps[0]})"
        
        print(f"{rank:<6} {description:<30} {time_range:<20} {score:<15.3f}")
    
    print()


def analyse_temporal_patterns():
    """Analyse temporal patterns across paths"""
    print("=== Temporal Pattern Analysis ===\n")
    
    weighting_func = TemporalWeightingFunction()
    ranker = TemporalPathRanker(weighting_func)
    
    paths = create_sample_temporal_paths()
    
    # Analyse patterns
    patterns = ranker.analyse_temporal_patterns(paths)
    
    print("Temporal Pattern Statistics:")
    for key, value in patterns.items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    
    print()


def create_comparison_visualisation():
    """Create visualisation comparing original vs enhanced scoring"""
    print("=== Creating Scoring Comparison Visualisation ===\n")
    
    weighting_func = TemporalWeightingFunction()
    query_time = "2023-01-01"
    paths = create_sample_temporal_paths()
    
    original_scores = [path.original_score for path in paths]
    enhanced_scores = []
    
    for path in paths:
        enhanced_score = weighting_func.enhanced_reliability_score(
            path, query_time, path.original_score
        )
        enhanced_scores.append(enhanced_score)
    
    # Create visualisation
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    path_labels = [f"Path {i+1}" for i in range(len(paths))]
    x_pos = np.arange(len(path_labels))
    
    # Original scores
    bars1 = ax1.bar(x_pos, original_scores, alpha=0.7, colour='lightblue', label='Original PathRAG Score')
    ax1.set_xlabel('Paths')
    ax1.set_ylabel('Score')
    ax1.set_title('Original PathRAG Scoring')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(path_labels)
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='centre', va='bottom')
    
    # Enhanced scores
    bars2 = ax2.bar(x_pos, enhanced_scores, alpha=0.7, colour='lightcoral', label='Enhanced Temporal Score')
    ax2.set_xlabel('Paths')
    ax2.set_ylabel('Score')
    ax2.set_title('Enhanced Temporal Scoring')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(path_labels)
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='centre', va='bottom')
    
    plt.tight_layout()
    
    # Save visualisation
    output_path = "/Users/hillcallum/Temporal_PathRAG/analysis_results/temporal_scoring_comparison.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualisation saved to: {output_path}")
    
    plt.show()


def main():
    """Main demonstration function"""
    print("Temporal PathRAG Weighting Functions Demonstration")
    print("=" * 60)
    print()
    
    try:
        # Demonstrate different components
        demonstrate_temporal_decay_functions()
        demonstrate_path_scoring()
        demonstrate_temporal_ranking()
        analyse_temporal_patterns()
        create_comparison_visualisation()
        
        # Demonstrate enhanced temporal flow pruning
        print("=== Enhanced Temporal Flow Pruning Demonstration ===\n")
        print("Note: This demonstrates the integration of temporal decay rate alpha and threshold theta")
        print("in PathRAG's resource propagation algorithm.\n")
        
        print("Demonstration completed successfully!")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()