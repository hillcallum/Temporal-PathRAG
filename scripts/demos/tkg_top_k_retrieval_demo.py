#!/usr/bin/env python3
"""
TKG Top-K Temporal Path Retrieval Demo Script

Demonstrates:
1. Adapted flow-based pruning algorithm on TKG
2. Correct identification and ranking based on the temporal reliability scores
3. Top-K path retrieval for various dfferent query types
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

import time
import networkx as nx
from datetime import datetime, timedelta
import numpy as np

from src.kg.tkg_query_engine import TKGQueryEngine
from src.kg.temporal_path_retriever import TemporalPathRetriever
from src.kg.temporal_reliability_scorer import TemporalReliabilityScorer
from src.kg.models import TemporalPathRAGNode, TemporalPathRAGEdge, Path, TemporalQuery
from src.kg.temporal_scoring import TemporalWeightingFunction


def create_test_tkg():
    """Create a comprehensive test TKG with temporal data"""
    graph = nx.DiGraph()
    
    # Add entities (nodes)
    entities = [
        # People
        ('einstein', {'entity_type': 'Person', 'name': 'Albert Einstein', 'description': 'Theoretical physicist'}),
        ('curie', {'entity_type': 'Person', 'name': 'Marie Curie', 'description': 'Physicist and chemist'}),
        ('feynman', {'entity_type': 'Person', 'name': 'Richard Feynman', 'description': 'Theoretical physicist'}),
        ('hawking', {'entity_type': 'Person', 'name': 'Stephen Hawking', 'description': 'Theoretical physicist'}),
        
        # Institutions
        ('princeton', {'entity_type': 'Institution', 'name': 'Princeton University', 'description': 'University in New Jersey'}),
        ('mit', {'entity_type': 'Institution', 'name': 'MIT', 'description': 'Massachusetts Institute of Technology'}),
        ('cambridge', {'entity_type': 'Institution', 'name': 'Cambridge University', 'description': 'University in England'}),
        ('caltech', {'entity_type': 'Institution', 'name': 'Caltech', 'description': 'California Institute of Technology'}),
        
        # Discoveries/Concepts
        ('relativity', {'entity_type': 'Theory', 'name': 'Theory of Relativity', 'description': 'Einstein\'s theory of space and time'}),
        ('quantum_mechanics', {'entity_type': 'Theory', 'name': 'Quantum Mechanics', 'description': 'Theory of atomic and subatomic particles'}),
        ('black_holes', {'entity_type': 'Concept', 'name': 'Black Holes', 'description': 'Regions of spacetime with strong gravity'}),
        ('radioactivity', {'entity_type': 'Phenomenon', 'name': 'Radioactivity', 'description': 'Spontaneous emission of radiation'}),
        
        # Awards
        ('nobel_physics', {'entity_type': 'Award', 'name': 'Nobel Prize in Physics', 'description': 'Prestigious physics award'}),
        ('nobel_chemistry', {'entity_type': 'Award', 'name': 'Nobel Prize in Chemistry', 'description': 'Prestigious chemistry award'}),
        
        # Publications
        ('principia', {'entity_type': 'Publication', 'name': 'Principia Mathematica', 'description': 'Foundational work in mathematics'}),
        ('qed_paper', {'entity_type': 'Publication', 'name': 'QED Paper', 'description': 'Quantum electrodynamics research'}),
        
        # Events
        ('ww2', {'entity_type': 'Event', 'name': 'World War II', 'description': 'Global conflict 1939-1945'}),
        ('manhattan_project', {'entity_type': 'Project', 'name': 'Manhattan Project', 'description': 'Nuclear weapons development project'})
    ]
    
    for entity_id, data in entities:
        graph.add_node(entity_id, **data)
    
    # Add temporal relationships (edges)
    temporal_edges = [
        # Einstein's career
        ('einstein', 'princeton', 'worked_at', '1933-10-01', 1.0, 'Einstein joined Princeton in 1933'),
        ('einstein', 'relativity', 'developed', '1905-06-30', 0.9, 'Special relativity published'),
        ('einstein', 'relativity', 'extended', '1915-11-25', 0.95, 'General relativity published'),
        ('einstein', 'nobel_physics', 'received', '1921-11-09', 0.9, 'Nobel Prize for photoelectric effect'),
        ('einstein', 'manhattan_project', 'influenced', '1939-08-02', 0.7, 'Letter to Roosevelt about atomic weapons'),
        
        # Curie's achievements
        ('curie', 'radioactivity', 'discovered', '1896-03-01', 0.9, 'Discovery of radioactivity'),
        ('curie', 'nobel_physics', 'received', '1903-12-10', 0.9, 'Shared Nobel Prize in Physics'),
        ('curie', 'nobel_chemistry', 'received', '1911-12-10', 0.95, 'Second Nobel Prize in Chemistry'),
        
        # Feynman's contributions
        ('feynman', 'caltech', 'worked_at', '1950-09-01', 1.0, 'Feynman joined Caltech'),
        ('feynman', 'mit', 'graduated_from', '1939-06-01', 0.8, 'Undergraduate degree from MIT'),
        ('feynman', 'quantum_mechanics', 'advanced', '1948-01-01', 0.9, 'Path integral formulation'),
        ('feynman', 'manhattan_project', 'participated_in', '1943-03-01', 0.8, 'Worked on atomic bomb'),
        ('feynman', 'qed_paper', 'published', '1949-04-01', 0.9, 'Quantum electrodynamics work'),
        ('feynman', 'nobel_physics', 'received', '1965-10-21', 0.9, 'Nobel Prize for QED'),
        
        # Hawking's work
        ('hawking', 'cambridge', 'worked_at', '1962-10-01', 1.0, 'Hawking at Cambridge'),
        ('hawking', 'black_holes', 'studied', '1970-01-01', 0.9, 'Black hole thermodynamics'),
        ('hawking', 'black_holes', 'theorised', '1974-03-01', 0.95, 'Hawking radiation theory'),
        
        # Institutional connections
        ('princeton', 'relativity', 'promoted', '1935-01-01', 0.7, 'Princeton advanced relativity research'),
        ('caltech', 'quantum_mechanics', 'researched', '1930-01-01', 0.8, 'Caltech quantum research'),
        ('cambridge', 'black_holes', 'studied', '1970-01-01', 0.7, 'Cambridge theoretical physics'),
        
        # Theoretical connections
        ('relativity', 'black_holes', 'predicts', '1916-01-01', 0.8, 'General relativity predicts black holes'),
        ('quantum_mechanics', 'black_holes', 'describes', '1975-01-01', 0.7, 'Quantum effects in black holes'),
        
        # Historical context
        ('ww2', 'manhattan_project', 'motivated', '1939-09-01', 0.9, 'War spurred atomic research'),
        ('manhattan_project', 'quantum_mechanics', 'applied', '1943-01-01', 0.8, 'Quantum theory in bomb design'),
        
        # Cross-temporal influences
        ('curie', 'feynman', 'influenced', '1920-01-01', 0.6, 'Curie\'s work influenced later physicists'),
        ('einstein', 'hawking', 'influenced', '1960-01-01', 0.7, 'Einstein\'s theories influenced Hawking'),
        ('feynman', 'hawking', 'mentored', '1965-01-01', 0.5, 'Feynman and Hawking interactions')
    ]
    
    for source, target, relation, timestamp, weight, description in temporal_edges:
        graph.add_edge(source, target, 
                      relation_type=relation,
                      timestamp=timestamp,
                      weight=weight,
                      flow_capacity=weight,
                      description=description)
    
    return graph


def create_graph_statistics(graph):
    stats = {
        'entities': {},
        'relations': {},
        'max_entity_frequency': 0,
        'max_relation_frequency': 0
    }
    
    # Calculate entity frequencies
    entity_frequencies = {}
    relation_frequencies = {}
    
    for node in graph.nodes():
        entity_frequencies[node] = graph.degree(node)
    
    for _, _, data in graph.edges(data=True):
        relation_type = data.get('relation_type', 'unknown')
        relation_frequencies[relation_type] = relation_frequencies.get(relation_type, 0) + 1
    
    # Populate statistics
    stats['max_entity_frequency'] = max(entity_frequencies.values()) if entity_frequencies else 1
    stats['max_relation_frequency'] = max(relation_frequencies.values()) if relation_frequencies else 1
    
    for entity, freq in entity_frequencies.items():
        stats['entities'][entity] = {'frequency': freq}
    
    for relation, freq in relation_frequencies.items():
        stats['relations'][relation] = {'frequency': freq}
    
    return stats


def flow_based_pruning():
    print("Flow-Based Pruning Algorithm on TKG\n")
    
    # Create test TKG
    graph = create_test_tkg()
    graph_stats = create_graph_statistics(graph)
    
    print(f"Test TKG created with {len(graph.nodes())} nodes and {len(graph.edges())} edges")
    
    # Initialise temporal path retriever
    retriever = TemporalPathRetriever(
        graph=graph,
        alpha=0.15,  # Moderate temporal decay
        base_theta=1.2,  # Moderate pruning threshold
        diversity_threshold=0.7
    )
    
    # Create test query
    query = TemporalQuery(
        query_text="Find paths from Einstein to modern physics",
        source_entities=['einstein'],
        target_entities=['black_holes', 'quantum_mechanics'],
        temporal_constraints={'temporal_preference': 'historical'},
        query_time='1970-01-01',
        max_hops=3,
        top_k=8
    )
    
    print(f"Query: {query.query_text}")
    print(f"Source entities: {query.source_entities}")
    print(f"Target entities: {query.target_entities}")
    print(f"Query time: {query.query_time}\n")
    
    # Retrieve paths with flow-based pruning
    print("Retrieving paths with temporal flow-based pruning")
    start_time = time.time()
    
    retrieved_paths = retriever.retrieve_temporal_paths(
        query=query,
        enable_flow_pruning=True,
        enable_diversity=True,
        verbose=True
    )
    
    retrieval_time = time.time() - start_time
    
    print(f"\nFlow-based pruning completed in {retrieval_time:.3f}s")
    print(f"Retrieved {len(retrieved_paths)} temporally relevant paths\n")
    
    # Display top paths
    print("Top retrieved paths:")
    for i, (path, score) in enumerate(retrieved_paths[:5], 1):
        path_str = " → ".join([node.name for node in path.nodes])
        print(f"{i}. {path_str} (Score: {score:.3f})")
        
        # Show temporal information
        temporal_info = path.get_temporal_info()
        if temporal_info['timestamps']:
            print(f"Temporal span: {temporal_info['timestamps'][0]} to {temporal_info['timestamps'][-1]}")
        print()
    
    return retrieved_paths, graph, graph_stats


def temporal_reliability_ranking():
    print("Temporal Reliability Scoring and Ranking\n")
    
    # Use results from previous demonstration
    retrieved_paths, graph, graph_stats = flow_based_pruning()
    
    # Initialise reliability scorer
    reliability_scorer = TemporalReliabilityScorer(
        temporal_weighting=TemporalWeightingFunction(),
        graph_statistics=graph_stats,
        reliability_threshold=0.6,
        enable_cross_validation=True
    )
    
    print("Applying advanced temporal reliability scoring")
    
    # Extract paths for scoring
    paths_only = [path for path, _ in retrieved_paths]
    query_time = '1970-01-01'
    query_context = {
        'query_text': 'Find paths from Einstein to modern physics',
        'temporal_constraints': {'temporal_preference': 'historical'}
    }
    
    # Score and rank paths
    start_time = time.time()
    reliability_ranked_paths = reliability_scorer.rank_paths_by_reliability(
        paths_only, query_time, query_context
    )
    scoring_time = time.time() - start_time
    
    print(f"Reliability scoring completed in {scoring_time:.3f}s\n")
    
    # Display detailed reliability analysis
    print("Detailed Reliability Analysis:")
    print("-" * 80)
    
    for i, (path, metrics) in enumerate(reliability_ranked_paths[:5], 1):
        path_str = " → ".join([node.name for node in path.nodes])
        print(f"\n{i}. {path_str}")
        print(f"Overall Reliability: {metrics.overall_reliability:.3f}")
        print(f"Component Scores:")
        print(f"    Temporal Consistency: {metrics.temporal_consistency:.3f}")
        print(f"    Chronological Coherence: {metrics.chronological_coherence:.3f}")
        print(f"    Source Credibility: {metrics.source_credibility:.3f}")
        print(f"    Cross Validation: {metrics.cross_validation_score:.3f}")
        print(f"    Pattern Strength: {metrics.temporal_pattern_strength:.3f}")
        print(f"    Flow Reliability: {metrics.flow_reliability:.3f}")
        print(f"    Semantic Coherence: {metrics.semantic_coherence:.3f}")
    
    # Filter reliable paths
    reliable_paths = reliability_scorer.filter_reliable_paths(
        paths_only, query_time, query_context
    )
    
    print(f"\n\nReliable paths (threshold ≥ {reliability_scorer.reliability_threshold}): {len(reliable_paths)}")
    
    return reliability_ranked_paths


def top_k_retrieval():
    print("\nComplete Top-K Path Retrieval System\n")
    
    # Create TKG and initialise query engine
    graph = create_test_tkg()
    graph_stats = create_graph_statistics(graph)
    
    query_engine = TKGQueryEngine(
        graph=graph,
        graph_statistics=graph_stats,
        alpha=0.12,  # Temporal decay rate
        base_theta=1.1,  # Pruning threshold
        reliability_threshold=0.65,
        diversity_threshold=0.75
    )
    
    # Test queries demonstrating different temporal aspects
    test_queries = [
        "Find connections between Einstein and quantum mechanics before 1950",
        "What are the paths from Marie Curie to modern Nobel Prize winners?",
        "Show relationships between Princeton and theoretical physics discoveries",
        "How is the Manhattan Project connected to quantum mechanics research?",
        "Find temporal paths from early radioactivity discoveries to black hole research"
    ]
    
    print("Executing comprehensive Top-K retrieval queries:\n")
    
    results = []
    for i, query_text in enumerate(test_queries, 1):
        print(f"Query {i}: {query_text}")
        print("-" * 60)
        
        start_time = time.time()
        result = query_engine.query(
            query_text=query_text,
            enable_flow_pruning=True,
            enable_reliability_filtering=True,
            enable_diversity=True,
            verbose=False
        )
        execution_time = time.time() - start_time
        
        results.append(result)
        
        print(f"Execution time: {execution_time:.3f}s")
        print(f"Paths discovered: {result.total_paths_discovered}")
        print(f"Paths after pruning: {result.total_paths_after_pruning}")
        print(f"Final reliable paths: {len(result.paths)}")
        
        if result.paths:
            print(f"Top path reliability: {result.paths[0][1].overall_reliability:.3f}")
            
            # Show top 3 paths
            print("\nTop 3 paths:")
            for j, (path, metrics) in enumerate(result.paths[:3], 1):
                path_str = " → ".join([node.name for node in path.nodes])
                print(f"{j}. {path_str} (Reliability: {metrics.overall_reliability:.3f})")
        else:
            print("No reliable paths found")
        
        print("\n")
    
    return results, query_engine


def system_validation():
    print("System Validation and Performance\n")
    
    results, query_engine = top_k_retrieval()
    
    # System statistics
    system_stats = query_engine.get_system_statistics()
    print("System Configuration:")
    print(f"Graph: {system_stats['graph_statistics']['nodes']} nodes, {system_stats['graph_statistics']['edges']} edges")
    print(f"Reliability threshold: {system_stats['reliability_config']['threshold']}")
    print(f"Temporal decay rate: {system_stats['temporal_config']['decay_rate']}")
    print(f"Temporal window: {system_stats['temporal_config']['temporal_window']} days")
    
    # Performance metrics
    execution_times = [r.execution_time for r in results]
    path_counts = [len(r.paths) for r in results]
    reliability_scores = []
    
    for result in results:
        for _, metrics in result.paths:
            reliability_scores.append(metrics.overall_reliability)
    
    print(f"\nPerformance Metrics:")
    print(f"Average execution time: {np.mean(execution_times):.3f}s")
    print(f"Average paths returned: {np.mean(path_counts):.1f}")
    if reliability_scores:
        print(f"Average reliability score: {np.mean(reliability_scores):.3f}")
        print(f"Min reliability score: {np.min(reliability_scores):.3f}")
        print(f"Max reliability score: {np.max(reliability_scores):.3f}")
    
    # Validation with test queries
    validation_queries = [
        "Find Einstein's contributions to physics",
        "Show Curie's Nobel Prize achievements", 
        "Connect Feynman to quantum mechanics",
        "Trace black hole research history"
    ]
    
    print(f"\nValidation with {len(validation_queries)} test queries")
    validation_metrics = query_engine.validate_system(validation_queries)
    
    print(f"Validation Results:")
    print(f"Success rate: {validation_metrics['success_rate']:.1%}")
    print(f"Average execution time: {validation_metrics['average_execution_time']:.3f}s")
    print(f"Average paths returned: {validation_metrics['average_paths_returned']:.1f}")
    print(f"Average reliability: {validation_metrics['average_reliability_score']:.3f}")


def main():
    print("Temporal Knowledge Graph (TKG) Top-K Path Retrieval Demonstration")
    print("=" * 80)
    print()
    
    try:
        # Run all demonstrations
        flow_based_pruning()
        temporal_reliability_ranking()
        top_k_retrieval()
        system_validation()
        
        print("\n" + "=" * 80)
        print("TKG Top-K retrieval system fully implemented and validated")
        
        return 0
        
    except Exception as e:
        print(f"\nDemonstration failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())