#!/usr/bin/env python3
"""
Temporal Path Retrieval Tests

This script creates and tests temporal multi-hop sample queries to validate:
- Temporal relevance of retrieved paths
- Chronological coherence and non-redundancy
- Temporal weighting and pruning logic effectiveness
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

import time
import networkx as nx
from datetime import datetime, timedelta
import numpy as np
from typing import List, Dict, Tuple, Any
import json

from src.kg.tkg_query_engine import TKGQueryEngine
from src.kg.temporal_path_retriever import TemporalPathRetriever
from src.kg.temporal_reliability_scorer import TemporalReliabilityScorer
from src.kg.models import TemporalPathRAGNode, TemporalPathRAGEdge, Path, TemporalQuery
from src.kg.temporal_scoring import TemporalWeightingFunction


def create_test_tkg():
    """Create a test TKG with diverse temporal relationships"""
    graph = nx.MultiDiGraph()
    
    # Add entities across different domains and time periods
    entities = [
        # Historical Scientists (1800s-1900s)
        ('darwin', {'entity_type': 'Person', 'name': 'Charles Darwin', 'description': 'Naturalist and biologist'}),
        ('mendel', {'entity_type': 'Person', 'name': 'Gregor Mendel', 'description': 'Father of genetics'}),
        ('pasteur', {'entity_type': 'Person', 'name': 'Louis Pasteur', 'description': 'Microbiologist'}),
        ('curie', {'entity_type': 'Person', 'name': 'Marie Curie', 'description': 'Physicist and chemist'}),
        ('einstein', {'entity_type': 'Person', 'name': 'Albert Einstein', 'description': 'Theoretical physicist'}),
        
        # Modern Scientists (1900s-2000s)
        ('watson', {'entity_type': 'Person', 'name': 'James Watson', 'description': 'Molecular biologist'}),
        ('crick', {'entity_type': 'Person', 'name': 'Francis Crick', 'description': 'Molecular biologist'}),
        ('hawking', {'entity_type': 'Person', 'name': 'Stephen Hawking', 'description': 'Theoretical physicist'}),
        ('turing', {'entity_type': 'Person', 'name': 'Alan Turing', 'description': 'Computer scientist'}),
        
        # Institutions
        ('cambridge', {'entity_type': 'Institution', 'name': 'Cambridge University', 'description': 'English university'}),
        ('oxford', {'entity_type': 'Institution', 'name': 'Oxford University', 'description': 'English university'}),
        ('harvard', {'entity_type': 'Institution', 'name': 'Harvard University', 'description': 'American university'}),
        ('princeton', {'entity_type': 'Institution', 'name': 'Princeton University', 'description': 'American university'}),
        ('mit', {'entity_type': 'Institution', 'name': 'MIT', 'description': 'Massachusetts Institute of Technology'}),
        
        # Scientific Concepts and Theories
        ('evolution', {'entity_type': 'Theory', 'name': 'Theory of Evolution', 'description': 'Darwin\'s theory of species development'}),
        ('genetics', {'entity_type': 'Field', 'name': 'Genetics', 'description': 'Study of heredity and genes'}),
        ('dna', {'entity_type': 'Molecule', 'name': 'DNA', 'description': 'Deoxyribonucleic acid'}),
        ('germ_theory', {'entity_type': 'Theory', 'name': 'Germ Theory', 'description': 'Microorganisms cause disease'}),
        ('relativity', {'entity_type': 'Theory', 'name': 'Theory of Relativity', 'description': 'Einstein\'s theory of space-time'}),
        ('quantum_mechanics', {'entity_type': 'Theory', 'name': 'Quantum Mechanics', 'description': 'Theory of atomic particles'}),
        ('computer_science', {'entity_type': 'Field', 'name': 'Computer Science', 'description': 'Study of computation'}),
        ('artificial_intelligence', {'entity_type': 'Field', 'name': 'Artificial Intelligence', 'description': 'Machine intelligence'}),
        
        # Technologies and Inventions
        ('microscope', {'entity_type': 'Technology', 'name': 'Microscope', 'description': 'Optical magnification device'}),
        ('x_ray', {'entity_type': 'Technology', 'name': 'X-ray', 'description': 'Electromagnetic radiation imaging'}),
        ('computer', {'entity_type': 'Technology', 'name': 'Computer', 'description': 'Electronic computation device'}),
        ('internet', {'entity_type': 'Technology', 'name': 'Internet', 'description': 'Global computer network'}),
        
        # Awards and Recognition
        ('nobel_physics', {'entity_type': 'Award', 'name': 'Nobel Prize in Physics', 'description': 'Prestigious physics award'}),
        ('nobel_chemistry', {'entity_type': 'Award', 'name': 'Nobel Prize in Chemistry', 'description': 'Prestigious chemistry award'}),
        ('nobel_medicine', {'entity_type': 'Award', 'name': 'Nobel Prize in Medicine', 'description': 'Prestigious medicine award'}),
        ('royal_society', {'entity_type': 'Honor', 'name': 'Royal Society Fellowship', 'description': 'British scientific honor'}),
        
        # Historical Events
        ('industrial_revolution', {'entity_type': 'Event', 'name': 'Industrial Revolution', 'description': 'Period of technological advancement'}),
        ('ww1', {'entity_type': 'Event', 'name': 'World War I', 'description': 'Global conflict 1914-1918'}),
        ('ww2', {'entity_type': 'Event', 'name': 'World War II', 'description': 'Global conflict 1939-1945'}),
        ('cold_war', {'entity_type': 'Event', 'name': 'Cold War', 'description': 'Political tension 1947-1991'}),
        ('space_race', {'entity_type': 'Event', 'name': 'Space Race', 'description': 'Competition in space exploration'}),
        
        # Publications and Works
        ('origin_of_species', {'entity_type': 'Publication', 'name': 'On the Origin of Species', 'description': 'Darwin\'s evolutionary theory book'}),
        ('principia', {'entity_type': 'Publication', 'name': 'Principia Mathematica', 'description': 'Mathematical foundations'}),
        ('nature_journal', {'entity_type': 'Publication', 'name': 'Nature Journal', 'description': 'Scientific journal'}),
        ('science_journal', {'entity_type': 'Publication', 'name': 'Science Journal', 'description': 'Scientific journal'})
    ]
    
    for entity_id, data in entities:
        graph.add_node(entity_id, **data)
    
    # Add temporal relationships
    temporal_edges = [
        # Early Scientific Foundations (1800s)
        ('darwin', 'cambridge', 'studied_at', '1828-01-01', 0.9, 'Darwin studied at Cambridge'),
        ('darwin', 'evolution', 'developed', '1859-11-24', 0.95, 'Published Origin of Species'),
        ('darwin', 'origin_of_species', 'published', '1859-11-24', 0.9, 'Origin of Species publication'),
        ('darwin', 'royal_society', 'elected_to', '1839-01-24', 0.8, 'Elected Fellow of Royal Society'),
        
        ('mendel', 'genetics', 'founded', '1865-02-08', 0.9, 'Mendel\'s laws of inheritance'),
        ('mendel', 'evolution', 'supported', '1865-02-08', 0.7, 'Genetics supports evolutionary theory'),
        
        ('pasteur', 'germ_theory', 'developed', '1862-01-01', 0.9, 'Pasteur developed germ theory'),
        ('pasteur', 'microscope', 'used', '1857-01-01', 0.8, 'Pasteur used microscopy'),
        
        # Early 20th Century Science
        ('curie', 'x_ray', 'discovered', '1895-11-08', 0.9, 'X-ray discovery'),
        ('curie', 'nobel_physics', 'received', '1903-12-10', 0.9, 'Nobel Prize in Physics'),
        ('curie', 'nobel_chemistry', 'received', '1911-12-10', 0.95, 'Nobel Prize in Chemistry'),
        ('curie', 'harvard', 'visited', '1921-05-01', 0.6, 'Curie visited Harvard'),
        
        ('einstein', 'princeton', 'worked_at', '1933-10-01', 0.9, 'Einstein at Princeton'),
        ('einstein', 'relativity', 'developed', '1905-06-30', 0.95, 'Special relativity'),
        ('einstein', 'quantum_mechanics', 'contributed_to', '1905-01-01', 0.8, 'Photoelectric effect'),
        ('einstein', 'nobel_physics', 'received', '1921-11-09', 0.9, 'Nobel Prize for photoelectric effect'),
        
        # Mid-20th Century Developments
        ('watson', 'cambridge', 'worked_at', '1951-10-01', 0.9, 'Watson at Cambridge'),
        ('watson', 'crick', 'collaborated_with', '1951-10-01', 0.9, 'Watson-Crick collaboration'),
        ('watson', 'dna', 'discovered_structure', '1953-04-25', 0.95, 'DNA double helix discovery'),
        ('crick', 'dna', 'discovered_structure', '1953-04-25', 0.95, 'DNA double helix discovery'),
        ('watson', 'nobel_medicine', 'received', '1962-10-18', 0.9, 'Nobel Prize for DNA structure'),
        ('crick', 'nobel_medicine', 'received', '1962-10-18', 0.9, 'Nobel Prize for DNA structure'),
        
        ('turing', 'cambridge', 'studied_at', '1931-10-01', 0.8, 'Turing at Cambridge'),
        ('turing', 'computer_science', 'founded', '1936-01-01', 0.9, 'Turing machine concept'),
        ('turing', 'computer', 'designed', '1945-01-01', 0.8, 'Early computer design'),
        ('turing', 'artificial_intelligence', 'conceptualised', '1950-01-01', 0.9, 'Turing test for AI'),
        
        ('hawking', 'cambridge', 'worked_at', '1962-10-01', 0.9, 'Hawking at Cambridge'),
        ('hawking', 'relativity', 'advanced', '1970-01-01', 0.8, 'Black hole physics'),
        ('hawking', 'quantum_mechanics', 'unified_with_gravity', '1974-01-01', 0.9, 'Hawking radiation'),
        
        # Institutional Connections
        ('cambridge', 'evolution', 'supported', '1860-01-01', 0.7, 'Cambridge supported evolutionary theory'),
        ('cambridge', 'genetics', 'researched', '1900-01-01', 0.8, 'Cambridge genetics research'),
        ('cambridge', 'computer_science', 'developed', '1930-01-01', 0.8, 'Cambridge computer science'),
        ('harvard', 'genetics', 'advanced', '1920-01-01', 0.8, 'Harvard genetics department'),
        ('mit', 'computer_science', 'pioneered', '1950-01-01', 0.9, 'MIT computer science'),
        ('mit', 'artificial_intelligence', 'developed', '1955-01-01', 0.9, 'MIT AI lab'),
        
        # Technological Evolution
        ('microscope', 'germ_theory', 'enabled', '1850-01-01', 0.8, 'Microscopy enabled germ theory'),
        ('x_ray', 'nobel_physics', 'led_to', '1901-01-01', 0.7, 'X-ray discovery Nobel Prize'),
        ('computer', 'artificial_intelligence', 'enabled', '1950-01-01', 0.9, 'Computers enabled AI'),
        ('computer', 'internet', 'led_to', '1960-01-01', 0.8, 'Computers led to internet'),
        
        # Scientific Field Evolution
        ('evolution', 'genetics', 'influenced', '1900-01-01', 0.8, 'Evolution influenced genetics'),
        ('genetics', 'dna', 'led_to', '1950-01-01', 0.9, 'Genetics led to DNA research'),
        ('quantum_mechanics', 'computer_science', 'influenced', '1940-01-01', 0.7, 'Quantum physics influenced computing'),
        ('computer_science', 'artificial_intelligence', 'spawned', '1950-01-01', 0.9, 'CS spawned AI'),
        
        # Historical Context
        ('industrial_revolution', 'microscope', 'enabled', '1830-01-01', 0.7, 'Industrial revolution enabled microscopy'),
        ('ww1', 'x_ray', 'advanced', '1915-01-01', 0.7, 'WWI advanced X-ray technology'),
        ('ww2', 'computer', 'accelerated', '1943-01-01', 0.8, 'WWII accelerated computing'),
        ('cold_war', 'space_race', 'caused', '1947-01-01', 0.9, 'Cold War led to space race'),
        ('space_race', 'computer_science', 'advanced', '1957-01-01', 0.8, 'Space race advanced computing'),
        
        # Publication and Knowledge Dissemination
        ('origin_of_species', 'evolution', 'documented', '1859-11-24', 0.9, 'Book documented evolution'),
        ('nature_journal', 'dna', 'published', '1953-04-25', 0.8, 'Nature published DNA structure'),
        ('science_journal', 'genetics', 'advanced', '1900-01-01', 0.7, 'Science journal advanced genetics'),
        
        # Cross-temporal Influences
        ('darwin', 'watson', 'influenced', '1920-01-01', 0.6, 'Darwin influenced later biologists'),
        ('curie', 'hawking', 'influenced', '1960-01-01', 0.5, 'Curie influenced later physicists'),
        ('einstein', 'turing', 'influenced', '1930-01-01', 0.6, 'Einstein influenced Turing'),
        ('mendel', 'watson', 'influenced', '1930-01-01', 0.7, 'Mendel influenced molecular biology'),
        
        # Modern Developments
        ('dna', 'artificial_intelligence', 'influences', '1980-01-01', 0.6, 'DNA research influences AI'),
        ('computer', 'genetics', 'revolutionised', '1970-01-01', 0.8, 'Computers revolutionised genetics'),
        ('internet', 'artificial_intelligence', 'enabled', '1990-01-01', 0.8, 'Internet enabled AI development')
    ]
    
    for source, target, relation, timestamp, weight, description in temporal_edges:
        graph.add_edge(source, target, 
                      relation_type=relation,
                      timestamp=timestamp,
                      weight=weight,
                      flow_capacity=weight,
                      description=description)
    
    return graph


def create_temporal_test_queries():
    """Create diverse temporal multi-hop sample queries"""
    queries = [
        # Historical Scientific Evolution (Long temporal spans)
        {
            'query_text': 'Trace the evolution from "darwin" to "dna" and "genetics"',
            'source_entities': ['darwin'],
            'target_entities': ['dna', 'genetics'],
            'temporal_constraints': {'temporal_preference': 'chronological'},
            'query_time': '1950-01-01',
            'max_hops': 4,
            'top_k': 10,
            'expected_temporal_pattern': 'progressive',
            'description': 'Tests long-term scientific evolution across multiple decades'
        },
        
        # Institution-based Scientific Development
        {
            'query_text': 'How did "cambridge" contribute to "computer_science" and "artificial_intelligence" development?',
            'source_entities': ['cambridge'],
            'target_entities': ['computer_science', 'artificial_intelligence'],
            'temporal_constraints': {'temporal_preference': 'institutional'},
            'query_time': '1960-01-01',
            'max_hops': 3,
            'top_k': 8,
            'expected_temporal_pattern': 'institutional_development',
            'description': 'Tests institutional influence on field development'
        },
        
        # Technology-driven Scientific Revolution
        {
            'query_text': 'Find connections between "microscope" and "dna", "watson", "crick"',
            'source_entities': ['microscope'],
            'target_entities': ['dna', 'watson', 'crick'],
            'temporal_constraints': {'temporal_preference': 'technological'},
            'query_time': '1955-01-01',
            'max_hops': 5,
            'top_k': 12,
            'expected_temporal_pattern': 'technological_enablement',
            'description': 'Tests how technology enables scientific breakthroughs'
        },
        
        # War-time Scientific Acceleration
        {
            'query_text': 'How did "ww2" accelerate "computer" and "artificial_intelligence" and "turing" development?',
            'source_entities': ['ww2'],
            'target_entities': ['computer', 'artificial_intelligence', 'turing'],
            'temporal_constraints': {'temporal_preference': 'wartime'},
            'query_time': '1945-01-01',
            'max_hops': 3,
            'top_k': 8,
            'expected_temporal_pattern': 'crisis_acceleration',
            'description': 'Tests how historical events accelerate scientific development'
        },
        
        # Multi-disciplinary Scientific Convergence
        {
            'query_text': 'Show how "einstein" and "curie" physics converged with "dna", "genetics", and "watson" biology',
            'source_entities': ['einstein', 'curie'],
            'target_entities': ['dna', 'genetics', 'watson'],
            'temporal_constraints': {'temporal_preference': 'interdisciplinary'},
            'query_time': '1960-01-01',
            'max_hops': 4,
            'top_k': 10,
            'expected_temporal_pattern': 'interdisciplinary_convergence',
            'description': 'Tests convergence of different scientific disciplines'
        },
        
        # Award and Recognition Networks
        {
            'query_text': 'Find paths from "curie" and "einstein" Nobel Prize winners to "watson" and "crick" laureates',
            'source_entities': ['curie', 'einstein'],
            'target_entities': ['watson', 'crick'],
            'temporal_constraints': {'temporal_preference': 'recognition'},
            'query_time': '1965-01-01',
            'max_hops': 3,
            'top_k': 8,
            'expected_temporal_pattern': 'recognition_influence',
            'description': 'Tests how early achievements influence later recognition'
        },
        
        # Publication and Knowledge Dissemination
        {
            'query_text': 'Trace how "origin_of_species" publishing evolved to "nature_journal" and "dna" genetics',
            'source_entities': ['origin_of_species'],
            'target_entities': ['nature_journal', 'dna'],
            'temporal_constraints': {'temporal_preference': 'publication'},
            'query_time': '1970-01-01',
            'max_hops': 4,
            'top_k': 10,
            'expected_temporal_pattern': 'knowledge_dissemination',
            'description': 'Tests evolution of scientific communication'
        },
        
        # Recent Historical Connections (Short temporal spans)
        {
            'query_text': 'Connect "dna" discovery and "watson" to "artificial_intelligence" and "computer" rise',
            'source_entities': ['dna', 'watson'],
            'target_entities': ['artificial_intelligence', 'computer'],
            'temporal_constraints': {'temporal_preference': 'recent'},
            'query_time': '1990-01-01',
            'max_hops': 3,
            'top_k': 8,
            'expected_temporal_pattern': 'recent_convergence',
            'description': 'Tests recent scientific convergence patterns'
        },
        
        # Cross-generational Scientific Influence
        {
            'query_text': 'How did "mendel" and "pasteur" 19th century science influence "hawking" and "turing" 20th century breakthroughs?',
            'source_entities': ['mendel', 'pasteur'],
            'target_entities': ['hawking', 'turing'],
            'temporal_constraints': {'temporal_preference': 'generational'},
            'query_time': '1980-01-01',
            'max_hops': 5,
            'top_k': 15,
            'expected_temporal_pattern': 'generational_influence',
            'description': 'Tests influence across different scientific generations'
        },
        
        # Complex Multi-hop Temporal Reasoning
        {
            'query_text': 'Find the longest scientifically coherent path from "evolution" to "artificial_intelligence"',
            'source_entities': ['evolution'],
            'target_entities': ['artificial_intelligence'],
            'temporal_constraints': {'temporal_preference': 'comprehensive'},
            'query_time': '2000-01-01',
            'max_hops': 6,
            'top_k': 20,
            'expected_temporal_pattern': 'comprehensive_evolution',
            'description': 'Tests complex multi-hop temporal reasoning'
        }
    ]
    
    return queries


def create_graph_statistics(graph):
    """Create graph statistics"""
    stats = {
        'entities': {},
        'relations': {},
        'temporal_distribution': {},
        'max_entity_frequency': 0,
        'max_relation_frequency': 0
    }
    
    # Calculate entity frequencies
    entity_frequencies = {}
    relation_frequencies = {}
    temporal_years = []
    
    for node in graph.nodes():
        entity_frequencies[node] = graph.degree(node)
    
    for _, _, data in graph.edges(data=True):
        relation_type = data.get('relation_type', 'unknown')
        relation_frequencies[relation_type] = relation_frequencies.get(relation_type, 0) + 1
        
        # Extract year from timestamp
        timestamp = data.get('timestamp', '1900-01-01')
        year = int(timestamp.split('-')[0])
        temporal_years.append(year)
    
    # Populate statistics
    stats['max_entity_frequency'] = max(entity_frequencies.values()) if entity_frequencies else 1
    stats['max_relation_frequency'] = max(relation_frequencies.values()) if relation_frequencies else 1
    
    for entity, freq in entity_frequencies.items():
        stats['entities'][entity] = {'frequency': freq}
    
    for relation, freq in relation_frequencies.items():
        stats['relations'][relation] = {'frequency': freq}
    
    # Temporal distribution
    if temporal_years:
        stats['temporal_distribution'] = {
            'min_year': min(temporal_years),
            'max_year': max(temporal_years),
            'span_years': max(temporal_years) - min(temporal_years),
            'average_year': np.mean(temporal_years)
        }
    
    return stats


def validate_temporal_relevance(path, query_time, temporal_window_days=365*50):
    """Validate that a path is temporally relevant"""
    temporal_info = path.get_temporal_info()
    
    if not temporal_info['timestamps']:
        return False, "No temporal information found"
    
    query_date = datetime.strptime(query_time, '%Y-%m-%d')
    path_timestamps = [datetime.strptime(ts, '%Y-%m-%d') for ts in temporal_info['timestamps']]
    
    # Check if all timestamps are within temporal window
    relevant_timestamps = []
    for ts in path_timestamps:
        days_diff = abs((query_date - ts).days)
        if days_diff <= temporal_window_days:
            relevant_timestamps.append(ts)
    
    if not relevant_timestamps:
        return False, f"No timestamps within {temporal_window_days} days of query time"
    
    # Check chronological coherence
    if len(path_timestamps) > 1:
        chronological = all(path_timestamps[i] <= path_timestamps[i+1] for i in range(len(path_timestamps)-1))
        if not chronological:
            return False, "Path not chronologically coherent"
    
    return True, f"Temporally relevant with {len(relevant_timestamps)} relevant timestamps"


def validate_path_non_redundancy(paths, similarity_threshold=0.7):
    """Validate that paths are non-redundant"""
    non_redundant_paths = []
    
    for i, (path1, score1) in enumerate(paths):
        is_redundant = False
        
        for j, (path2, score2) in enumerate(non_redundant_paths):
            # Calculate path similarity based on node overlap
            nodes1 = set(node.name for node in path1.nodes)
            nodes2 = set(node.name for node in path2.nodes)
            
            if nodes1 and nodes2:
                similarity = len(nodes1.intersection(nodes2)) / len(nodes1.union(nodes2))
                if similarity >= similarity_threshold:
                    is_redundant = True
                    break
        
        if not is_redundant:
            non_redundant_paths.append((path1, score1))
    
    return non_redundant_paths


def run_temporal_path_tests():
    """Run temporal path retrieval tests"""
    print("Temporal Path Retrieval Tests")
    print("=" * 80)
    
    # Create test TKG
    print("\n1. Creating test TKG")
    graph = create_test_tkg()
    graph_stats = create_graph_statistics(graph)
    
    print(f"TKG created with {len(graph.nodes())} nodes and {len(graph.edges())} edges")
    print(f"Temporal span: {graph_stats['temporal_distribution']['min_year']}-{graph_stats['temporal_distribution']['max_year']}")
    print(f"({graph_stats['temporal_distribution']['span_years']} years)")
    
    # Create test queries
    print("\n2. Creating temporal multi-hop sample queries")
    test_queries = create_temporal_test_queries()
    print(f"Created {len(test_queries)} diverse temporal queries")
    
    # Initialise query engine
    print("\n3. Initialising TKG query engine")
    query_engine = TKGQueryEngine(
        graph=graph,
        graph_statistics=graph_stats,
        alpha=0.01,  # Temporal decay rate (optimised)
        base_theta=0.1,  # Pruning threshold (optimised)
        reliability_threshold=0.4,  # Lowered from 0.6 to allow more paths
        diversity_threshold=0.7,
        use_updated_scoring=False  # Disable updated scoring for now
    )
    
    # Run tests
    print("\n4. Running temporal path retrieval tests")
    test_results = []
    
    for i, query_config in enumerate(test_queries, 1):
        print(f"\nTest {i}: {query_config['description']}")
        print(f"Query: {query_config['query_text']}")
        print(f"Source: {query_config['source_entities']}")
        print(f"Target: {query_config['target_entities']}")
        print(f"Query time: {query_config['query_time']}")
        
        # Execute query
        start_time = time.time()
        result = query_engine.query(
            query_text=query_config['query_text'],
            source_entities=query_config.get('source_entities'),
            target_entities=query_config.get('target_entities'),
            temporal_constraints=query_config.get('temporal_constraints'),
            query_time=query_config.get('query_time'),
            max_hops=query_config.get('max_hops'),
            top_k=query_config.get('top_k'),
            enable_flow_pruning=True,
            enable_reliability_filtering=True,  # Re-enabled with lower threshold
            enable_diversity=True,
            verbose=False
        )
        execution_time = time.time() - start_time
        
        # Validate results
        temporal_validation_results = []
        for path, metrics in result.paths:
            is_valid, reason = validate_temporal_relevance(path, query_config['query_time'])
            temporal_validation_results.append((path, metrics, is_valid, reason))
        
        # Check non-redundancy
        non_redundant_paths = validate_path_non_redundancy(result.paths)
        
        # Store results
        test_result = {
            'query_config': query_config,
            'execution_time': execution_time,
            'total_paths_discovered': result.total_paths_discovered,
            'total_paths_after_pruning': result.total_paths_after_pruning,
            'final_paths': len(result.paths),
            'temporally_valid_paths': sum(1 for _, _, valid, _ in temporal_validation_results if valid),
            'non_redundant_paths_count': len(non_redundant_paths),
            'average_reliability': np.mean([metrics.overall_reliability for _, metrics in result.paths]) if result.paths else 0.0,
            'temporal_validation_results': temporal_validation_results,
            'non_redundant_paths': non_redundant_paths
        }
        test_results.append(test_result)
        
        # Print results
        print(f"Execution time: {execution_time:.3f}s")
        print(f"Paths discovered: {result.total_paths_discovered}")
        print(f"Paths after pruning: {result.total_paths_after_pruning}")
        print(f"Final paths: {len(result.paths)}")
        print(f"Temporally valid: {test_result['temporally_valid_paths']}")
        print(f"Non-redundant: {test_result['non_redundant_paths_count']}")
        print(f"Average reliability: {test_result['average_reliability']:.3f}")
        
        if result.paths:
            print(f"Top 3 paths:")
            for j, (path, metrics) in enumerate(result.paths[:3], 1):
                path_str = " -> ".join([node.name for node in path.nodes])
                print(f"{j}. {path_str} (R: {metrics.overall_reliability:.3f})")
    
    return test_results, graph, graph_stats


def analyse_test_results(test_results):
    """Analyse and report test results"""
    print("\n5. Analysing test results")
    print("=" * 80)
    
    # Overall statistics
    total_queries = len(test_results)
    successful_queries = sum(1 for result in test_results if result['final_paths'] > 0)
    success_rate = successful_queries / total_queries if total_queries > 0 else 0
    
    avg_execution_time = np.mean([result['execution_time'] for result in test_results])
    avg_paths_discovered = np.mean([result['total_paths_discovered'] for result in test_results])
    avg_paths_after_pruning = np.mean([result['total_paths_after_pruning'] for result in test_results])
    avg_final_paths = np.mean([result['final_paths'] for result in test_results])
    avg_temporal_validity = np.mean([result['temporally_valid_paths'] / max(1, result['final_paths']) for result in test_results])
    avg_non_redundancy = np.mean([result['non_redundant_paths_count'] / max(1, result['final_paths']) for result in test_results])
    avg_reliability = np.mean([result['average_reliability'] for result in test_results if result['average_reliability'] > 0])
    
    print(f"Overall Performance:")
    print(f"Success rate: {success_rate:.1%} ({successful_queries}/{total_queries})")
    print(f"Average execution time: {avg_execution_time:.3f}s")
    print(f"Average paths discovered: {avg_paths_discovered:.1f}")
    print(f"Average paths after pruning: {avg_paths_after_pruning:.1f}")
    print(f"Average final paths: {avg_final_paths:.1f}")
    print(f"Average temporal validity: {avg_temporal_validity:.1%}")
    print(f"Average non-redundancy: {avg_non_redundancy:.1%}")
    print(f"Average reliability score: {avg_reliability:.3f}")
    
    # Query type analysis
    print(f"\nQuery Type Analysis:")
    query_types = {}
    for result in test_results:
        pattern = result['query_config']['expected_temporal_pattern']
        if pattern not in query_types:
            query_types[pattern] = []
        query_types[pattern].append(result)
    
    for pattern, results in query_types.items():
        avg_success = np.mean([1 if r['final_paths'] > 0 else 0 for r in results])
        avg_time = np.mean([r['execution_time'] for r in results])
        avg_reliability = np.mean([r['average_reliability'] for r in results if r['average_reliability'] > 0])
        print(f"{pattern}: {avg_success:.1%} success, {avg_time:.3f}s avg, {avg_reliability:.3f} reliability")
    
    # Performance insights
    print(f"\nPerformance Insights:")
    
    # Best performing queries
    best_queries = sorted(test_results, key=lambda x: x['average_reliability'], reverse=True)[:3]
    print(f"Top 3 performing queries:")
    for i, result in enumerate(best_queries, 1):
        print(f"{i}. {result['query_config']['description']} (R: {result['average_reliability']:.3f})")
    
    # Most challenging queries
    challenging_queries = [r for r in test_results if r['final_paths'] == 0]
    if challenging_queries:
        print(f"Challenging queries ({len(challenging_queries)}):")
        for result in challenging_queries:
            print(f" - {result['query_config']['description']}")
    
    # Temporal validation issues
    temporal_issues = []
    for result in test_results:
        invalid_paths = [tvr for tvr in result['temporal_validation_results'] if not tvr[2]]
        if invalid_paths:
            temporal_issues.append((result['query_config']['description'], len(invalid_paths)))
    
    if temporal_issues:
        print(f"Temporal validation issues:")
        for desc, count in temporal_issues:
            print(f" - {desc}: {count} invalid paths")
    
    return {
        'success_rate': success_rate,
        'avg_execution_time': avg_execution_time,
        'avg_temporal_validity': avg_temporal_validity,
        'avg_non_redundancy': avg_non_redundancy,
        'avg_reliability': avg_reliability,
        'query_type_analysis': query_types,
        'best_queries': best_queries,
        'challenging_queries': challenging_queries,
        'temporal_issues': temporal_issues
    }


def debug_temporal_weighting_and_pruning(graph, graph_stats):
    """Debug and refine temporal weighting and pruning logic"""
    print("\n6. Debugging temporal weighting and pruning logic")
    print("=" * 80)
    
    # Test different alpha values
    alpha_values = [0.05, 0.1, 0.15, 0.2, 0.3]
    theta_values = [0.5, 1.0, 1.5, 2.0]
    
    test_query = TemporalQuery(
        query_text="Test query for parameter tuning",
        source_entities=['darwin'],
        target_entities=['dna'],
        temporal_constraints={'temporal_preference': 'chronological'},
        query_time='1950-01-01',
        max_hops=4,
        top_k=10
    )
    
    print(f"Testing parameter combinations:")
    print(f"Alpha values: {alpha_values}")
    print(f"Theta values: {theta_values}")
    
    parameter_results = []
    
    for alpha in alpha_values:
        for theta in theta_values:
            print(f"\nTesting alpha={alpha}, theta={theta}")
            
            # Create retriever with specific parameters
            retriever = TemporalPathRetriever(
                graph=graph,
                alpha=alpha,
                base_theta=theta,
                diversity_threshold=0.7
            )
            
            start_time = time.time()
            try:
                paths = retriever.retrieve_temporal_paths(
                    query=test_query,
                    enable_flow_pruning=True,
                    enable_diversity=True,
                    verbose=False
                )
                execution_time = time.time() - start_time
                
                # Calculate performance metrics
                avg_reliability = np.mean([score for _, score in paths]) if paths else 0.0
                
                parameter_results.append({
                    'alpha': alpha,
                    'theta': theta,
                    'execution_time': execution_time,
                    'paths_found': len(paths),
                    'avg_reliability': avg_reliability,
                    'success': len(paths) > 0
                })
                
                print(f"Paths found: {len(paths)}, Avg reliability: {avg_reliability:.3f}, Time: {execution_time:.3f}s")
                
            except Exception as e:
                print(f"Error: {e}")
                parameter_results.append({
                    'alpha': alpha,
                    'theta': theta,
                    'execution_time': 0.0,
                    'paths_found': 0,
                    'avg_reliability': 0.0,
                    'success': False,
                    'error': str(e)
                })
    
    # Analyse parameter effects
    print(f"\nParameter Analysis:")
    
    # Best parameter combinations
    successful_results = [r for r in parameter_results if r['success']]
    if successful_results:
        best_by_reliability = max(successful_results, key=lambda x: x['avg_reliability'])
        best_by_speed = min(successful_results, key=lambda x: x['execution_time'])
        best_by_paths = max(successful_results, key=lambda x: x['paths_found'])
        
        print(f"Best reliability: alpha={best_by_reliability['alpha']}, theta={best_by_reliability['theta']} (R: {best_by_reliability['avg_reliability']:.3f})")
        print(f"Fastest execution: alpha={best_by_speed['alpha']}, theta={best_by_speed['theta']} (T: {best_by_speed['execution_time']:.3f}s)")
        print(f"Most paths found: alpha={best_by_paths['alpha']}, theta={best_by_paths['theta']} (P: {best_by_paths['paths_found']})")
    
    # Alpha effect analysis
    alpha_effects = {}
    for alpha in alpha_values:
        alpha_results = [r for r in parameter_results if r['alpha'] == alpha and r['success']]
        if alpha_results:
            alpha_effects[alpha] = {
                'avg_reliability': np.mean([r['avg_reliability'] for r in alpha_results]),
                'avg_paths': np.mean([r['paths_found'] for r in alpha_results]),
                'avg_time': np.mean([r['execution_time'] for r in alpha_results])
            }
    
    print(f"Alpha effects:")
    for alpha, effects in alpha_effects.items():
        print(f"Alpha={alpha}: R={effects['avg_reliability']:.3f}, P={effects['avg_paths']:.1f}, T={effects['avg_time']:.3f}s")
    
    # Theta effect analysis
    theta_effects = {}
    for theta in theta_values:
        theta_results = [r for r in parameter_results if r['theta'] == theta and r['success']]
        if theta_results:
            theta_effects[theta] = {
                'avg_reliability': np.mean([r['avg_reliability'] for r in theta_results]),
                'avg_paths': np.mean([r['paths_found'] for r in theta_results]),
                'avg_time': np.mean([r['execution_time'] for r in theta_results])
            }
    
    print(f"Theta effects:")
    for theta, effects in theta_effects.items():
        print(f"Theta={theta}: R={effects['avg_reliability']:.3f}, P={effects['avg_paths']:.1f}, T={effects['avg_time']:.3f}s")
    
    return parameter_results, alpha_effects, theta_effects


def generate_test_report(test_results, analysis_results, debug_results):
    """Generate test report"""
    print("\n7. Generating test report")
    print("=" * 80)
    
    report = {
        'test_summary': {
            'total_queries': len(test_results),
            'success_rate': analysis_results['success_rate'],
            'avg_execution_time': analysis_results['avg_execution_time'],
            'avg_temporal_validity': analysis_results['avg_temporal_validity'],
            'avg_non_redundancy': analysis_results['avg_non_redundancy'],
            'avg_reliability': analysis_results['avg_reliability']
        },
        'query_type_performance': analysis_results['query_type_analysis'],
        'best_performing_queries': analysis_results['best_queries'],
        'challenging_queries': analysis_results['challenging_queries'],
        'temporal_issues': analysis_results['temporal_issues'],
        'parameter_optimisation': debug_results
    }
    
    # Save report to file
    report_file = '/Users/hillcallum/Temporal_PathRAG/test_results/temporal_path_retrieval_report.json'
    os.makedirs(os.path.dirname(report_file), exist_ok=True)
    
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"Test report saved to: {report_file}")
    
    # Print summary
    print(f"\nTest Summary:")
    print(f"Total queries tested: {report['test_summary']['total_queries']}")
    print(f"Success rate: {report['test_summary']['success_rate']:.1%}")
    print(f"Average execution time: {report['test_summary']['avg_execution_time']:.3f}s")
    print(f"Temporal validity: {report['test_summary']['avg_temporal_validity']:.1%}")
    print(f"Non-redundancy: {report['test_summary']['avg_non_redundancy']:.1%}")
    print(f"Average reliability: {report['test_summary']['avg_reliability']:.3f}")
    
    return report


def main():
    """Main function to run all tests"""
    print("Starting Temporal Path Retrieval Tests")
    print("=" * 80)
    
    try:
        # Create test results directory
        os.makedirs('/Users/hillcallum/Temporal_PathRAG/test_results', exist_ok=True)
        
        # Run temporal path tests
        test_results, graph, graph_stats = run_temporal_path_tests()
        
        # Analyse results
        analysis_results = analyse_test_results(test_results)
        
        # Debug and refine parameters
        debug_results = debug_temporal_weighting_and_pruning(graph, graph_stats)
        
        # Generate report
        report = generate_test_report(test_results, analysis_results, debug_results)
        
        print("\n" + "=" * 80)
        print("Temporal Path Retrieval Tests Completed")
        print("=" * 80)
        
        return 0
        
    except Exception as e:
        print(f"\nTest suite failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())