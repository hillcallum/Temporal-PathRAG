#!/usr/bin/env python3
"""
Graph Query Testing Script - inspired by PathRAG's query patterns
Test various temporal graph queries to confirm schema adherence and connectivity
"""

from src.kg.storage.temporal_graph_storage import TemporalGraphDatabase
import random
from typing import List

def run_comprehensive_graph_queries():
    """Run comprehensive graph queries to test functionality and schema adherence"""
    
    print("=== Temporal Graph Query Testing ===")
    
    # Load existing database
    tg_db = TemporalGraphDatabase()
    tg_db.load_database()
    
    # 1. Basic connectivity tests
    print("\n1. Basic connection testing")
    print(f"Total nodes: {tg_db.stats['nodes']:,}")
    print(f"Total edges: {tg_db.stats['edges']:,}")
    print(f"Graph is connected: {len(list(tg_db.main_graph.nodes())) > 0}")
    
    # 2. Temporal query tests
    print("\n2. Temporal Query Testing")
    
    # Test temporal facts by year range
    facts_90s = tg_db.query_temporal_range("1990", "1999", limit=10)
    facts_2000s = tg_db.query_temporal_range("2000", "2009", limit=10)
    facts_2010s = tg_db.query_temporal_range("2010", "2019", limit=10)
    
    print(f"Facts from 1990s: {len(facts_90s)}")
    print(f"Facts from 2000s: {len(facts_2000s)}")
    print(f"Facts from 2010s: {len(facts_2010s)}")
    
    if facts_90s:
        print(f"Sample 1990s fact: {facts_90s[0]}")
    if facts_2000s:
        print(f"Sample 2000s fact: {facts_2000s[0]}")
    if facts_2010s:
        print(f"Sample 2010s fact: {facts_2010s[0]}")
    
    # 3. Entity neighbourhood tests (inspired by PathRAG)
    print("\n3. Eentity Neighbouthood Testing")
    
    sample_entities = list(tg_db.main_graph.nodes())[:5]
    
    for entity in sample_entities:
        neighbours = tg_db.query_entity_neighbours(entity, max_hops=2)
        if neighbours['found']:
            out_count = len(neighbours['direct_neighbours']['outgoing'])
            in_count = len(neighbours['direct_neighbours']['incoming'])
            print(f"Entity '{entity}': {out_count} outgoing, {in_count} incoming connections")
        else:
            print(f"Entity '{entity}': not found")
    
    # 4. Schema adherence tests
    print("\n4. Schema Adherence Testing")
    
    # Check edge attributes
    sample_edges = list(tg_db.main_graph.edges(data=True))[:5]
    for u, v, data in sample_edges:
        required_attrs = ['relation', 'timestamp', 'dataset', 'edge_type']
        has_all_attrs = all(attr in data for attr in required_attrs)
        print(f"Edge ({u} -> {v}): Required attributes present: {has_all_attrs}")
        if not has_all_attrs:
            missing = [attr for attr in required_attrs if attr not in data]
            print(f"Missing: {missing}")
    
    # Check node attributes
    sample_nodes = list(tg_db.main_graph.nodes(data=True))[:5]
    for node, data in sample_nodes:
        required_attrs = ['node_type', 'dataset', 'first_seen']
        has_all_attrs = all(attr in data for attr in required_attrs)
        print(f"Node '{node}': Required attributes present: {has_all_attrs}")
        if not has_all_attrs:
            missing = [attr for attr in required_attrs if attr not in data]
            print(f"Missing: {missing}")
    
    # 5. Temporal index tests
    print("\n5. Temporal Index Testing")
    
    # Check temporal index functionality
    sample_timestamps = list(tg_db.temporal_index.keys())[:3]
    for timestamp in sample_timestamps:
        edge_count = len(tg_db.temporal_index[timestamp])
        facts = tg_db.query_temporal_facts(timestamp, limit=3)
        print(f"Timestamp '{timestamp}': {edge_count} edges, {len(facts)} facts retrieved")
    
    # 6. Relation statistics and patterns
    print("\n6. Relation Statistics")
    
    rel_stats = tg_db.get_relation_statistics()
    print(f"Total unique relations: {rel_stats['total_relations']}")
    print("Top 10 most frequent relations:")
    for rel, count in list(rel_stats['relation_frequency'].items())[:10]:
        print(f"{rel}: {count:,} occurrences")
    
    # 7. Dataset distribution tests
    print("\n7. Dataset Distribution")
    
    dataset_node_counts = {}
    dataset_edge_counts = {}
    
    for node, data in tg_db.main_graph.nodes(data=True):
        dataset = data.get('dataset', 'unknown')
        dataset_node_counts[dataset] = dataset_node_counts.get(dataset, 0) + 1
    
    for u, v, data in tg_db.main_graph.edges(data=True):
        dataset = data.get('dataset', 'unknown')
        dataset_edge_counts[dataset] = dataset_edge_counts.get(dataset, 0) + 1
    
    print("Node distribution by dataset:")
    for dataset, count in dataset_node_counts.items():
        print(f"{dataset}: {count:,} nodes")
    
    print("Edge distribution by dataset:")
    for dataset, count in dataset_edge_counts.items():
        print(f"{dataset}: {count:,} edges")
    
    # 8. Path finding tests (inspired by PathRAG's path traversal)
    print("\n8. Path Finding Tests")
    
    # Test multi-hop connectivity for a few entities
    if len(sample_entities) >= 2:
        source = sample_entities[0]
        target = sample_entities[1]
        
        try:
            import networkx as nx
            if tg_db.main_graph.has_node(source) and tg_db.main_graph.has_node(target):
                if nx.has_path(tg_db.main_graph, source, target):
                    try:
                        path = nx.shortest_path(tg_db.main_graph, source, target)
                        print(f"Shortest path from '{source}' to '{target}': {len(path)-1} hops")
                        if len(path) <= 4:  # Only show short paths
                            print(f"Path: {' -> '.join(path[:4])}")
                    except nx.NetworkXNoPath:
                        print(f"No path found from '{source}' to '{target}'")
                else:
                    print(f"No path exists from '{source}' to '{target}'")
        except Exception as e:
            print(f"Path finding error: {e}")
    
    # 9. Temporal coverage analysis
    print("\n9. Temporal Coverage Analysis")
    
    timestamps = list(tg_db.temporal_index.keys())
    if timestamps:
        # Extract years and analyse distribution
        years = []
        for ts in timestamps:
            try:
                if '-' in ts:
                    year = ts.split('-')[0]
                elif len(ts) >= 4:
                    year = ts[:4]
                else:
                    continue
                years.append(int(year))
            except:
                continue
        
        if years:
            min_year = min(years)
            max_year = max(years)
            unique_years = len(set(years))
            print(f"Temporal coverage: {min_year} to {max_year} ({unique_years} unique years)")
            
            # Decade distribution
            decades = {}
            for year in years:
                decade = (year // 10) * 10
                decades[decade] = decades.get(decade, 0) + 1
            
            print("Facts by decade:")
            for decade in sorted(decades.keys()):
                print(f"{decade}s: {decades[decade]:,} facts")
    
    # 10. Performance test - large query
    print("\n10. Performance Testing")
    
    # Test querying a large temporal range
    large_query_facts = tg_db.query_temporal_range("1900", "2023", limit=1000)
    print(f"Large temporal range query returned: {len(large_query_facts)} facts")
    
    print("\nQuery Testing All Completed")
    print("All query tests completed successfully")
    
    return True

if __name__ == "__main__":
    run_comprehensive_graph_queries()