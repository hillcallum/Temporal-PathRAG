#!/usr/bin/env python3
"""
Temporal Graph Database Implementation inspired by PathRAG's NetworkX-based storage 
Handles (S,P,O,T) quadruplets with temporal indexing
"""

import json
import networkx as nx
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional, Any, Union
from collections import defaultdict, Counter
import pandas as pd
import pickle
from datetime import datetime
import re

class TemporalGraphDatabase:
    """NetworkX-based temporal knowledge graph database"""
    
    def __init__(self, db_path: str = "analysis_results/temporal_graph_db"):
        self.db_path = Path(db_path)
        self.db_path.mkdir(exist_ok=True)
        
        # Initialise NetworkX graphs
        self.main_graph = nx.MultiDiGraph()  # Primary temporal graph
        self.entity_graph = nx.Graph()       # Entity relationship graph
        self.temporal_index = defaultdict(set)  # Time -> set of edge IDs
        
        # Metadata tracking
        self.metadata = {
            "created": datetime.now().isoformat(),
            "datasets": [],
            "total_quadruplets": 0,
            "unique_entities": 0,
            "unique_relations": 0,
            "unique_timestamps": 0
        }
        
        # Statistics for verification
        self.stats = {
            "nodes": 0,
            "edges": 0,
            "temporal_edges": 0,
            "datasets_loaded": []
        }
        
        print(f"Temporal Graph Database initialised at {self.db_path}")
    
    def load_temporal_quadruplets(self, quadruplets_file: str, dataset_name: str):
        """Load (S,P,O,T) quadruplets into the graph database"""
        print(f"\nLoading quadruplets from {quadruplets_file} for {dataset_name}")
        
        if not Path(quadruplets_file).exists():
            raise FileNotFoundError(f"Quadruplets file not found: {quadruplets_file}")
        
        with open(quadruplets_file, 'r') as f:
            quadruplets = json.load(f)
        
        loaded_count = 0
        skipped_count = 0
        
        for quad in quadruplets:
            subject, predicate, obj, timestamp = quad
            
            # Add nodes if they don't exist
            if not self.main_graph.has_node(subject):
                self.main_graph.add_node(subject, 
                                       node_type="entity",
                                       dataset=dataset_name,
                                       first_seen=timestamp)
            
            if not self.main_graph.has_node(obj):
                self.main_graph.add_node(obj,
                                       node_type="entity", 
                                       dataset=dataset_name,
                                       first_seen=timestamp)
            
            # Create unique edge key for temporal indexing
            edge_key = f"{subject}-{predicate}-{obj}-{timestamp}"
            
            # Add temporal edge with comprehensive attributes
            self.main_graph.add_edge(subject, obj,
                                   key=edge_key,
                                   relation=predicate,
                                   timestamp=timestamp,
                                   dataset=dataset_name,
                                   edge_type="temporal")
            
            # Add to temporal index for efficient time-based queries
            self.temporal_index[timestamp].add(edge_key)
            
            # Add to entity relationship graph (non-temporal)
            if not self.entity_graph.has_edge(subject, obj):
                self.entity_graph.add_edge(subject, obj, 
                                         relations=set([predicate]),
                                         datasets=set([dataset_name]))
            else:
                self.entity_graph[subject][obj]['relations'].add(predicate)
                self.entity_graph[subject][obj]['datasets'].add(dataset_name)
            
            loaded_count += 1
            
            if loaded_count % 50000 == 0:
                print(f"Loaded {loaded_count:,} quadruplets")
        
        # Update metadata
        self.metadata["datasets"].append(dataset_name)
        self.metadata["total_quadruplets"] += loaded_count
        self.stats["datasets_loaded"].append(dataset_name)
        
        print(f"Successfully loaded {loaded_count:,} quadruplets from {dataset_name}")
        if skipped_count > 0:
            print(f"Skipped {skipped_count:,} invalid quadruplets")
        
        return loaded_count
    
    def load_dataset_from_structure(self, dataset_path: str):
        """Load dataset from directory structure (MultiTQ/TimeQuestions format)"""
        dataset_path = Path(dataset_path)
        
        # Look for knowledge graph file in typical locations
        possible_kg_files = [
            dataset_path / "kg" / "full.txt",
            dataset_path / "full.txt", 
            dataset_path / "kg" / "temporal_kg.txt"
        ]
        
        kg_file = None
        for path in possible_kg_files:
            if path.exists():
                kg_file = path
                break
                
        if not kg_file:
            raise FileNotFoundError(f"No knowledge graph file found in {dataset_path}")
            
        print(f"Loading temporal KG from {kg_file}")
        
        # Load quadruplets directly without temporary file to avoid disk space issues
        dataset_name = dataset_path.name
        
        print(f"Loading quadruplets from {kg_file} for {dataset_name}")
        
        loaded_count = 0
        skipped_count = 0
        
        with open(kg_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                    
                parts = line.split('\t')
                if len(parts) >= 4:
                    subject, predicate, obj, timestamp = parts[:4]
                elif len(parts) == 3:
                    # Handle quadruplets with timestamp
                    subject, predicate, obj = parts
                    timestamp = "unknown"
                else:
                    skipped_count += 1
                    continue
                
                # Add nodes if they don't exist
                if not self.main_graph.has_node(subject):
                    self.main_graph.add_node(subject, 
                                           node_type="entity",
                                           dataset=dataset_name,
                                           first_seen=timestamp)
                
                if not self.main_graph.has_node(obj):
                    self.main_graph.add_node(obj,
                                           node_type="entity", 
                                           dataset=dataset_name,
                                           first_seen=timestamp)
                
                # Create unique edge key for temporal indexing
                edge_key = f"{subject}-{predicate}-{obj}-{timestamp}"
                
                # Add temporal edge with comprehensive attributes
                self.main_graph.add_edge(subject, obj,
                                       key=edge_key,
                                       relation=predicate,
                                       timestamp=timestamp,
                                       dataset=dataset_name,
                                       edge_type="temporal")
                
                # Add to temporal index for efficient time-based queries
                self.temporal_index[timestamp].add(edge_key)
                
                # Add to entity relationship graph (non-temporal)
                if not self.entity_graph.has_edge(subject, obj):
                    self.entity_graph.add_edge(subject, obj, 
                                             relations=set([predicate]),
                                             datasets=set([dataset_name]))
                else:
                    self.entity_graph[subject][obj]['relations'].add(predicate)
                    self.entity_graph[subject][obj]['datasets'].add(dataset_name)
                
                loaded_count += 1
                
                if loaded_count % 50000 == 0:
                    print(f"Loaded {loaded_count:,} quadruplets")
        
        # Update metadata
        self.metadata["datasets"].append(dataset_name)
        self.metadata["total_quadruplets"] += loaded_count
        self.stats["datasets_loaded"].append(dataset_name)
        
        print(f"Successfully loaded {loaded_count:,} quadruplets from {dataset_name}")
        if skipped_count > 0:
            print(f"Skipped {skipped_count:,} invalid quadruplets")
        
        return loaded_count
    
    def update_statistics(self):
        """Update graph statistics for verification"""
        self.stats["nodes"] = self.main_graph.number_of_nodes()
        self.stats["edges"] = self.main_graph.number_of_edges()
        self.stats["temporal_edges"] = len([(u, v) for u, v, d in self.main_graph.edges(data=True) 
                                          if d.get('edge_type') == 'temporal'])
        
        self.metadata["unique_entities"] = self.stats["nodes"]
        self.metadata["unique_relations"] = len(set([d['relation'] for u, v, d in self.main_graph.edges(data=True)]))
        self.metadata["unique_timestamps"] = len(self.temporal_index)
        
        print(f"\nGraph Statistics:")
        print(f"Nodes: {self.stats['nodes']:,}")
        print(f"Edges: {self.stats['edges']:,}")
        print(f"Temporal edges: {self.stats['temporal_edges']:,}")
        print(f"Unique timestamps: {self.metadata['unique_timestamps']:,}")
        print(f"Unique relations: {self.metadata['unique_relations']:,}")
    
    def query_temporal_facts(self, timestamp: str, limit: int = 10) -> List[Tuple]:
        """Query facts by timestamp - inspired by PathRAG's query patterns"""
        if timestamp not in self.temporal_index:
            return []
        
        edge_keys = list(self.temporal_index[timestamp])[:limit]
        facts = []
        
        for edge_key in edge_keys:
            # Parse edge key to get components
            parts = edge_key.rsplit('-', 1)  # Split on last dash to preserve timestamp
            if len(parts) == 2:
                spo_part, ts_part = parts
                spo_components = spo_part.split('-')
                if len(spo_components) >= 3:
                    subject = '-'.join(spo_components[:-2])
                    predicate = spo_components[-2]
                    obj = spo_components[-1]
                    facts.append((subject, predicate, obj, timestamp))
        
        return facts
    
    def query_entity_neighbours(self, entity: str, max_hops: int = 2) -> Dict[str, Any]:
        """Query entity neighbourhood - similar to PathRAG's path finding"""
        if entity not in self.main_graph:
            return {"entity": entity, "found": False}
        
        # Get direct neighbours
        neighbours = {
            "outgoing": list(self.main_graph.successors(entity)),
            "incoming": list(self.main_graph.predecessors(entity))
        }
        
        # Get relations (handle MultiDiGraph edge access)
        relations = {
            "outgoing": [],
            "incoming": []
        }
        
        for neighbour in neighbours["outgoing"]:
            for key, data in self.main_graph[entity][neighbour].items():
                relations["outgoing"].append(data['relation'])
                break  # Take first relation for each neighbour
        
        for pred in neighbours["incoming"]:
            for key, data in self.main_graph[pred][entity].items():
                relations["incoming"].append(data['relation'])
                break  # Take first relation for each neighbour
        
        # Multi-hop traversal
        if max_hops > 1:
            two_hop = {
                "outgoing": [],
                "incoming": []
            }
            
            for neighbour in neighbours["outgoing"][:5]:  # Limit to prevent explosion
                two_hop["outgoing"].extend(list(self.main_graph.successors(neighbour))[:3])
            
            for neighbour in neighbours["incoming"][:5]:
                two_hop["incoming"].extend(list(self.main_graph.predecessors(neighbour))[:3])
        
        return {
            "entity": entity,
            "found": True,
            "direct_neighbours": neighbours,
            "relations": relations,
            "two_hop": two_hop if max_hops > 1 else None
        }
    
    def query_temporal_range(self, start_year: str, end_year: str, limit: int = 100) -> List[Tuple]:
        """Query facts within temporal range"""
        facts = []
        
        for timestamp in self.temporal_index:
            # Extract year from timestamp
            year_match = re.match(r'^(\d{4})', timestamp)
            if year_match:
                year = year_match.group(1)
                if start_year <= year <= end_year:
                    facts.extend(self.query_temporal_facts(timestamp, limit=10))
            
            if len(facts) >= limit:
                break
        
        return facts[:limit]
    
    def get_relation_statistics(self) -> Dict[str, Any]:
        """Get comprehensive relation statistics"""
        relation_counts = Counter()
        temporal_relations = defaultdict(set)
        
        for u, v, data in self.main_graph.edges(data=True):
            relation = data.get('relation', 'unknown')
            timestamp = data.get('timestamp', 'unknown')
            
            relation_counts[relation] += 1
            temporal_relations[relation].add(timestamp)
        
        return {
            "total_relations": len(relation_counts),
            "relation_frequency": dict(relation_counts.most_common(20)),
            "temporal_coverage": {rel: len(timestamps) 
                                for rel, timestamps in temporal_relations.items()}
        }
    
    def verify_data_integrity(self, expected_counts: Dict[str, int]) -> Dict[str, bool]:
        """Verify loaded data matches expected counts"""
        verification = {}
        
        if "quadruplets" in expected_counts:
            actual_edges = self.stats["temporal_edges"]
            expected_edges = expected_counts["quadruplets"]
            verification["quadruplets_match"] = actual_edges == expected_edges
            print(f"Quadruplets: {actual_edges:,} loaded, {expected_edges:,} expected")
        
        if "entities" in expected_counts:
            actual_nodes = self.stats["nodes"]
            expected_nodes = expected_counts["entities"]
            verification["entities_match"] = actual_nodes == expected_nodes
            print(f"Entities: {actual_nodes:,} loaded, {expected_nodes:,} expected")
        
        if "timestamps" in expected_counts:
            actual_timestamps = len(self.temporal_index)
            expected_timestamps = expected_counts["timestamps"]
            verification["timestamps_match"] = actual_timestamps == expected_timestamps
            print(f"Timestamps: {actual_timestamps:,} loaded, {expected_timestamps:,} expected")
        
        return verification
    
    def save_database(self):
        """Save graph database to disk"""
        print(f"\nSaving temporal graph database to {self.db_path}")
        
        # Save main graph
        main_graph_file = self.db_path / "main_graph.pkl"
        with open(main_graph_file, 'wb') as f:
            pickle.dump(self.main_graph, f)
        
        # Save entity graph
        entity_graph_file = self.db_path / "entity_graph.pkl"
        with open(entity_graph_file, 'wb') as f:
            pickle.dump(self.entity_graph, f)
        
        # Save temporal index
        temporal_index_file = self.db_path / "temporal_index.pkl"
        with open(temporal_index_file, 'wb') as f:
            pickle.dump(dict(self.temporal_index), f)
        
        # Save metadata
        metadata_file = self.db_path / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        # Save statistics
        stats_file = self.db_path / "statistics.json"
        with open(stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        print(f"Database saved successfully")
    
    def load_database(self):
        """Load existing graph database from disk"""
        print(f"Loading temporal graph database from {self.db_path}")
        
        try:
            # Load main graph
            main_graph_file = self.db_path / "main_graph.pkl"
            if main_graph_file.exists():
                with open(main_graph_file, 'rb') as f:
                    self.main_graph = pickle.load(f)
            
            # Load entity graph
            entity_graph_file = self.db_path / "entity_graph.pkl"
            if entity_graph_file.exists():
                with open(entity_graph_file, 'rb') as f:
                    self.entity_graph = pickle.load(f)
            
            # Load temporal index
            temporal_index_file = self.db_path / "temporal_index.pkl"
            if temporal_index_file.exists():
                with open(temporal_index_file, 'rb') as f:
                    self.temporal_index = defaultdict(set, pickle.load(f))
            
            # Load metadata
            metadata_file = self.db_path / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    self.metadata = json.load(f)
            
            # Load statistics
            stats_file = self.db_path / "statistics.json"
            if stats_file.exists():
                with open(stats_file, 'r') as f:
                    self.stats = json.load(f)
            
            print(f"Database loaded successfully")
            self.update_statistics()
            
        except Exception as e:
            print(f"Error loading database: {e}")
            print("Starting with fresh database")
    
    def get_main_graph(self):
        """Return the main temporal graph"""
        return self.main_graph
    
    def get_statistics(self):
        """Return current statistics"""
        # Update statistics to include metadata
        stats = self.stats.copy()
        stats.update({
            "total_quadruplets": self.metadata.get("total_quadruplets", 0),
            "unique_entities": self.metadata.get("unique_entities", 0),
            "unique_relations": self.metadata.get("unique_relations", 0),
            "unique_timestamps": self.metadata.get("unique_timestamps", 0)
        })
        return stats


def main():
    """Main execution - load quadruplets and create temporal graph database"""
    
    # Initialise database
    tg_db = TemporalGraphDatabase()
    
    # Expected data from dataset_parser.py results
    expected_counts = {
        "MultiTQ": {"quadruplets": 461329, "entities": 77943, "timestamps": 3893},
        "TimeQuestions": {"quadruplets": 240597, "entities": 86222, "timestamps": 2877}
    }
    
    # Load quadruplets from temporal analysis results
    datasets = ["MultiTQ", "TimeQuestions"]
    
    for dataset in datasets:
        quadruplets_file = f"analysis_results/temporal_kg_analysis/{dataset}_quadruplets.json"
        
        if Path(quadruplets_file).exists():
            print(f"Loading {dataset} dataset")
            
            tg_db.load_temporal_quadruplets(quadruplets_file, dataset)
            tg_db.update_statistics()
            
            # Verify data integrity
            if dataset in expected_counts:
                verification = tg_db.verify_data_integrity(expected_counts[dataset])
                print(f"Verification results: {verification}")
        else:
            print(f"Quadruplets file not found: {quadruplets_file}")
            print("Please run dataset_parser.py first to generate quadruplets")
    
    # Final statistics and sample queries
    print("Final Temporal Graph Database Statistics")
    tg_db.update_statistics()
    
    # Sample queries to demonstrate functionality
    print("Sample Temporal Queries")
    
    # Query facts from a specific year
    sample_facts = tg_db.query_temporal_range("1990", "1995", limit=5)
    print(f"\nSample facts from 1990-1995:")
    for fact in sample_facts[:3]:
        print(f"{fact}")
    
    # Query entity neighbourhood
    if tg_db.stats["nodes"] > 0:
        # Get a sample entity
        sample_entity = list(tg_db.main_graph.nodes())[0]
        neighbours = tg_db.query_entity_neighbours(sample_entity, max_hops=2)
        print(f"\nSample entity '{sample_entity}' neighbours:")
        print(f"Outgoing: {neighbours['direct_neighbours']['outgoing'][:3]}")
        print(f"Incoming: {neighbours['direct_neighbours']['incoming'][:3]}")
    
    # Relation statistics
    rel_stats = tg_db.get_relation_statistics()
    print(f"\nTop 5 most frequent relations:")
    for rel, count in list(rel_stats["relation_frequency"].items())[:5]:
        print(f"{rel}: {count:,} occurrences")
    
    # Save database
    tg_db.save_database()
    
    print("Temporal Graph Database Creation Complete")
    print(f"Database saved to: {tg_db.db_path}")
    print(f"Total nodes: {tg_db.stats['nodes']:,}")
    print(f"Total edges: {tg_db.stats['edges']:,}")
    print(f"Temporal coverage: {len(tg_db.temporal_index):,} unique timestamps")


if __name__ == "__main__":
    main()