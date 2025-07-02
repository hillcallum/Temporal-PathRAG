#!/usr/bin/env python3
"""
Data Count Verification Script
Verifoes that the temporal graph database has loaded the expected number of entities and quadruplets
"""

import json
from pathlib import Path
from src.kg.temporal_graph_storage import TemporalGraphDatabase

def verify_data_counts():
    """Compare actual vs expected data counts"""
    
    # Load existing database
    tg_db = TemporalGraphDatabase()
    tg_db.load_database()
    
    # Expected counts from dataset analysis
    expected_counts = {
        "MultiTQ": {"quadruplets": 461329, "entities": 77943, "timestamps": 3893},
        "TimeQuestions": {"quadruplets": 240597, "entities": 86222, "timestamps": 2877}
    }
    
    # Get actual counts
    actual_counts = {
        "total_quadruplets": tg_db.stats["temporal_edges"],
        "total_nodes": tg_db.stats["nodes"],
        "total_timestamps": len(tg_db.temporal_index),
        "total_relations": tg_db.metadata["unique_relations"]
    }
    
    # Calculate expected totals
    expected_total_quadruplets = sum(d["quadruplets"] for d in expected_counts.values())
    expected_total_entities = sum(d["entities"] for d in expected_counts.values())
    expected_total_timestamps = sum(d["timestamps"] for d in expected_counts.values())
    
    print("=== Data Count Verification ===")
    print(f"Expected total quadruplets: {expected_total_quadruplets:,}")
    print(f"Actual total quadruplets: {actual_counts['total_quadruplets']:,}")
    print(f"Match: {expected_total_quadruplets == actual_counts['total_quadruplets']}")
    
    print(f"\nExpected total entities: {expected_total_entities:,}")
    print(f"Actual total entities: {actual_counts['total_nodes']:,}")
    print(f"Match: {expected_total_entities == actual_counts['total_nodes']}")
    
    print(f"\nExpected total timestamps: {expected_total_timestamps:,}")
    print(f"Actual total timestamps: {actual_counts['total_timestamps']:,}")
    print(f"Match: {expected_total_timestamps == actual_counts['total_timestamps']}")
    
    print(f"\nActual relations: {actual_counts['total_relations']:,}")
    
    # Analyse discrepancies
    print("\n=== Discrepancy Analysis ===")
    
    # Check if there are overlapping entities between datasets
    print("Checking for entity overlap between datasets")
    
    # Load quadruplets to analyse entity overlap
    multitq_file = "analysis_results/temporal_kg_analysis/MultiTQ_quadruplets.json"
    timequestions_file = "analysis_results/temporal_kg_analysis/TimeQuestions_quadruplets.json"
    
    if Path(multitq_file).exists() and Path(timequestions_file).exists():
        with open(multitq_file, 'r') as f:
            multitq_quads = json.load(f)
        with open(timequestions_file, 'r') as f:
            timequestions_quads = json.load(f)
        
        # Extract unique entities from each dataset
        multitq_entities = set()
        timequestions_entities = set()
        
        for quad in multitq_quads:
            subject, predicate, obj, timestamp = quad
            multitq_entities.add(subject)
            multitq_entities.add(obj)
        
        for quad in timequestions_quads:
            subject, predicate, obj, timestamp = quad
            timequestions_entities.add(subject)
            timequestions_entities.add(obj)
        
        # Find overlapping entities
        overlapping_entities = multitq_entities.intersection(timequestions_entities)
        
        print(f"MultiTQ unique entities: {len(multitq_entities):,}")
        print(f"TimeQuestions unique entities: {len(timequestions_entities):,}")
        print(f"Overlapping entities: {len(overlapping_entities):,}")
        print(f"Combined unique entities: {len(multitq_entities.union(timequestions_entities)):,}")
        
        # This below explains why we are getting fewer total entities than we expected (look to see if can fix in future)
        print(f"\nEntity count explanation:")
        print(f"Expected: {len(multitq_entities)} + {len(timequestions_entities)} = {len(multitq_entities) + len(timequestions_entities):,}")
        print(f"Actual: {len(multitq_entities.union(timequestions_entities)):,} (due to {len(overlapping_entities):,} overlaps)")
        
        # Check timestamps overlap
        multitq_timestamps = set()
        timequestions_timestamps = set()
        
        for quad in multitq_quads:
            multitq_timestamps.add(quad[3])
        
        for quad in timequestions_quads:
            timequestions_timestamps.add(quad[3])
        
        overlapping_timestamps = multitq_timestamps.intersection(timequestions_timestamps)
        
        print(f"\nTimestamp overlap:")
        print(f"MultiTQ unique timestamps: {len(multitq_timestamps):,}")
        print(f"TimeQuestions unique timestamps: {len(timequestions_timestamps):,}")
        print(f"Overlapping timestamps: {len(overlapping_timestamps):,}")
        print(f"Combined unique timestamps: {len(multitq_timestamps.union(timequestions_timestamps)):,}")
    
    # Summary
    print("\n=== Verification Summary ===")
    quadruplet_mismatch = expected_total_quadruplets != actual_counts['total_quadruplets']
    entity_mismatch = expected_total_entities != actual_counts['total_nodes']
    timestamp_mismatch = expected_total_timestamps != actual_counts['total_timestamps']
    
    if not quadruplet_mismatch and not entity_mismatch and not timestamp_mismatch:
        print("All data counts match expected values")
        return True
    else:
        print("Data count discrepancies found:")
        if quadruplet_mismatch:
            print(f"- Quadruplet count mismatch")
        if entity_mismatch:
            print(f"- Entity count mismatch (likely due to entity overlap between datasets)")
        if timestamp_mismatch:
            print(f"- Timestamp count mismatch (likely due to timestamp overlap between datasets)")
        return False

if __name__ == "__main__":
    verify_data_counts()