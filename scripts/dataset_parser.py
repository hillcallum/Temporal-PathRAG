#!/usr/bin/env python3
"""
Lightweight temporal knowledge graph parser for PathRAG
Extracts (S,P,O,T) quadruplets with timestamp normalisation
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Set
from datetime import datetime


class TemporalKGParser:
    """Parse temporal knowledge graph data following TimeR4 methodology"""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.entities: Set[str] = set()
        self.relations: Set[str] = set()
        self.timestamps: Set[str] = set()
        
    def normalise_timestamp(self, timestamp: str) -> str:
        """Normalise timestamps to ISO 8601 format where possible"""
        if not timestamp or timestamp == "####":
            return "UNKNOWN"
            
        timestamp = timestamp.strip()
        
        # Already ISO format
        if re.match(r'^\d{4}-\d{2}-\d{2}$', timestamp):
            return timestamp
            
        # Year only 
        if re.match(r'^\d{4}$', timestamp):
            return f"{timestamp}-01-01"
            
        # Year-month
        if re.match(r'^\d{4}-\d{2}$', timestamp):
            return f"{timestamp}-01"
            
        return timestamp
    
    def extract_quadruplets(self, kg_file: str) -> List[Tuple[str, str, str, str]]:
        """Extract (Subject, Predicate, Object, Timestamp) quadruplets"""
        quadruplets = []
        kg_path = self.dataset_path / kg_file
        
        if not kg_path.exists():
            print(f"Warning: {kg_file} not found in {self.dataset_path}")
            return quadruplets
            
        print(f"Processing {kg_file}")
        
        with open(kg_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                    
                parts = line.split('\t')
                
                if len(parts) >= 4:
                    # Standard temporal KG format: S P O T
                    subject = parts[0].strip()
                    predicate = parts[1].strip()
                    obj = parts[2].strip()
                    timestamp = self.normalise_timestamp(parts[3])
                    
                elif len(parts) == 3:
                    # Static KG format: S P O
                    subject = parts[0].strip()
                    predicate = parts[1].strip()
                    obj = parts[2].strip()
                    timestamp = "UNKNOWN"
                    
                else:
                    continue
                
                # Track unique elements
                self.entities.add(subject)
                self.entities.add(obj)
                self.relations.add(predicate)
                self.timestamps.add(timestamp)
                
                quadruplets.append((subject, predicate, obj, timestamp))
                
        print(f"Extracted {len(quadruplets)} quadruplets")
        return quadruplets
    
    def define_entity_types(self) -> Dict[str, str]:
        """Define entity types based on common patterns"""
        entity_types = {}
        
        # Simple pattern-based classification
        for entity in self.entities:
            entity_lower = entity.lower()
            
            if any(pattern in entity_lower for pattern in ['person', 'human']):
                entity_types[entity] = 'PERSON'
            elif any(pattern in entity_lower for pattern in ['location', 'place', 'city', 'country']):
                entity_types[entity] = 'LOCATION'
            elif any(pattern in entity_lower for pattern in ['time', 'date', 'year']):
                entity_types[entity] = 'TIME'
            elif any(pattern in entity_lower for pattern in ['organisation', 'organization', 'company']):
                entity_types[entity] = 'ORGANISATION'
            else:
                entity_types[entity] = 'ENTITY'
                
        return entity_types
    
    def define_relation_types(self) -> Dict[str, str]:
        """Define relation types and semantic categories"""
        relation_types = {}
        
        temporal_patterns = ['before', 'after', 'during', 'start', 'end']
        spatial_patterns = ['located', 'born', 'in', 'at']
        social_patterns = ['married', 'friend', 'colleague', 'member']
        
        for relation in self.relations:
            rel_lower = relation.lower()
            
            if any(pattern in rel_lower for pattern in temporal_patterns):
                relation_types[relation] = 'TEMPORAL'
            elif any(pattern in rel_lower for pattern in spatial_patterns):
                relation_types[relation] = 'SPATIAL'
            elif any(pattern in rel_lower for pattern in social_patterns):
                relation_types[relation] = 'SOCIAL'
            else:
                relation_types[relation] = 'FACTUAL'
                
        return relation_types
    
    def generate_statistics(self, quadruplets: List[Tuple[str, str, str, str]]) -> Dict[str, int]:
        """Generate dataset statistics"""
        return {
            'total_quadruplets': len(quadruplets),
            'unique_entities': len(self.entities),
            'unique_relations': len(self.relations),
            'unique_timestamps': len(self.timestamps),
            'temporal_facts': len([q for q in quadruplets if q[3] != "UNKNOWN"])
        }


def process_multitq():
    """Process MultiTQ dataset"""
    print("\n" + "="*50)
    print("Processing MultiTQ Dataset")
    print("="*50)
    
    parser = TemporalKGParser("datasets/MultiTQ")
    quadruplets = parser.extract_quadruplets("kg/full.txt")
    
    if quadruplets:
        entity_types = parser.define_entity_types()
        relation_types = parser.define_relation_types()
        stats = parser.generate_statistics(quadruplets)
        
        print(f"MultiTQ Statistics:")
        print(f"Quadruplets: {stats['total_quadruplets']:,}")
        print(f"Entities: {stats['unique_entities']:,}")
        print(f"Relations: {stats['unique_relations']:,}")
        print(f"Timestamps: {stats['unique_timestamps']:,}")
        print(f"Temporal Facts: {stats['temporal_facts']:,}")


def process_timequestions():
    """Process TimeQuestions dataset"""
    print("\n" + "="*50)
    print("Processing TimeQuestions Dataset")
    print("="*50)
    
    parser = TemporalKGParser("datasets/TimeQuestions")
    quadruplets = parser.extract_quadruplets("kg/full.txt")
    
    if quadruplets:
        entity_types = parser.define_entity_types()
        relation_types = parser.define_relation_types()
        stats = parser.generate_statistics(quadruplets)
        
        print(f"TimeQuestions Statistics:")
        print(f"Quadruplets: {stats['total_quadruplets']:,}")
        print(f"Entities: {stats['unique_entities']:,}")
        print(f"Relations: {stats['unique_relations']:,}")
        print(f"Timestamps: {stats['unique_timestamps']:,}")
        print(f"Temporal Facts: {stats['temporal_facts']:,}")


def main():
    """Main processing function"""
    print("Temporal PathRAG Dataset Parser")
    print("Lightweight (S,P,O,T) quadruplet extraction")
    
    # Process both datasets
    process_multitq()
    process_timequestions()
    
    print("\nDataset parsing complete")


if __name__ == "__main__":
    main()