#!/usr/bin/env python3
"""
Lightweight temporal knowledge graph parser for PathRAG
Extracts (S,P,O,T) quadruplets with timestamp normalisation
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional, Any
from datetime import datetime
from collections import defaultdict, Counter
import pandas as pd


class TemporalKGParser:
    """Parsing tenporal KG graph data"""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.dataset_name = dataset_path.split('/')[-1]
        
        # Initialise collections for analysis
        self.entities: Set[str] = set()
        self.entity_types: Dict[str, str] = {}  # entity -> type
        self.relations: Set[str] = set()
        self.relation_types: Dict[str, str] = {}  # relation -> category
        self.timestamps: Set[str] = set()
        self.quadruplets: List[Tuple[str, str, str, str]] = []
        
        print(f"Analysing {self.dataset_name} temporal knowledge graph")
    
    def analyse_file_formats(self) -> Dict[str, Any]:
        """Check if everyhting is in the right format"""
        print(f"\nAnalysing file formats for {self.dataset_name}")
        
        analysis = {
            "dataset_name": self.dataset_name,
            "files_found": {},
            "kg_format": {},
            "questions_format": {},
            "temporal_representation": {}
        }
        
        # Analyse directory structure
        for file_path in self.dataset_path.rglob("*"):
            if file_path.is_file():
                rel_path = str(file_path.relative_to(self.dataset_path))
                file_size = file_path.stat().st_size
                analysis["files_found"][rel_path] = {
                    "size_bytes": file_size,
                    "size_mb": round(file_size / 1024 / 1024, 2)
                }
        
        # Analyse KG format
        kg_file = self.dataset_path / "kg/full.txt"
        if kg_file.exists():
            with open(kg_file, 'r', encoding='utf-8') as f:
                sample_lines = [f.readline().strip() for _ in range(5)]
            
            analysis["kg_format"] = {
                "file": "kg/full.txt",
                "sample_lines": sample_lines,
                "total_lines": sum(1 for _ in open(kg_file, 'r', encoding='utf-8')),
                "format": "Tab-separated (Subject, Predicate, Object, Timestamp)"
            }
        
        # Analyse questions format
        for split in ["test", "dev", "train"]:
            q_file = self.dataset_path / f"questions/{split}.json"
            if q_file.exists():
                with open(q_file, 'r') as f:
                    questions = json.load(f)
                
                if questions:
                    analysis["questions_format"][split] = {
                        "count": len(questions),
                        "sample_keys": list(questions[0].keys()),
                        "sample_question": questions[0]
                    }
        
        return analysis
    
    def extract_quadruplets(self) -> List[Tuple[str, str, str, str]]:
        """Parse raw facts into (S, P, O, T) quadruplets"""
        print(f"\nExtracting (S, P, O, T) quadruplets from {self.dataset_name}")
        
        kg_file = self.dataset_path / "kg/full.txt"
        quadruplets = []
        
        with open(kg_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split('\t')
                if len(parts) >= 4:
                    subject = parts[0].strip()
                    predicate = parts[1].strip()
                    obj = parts[2].strip()
                    timestamp = parts[3].strip()
                    
                    # Handle TimeQuestions format with end timestamps
                    if len(parts) >= 5 and self.dataset_name == "TimeQuestions":
                        end_timestamp = parts[4].strip()
                        # Use start timestamp for primary temporal info
                        timestamp = timestamp
                    
                    quadruplet = (subject, predicate, obj, timestamp)
                    quadruplets.append(quadruplet)
                    
                    # Collect entities and relations
                    self.entities.add(subject)
                    self.entities.add(obj)
                    self.relations.add(predicate)
                    self.timestamps.add(timestamp)
                
                if line_num % 100000 == 0:
                    print(f"Processed {line_num:,} facts")
        
        self.quadruplets = quadruplets
        print(f"Extracted {len(quadruplets):,} quadruplets")
        return quadruplets
    
    def normalise_timestamps(self) -> Dict[str, str]:
        """Handle timestamp normalisation"""
        print(f"\nNormalising timestamps for {self.dataset_name}")
        
        normalised_map = {}
        normalisation_stats = defaultdict(int)
        
        for timestamp in self.timestamps:
            normalised = self.normalise_single_timestamp(timestamp)
            normalised_map[timestamp] = normalised
            
            # Track normalisation patterns
            if normalised == "UNKNOWN":
                normalisation_stats["unknown"] += 1
            elif re.match(r'^\d{4}$', normalised):
                normalisation_stats["year_only"] += 1
            elif re.match(r'^\d{4}-\d{2}$', normalised):
                normalisation_stats["year_month"] += 1
            elif re.match(r'^\d{4}-\d{2}-\d{2}$', normalised):
                normalisation_stats["full_date"] += 1
            else:
                normalisation_stats["other"] += 1
        
        print(f"Normalised {len(normalised_map)} unique timestamps")
        print(f"Granularity distribution:")
        for pattern, count in normalisation_stats.items():
            percentage = (count / len(normalised_map)) * 100
            print(f"{pattern}: {count} ({percentage:.1f}%)")
        
        return normalised_map
    
    def normalise_single_timestamp(self, timestamp: str) -> str:
        """Normalise a single timestamp to ISO format"""
        if not timestamp or timestamp in ["####", "UNKNOWN", ""]:
            return "UNKNOWN"
        
        timestamp = timestamp.strip()
        
        # Already ISO format (YYYY-MM-DD)
        if re.match(r'^\d{4}-\d{2}-\d{2}$', timestamp):
            return timestamp
        
        # Year only
        if re.match(r'^\d{4}$', timestamp):
            return timestamp
        
        # Year-month
        if re.match(r'^\d{4}-\d{2}$', timestamp):
            return timestamp
        
        # Handle special dataset formats
        if self.dataset_name == "TimeQuestions":
            # TimeQuestions uses year IDs (1-2916 maps to years)
            try:
                year_id = int(timestamp)
                if 1 <= year_id <= 2916:
                    # Simple mapping: year_id maps to actual year
                    # This is dataset-specific logic
                    return str(year_id) if year_id > 1000 else f"{year_id:04d}"
            except ValueError:
                pass
        
        # Try common date formats
        date_patterns = [
            ('%Y-%m-%d', r'^\d{4}-\d{1,2}-\d{1,2}$'),
            ('%d/%m/%Y', r'^\d{1,2}/\d{1,2}/\d{4}$'),
            ('%m/%d/%Y', r'^\d{1,2}/\d{1,2}/\d{4}$'),
            ('%Y', r'^\d{4}$'),
        ]
        
        for fmt, pattern in date_patterns:
            if re.match(pattern, timestamp):
                try:
                    dt = datetime.strptime(timestamp, fmt)
                    return dt.strftime('%Y-%m-%d')
                except ValueError:
                    continue
        
        return "UNKNOWN"
    
    def analyse_entity_types(self) -> Dict[str, List[str]]:
        """Define specific entity types"""
        print(f"\nAnalysing entity types for {self.dataset_name}")
        
        entity_types = defaultdict(list)
        
        # Pattern-based entity type classification
        for entity in self.entities:
            entity_clean = entity.replace('_', ' ')
            entity_type = self.classify_entity_type(entity_clean)
            entity_types[entity_type].append(entity)
            self.entity_types[entity] = entity_type
        
        print(f"Classified {len(self.entities):,} entities into {len(entity_types)} types")
        for entity_type, entities in entity_types.items():
            print(f"{entity_type}: {len(entities):,} entities")
            # Show sample entities
            samples = entities[:3]
            print(f"Examples: {', '.join(samples)}")
        
        return dict(entity_types)
    
    def classify_entity_type(self, entity: str) -> str:
        """Classify entity into types based on patterns"""
        entity_lower = entity.lower()
        
        # Person patterns
        person_patterns = [
            r'\b(mr|mrs|ms|dr|prof|president|minister|secretary|director)\b',
            r'\b[A-Z][a-z]+ [A-Z][a-z]+$',  # First Last name pattern
        ]
        for pattern in person_patterns:
            if re.search(pattern, entity, re.IGNORECASE):
                return "Person"
        
        # Organisation patterns
        org_patterns = [
            r'\b(company|corp|corporation|inc|llc|ltd|university|college|school)\b',
            r'\b(government|ministry|department|agency|bureau|commission)\b',
            r'\b(party|organization|association|union|group)\b'
        ]
        for pattern in org_patterns:
            if re.search(pattern, entity, re.IGNORECASE):
                return "Organisation"
        
        # Location patterns
        location_patterns = [
            r'\b(city|town|village|county|state|province|country|nation)\b',
            r'\b(street|avenue|road|boulevard|square|park)\b',
            r'\b(north|south|east|west|central|upper|lower)\b.*\b(region|area|district)\b'
        ]
        for pattern in location_patterns:
            if re.search(pattern, entity, re.IGNORECASE):
                return "Location"
        
        # Event patterns
        event_patterns = [
            r'\b(conference|meeting|summit|election|war|battle|crisis)\b',
            r'\b(festival|ceremony|celebration|protest|demonstration)\b'
        ]
        for pattern in event_patterns:
            if re.search(pattern, entity, re.IGNORECASE):
                return "Event"
        
        # Time patterns (for temporal entities)
        time_patterns = [
            r'\b\d{4}\b',  # Years
            r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b',
            r'\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b'
        ]
        for pattern in time_patterns:
            if re.search(pattern, entity, re.IGNORECASE):
                return "Time"
        
        # Default classification
        if entity_lower.startswith(('the ', 'a ', 'an ')):
            return "Concept"
        
        return "Entity"  # Generic fallback
    
    def analyse_relation_types(self) -> Dict[str, List[str]]:
        """Define relation types"""
        print(f"\nAnalysing relation types for {self.dataset_name}")
        
        relation_categories = defaultdict(list)
        
        for relation in self.relations:
            category = self._classify_relation_type(relation)
            relation_categories[category].append(relation)
            self.relation_types[relation] = category
        
        print(f"Classified {len(self.relations):,} relations into {len(relation_categories)} categories")
        for category, relations in relation_categories.items():
            print(f"{category}: {len(relations):,} relations")
            samples = relations[:3]
            print(f"Examples: {', '.join(samples)}")
        
        return dict(relation_categories)
    
    def classify_relation_type(self, relation: str) -> str:
        """Classify relation into categories"""
        relation_clean = relation.replace('_', ' ').lower()
        
        # Temporal relations
        temporal_patterns = [
            r'\b(occurred|started|ended|began|finished|happened)\b',
            r'\b(before|after|during|until|since|from|to)\b',
            r'\b(born|died|founded|established|created|destroyed)\b'
        ]
        for pattern in temporal_patterns:
            if re.search(pattern, relation_clean):
                return "Temporal"
        
        # Social relations
        social_patterns = [
            r'\b(married|divorced|friend|enemy|colleague|partner)\b',
            r'\b(parent|child|sibling|spouse|family)\b',
            r'\b(knows|meets|visits|contacts)\b'
        ]
        for pattern in social_patterns:
            if re.search(pattern, relation_clean):
                return "Social"
        
        # Professional relations
        professional_patterns = [
            r'\b(works|employed|hired|fired|promoted)\b',
            r'\b(ceo|director|manager|employee|staff)\b',
            r'\b(position|role|job|career|occupation)\b'
        ]
        for pattern in professional_patterns:
            if re.search(pattern, relation_clean):
                return "Professional"
        
        # Location relations
        location_patterns = [
            r'\b(located|situated|based|headquartered)\b',
            r'\b(capital|member|part|belongs)\b',
            r'\b(visit|travel|move|migrate)\b'
        ]
        for pattern in location_patterns:
            if re.search(pattern, relation_clean):
                return "Locational"
        
        # Ownership/possession
        ownership_patterns = [
            r'\b(owns|owned|possesses|has|belongs)\b',
            r'\b(property|asset|wealth|inheritance)\b'
        ]
        for pattern in ownership_patterns:
            if re.search(pattern, relation_clean):
                return "Ownership"
        
        return "General"  # Generic fallback
    
    def analyse_temporal_granularity(self) -> Dict[str, Any]:
        """Implement multi-granularity timestamp handling"""
        print(f"\nAnalysing temporal granularity for {self.dataset_name}")
        
        granularity_analysis = {
            "year_only": [],
            "year_month": [],
            "full_date": [],
            "unknown": [],
            "statistics": {}
        }
        
        for timestamp in self.timestamps:
            normalised = self.normalise_single_timestamp(timestamp)
            
            if normalised == "UNKNOWN":
                granularity_analysis["unknown"].append(timestamp)
            elif re.match(r'^\d{4}$', normalised):
                granularity_analysis["year_only"].append(timestamp)
            elif re.match(r'^\d{4}-\d{2}$', normalised):
                granularity_analysis["year_month"].append(timestamp)
            elif re.match(r'^\d{4}-\d{2}-\d{2}$', normalised):
                granularity_analysis["full_date"].append(timestamp)
        
        # Calculate statistics
        total = len(self.timestamps)
        granularity_analysis["statistics"] = {
            "total_timestamps": total,
            "year_only_count": len(granularity_analysis["year_only"]),
            "year_month_count": len(granularity_analysis["year_month"]),
            "full_date_count": len(granularity_analysis["full_date"]),
            "unknown_count": len(granularity_analysis["unknown"]),
            "year_only_percent": len(granularity_analysis["year_only"]) / total * 100,
            "year_month_percent": len(granularity_analysis["year_month"]) / total * 100,
            "full_date_percent": len(granularity_analysis["full_date"]) / total * 100,
            "unknown_percent": len(granularity_analysis["unknown"]) / total * 100,
        }
        
        print(f"Temporal granularity analysis complete:")
        stats = granularity_analysis["statistics"]
        print(f"Year only: {stats['year_only_count']:,} ({stats['year_only_percent']:.1f}%)")
        print(f"Year-Month: {stats['year_month_count']:,} ({stats['year_month_percent']:.1f}%)")
        print(f"Full Date: {stats['full_date_count']:,} ({stats['full_date_percent']:.1f}%)")
        print(f"Unknown: {stats['unknown_count']:,} ({stats['unknown_percent']:.1f}%)")
        
        return granularity_analysis
    
    def save_analysis_results(self, output_dir: str = "analysis_results"):
        """Save all analysis results to files"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save quadruplets
        quadruplets_file = output_path / f"{self.dataset_name}_quadruplets.json"
        with open(quadruplets_file, 'w') as f:
            json.dump(self.quadruplets, f, indent=2)
        
        # Save entity types
        entity_types_file = output_path / f"{self.dataset_name}_entity_types.json"
        entity_types_dict = defaultdict(list)
        for entity, entity_type in self.entity_types.items():
            entity_types_dict[entity_type].append(entity)
        
        with open(entity_types_file, 'w') as f:
            json.dump(dict(entity_types_dict), f, indent=2)
        
        # Save relation types
        relation_types_file = output_path / f"{self.dataset_name}_relation_types.json"
        relation_types_dict = defaultdict(list)
        for relation, relation_type in self.relation_types.items():
            relation_types_dict[relation_type].append(relation)
        
        with open(relation_types_file, 'w') as f:
            json.dump(dict(relation_types_dict), f, indent=2)
        
        print(f"Analysis results saved to {output_path}/")
    
    def run_complete_analysis(self):
        """Run all analysis sub-tasks"""
        print(f"Running complete temporal KG parsing for {self.dataset_name}")
        print("=" * 60)
        
        # File format inspection
        format_analysis = self.analyse_file_formats()
        
        # Extract quadruplets
        quadruplets = self.extract_quadruplets()
        
        # Normalise timestamps
        normalised_timestamps = self.normalise_timestamps()
        
        # Analyse entity types
        entity_types = self.analyse_entity_types()
        
        # Analyse relation types
        relation_types = self.analyse_relation_types()
        
        # Temporal granularity
        temporal_analysis = self.analyse_temporal_granularity()
        
        # Save results
        self.save_analysis_results()
        
        print(f"\nComplete analysis finished for {self.dataset_name}")
        print(f"{len(quadruplets):,} quadruplets extracted")
        print(f"{len(entity_types)} entity categories")
        print(f"{len(relation_types)} relation categories")
        print(f"{len(self.timestamps)} unique timestamps")


def main():
    """Main execution function"""

    # Analyse both datasets
    datasets = ["datasets/MultiTQ", "datasets/TimeQuestions"]
    
    for dataset_path in datasets:
        if Path(dataset_path).exists():
            analyser = TemporalKGParser(dataset_path)
            analyser.run_complete_analysis()
            print()
        else:
            print(f"Dataset not found: {dataset_path}")


if __name__ == "__main__":
    main()