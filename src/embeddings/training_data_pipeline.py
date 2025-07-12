"""
Training data pipeline for temporal embeddings for fine-tuning
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
import networkx as nx
from dataclasses import dataclass, asdict
from collections import defaultdict
import numpy as np
from tqdm import tqdm


@dataclass
class TemporalQuadruplet:
    """Represents a temporal quadruplet for training"""
    subject: str
    relation: str
    object: str
    timestamp: str
    
    def to_text(self) -> str:
        """Convert to natural language text"""
        return f"At {self.timestamp}, {self.subject} {self.relation.replace('_', ' ')} {self.object}"
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class TrainingExample:
    """A training example for temporal embeddings"""
    anchor: TemporalQuadruplet
    positive: Optional[TemporalQuadruplet] = None
    negative: Optional[TemporalQuadruplet] = None
    example_type: str = "quadruplet"  # quadruplet, contrastive, or reconstruction
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialisation"""
        data = {
            "anchor": self.anchor.to_dict(),
            "example_type": self.example_type
        }
        if self.positive:
            data["positive"] = self.positive.to_dict()
        if self.negative:
            data["negative"] = self.negative.to_dict()
        return data


class TemporalEmbeddingDataPipeline:
    """
    Creates training data for temporal embeddings from KGs
    """
    
    def __init__(self, graph: nx.DiGraph, dataset_name: str):
        """
        Initialise the training data pipeline
        """
        self.graph = graph
        self.dataset_name = dataset_name
        self.quadruplets = self.extract_temporal_quadruplets()
        
        # Index structures for efficient sampling
        self._build_indices()
        
    def extract_temporal_quadruplets(self) -> List[TemporalQuadruplet]:
        """Extract all temporal quadruplets from the graph"""
        quadruplets = []
        
        for source, target, data in self.graph.edges(data=True):
            if 'relation' in data and 'timestamps' in data:
                for timestamp in data['timestamps']:
                    quadruplet = TemporalQuadruplet(
                        subject=source,
                        relation=data['relation'],
                        object=target,
                        timestamp=str(timestamp)
                    )
                    quadruplets.append(quadruplet)
        
        print(f"Extracted {len(quadruplets)} temporal quadruplets from graph")
        return quadruplets
    
    def _build_indices(self):
        """Build indices for efficient sampling"""
        # Entity to quadruplets mapping
        self.entity_quadruplets = defaultdict(list)
        
        # Relation to quadruplets mapping
        self.relation_quadruplets = defaultdict(list)
        
        # Temporal index
        self.temporal_quadruplets = defaultdict(list)
        
        # Entity pairs that are connected
        self.connected_pairs = set()
        
        for quadruplet in self.quadruplets:
            # Entity indices
            self.entity_quadruplets[quadruplet.subject].append(quadruplet)
            self.entity_quadruplets[quadruplet.object].append(quadruplet)
            
            # Relation index
            self.relation_quadruplets[quadruplet.relation].append(quadruplet)
            
            # Temporal index
            self.temporal_quadruplets[quadruplet.timestamp].append(quadruplet)
            
            # Connected pairs
            self.connected_pairs.add((quadruplet.subject, quadruplet.object))
            self.connected_pairs.add((quadruplet.object, quadruplet.subject))
        
        # All entities
        self.entities = list(self.entity_quadruplets.keys())
        
        print(f"Built indices: {len(self.entities)} entities, "
              f"{len(self.relation_quadruplets)} relations, "
              f"{len(self.temporal_quadruplets)} timestamps")
    
    def generate_quadruplets_examples(self, num_examples: int) -> List[TrainingExample]:
        """
        Generate quadruplet training examples (anchor, positive, negative)
        
        Positive examples share entities or occur at similar times
        Negative examples are randomly sampled dissimilar quadruplets
        """
        examples = []
        
        for _ in tqdm(range(num_examples), desc="Generating quadruplet examples"):
            # Sample anchor
            anchor = random.choice(self.quadruplets)
            
            # Generate positive example
            positive = self.sample_positive_quadruplet(anchor)
            
            # Generate negative example
            negative = self.sample_negative_quadruplet(anchor)
            
            example = TrainingExample(
                anchor=anchor,
                positive=positive,
                negative=negative,
                example_type="quadruplet"
            )
            examples.append(example)
        
        return examples
    
    def sample_positive_quadruplet(self, anchor: TemporalQuadruplet) -> TemporalQuadruplet:
        """Sample a positive quadruplet related to the anchor"""
        strategies = [
            self.positive_same_entity,
            self.positive_same_relation,
            self.positive_temporal_proximity,
            self.positive_connected_entity
        ]
        
        # Try strategies in random order
        random.shuffle(strategies)
        
        for strategy in strategies:
            positive = strategy(anchor)
            if positive and positive != anchor:
                return positive
        
        # Fallback: return a random quadruplet (not ideal but ensures we have something)
        return random.choice(self.quadruplets)
    
    def positive_same_entity(self, anchor: TemporalQuadruplet) -> Optional[TemporalQuadruplet]:
        """Positive example sharing an entity with anchor"""
        # Get quadruplets with same subject or object
        candidates = (self.entity_quadruplets[anchor.subject] + 
                     self.entity_quadruplets[anchor.object])
        
        candidates = [t for t in candidates if t != anchor]
        
        if candidates:
            return random.choice(candidates)
        return None
    
    def positive_same_relation(self, anchor: TemporalQuadruplet) -> Optional[TemporalQuadruplet]:
        """Positive example with same relation"""
        candidates = [t for t in self.relation_quadruplets[anchor.relation] 
                     if t != anchor]
        
        if candidates:
            return random.choice(candidates)
        return None
    
    def positive_temporal_proximity(self, anchor: TemporalQuadruplet) -> Optional[TemporalQuadruplet]:
        """Positive example from temporally close timestamp"""
        # Find nearby timestamps
        try:
            anchor_time = self.parse_timestamp(anchor.timestamp)
            
            # Find quadruplets within temporal window
            candidates = []
            for ts, quadruplets in self.temporal_quadruplets.items():
                try:
                    time = self.parse_timestamp(ts)
                    if abs(time - anchor_time) <= 365:  # Within a year
                        candidates.extend([t for t in quadruplets if t != anchor])
                except:
                    continue
            
            if candidates:
                return random.choice(candidates)
        except:
            pass
        
        return None
    
    def positive_connected_entity(self, anchor: TemporalQuadruplet) -> Optional[TemporalQuadruplet]:
        """Positive example involving entities connected to anchor entities"""
        # Find entities connected to anchor's entities
        connected_entities = set()
        
        # Get neighbours of subject and object
        if anchor.subject in self.graph:
            connected_entities.update(self.graph.neighbors(anchor.subject))
        if anchor.object in self.graph:
            connected_entities.update(self.graph.neighbors(anchor.object))
        
        # Find quadruplets involving these entities
        candidates = []
        for entity in connected_entities:
            candidates.extend(self.entity_quadruplets[entity])
        
        candidates = [t for t in candidates if t != anchor]
        
        if candidates:
            return random.choice(candidates)
        return None
    
    def sample_negative_quadruplet(self, anchor: TemporalQuadruplet) -> TemporalQuadruplet:
        """Sample a negative quadruplet unrelated to anchor"""
        # Strategy: sample quadruplet that don't share entities or relations
        max_attempts = 100
        
        for _ in range(max_attempts):
            candidate = random.choice(self.quadruplets)
            
            # Check if it's a good negative
            if (candidate.subject != anchor.subject and
                candidate.object != anchor.object and
                candidate.subject != anchor.object and
                candidate.object != anchor.subject and
                candidate.relation != anchor.relation):
                
                # Also check they're not directly connected
                if ((candidate.subject, candidate.object) not in self.connected_pairs and
                    (candidate.object, candidate.subject) not in self.connected_pairs):
                    return candidate
        
        # Fallback: return any different quadruplet
        candidates = [t for t in self.quadruplets if t != anchor]
        return random.choice(candidates) if candidates else anchor
    
    def generate_contrastive_examples(self, num_examples: int) -> List[TrainingExample]:
        """
        Generate contrastive learning examples focusing on temporal patterns and sequences
        """
        examples = []
        
        for _ in tqdm(range(num_examples), desc="Generating contrastive examples"):
            # Sample anchor
            anchor = random.choice(self.quadruplets)
            
            # For contrastive learning, we create pairs
            if random.random() < 0.5:
                # Positive pair: temporally coherent
                positive = self.sample_temporal_coherent(anchor)
                example = TrainingExample(
                    anchor=anchor,
                    positive=positive,
                    example_type="contrastive_positive"
                )
            else:
                # Negative pair: temporally incoherent
                negative = self.sample_temporal_incoherent(anchor)
                example = TrainingExample(
                    anchor=anchor,
                    negative=negative,
                    example_type="contrastive_negative"
                )
            
            examples.append(example)
        
        return examples
    
    def sample_temporal_coherent(self, anchor: TemporalQuadruplet) -> TemporalQuadruplet:
        """Sample a temporally coherent quadruplet"""
        # Find quadruplet that form a temporal sequence
        candidates = []
        
        # Same entities, different times (temporal evolution)
        for quadruplet in self.entity_quadruplets[anchor.subject]:
            if quadruplet != anchor and quadruplet.object in self.entities:
                candidates.append(quadruplet)
        
        # Causal patterns (A->B at t1, B->C at t2)
        if anchor.object in self.entity_quadruplets:
            for quadruplet in self.entity_quadruplets[anchor.object]:
                if quadruplet.subject == anchor.object:  # Continuation
                    candidates.append(quadruplet)
        
        if candidates:
            return random.choice(candidates)
        
        # Fallback to positive sampling
        return self.sample_positive_quadruplet(anchor)
    
    def sample_temporal_incoherent(self, anchor: TemporalQuadruplet) -> TemporalQuadruplet:
        """Sample a temporally incoherent quadruplet"""
        # Find quadruplets that break temporal logic
        
        # Different time period, unrelated entities
        candidates = []
        for ts, quadruplets in self.temporal_quadruplets.items():
            if ts != anchor.timestamp:
                for quadruplet in quadruplets:
                    if (quadruplet.subject not in [anchor.subject, anchor.object] and
                        quadruplet.object not in [anchor.subject, anchor.object]):
                        candidates.append(quadruplet)
        
        if candidates:
            return random.choice(candidates[:100])  # Limit for efficiency
        
        # Fallback to negative sampling
        return self.sample_negative_quadruplet(anchor)
    
    def generate_reconstruction_examples(self, num_examples: int) -> List[TrainingExample]:
        """
        Generate reconstruction examples for self-supervised learning, so the model learns
        to reconstruct masked parts of temporal quadruplet
        """
        examples = []
        
        for _ in tqdm(range(num_examples), desc="Generating reconstruction examples"):
            # Sample a quadruplet
            quadruplet = random.choice(self.quadruplets)
            
            # Create masked version
            mask_type = random.choice(['entity', 'relation', 'temporal'])
            
            if mask_type == 'entity':
                # Mask subject or object
                if random.random() < 0.5:
                    masked = TemporalQuadruplet(
                        subject="[MASK]",
                        relation=quadruplet.relation,
                        object=quadruplet.object,
                        timestamp=quadruplet.timestamp
                    )
                else:
                    masked = TemporalQuadruplet(
                        subject=quadruplet.subject,
                        relation=quadruplet.relation,
                        object="[MASK]",
                        timestamp=quadruplet.timestamp
                    )
            elif mask_type == 'relation':
                # Mask relation
                masked = TemporalQuadruplet(
                    subject=quadruplet.subject,
                    relation="[MASK]",
                    object=quadruplet.object,
                    timestamp=quadruplet.timestamp
                )
            else:
                # Mask timestamp
                masked = TemporalQuadruplet(
                    subject=quadruplet.subject,
                    relation=quadruplet.relation,
                    object=quadruplet.object,
                    timestamp="[MASK]"
                )
            
            example = TrainingExample(
                anchor=masked,
                positive=quadruplet,  # Original quadruplet is the target
                example_type=f"reconstruction_{mask_type}"
            )
            examples.append(example)
        
        return examples
    
    def parse_timestamp(self, timestamp: str) -> int:
        """Parse timestamp to numeric value for comparison"""
        # Simple parsing - can be enhanced based on timestamp format
        if '-' in timestamp:
            # Assume YYYY-MM-DD format
            parts = timestamp.split('-')
            if len(parts) >= 1 and parts[0].isdigit():
                return int(parts[0])
        
        # Fallback: hash to number
        return hash(timestamp) % 10000
    
    def create_training_dataset(self, 
                              num_quadruplet: int = 10000,
                              num_contrastive: int = 5000,
                              num_reconstruction: int = 5000,
                              output_dir: Optional[Path] = None) -> Dict[str, List[TrainingExample]]:
        """
        Create a complete training dataset with multiple objectives
        """
        print(f"Creating training dataset for {self.dataset_name}")
        
        # Generate examples
        all_examples = []
        
        if num_quadruplet > 0:
            all_examples.extend(self.generate_quadruplets_examples(num_quadruplet))
        
        if num_contrastive > 0:
            all_examples.extend(self.generate_contrastive_examples(num_contrastive))
        
        if num_reconstruction > 0:
            all_examples.extend(self.generate_reconstruction_examples(num_reconstruction))
        
        # Shuffle all examples
        random.shuffle(all_examples)
        
        # Split into train/val/test (80/10/10)
        n = len(all_examples)
        train_end = int(0.8 * n)
        val_end = int(0.9 * n)
        
        dataset = {
            'train': all_examples[:train_end],
            'validation': all_examples[train_end:val_end],
            'test': all_examples[val_end:]
        }
        
        print(f"Created dataset with {len(dataset['train'])} train, "
              f"{len(dataset['validation'])} val, {len(dataset['test'])} test examples")
        
        # Save if output directory specified
        if output_dir:
            self.save_dataset(dataset, output_dir)
        
        return dataset
    
    def save_dataset(self, dataset: Dict[str, List[TrainingExample]], output_dir: Path):
        """Save dataset to disk"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for split, examples in dataset.items():
            output_file = output_dir / f"{split}.json"
            
            # Convert to serialisable format
            data = [ex.to_dict() for ex in examples]
            
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            print(f"Saved {len(examples)} examples to {output_file}")
        
        # Save metadata
        metadata = {
            'dataset_name': self.dataset_name,
            'num_entities': len(self.entities),
            'num_relations': len(self.relation_quadruplets),
            'num_timestamps': len(self.temporal_quadruplets),
            'num_quadruplets': len(self.quadruplets),
            'splits': {split: len(examples) for split, examples in dataset.items()}
        }
        
        with open(output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Dataset saved to {output_dir}")


def create_training_pipeline(graph: nx.DiGraph, dataset_name: str) -> TemporalEmbeddingDataPipeline:
    """Create training data pipeline"""
    return TemporalEmbeddingDataPipeline(graph, dataset_name)