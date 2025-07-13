#!/usr/bin/env python3
"""
Test trained temporal embeddings by evaluating similarity and temporal reasoning
"""

import sys
import argparse
from pathlib import Path
import torch
import numpy as np
from typing import List, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.training.train_temporal_embeddings import TemporalEmbeddingModel, TrainingConfig
from src.embeddings.temporal_embeddings import TemporalEmbeddingConfig


def load_trained_model(model_path: str) -> TemporalEmbeddingModel:
    """Load trained temporal embedding model"""
    print(f"Loading model from {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Create config from checkpoint
    config_dict = checkpoint.get('config', {})
    config = TrainingConfig(**{k: v for k, v in config_dict.items() if k in TrainingConfig.__dataclass_fields__})
    
    # Create model
    model = TemporalEmbeddingModel(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model


def test_temporal_similarity(model: TemporalEmbeddingModel):
    """Test temporal similarity reasoning"""
    print("\nTesting Temporal Similarity")
    
    # Test cases
    test_cases = [
        # Similar temporal events
        (
            "In 1969, Neil Armstrong walked on the moon",
            "In 1969, the Apollo 11 mission landed on the moon",
            "In 2020, SpaceX launched astronauts to ISS"
        ),
        # Entity evolution
        (
            "In 1990, Germany was reunified",
            "In 1991, Germany joined the European Union",
            "In 1990, Japan experienced economic bubble"
        ),
        # Causal relationships
        (
            "In 2008, the financial crisis began",
            "In 2009, many banks received bailouts",
            "In 2008, Olympics were held in Beijing"
        )
    ]
    
    for anchor, positive, negative in test_cases:
        # Get embeddings
        with torch.no_grad():
            anchor_emb = model([anchor])
            positive_emb = model([positive])
            negative_emb = model([negative])
            
        # Compute distances
        pos_dist = torch.nn.functional.pairwise_distance(anchor_emb, positive_emb).item()
        neg_dist = torch.nn.functional.pairwise_distance(anchor_emb, negative_emb).item()
        
        print(f"\nAnchor: {anchor}")
        print(f"Positive: {positive} (dist: {pos_dist:.4f})")
        print(f"Negative: {negative} (dist: {neg_dist:.4f})")
        print(f"Margin: {neg_dist - pos_dist:.4f} (should be positive)")


def test_temporal_encoding(model: TemporalEmbeddingModel):
    """Test temporal encoding capabilities"""
    print("\nTesting Temporal Encoding")
    
    # Same event at different times
    events = [
        "In 1900, the automobile industry emerged",
        "In 1950, the automobile industry expanded globally",
        "In 2000, the automobile industry embraced electric vehicles",
        "In 2020, the automobile industry focused on autonomous driving"
    ]
    
    # Get embeddings
    with torch.no_grad():
        embeddings = model(events)
        
    # Compute pairwise similarities
    print("\nTemporal progression similarity matrix:")
    print("(Higher values = more similar)")
    
    n = len(events)
    similarity_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            sim = torch.nn.functional.cosine_similarity(
                embeddings[i:i+1], embeddings[j:j+1]
            ).item()
            similarity_matrix[i, j] = sim
            
    # Print matrix
    print("\n ", end="")
    for i in range(n):
        print(f"{1900 + i*50:>8}", end="")
    print()
    
    for i in range(n):
        print(f"{1900 + i*50:>5}", end="")
        for j in range(n):
            print(f"{similarity_matrix[i, j]:>8.4f}", end="")
        print()


def test_entity_temporal_consistency(model: TemporalEmbeddingModel):
    """Test entity consistency across time"""
    print("\nTesting Entity Temporal Consistency")
    
    # Different entities at same time vs same entity at different times
    test_groups = [
        {
            "name": "Apple Inc evolution",
            "events": [
                "In 1976, Apple was founded by Steve Jobs",
                "In 1984, Apple released the Macintosh",
                "In 2007, Apple launched the iPhone",
                "In 2020, Apple transitioned to ARM processors"
            ]
        },
        {
            "name": "Mixed entities in 2007",
            "events": [
                "In 2007, Apple launched the iPhone",
                "In 2007, the financial crisis began brewing",
                "In 2007, Facebook opened to the public",
                "In 2007, Netflix started streaming service"
            ]
        }
    ]
    
    for group in test_groups:
        print(f"\n{group['name']}:")
        
        # Get embeddings
        with torch.no_grad():
            embeddings = model(group['events'])
            
        # Compute average pairwise similarity
        n = len(embeddings)
        total_sim = 0
        count = 0
        
        for i in range(n):
            for j in range(i+1, n):
                sim = torch.nn.functional.cosine_similarity(
                    embeddings[i:i+1], embeddings[j:j+1]
                ).item()
                total_sim += sim
                count += 1
                
        avg_sim = total_sim / count if count > 0 else 0
        print(f"Average similarity: {avg_sim:.4f}")
        
        # Show individual similarities
        for i, event in enumerate(group['events']):
            print(f" {i+1}. {event}")


def test_retrieval_quality(model: TemporalEmbeddingModel):
    """Test retrieval quality with temporal queries"""
    print("\nTesting Retrieval Quality")
    
    # Knowledge base
    kb_events = [
        "In 1969, Neil Armstrong walked on the moon",
        "In 1961, Yuri Gagarin became the first human in space",
        "In 1957, Sputnik was launched by the Soviet Union",
        "In 1990, Hubble Space Telescope was launched",
        "In 2021, James Webb Space Telescope was launched",
        "In 1492, Christopher Columbus reached the Americas",
        "In 1776, the United States declared independence",
        "In 1789, the French Revolution began",
        "In 1945, World War II ended",
        "In 1989, the Berlin Wall fell"
    ]
    
    # Queries
    queries = [
        "What happened in space exploration during the 1960s?",
        "Events related to telescopes in the late 20th century",
        "Major political changes in the late 1980s"
    ]
    
    # Get KB embeddings
    with torch.no_grad():
        kb_embeddings = model(kb_events)
        
    for query in queries:
        print(f"\nQuery: {query}")
        
        # Get query embedding
        with torch.no_grad():
            query_emb = model([query])
            
        # Compute similarities
        similarities = []
        for i, kb_emb in enumerate(kb_embeddings):
            sim = torch.nn.functional.cosine_similarity(
                query_emb, kb_emb.unsqueeze(0)
            ).item()
            similarities.append((sim, i))
            
        # Sort by similarity
        similarities.sort(reverse=True)
        
        # Show top 3 results
        print("Top 3 results:")
        for rank, (sim, idx) in enumerate(similarities[:3], 1):
            print(f" {rank}. {kb_events[idx]} (sim: {sim:.4f})")


def main():
    parser = argparse.ArgumentParser(description="Test trained temporal embeddings")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained model checkpoint")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to use")
    
    args = parser.parse_args()
    
    # Load model
    model = load_trained_model(args.model_path)
    model = model.to(args.device)
    
    print(f"Model loaded on {args.device}")
    
    # Run tests
    test_temporal_similarity(model)
    test_temporal_encoding(model)
    test_entity_temporal_consistency(model)
    test_retrieval_quality(model)
    
    print("\nTesting Complete")


if __name__ == "__main__":
    main()