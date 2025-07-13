#!/usr/bin/env python3
"""
Test enhanced query functionality with embeddings
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.kg.tkg_query_engine import TKGQueryEngine
from src.utils.dataset_loader import load_dataset
from src.embeddings.embedding_integration import EmbeddingIntegration
from src.embeddings.temporal_embeddings import TemporalEmbeddingConfig


def main():
    print("Testing Enhanced Query with Embeddings\n")
    
    # Load dataset
    print("Loading dataset")
    try:
        graph = load_dataset("MultiTQ")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return
    print(f"Loaded graph with {len(graph.nodes())} nodes and {len(graph.edges())} edges")
    
    # Create embedding integration
    print("\nInitialising embedding integration")
    embedding_config = TemporalEmbeddingConfig(
        model_name="all-MiniLM-L6-v2"
    )
    
    embedding_integration = EmbeddingIntegration(
        graph=graph,
        dataset_name="MultiTQ",
        config=embedding_config
    )
    
    # Initialise TKG Query Engine with enhanced scoring
    print("\nInitialising TKG Query Engine with enhanced scoring")
    engine = TKGQueryEngine(
        graph=graph,
        use_enhanced_scoring=True,
        embedding_integration=embedding_integration
    )
    
    # Test queries
    test_queries = [
        "When did Barack Obama become president?",
        "Who was the president in 2015?",
        "What happened in Washington in 2009?"
    ]
    
    print("\nTesting queries with enhanced scoring:\n")
    
    for query in test_queries:
        print(f"Query: {query}")
        try:
            result = engine.query(query, verbose=False)
            
            if result.paths:
                print(f"Found {len(result.paths)} paths")
                # Show top path
                path, metrics = result.paths[0]
                if hasattr(metrics, 'combined_score'):
                    print(f"Top path score: {metrics.combined_score:.3f}")
                    print(f"  - Reliability: {metrics.reliability_score:.3f}")
                    print(f"  - Embedding: {metrics.embedding_score:.3f}")
                    print(f"  - Confidence: {metrics.confidence:.3f}")
                else:
                    print(f"Top path score: {metrics:.3f}")
            else:
                print("No paths found")
                
        except Exception as e:
            print(f"Error: {e}")
        
        print()
    
    print("\nEnhanced query test complete")
    
    # Check if temporal adapter exists
    adapter_path = Path("./models/temporal_adapter")
    if adapter_path.exists():
        print(f"\nTemporal adapter found at: {adapter_path}")
    else:
        print("\nNo trained temporal adapter found")


if __name__ == "__main__":
    main()