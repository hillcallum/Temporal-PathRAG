#!/usr/bin/env python3
"""
Integrate trained temporal embeddings with the existing Temporal PathRAG system
"""

import sys
import argparse
from pathlib import Path
import torch
import logging
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.training.train_temporal_embeddings import TemporalEmbeddingModel, TrainingConfig
from src.embeddings.temporal_embeddings import (
    TemporalEmbeddings, 
    TemporalEmbeddingConfig
)
from src.utils.dataset_loader import load_dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TrainedTemporalEmbeddings(TemporalEmbeddings):
    """
    Enhanced temporal embeddings using trained model
    """
    
    def __init__(self, model_path: str, config: TemporalEmbeddingConfig = None):
        """Initialise with trained model"""
        super().__init__(config)
        
        # Load trained model
        logger.info(f"Loading trained model from {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Create training config from checkpoint
        training_config_dict = checkpoint.get('config', {})
        training_config = TrainingConfig(
            **{k: v for k, v in training_config_dict.items() 
               if k in TrainingConfig.__dataclass_fields__}
        )
        
        # Create and load model
        self.trained_model = TemporalEmbeddingModel(training_config)
        self.trained_model.load_state_dict(checkpoint['model_state_dict'])
        self.trained_model.to(self.device)
        self.trained_model.eval()
        
        logger.info("Trained model loaded successfully")
        
        # Update encoder to use the trained model's encoder
        self.encoder = self.trained_model.encoder
        
    def enhance_entity_text(self, entity: str) -> str:
        """Enhanced entity text representation"""
        # Keep it simple for trained model
        return entity
    
    def enhance_relation_text(self, relation: str) -> str:
        """Enhanced relation text representation"""
        return relation.replace('_', ' ').replace('-', ' ')
    
    def get_path_embedding(self, path_nodes, path_relations, path_timestamps):
        """
        Get embedding for a path using the trained model
        """
        # Create text representation of the path
        path_texts = []
        
        for i in range(len(path_relations)):
            # Create temporal quadruplet text
            if i < len(path_timestamps) and path_timestamps[i]:
                text = f"At {path_timestamps[i]}, {path_nodes[i]} {self.enhance_relation_text(path_relations[i])} {path_nodes[i+1]}"
            else:
                text = f"{path_nodes[i]} {self.enhance_relation_text(path_relations[i])} {path_nodes[i+1]}"
            path_texts.append(text)
            
        if not path_texts:
            return super().get_path_embedding(path_nodes, path_relations, path_timestamps)
            
        # Get embeddings from trained model
        with torch.no_grad():
            embeddings = self.trained_model(path_texts)
            
        # Average pool the embeddings
        path_embedding = embeddings.mean(dim=0).cpu().numpy()
        
        # Normalise
        norm = np.linalg.norm(path_embedding)
        if norm > 0:
            path_embedding = path_embedding / norm
            
        return path_embedding


def compare_embeddings(dataset_name: str, model_path: str = None):
    """Compare original vs trained embeddings"""
    logger.info(f"Comparing embeddings on {dataset_name}")
    
    # Load dataset
    dataset = load_dataset(dataset_name)
    
    # Create both embedding systems
    config = TemporalEmbeddingConfig()
    
    logger.info("Creating original embeddings")
    original_embeddings = TemporalEmbeddings(config)
    original_embeddings.precompute_graph_embeddings(dataset.graph, f"{dataset_name}_original")
    
    if model_path:
        logger.info("Creating trained embeddings")
        trained_embeddings = TrainedTemporalEmbeddings(model_path, config)
        trained_embeddings.precompute_graph_embeddings(dataset.graph, f"{dataset_name}_trained")
    
    # Test on some example paths
    test_paths = [
        {
            "nodes": ["Barack_Obama", "United_States"],
            "relations": ["president_of"],
            "timestamps": ["2009"]
        },
        {
            "nodes": ["World_War_II", "1945", "end"],
            "relations": ["occurred_in", "event"],
            "timestamps": ["1945", "1945"]
        }
    ]
    
    for i, path in enumerate(test_paths):
        logger.info(f"\nTest path {i+1}:")
        logger.info(f"Nodes: {path['nodes']}")
        logger.info(f"Relations: {path['relations']}")
        logger.info(f"Timestamps: {path['timestamps']}")
        
        # Get embeddings
        orig_emb = original_embeddings.get_path_embedding(
            path['nodes'], path['relations'], path['timestamps']
        )
        
        if model_path:
            trained_emb = trained_embeddings.get_path_embedding(
                path['nodes'], path['relations'], path['timestamps']
            )
            
            # Compare
            similarity = original_embeddings.compute_similarity(orig_emb, trained_emb)
            logger.info(f"Similarity between original and trained: {similarity:.4f}")


def create_enhanced_config(model_path: str, output_path: str):
    """Create configuration file for using trained embeddings"""
    config = {
        "embeddings": {
            "type": "trained",
            "model_path": model_path,
            "base_config": {
                "model_name": "all-MiniLM-L6-v2",
                "embedding_dim": 384,
                "batch_size": 512,
                "use_gpu": True,
                "temporal_encoding_method": "learned"
            }
        },
        "integration": {
            "use_for_path_scoring": True,
            "use_for_retrieval": True,
            "similarity_weight": 0.4
        }
    }
    
    import json
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
        
    logger.info(f"Created configuration at {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Integrate trained embeddings")
    
    parser.add_argument("--model_path", type=str, help="Path to trained model")
    parser.add_argument("--dataset", type=str, default="MultiTQ", help="Dataset to test on")
    parser.add_argument("--create_config", type=str, help="Create config file at this path")
    parser.add_argument("--compare", action="store_true", help="Compare with original embeddings")
    
    args = parser.parse_args()
    
    if args.create_config and args.model_path:
        create_enhanced_config(args.model_path, args.create_config)
        
    if args.compare:
        compare_embeddings(args.dataset, args.model_path)
        
    if not args.create_config and not args.compare:
        logger.info("Please specify --create_config or --compare")


if __name__ == "__main__":
    main()