"""
Temporal Embedding Retriever that integrates trained temporal embeddings
into the PathRAG retrieval pipeline
"""

import torch
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path
import logging
from transformers import AutoModel, AutoTokenizer
import json
from datetime import datetime
import networkx as nx

from ...training.train_temporal_embeddings import TemporalEmbeddingModel
from ..utils.graph_utils import safe_get_edge_data

logger = logging.getLogger(__name__)


class TemporalEmbeddingRetriever:
    """Retriever that uses trained temporal embeddings for path scoring"""
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        cache_embeddings: bool = True
    ):
        """
        Initialise the temporal embedding retriever
        """
        self.device = device
        self.cache_embeddings = cache_embeddings
        self.embedding_cache = {} if cache_embeddings else None
        
        # Load the trained model
        self.load_model(model_path)
        
    def load_model(self, model_path: str):
        """Load the trained temporal embedding model"""
        logger.info(f"Loading temporal embedding model from {model_path}")
        
        # Load model configuration
        config_path = Path(model_path) / "config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            # Default configuration
            config = {
                "base_model": "alibaba-damo/gte-Qwen2-1.5B-instruct",
                "temporal_dim": 128,
                "num_heads": 8,
                "use_lora": True
            }
        
        # Initialise model
        self.model = TemporalEmbeddingModel(
            model_name=config["base_model"],
            temporal_dim=config["temporal_dim"],
            num_heads=config["num_heads"],
            use_lora=config.get("use_lora", True)
        )
        
        # Load trained weights
        model_file = Path(model_path) / "pytorch_model.bin"
        if model_file.exists():
            state_dict = torch.load(model_file, map_location=self.device)
            self.model.load_state_dict(state_dict, strict=False)
            logger.info("Loaded trained model weights")
        else:
            logger.warning("No trained weights found, using base model")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Initialise tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config["base_model"])
        
    def encode_query(self, query: str, temporal_context: Optional[Dict] = None) -> torch.Tensor:
        """
        Encode a query with temporal context
        """
        # Check cache
        cache_key = f"query_{query}_{str(temporal_context)}"
        if self.cache_embeddings and cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
        
        # Tokenize
        inputs = self.tokenizer(
            query,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Extract temporal signals
        temporal_signals = self.extract_temporal_signals(query, temporal_context)
        
        # Encode
        with torch.no_grad():
            embeddings = self.model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                temporal_signals=temporal_signals
            )
        
        # Cache if enabled
        if self.cache_embeddings:
            self.embedding_cache[cache_key] = embeddings
        
        return embeddings
    
    def encode_path(
        self,
        path: List[Tuple[str, str, str]],
        graph: nx.Graph,
        include_context: bool = True
    ) -> torch.Tensor:
        """
        Encode a knowledge graph path
        """
        # Check cache
        cache_key = f"path_{str(path)}"
        if self.cache_embeddings and cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
        
        # Convert path to text representation
        path_text = self.path_to_text(path, graph)
        
        # Tokenize
        inputs = self.tokenizer(
            path_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Extract temporal signals from path
        temporal_signals = self.extract_path_temporal_signals(path, graph)
        
        # Encode
        with torch.no_grad():
            embeddings = self.model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                temporal_signals=temporal_signals
            )
        
        # Cache if enabled
        if self.cache_embeddings:
            self.embedding_cache[cache_key] = embeddings
        
        return embeddings
    
    def score_paths(
        self,
        query: str,
        paths: List[List[Tuple[str, str, str]]],
        graph: nx.Graph,
        temporal_context: Optional[Dict] = None,
        return_embeddings: bool = False
    ) -> List[float]:
        """
        Score paths based on their relevance to the query
        """
        # Encode query
        query_embedding = self.encode_query(query, temporal_context)
        
        scores = []
        embeddings = [] if return_embeddings else None
        
        # Score each path
        for path in paths:
            # Encode path
            path_embedding = self.encode_path(path, graph)
            
            # Compute similarity score
            score = torch.cosine_similarity(
                query_embedding.mean(dim=1),
                path_embedding.mean(dim=1)
            ).item()
            
            scores.append(score)
            
            if return_embeddings:
                embeddings.append({
                    'query': query_embedding.cpu().numpy(),
                    'path': path_embedding.cpu().numpy()
                })
        
        if return_embeddings:
            return scores, embeddings
        return scores
    
    def rerank_paths(
        self,
        query: str,
        paths: List[List[Tuple[str, str, str]]],
        graph: nx.Graph,
        temporal_context: Optional[Dict] = None,
        top_k: int = 10
    ) -> List[Tuple[List[Tuple[str, str, str]], float]]:
        """
        Rerank paths based on temporal relevance
        """
        # Score all paths
        scores = self.score_paths(query, paths, graph, temporal_context)
        
        # Sort by score
        path_scores = list(zip(paths, scores))
        path_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top k
        return path_scores[:top_k]
    
    def extract_temporal_signals(
        self,
        text: str,
        temporal_context: Optional[Dict] = None
    ) -> torch.Tensor:
        """
        Extract temporal signals from text and context
        """
        # Simple temporal signal extraction
        # In future, will use more sophisticated NLP
        temporal_keywords = [
            'before', 'after', 'during', 'since', 'until',
            'when', 'while', 'year', 'month', 'day'
        ]
        
        # Count temporal keywords
        text_lower = text.lower()
        temporal_count = sum(1 for kw in temporal_keywords if kw in text_lower)
        
        # Create signal vector
        signal = torch.zeros(1, 128).to(self.device)
        signal[0, 0] = temporal_count / len(temporal_keywords)
        
        # Add context signals if available
        if temporal_context:
            if 'start_time' in temporal_context:
                signal[0, 1] = 1.0
            if 'end_time' in temporal_context:
                signal[0, 2] = 1.0
        
        return signal
    
    def extract_path_temporal_signals(
        self,
        path: List[Tuple[str, str, str]],
        graph: nx.Graph
    ) -> torch.Tensor:
        """
        Extract temporal signals from a path
        """
        signal = torch.zeros(1, 128).to(self.device)
        
        # Check for temporal edges
        temporal_edge_count = 0
        for source, rel, target in path:
            if graph.has_edge(source, target):
                edge_data = safe_get_edge_data(graph, source, target)
                if 'te' in edge_data:  # Temporal edge
                    temporal_edge_count += 1
        
        signal[0, 0] = temporal_edge_count / max(len(path), 1)
        
        # Check for temporal nodes
        temporal_node_count = 0
        nodes = set()
        for source, _, target in path:
            nodes.add(source)
            nodes.add(target)
        
        for node in nodes:
            if node in graph and 'tv' in graph.nodes[node]:  # Temporal validity
                temporal_node_count += 1
        
        signal[0, 1] = temporal_node_count / max(len(nodes), 1)
        
        return signal
    
    def path_to_text(
        self,
        path: List[Tuple[str, str, str]],
        graph: nx.Graph
    ) -> str:
        """
        Convert a path to text representation
        """
        text_parts = []
        
        for source, rel, target in path:
            # Add relation text
            text_parts.append(f"{source} {rel} {target}")
            
            # Add temporal context if available
            if graph.has_edge(source, target):
                edge_data = safe_get_edge_data(graph, source, target)
                if 'te' in edge_data:
                    start, end = edge_data['te']
                    if start and end:
                        text_parts.append(f"[from {start} to {end}]")
                    elif start:
                        text_parts.append(f"[since {start}]")
                    elif end:
                        text_parts.append(f"[until {end}]")
        
        return " ".join(text_parts)
    
    def clear_cache(self):
        """Clear the embedding cache"""
        if self.embedding_cache:
            self.embedding_cache.clear()
            logger.info("Cleared embedding cache")