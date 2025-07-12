"""
Integration for temporal embeddings with the existing Temporal PathRAG system
"""

import numpy as np
from typing import List, Optional
import networkx as nx

from .temporal_embeddings import (
    TemporalEmbeddingConfig,
    TemporalEmbeddings
)
from ..kg.models import Path


class EmbeddingIntegration:
    """
    Integrates temporal embeddings with the Temporal PathRAG system
    Provides optimised semantic similarity calculations for path scoring
    """
    
    def __init__(self, graph: nx.DiGraph, dataset_name: str, 
                 config: Optional[TemporalEmbeddingConfig] = None):
        """
        Initialise embedding integration
        """
        self.graph = graph
        self.dataset_name = dataset_name
        self.embeddings = TemporalEmbeddings(config)
        
        # Precompute embeddings for the entire graph
        self.embeddings.precompute_graph_embeddings(graph, dataset_name)
        
    def compute_path_semantic_similarity(self, path1: Path, 
                                       path2: Path) -> float:
        """
        Compute semantic similarity between two temporal paths using pre-computed embeddings
        """
        # Extract path components
        nodes1 = [edge[0] for edge in path1.edges] + [path1.edges[-1][1] if path1.edges else ""]
        relations1 = [edge[2]['relation'] for edge in path1.edges if 'relation' in edge[2]]
        timestamps1 = []
        for edge in path1.edges:
            if 'timestamps' in edge[2]:
                timestamps1.extend([str(ts) for ts in edge[2]['timestamps']])
        
        nodes2 = [edge[0] for edge in path2.edges] + [path2.edges[-1][1] if path2.edges else ""]
        relations2 = [edge[2]['relation'] for edge in path2.edges if 'relation' in edge[2]]
        timestamps2 = []
        for edge in path2.edges:
            if 'timestamps' in edge[2]:
                timestamps2.extend([str(ts) for ts in edge[2]['timestamps']])
        
        # Get aggregated path embeddings
        embedding1 = self.embeddings.get_path_embedding(nodes1, relations1, timestamps1)
        embedding2 = self.embeddings.get_path_embedding(nodes2, relations2, timestamps2)
        
        # Compute similarity
        return self.embeddings.compute_similarity(embedding1, embedding2)
    
    def compute_query_path_similarity(self, query: str, path: Path) -> float:
        """
        Compute semantic similarity between a query and a temporal path
        """
        # Get query embedding
        query_embedding = self.embeddings.encoder.encode(query, convert_to_numpy=True)
        
        # Get path components
        nodes = [edge[0] for edge in path.edges] + [path.edges[-1][1] if path.edges else ""]
        relations = [edge[2]['relation'] for edge in path.edges if 'relation' in edge[2]]
        timestamps = []
        for edge in path.edges:
            if 'timestamps' in edge[2]:
                timestamps.extend([str(ts) for ts in edge[2]['timestamps']])
        
        # Get path embedding
        path_embedding = self.embeddings.get_path_embedding(nodes, relations, timestamps)
        
        # Compute similarity
        return self.embeddings.compute_similarity(query_embedding, path_embedding)
    
    def batch_compute_path_similarities(self, paths: List[Path], 
                                      reference_path: Optional[Path] = None,
                                      query: Optional[str] = None) -> List[float]:
        """
        Efficiently compute similarities for multiple paths
        """
        if reference_path is None and query is None:
            raise ValueError("Either reference_path or query must be provided")
        
        # Get reference embedding
        if query:
            reference_embedding = self.embeddings.encoder.encode(query, convert_to_numpy=True)
        else:
            # Extract reference path components
            ref_nodes = [edge[0] for edge in reference_path.edges] + \
                       [reference_path.edges[-1][1] if reference_path.edges else ""]
            ref_relations = [edge[2]['relation'] for edge in reference_path.edges 
                           if 'relation' in edge[2]]
            ref_timestamps = []
            for edge in reference_path.edges:
                if 'timestamps' in edge[2]:
                    ref_timestamps.extend([str(ts) for ts in edge[2]['timestamps']])
            
            reference_embedding = self.embeddings.get_path_embedding(
                ref_nodes, ref_relations, ref_timestamps
            )
        
        # Compute similarities for all paths
        similarities = []
        for path in paths:
            # Extract path components
            nodes = [edge[0] for edge in path.edges] + [path.edges[-1][1] if path.edges else ""]
            relations = [edge[2]['relation'] for edge in path.edges if 'relation' in edge[2]]
            timestamps = []
            for edge in path.edges:
                if 'timestamps' in edge[2]:
                    timestamps.extend([str(ts) for ts in edge[2]['timestamps']])
            
            # Get path embedding
            path_embedding = self.embeddings.get_path_embedding(nodes, relations, timestamps)
            
            # Compute similarity
            similarity = self.embeddings.compute_similarity(reference_embedding, path_embedding)
            similarities.append(similarity)
        
        return similarities
    
    def get_node_neighbourhood_embedding(self, node: str, k_hop: int = 2) -> np.ndarray:
        """
        Get embedding for a node's k-hop neighbourhood
        """
        # Get k-hop neighbours
        neighbours = {node}
        current_layer = {node}
        
        for _ in range(k_hop):
            next_layer = set()
            for n in current_layer:
                # Get successors and predecessors
                next_layer.update(self.graph.successors(n))
                next_layer.update(self.graph.predecessors(n))
            neighbours.update(next_layer)
            current_layer = next_layer
        
        # Collect embeddings
        embeddings = []
        for neighbour in neighbours:
            emb = self.embeddings.get_node_embedding(neighbour)
            if emb is not None:
                embeddings.append(emb)
        
        if not embeddings:
            return np.zeros(self.embeddings.config.embedding_dim)
        
        # Aggregate with distance-based weighting
        # Closer nodes get higher weights
        weighted_embeddings = []
        for neighbour in neighbours:
            try:
                distance = nx.shortest_path_length(self.graph, node, neighbour)
                weight = 1.0 / (1.0 + distance)  # Inverse distance weighting
                emb = self.embeddings.get_node_embedding(neighbour)
                if emb is not None:
                    weighted_embeddings.append(weight * emb)
            except:
                # No path exists
                pass
        
        if not weighted_embeddings:
            return np.mean(embeddings, axis=0)
        
        # Weighted average
        neighbourhood_embedding = np.sum(weighted_embeddings, axis=0)
        neighbourhood_embedding = neighbourhood_embedding / np.sum([1.0 / (1.0 + i) 
                                                                 for i in range(len(weighted_embeddings))])
        
        # Normalise
        norm = np.linalg.norm(neighbourhood_embedding)
        if norm > 0:
            neighbourhood_embedding = neighbourhood_embedding / norm
        
        return neighbourhood_embedding
    
    def clear_cache(self):
        """Clear embedding cache"""
        self.embeddings.clear_cache()


def create_embedding_integration(graph: nx.DiGraph, dataset_name: str,
                               use_gpu: bool = True,
                               temporal_encoding: str = "sinusoidal") -> EmbeddingIntegration:
    """
    Create embedding integration
    """
    config = TemporalEmbeddingConfig(
        use_gpu=use_gpu,
        temporal_encoding_method=temporal_encoding
    )
    
    return EmbeddingIntegration(graph, dataset_name, config)