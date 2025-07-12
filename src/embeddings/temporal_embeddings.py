"""
Temporal embeddings using pre-trained models
"""

import numpy as np
import torch
from typing import Dict, List, Optional
from pathlib import Path
import pickle
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass
import networkx as nx
from tqdm import tqdm


@dataclass
class TemporalEmbeddingConfig:
    """Configuration for temporal embeddings"""
    model_name: str = "all-MiniLM-L6-v2"  
    embedding_dim: int = 384  # Dimension of all-MiniLM-L6-v2
    cache_dir: Path = Path.home() / ".temporal_pathrag_cache" / "embeddings"
    batch_size: int = 512
    use_gpu: bool = True
    temporal_encoding_method: str = "sinusoidal"  # Options: sinusoidal, learned, none
    max_temporal_positions: int = 10000  # For sinusoidal encoding


class TemporalEmbeddings:
    """
    Temporal embeddings using pre-trained models
    """
    
    def __init__(self, config: Optional[TemporalEmbeddingConfig] = None):
        """Initialise the temporal embedding system"""
        self.config = config or TemporalEmbeddingConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.config.use_gpu else "cpu")
        
        # Initialise sentence transformer
        print(f"Loading pre-trained model: {self.config.model_name}")
        self.encoder = SentenceTransformer(self.config.model_name, device=self.device)
        
        # Create cache directory
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Caches for embeddings
        self.node_embeddings: Dict[str, np.ndarray] = {}
        self.relation_embeddings: Dict[str, np.ndarray] = {}
        self.temporal_embeddings: Dict[str, np.ndarray] = {}
        
    def get_cache_key(self, graph_id: str) -> str:
        """Generate a unique cache key for a graph"""
        return f"{graph_id}_{self.config.model_name}_{self.config.temporal_encoding_method}"
    
    def load_cached_embeddings(self, graph_id: str) -> bool:
        """Load embeddings from cache if available"""
        cache_file = self.config.cache_dir / f"{self.get_cache_key(graph_id)}.pkl"
        
        if cache_file.exists():
            try:
                print(f"Loading cached embeddings from {cache_file}")
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                    self.node_embeddings = data['nodes']
                    self.relation_embeddings = data['relations']
                    self.temporal_embeddings = data.get('temporal', {})
                print(f"Loaded {len(self._node_embeddings)} node embeddings from cache")
                return True
            except Exception as e:
                print(f"Failed to load cache: {e}")
        return False
    
    def save_embeddings_to_cache(self, graph_id: str):
        """Save embeddings to cache"""
        cache_file = self.config.cache_dir / f"{self.get_cache_key(graph_id)}.pkl"
        
        try:
            print(f"Saving embeddings to cache: {cache_file}")
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'nodes': self.node_embeddings,
                    'relations': self.relation_embeddings,
                    'temporal': self.temporal_embeddings
                }, f, protocol=pickle.HIGHEST_PROTOCOL)
            print("Embeddings saved successfully")
        except Exception as e:
            print(f"Failed to save cache: {e}")
    
    def precompute_graph_embeddings(self, graph: nx.DiGraph, graph_id: str, 
                                   force_recompute: bool = False) -> None:
        """
        Precompute embeddings for all nodes and edges in the graph
        """
        # Try to load from cache first
        if not force_recompute and self.load_cached_embeddings(graph_id):
            return
        
        print("Precomputing graph embeddings")
        
        # Extract unique nodes and relations
        nodes = list(graph.nodes())
        relations = set()
        timestamps = set()
        
        for _, _, data in graph.edges(data=True):
            if 'relation' in data:
                relations.add(data['relation'])
            if 'timestamps' in data:
                for ts in data['timestamps']:
                    timestamps.add(str(ts))
        
        relations = list(relations)
        timestamps = list(timestamps)
        
        # Compute node embeddings
        print(f"Computing embeddings for {len(nodes)} nodes")
        self.compute_node_embeddings(nodes)
        
        # Compute relation embeddings
        print(f"Computing embeddings for {len(relations)} relations")
        self.compute_relation_embeddings(relations)
        
        # Compute temporal embeddings
        if self.config.temporal_encoding_method != "none":
            print(f"Computing temporal embeddings for {len(timestamps)} timestamps")
            self.compute_temporal_embeddings(timestamps)
        
        # Save to cache
        self.save_embeddings_to_cache(graph_id)
        
    def compute_node_embeddings(self, nodes: List[str]) -> None:
        """Compute embeddings for nodes in batches"""
        self.node_embeddings.clear()
        
        # Process in batches
        for i in tqdm(range(0, len(nodes), self.config.batch_size), desc="Node embeddings"):
            batch = nodes[i:i + self.config.batch_size]
            
            # For entities, we can enhance the text representation
            texts = [self.enhance_entity_text(node) for node in batch]
            
            # Compute embeddings
            embeddings = self.encoder.encode(texts, convert_to_numpy=True, 
                                           show_progress_bar=False)
            
            # Store embeddings
            for node, embedding in zip(batch, embeddings):
                self.node_embeddings[node] = embedding
    
    def compute_relation_embeddings(self, relations: List[str]) -> None:
        """Compute embeddings for relations"""
        self.relation_embeddings.clear()
        
        if not relations:
            return
        
        # Enhance relation text representation
        texts = [self.enhance_relation_text(rel) for rel in relations]
        
        # Compute embeddings in batches
        for i in range(0, len(relations), self.config.batch_size):
            batch_texts = texts[i:i + self.config.batch_size]
            batch_relations = relations[i:i + self.config.batch_size]
            
            embeddings = self.encoder.encode(batch_texts, convert_to_numpy=True,
                                           show_progress_bar=False)
            
            for rel, embedding in zip(batch_relations, embeddings):
                self.relation_embeddings[rel] = embedding
    
    def compute_temporal_embeddings(self, timestamps: List[str]) -> None:
        """Compute temporal embeddings using sinusoidal encoding"""
        self.temporal_embeddings.clear()
        
        if self.config.temporal_encoding_method == "sinusoidal":
            # Use sinusoidal positional encoding for temporal information
            for ts in timestamps:
                self.temporal_embeddings[ts] = self.sinusoidal_temporal_encoding(ts)
        elif self.config.temporal_encoding_method == "learned":
            # Use the language model to encode temporal descriptions
            texts = [f"time: {ts}" for ts in timestamps]
            
            for i in range(0, len(timestamps), self.config.batch_size):
                batch_texts = texts[i:i + self.config.batch_size]
                batch_timestamps = timestamps[i:i + self.config.batch_size]
                
                embeddings = self.encoder.encode(batch_texts, convert_to_numpy=True,
                                               show_progress_bar=False)
                
                for ts, embedding in zip(batch_timestamps, embeddings):
                    self.temporal_embeddings[ts] = embedding
    
    def enhance_entity_text(self, entity: str) -> str:
        """Enhance entity text for better embedding"""
        # Simple enhancement - can be improved with entity types, descriptions, etc. which will do in future
        return f"entity: {entity}"
    
    def enhance_relation_text(self, relation: str) -> str:
        """Enhance relation text for better embedding"""
        # Convert underscores to spaces and add context
        enhanced = relation.replace('_', ' ').replace('-', ' ')
        return f"relation: {enhanced}"
    
    def sinusoidal_temporal_encoding(self, timestamp: str) -> np.ndarray:
        """
        Generate sinusoidal temporal encoding for a timestamp
        Similar to positional encoding in Transformers
        """
        # Convert timestamp to a numeric position
        # This is a simple approach - will be enhanced based on timestamp format
        try:
            # Try to extract year/date information
            if '-' in timestamp:
                parts = timestamp.split('-')
                year = int(parts[0]) if parts[0].isdigit() else 2000
                position = year - 1900  # Normalise to position
            else:
                position = hash(timestamp) % self.config.max_temporal_positions
        except:
            position = hash(timestamp) % self.config.max_temporal_positions
        
        # Generate sinusoidal encoding
        encoding = np.zeros(self.config.embedding_dim)
        for i in range(0, self.config.embedding_dim, 2):
            encoding[i] = np.sin(position / (10000 ** (i / self.config.embedding_dim)))
            if i + 1 < self.config.embedding_dim:
                encoding[i + 1] = np.cos(position / (10000 ** (i / self.config.embedding_dim)))
        
        return encoding
    
    def get_node_embedding(self, node: str) -> Optional[np.ndarray]:
        """Get embedding for a node"""
        if node in self.node_embeddings:
            return self.node_embeddings[node]
        
        # Compute off the cuff if not cached
        text = self.enhance_entity_text(node)
        embedding = self.encoder.encode(text, convert_to_numpy=True)
        self.node_embeddings[node] = embedding
        return embedding
    
    def get_relation_embedding(self, relation: str) -> Optional[np.ndarray]:
        """Get embedding for a relation"""
        if relation in self.relation_embeddings:
            return self.relation_embeddings[relation]
        
        # Compute off the cuff again if not cached
        text = self.enhance_relation_text(relation)
        embedding = self.encoder.encode(text, convert_to_numpy=True)
        self.relation_embeddings[relation] = embedding
        return embedding
    
    def get_temporal_embedding(self, timestamp: str) -> Optional[np.ndarray]:
        """Get temporal embedding"""
        ts_str = str(timestamp)
        
        if ts_str in self.temporal_embeddings:
            return self.temporal_embeddings[ts_str]
        
        # Compute off the cuff
        if self.config.temporal_encoding_method == "sinusoidal":
            embedding = self.sinusoidal_temporal_encoding(ts_str)
        elif self.config.temporal_encoding_method == "learned":
            text = f"time: {ts_str}"
            embedding = self.encoder.encode(text, convert_to_numpy=True)
        else:
            # No temporal encoding
            embedding = np.zeros(self.config.embedding_dim)
        
        self.temporal_embeddings[ts_str] = embedding
        return embedding
    
    def get_path_embedding(self, path_nodes: List[str], path_relations: List[str],
                          path_timestamps: List[str]) -> np.ndarray:
        """
        Get embedding for an entire path by aggregating node, relation, and temporal embeddings
        """
        embeddings = []
        
        # Add node embeddings
        for node in path_nodes:
            node_emb = self.get_node_embedding(node)
            if node_emb is not None:
                embeddings.append(node_emb)
        
        # Add relation embeddings
        for rel in path_relations:
            rel_emb = self.get_relation_embedding(rel)
            if rel_emb is not None:
                embeddings.append(rel_emb)
        
        # Add temporal embeddings
        if self.config.temporal_encoding_method != "none":
            for ts in path_timestamps:
                ts_emb = self.get_temporal_embedding(ts)
                if ts_emb is not None:
                    embeddings.append(ts_emb)
        
        if not embeddings:
            return np.zeros(self.config.embedding_dim)
        
        # Aggregate embeddings (mean pooling)
        path_embedding = np.mean(embeddings, axis=0)
        
        # Normalise
        norm = np.linalg.norm(path_embedding)
        if norm > 0:
            path_embedding = path_embedding / norm
        
        return path_embedding
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings"""
        # Ensure embeddings are normalised
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        embedding1 = embedding1 / norm1
        embedding2 = embedding2 / norm2
        
        return np.dot(embedding1, embedding2)
    
    def clear_cache(self):
        """Clear all cached embeddings"""
        self.node_embeddings.clear()
        self.relation_embeddings.clear()
        self.temporal_embeddings.clear()
        print("Cleared all cached embeddings from memory")


def create_embeddings(config: Optional[TemporalEmbeddingConfig] = None) -> TemporalEmbeddings:
    """Create temporal embeddings"""
    return TemporalEmbeddings(config)