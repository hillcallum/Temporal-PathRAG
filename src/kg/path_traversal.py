import networkx as nx
from collections import defaultdict, deque
from typing import List, Dict, Optional, Tuple
import heapq
import torch
import numpy as np

from .models import TemporalPathRAGNode, TemporalPathRAGEdge, Path
from .temporal_scoring import TemporalWeightingFunction, TemporalPathRanker, TemporalPath, TemporalRelevanceMode
from .temporal_flow_pruning import TemporalFlowPruning

class TemporalPathTraversal:
    """
    GPU-accelerated Temporal PathRAG path traversal implementation
    """
    
    def __init__(self, graph: nx.DiGraph, device: torch.device = None, 
                 temporal_mode: TemporalRelevanceMode = TemporalRelevanceMode.EXPONENTIAL_DECAY):
        self.graph = graph
        self.path_cache: Dict[str, List[Path]] = {}
        
        # Initialise temporal weighting system
        self.temporal_weighting = TemporalWeightingFunction(
            decay_rate=0.1,
            temporal_window=365,  # 1 year window
            chronological_weight=0.3,
            proximity_weight=0.4, 
            consistency_weight=0.3
        )
        self.temporal_ranker = TemporalPathRanker(self.temporal_weighting)
        self.temporal_mode = temporal_mode
        
        # Initialise temporal flow pruning system
        self.temporal_flow_pruning = TemporalFlowPruning(
            temporal_weighting=self.temporal_weighting,
            temporal_mode=temporal_mode,
            alpha=0.1,  # Temporal decay rate alpha
            base_theta=1.0  # Base pruning threshold theta
        )
        
        # Setup device for GPU acceleration
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        print(f"Temporal PathRAG using device: {self.device}")
        
        # Initialise sentence transformer for semantic similarity (GPU-accelerated)
        self.init_sentence_transformer()
        
    def get_gpu_memory_usage(self) -> Dict[str, float]:
        """
        Get current GPU memory usage for monitoring
        """
        if self.device.type == 'cuda' and torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(self.device) / 1e9
            reserved = torch.cuda.memory_reserved(self.device) / 1e9
            total = torch.cuda.get_device_properties(self.device).total_memory / 1e9
            
            return {
                'allocated_gb': allocated,
                'reserved_gb': reserved,
                'total_gb': total,
                'free_gb': total - reserved,
                'utilisation_percent': (reserved / total) * 100
            }
        else:
            return {'device': 'cpu', 'message': 'GPU not available'}
    
    def cleanup_gpu_memory(self):
        """
        Clean up GPU memory cache
        """
        if self.device.type == 'cuda' and torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("GPU memory cache cleared")
        
    def init_sentence_transformer(self):
        """Initialise sentence transformer for GPU-accelerated semantic similarity"""
        try:
            from sentence_transformers import SentenceTransformer
            model_name = 'all-MiniLM-L6-v2'  
            print(f"Loading sentence transformer: {model_name}")
            self.sentence_transformer = SentenceTransformer(model_name, device=self.device)
            self.use_gpu_embeddings = True
            print(f"Successfully loaded {model_name} on {self.device}")
        except ImportError:
            print("sentence-transformers not available, using rule-based similarity")
            self.sentence_transformer = None
            self.use_gpu_embeddings = False
        except Exception as e:
            print(f"Error loading sentence transformer: {e}")
            self.sentence_transformer = None
            self.use_gpu_embeddings = False
    
    def find_paths(self, 
                   source_id: str, 
                   target_id: str, 
                   max_depth: int = 3, 
                   max_paths: int = 10,
                   query_time: str = None) -> List[Path]:
        """
        Find paths between source and target nodes using BFS with temporal awareness
        """
        # Check cache
        cache_key = f"{source_id}->{target_id}-{max_depth}-{max_paths}"
        if cache_key in self.path_cache:
            cached_paths = self.path_cache[cache_key]
            return self.score_paths(cached_paths, query_time)
        
        if not self.graph.has_node(source_id) or not self.graph.has_node(target_id):
            print(f"Source {source_id} or target {target_id} not found in graph")
            return []
        
        # BFS to find all paths
        all_paths = []
        queue = deque([(source_id, [source_id], [])])
        visited_paths = set()
        
        while queue and len(all_paths) < max_paths * 3:  # Get more paths for better filtering
            current_node, path_nodes, path_edges = queue.popleft()
            
            # Check if we've reached target
            if current_node == target_id and len(path_nodes) > 1:
                path = self.construct_path_from_traversal(path_nodes, path_edges)
                if path:
                    all_paths.append(path)
                continue
            
            # Check depth limit
            if len(path_nodes) >= max_depth + 1:
                continue
            
            # Explore neighbours
            if current_node in self.graph:
                for neighbour in self.graph.neighbors(current_node):
                    new_path_nodes = path_nodes + [neighbour]
                    path_signature = tuple(new_path_nodes)
                    
                    if path_signature not in visited_paths and neighbour not in path_nodes:
                        visited_paths.add(path_signature)
                        
                        # Get edge data
                        edge_data = self.graph.get_edge_data(current_node, neighbour, {})
                        new_path_edges = path_edges + [(current_node, neighbour, edge_data)]
                        
                        queue.append((neighbour, new_path_nodes, new_path_edges))
        
        # Score and cache results
        scored_paths = self.score_paths(all_paths, query_time)
        self.path_cache[cache_key] = scored_paths
        
        return scored_paths[:max_paths]
    
    def dijkstra_paths(self, 
                       source: str, 
                       target: str, 
                       max_hops: int) -> List[Path]:
        """
        Modified Dijkstra's algorithm to find multiple paths with cost-based prioritisation
        """
        # Priority queue: (cost, path_nodes, path_edges)
        pq = [(0.0, [source], [])]
        visited_paths = set()
        found_paths = []
        
        while pq and len(found_paths) < 50:  
            cost, path_nodes, path_edges = heapq.heappop(pq)
            
            # Convert to tuple for hashing
            path_signature = tuple(path_nodes)
            if path_signature in visited_paths:
                continue
            visited_paths.add(path_signature)
            
            current_node = path_nodes[-1]
            
            # Check if we reached target
            if current_node == target and len(path_nodes) > 1:
                path = self.construct_path_from_traversal(path_nodes, path_edges)
                if path:
                    found_paths.append(path)
                continue
            
            # Explore neighbours if within hop limit
            if len(path_nodes) < max_hops + 1:
                neighbours = self.get_neighbours(current_node)
                
                for neighbour_id, edge_data in neighbours:
                    if neighbour_id not in path_nodes:  # Avoid cycles
                        edge_cost = self.calculate_edge_cost(edge_data)
                        new_cost = cost + edge_cost
                        new_path_nodes = path_nodes + [neighbour_id]
                        new_path_edges = path_edges + [edge_data]
                        
                        heapq.heappush(pq, (new_cost, new_path_nodes, new_path_edges))
        
        return found_paths
    
    def get_neighbours(self, node_id: str) -> List[Tuple[str, Dict]]:
        """Get neighbouring nodes and edge data"""
        neighbours = []
        
        if self.graph.has_node(node_id):
            for neighbour in self.graph.neighbors(node_id):
                edge_data = self.graph.get_edge_data(node_id, neighbour, {})
                neighbours.append((neighbour, edge_data))
        
        return neighbours
    
    def calculate_edge_cost(self, edge_data: Dict) -> float:
        """Calculate cost for edge traversal in Dijkstra"""
        # Lower weight = higher cost (invert for shortest path)
        base_cost = 1.0 / max(edge_data.get('weight', 1.0), 0.1)
        
        # Add temporal penalty if edge is very old or future
        if 'timestamp' in edge_data:
            try:
                from datetime import datetime
                edge_time = datetime.fromisoformat(edge_data['timestamp'].replace('T', ' '))
                current_time = datetime.now()
                time_diff_years = abs((current_time - edge_time).days) / 365.25
                
                # Add cost for very old or future events
                temporal_penalty = min(time_diff_years * 0.1, 1.0)
                base_cost += temporal_penalty
            except (ValueError, TypeError):
                pass
        
        return base_cost
    
    def construct_path_from_traversal(self, node_ids: List[str], edge_data: List) -> Optional[Path]:
        """Construct a Path object from traversal results"""
        try:
            path = Path()
            
            # Add nodes
            for node_id in node_ids:
                if self.graph.has_node(node_id):
                    node_data = self.graph.nodes[node_id]
                    node = TemporalPathRAGNode(
                        id=node_id,
                        entity_type=node_data.get('entity_type', 'Unknown'),
                        name=node_data.get('name', node_id),
                        description=node_data.get('description', ''),
                        properties=node_data
                    )
                    path.add_node(node)
                else:
                    return None
            
            # Add edges - handle different edge_data formats
            for i, edge_info in enumerate(edge_data):
                if isinstance(edge_info, tuple) and len(edge_info) == 3:
                    # Format: (source, target, data)
                    source, target, data = edge_info
                elif isinstance(edge_info, dict):
                    # Format: edge data dict, infer source/target from node sequence
                    data = edge_info
                    if i < len(node_ids) - 1:
                        source = node_ids[i]
                        target = node_ids[i + 1]
                    else:
                        continue
                else:
                    continue
                
                edge = TemporalPathRAGEdge(
                    source_id=source,
                    target_id=target,
                    relation_type=data.get('relation_type', 'related_to'),
                    weight=data.get('weight', 1.0),
                    description=data.get('description', ''),
                    timestamp=data.get('timestamp'),
                    flow_capacity=data.get('flow_capacity', 1.0),
                    properties=data
                )
                path.add_edge(edge)
            
            return path
            
        except Exception as e:
            print(f"Error constructing path: {e}")
            return None
    
    def score_paths(self, paths: List[Path], query_time: str = None) -> List[Path]:
        """
        Enhanced path scoring with temporal weighting integration
        Combines PathRAG's structural flow with sophisticated temporal scoring
        """
        if not paths:
            return paths
            
        # Use current time if no query time provided
        if query_time is None:
            from datetime import datetime
            query_time = datetime.now().isoformat()
            
        # Pre-calculate all basic scores 
        length_penalties = []
        edge_qualities = []
        basic_temporal_scores = []
        
        for path in paths:
            # Basic scoring considering path length and edge weights
            length_penalty = 1.0 / (len(path.nodes) + 1)
            length_penalties.append(length_penalty)
            
            edge_quality = 1.0
            if path.edges:
                edge_weights = [edge.weight for edge in path.edges]
                edge_quality = sum(edge_weights) / len(edge_weights)
            edge_qualities.append(edge_quality)
            
            # Basic temporal constraint scoring (legacy compatibility)
            basic_temporal_score = self.calculate_temporal_coherence(path)
            basic_temporal_scores.append(basic_temporal_score)
        
        # GPU-accelerated batch semantic similarity calculation
        semantic_scores = self.batch_calculate_semantic_similarity(paths)
        
        # Convert to TemporalPath objects and apply enhanced temporal scoring
        temporal_paths = []
        for i, path in enumerate(paths):
            # Extract timestamps from path edges
            timestamps = []
            edges_with_timestamps = []
            
            for edge in path.edges:
                # Check if edge has timestamp information
                if hasattr(edge, 'timestamp') and edge.timestamp:
                    timestamps.append(edge.timestamp)
                    # Create edge tuple for TemporalPath
                    edges_with_timestamps.append((
                        edge.source_id, edge.relation_type, 
                        edge.target_id, edge.timestamp
                    ))
                else:
                    # Use a default timestamp if none available (current approach fallback)
                    timestamps.append(query_time)
                    edges_with_timestamps.append((
                        edge.source_id, edge.relation_type,
                        edge.target_id, query_time
                    ))
            
            # Calculate original PathRAG score (structural component)
            original_score = (length_penalties[i] * 0.3 + 
                            edge_qualities[i] * 0.4 + 
                            semantic_scores[i] * 0.2 + 
                            basic_temporal_scores[i] * 0.1)
            
            # Create TemporalPath for enhanced scoring
            temporal_path = TemporalPath(
                nodes=[node.id for node in path.nodes],
                edges=edges_with_timestamps,
                timestamps=timestamps,
                original_score=original_score
            )
            
            temporal_paths.append((path, temporal_path))
        
        # Apply enhanced temporal scoring
        enhanced_scores = []
        for path, temporal_path in temporal_paths:
            enhanced_score = self.temporal_weighting.enhanced_reliability_score(
                temporal_path, query_time, temporal_path.original_score
            )
            enhanced_scores.append(enhanced_score)
            path.score = enhanced_score  # Update path score
        
        return sorted(paths, key=lambda p: p.score, reverse=True)
    
    def calculate_semantic_similarity(self, path: Path) -> float:
        """
        Calculate semantic similarity score using GPU-accelerated embeddings and rule-based patterns
        """
        if not path.nodes or not path.edges:
            return 0.5  # neutral score
        
        if self.use_gpu_embeddings and self.sentence_transformer:
            return self.gpu_semantic_similarity(path)
        else:
            return self.rule_based_semantic_similarity(path)
    
    def gpu_semantic_similarity(self, path: Path) -> float:
        """GPU-accelerated semantic similarity using sentence transformers"""
        try:
            # Extract textual content from path
            text_chunks = []
            for node in path.nodes:
                text_chunks.append(node.tv if hasattr(node, 'tv') and node.tv else node.name)
            for edge in path.edges:
                text_chunks.append(edge.te if hasattr(edge, 'te') and edge.te else edge.relation_type)
            
            if len(text_chunks) < 2:
                return 0.5
            
            # Calculate embeddings
            embeddings = self.sentence_transformer.encode(text_chunks, convert_to_tensor=True)
            
            # Calculate pairwise similarities
            similarities = []
            for i in range(len(embeddings) - 1):
                sim = torch.cosine_similarity(embeddings[i].unsqueeze(0), embeddings[i+1].unsqueeze(0))
                similarities.append(sim.item())
            
            return np.mean(similarities) if similarities else 0.5
            
        except Exception as e:
            print(f"GPU semantic similarity error: {e}")
            return self.rule_based_semantic_similarity(path)
    
    def rule_based_semantic_similarity(self, path: Path) -> float:
        """Rule-based semantic similarity fallback"""
        try:
            score = 0.5  # base score
            
            # Check for common entity types
            entity_types = [node.entity_type for node in path.nodes if hasattr(node, 'entity_type')]
            if len(set(entity_types)) < len(entity_types):  # repeated types
                score += 0.1
            
            # Check for meaningful relation types
            meaningful_relations = ['worked_at', 'born_in', 'led_project', 'graduated_from', 'founded']
            relation_types = [edge.relation_type for edge in path.edges]
            meaningful_count = sum(1 for rel in relation_types if rel in meaningful_relations)
            score += (meaningful_count / len(relation_types)) * 0.2 if relation_types else 0
            
            # Path length bonus (shorter paths often more coherent)
            length_bonus = max(0, 0.2 - len(path.nodes) * 0.05)
            score += length_bonus
            
            return min(1.0, max(0.0, score))
            
        except Exception:
            return 0.5
    
    def batch_calculate_semantic_similarity(self, paths: List[Path]) -> List[float]:
        """Calculate semantic similarity for multiple paths efficiently"""
        if self.use_gpu_embeddings and self.sentence_transformer:
            return self.batch_gpu_semantic_similarity(paths)
        else:
            return [self.calculate_semantic_similarity(path) for path in paths]
    
    def batch_gpu_semantic_similarity(self, paths: List[Path]) -> List[float]:
        """
        GPU-accelerated batch semantic similarity calculation
        """
        try:
            # Extract all path texts
            path_texts = []
            reference_queries = []
            
            for path in paths:
                # Extract semantic content from the path
                path_text = path.path_text if hasattr(path, 'path_text') and path.path_text else ""
                if not path_text:
                    # Fallback: construct text from nodes and edges
                    node_texts = [f"{node.name}: {getattr(node, 'description', '')}" for node in path.nodes]
                    edge_texts = [edge.relation_type for edge in path.edges]
                    path_text = " -> ".join([f"{node_texts[i]} --[{edge_texts[i]}]-->" if i < len(edge_texts) else node_texts[i] 
                                           for i in range(len(node_texts))])
                
                # Create reference query
                source_context = f"{path.nodes[0].name} {getattr(path.nodes[0], 'description', '')}"
                target_context = f"{path.nodes[-1].name} {getattr(path.nodes[-1], 'description', '')}"
                reference_query = f"How is {source_context} related to {target_context}?"
                
                path_texts.append(path_text)
                reference_queries.append(reference_query)
            
            # Batch encode all texts using GPU
            with torch.no_grad():
                # Process in smaller batches to avoid memory issues
                batch_size = 32
                all_similarities = []
                
                for i in range(0, len(paths), batch_size):
                    batch_paths = path_texts[i:i+batch_size]
                    batch_queries = reference_queries[i:i+batch_size]
                    
                    # Encode batch
                    path_embeddings = self.sentence_transformer.encode(
                        batch_paths, 
                        convert_to_tensor=True,
                        device=self.device,
                        batch_size=len(batch_paths)
                    )
                    query_embeddings = self.sentence_transformer.encode(
                        batch_queries, 
                        convert_to_tensor=True,
                        device=self.device,
                        batch_size=len(batch_queries)
                    )
                    
                    # Calculate cosine similarities for the batch
                    similarities = torch.nn.functional.cosine_similarity(
                        path_embeddings, query_embeddings, dim=1
                    ).cpu().numpy()
                    
                    # Normalise to [0, 1] range
                    normalised_similarities = (similarities + 1) / 2
                    all_similarities.extend(normalised_similarities)
                
                # Apply path quality bonuses
                final_scores = []
                for i, path in enumerate(paths):
                    quality_bonus = self.calculate_path_quality_bonus(path)
                    # 70% semantic similarity + 30% structural quality
                    final_score = 0.7 * all_similarities[i] + 0.3 * quality_bonus
                    final_scores.append(min(final_score, 1.0))
                
                return final_scores
                
        except Exception as e:
            print(f"Warning: Batch GPU semantic similarity failed ({e}), falling back to individual")
            return [self.calculate_semantic_similarity(path) for path in paths]
    
    def calculate_path_quality_bonus(self, path: Path) -> float:
        """Calculate path quality bonus based on structural features"""
        try:
            score = 0.5  # base score
            
            # Length bonus (shorter paths often more relevant)
            length_bonus = max(0, 0.3 - len(path.nodes) * 0.1)
            score += length_bonus
            
            # Edge weight bonus
            if path.edges:
                avg_weight = sum(edge.weight for edge in path.edges) / len(path.edges)
                weight_bonus = min(0.2, avg_weight * 0.2)
                score += weight_bonus
            
            # Entity type diversity bonus
            if path.nodes:
                entity_types = [node.entity_type for node in path.nodes if hasattr(node, 'entity_type')]
                type_diversity = len(set(entity_types)) / len(entity_types) if entity_types else 0
                score += type_diversity * 0.2
            
            return min(1.0, max(0.0, score))
            
        except Exception:
            return 0.5
    
    def calculate_temporal_coherence(self, path: Path) -> float:
        """
        Calculate basic temporal coherence score for legacy compatibility
        """
        if not path.edges:
            return 1.0
        
        temporal_edges = 0
        valid_temporal_constraints = 0
        
        for edge in path.edges:
            if hasattr(edge, 'timestamp') and edge.timestamp:
                temporal_edges += 1
                # Basic validation - if timestamp exists and is parseable, it's valid
                try:
                    from datetime import datetime
                    datetime.fromisoformat(edge.timestamp.replace('T', ' '))
                    valid_temporal_constraints += 1
                except (ValueError, TypeError):
                    pass
        
        if temporal_edges == 0:
            return 0.5  # Neutral score for non-temporal paths
        
        return valid_temporal_constraints / temporal_edges
    
    def enhanced_temporal_flow_pruning(self, paths: List[Path], top_k: int, query_time: str = None) -> List[Path]:
        """
        Apply enhanced temporal-aware flow-based pruning using the dedicated TemporalFlowPruning system
        
        This method integrates temporal weighting into PathRAG's resource propagation by:
        1. Modifying edge capacities with temporal decay factors
        2. Using adaptive thresholds theta based on temporal context
        3. Applying temporal decay rate alpha in resource propagation
        """
        if query_time is None:
            from datetime import datetime
            query_time = datetime.now().isoformat()
        
        return self.temporal_flow_pruning.flow_based_pruning_with_temporal_weighting(
            paths, top_k, query_time
        )