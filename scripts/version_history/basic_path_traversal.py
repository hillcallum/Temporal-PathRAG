import networkx as nx
from collections import defaultdict, deque
from typing import List, Dict, Optional, Tuple
import heapq
import torch
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from src.kg.models import TemporalPathRAGNode, TemporalPathRAGEdge, Path
from src.kg.temporal_scoring import TemporalWeightingFunction, TemporalPathRanker, TemporalPath, TemporalRelevanceMode

class BasicPathTraversal:
    """
    GPU-accelerated PathRAG path traversal implementation
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
        
        # Setup device for GPU acceleration
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        print(f"PathRAG using device: {self.device}")
        
        # Initialise sentence transformer for semantic similarity (GPU-accelerated)
        self.init_sentence_transformer()
        
    def init_sentence_transformer(self):
        """Initialise sentence transformer for GPU-accelerated semantic similarity"""
        try:
            from sentence_transformers import SentenceTransformer
            model_name = 'all-MiniLM-L6-v2'  
            print(f"Loading sentence transformer: {model_name}")
            self.sentence_transformer = SentenceTransformer(model_name, device=self.device)
            self.use_gpu_embeddings = True
            print("GPU-accelerated semantic similarity enabled")
        except ImportError:
            print("Warning: sentence-transformers not available, using basic similarity")
            self.sentence_transformer = None
            self.use_gpu_embeddings = False
    
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
    
    def find_paths(self, 
                   source_node_id: str, 
                   target_node_id: str,
                   max_hops: int = 3,
                   top_k: int = 10,
                   via_nodes: List[str] = None,
                   fallback_search: bool = True,
                   query_time: str = None,
                   temporal_constraints: Dict = None) -> List[Path]:
        """
        Find paths between source and target nodes using our basic PathRAG approach
        """
        try:
            # Validate input nodes
            validation_result = self.validate_nodes([source_node_id, target_node_id])
            if not validation_result['all_valid']:
                return self.handle_missing_nodes(source_node_id, target_node_id, validation_result, fallback_search, max_hops, top_k)
            
            # Handle via-node queries
            if via_nodes:
                return self.find_paths_via_nodes(source_node_id, target_node_id, via_nodes, max_hops, top_k)
            
            # Check cache first
            cache_key = f"{source_node_id}_{target_node_id}_{max_hops}"
            if cache_key in self.path_cache:
                return self.path_cache[cache_key][:top_k]
            
            # Find all paths
            all_paths = self.dijkstra_paths(source_node_id, target_node_id, max_hops)
            
            # If no paths found and fallback enabled, try alternative searches
            if not all_paths and fallback_search:
                all_paths = self.fallback_path_search(source_node_id, target_node_id, max_hops)
            
            # Score paths with temporal enhancement
            scored_paths = self.score_paths(all_paths, query_time)
            
            # Apply basic flow-based pruning
            pruned_paths = self.flow_based_pruning(scored_paths, top_k)
            
            # Cache results
            self.path_cache[cache_key] = pruned_paths
            
            return pruned_paths[:top_k]
            
        except Exception as e:
            print(f"Error in find_paths: {e}")
            if fallback_search:
                # Graceful degradation - try neighbourhood exploration
                return self.graceful_fallback(source_node_id, target_node_id, max_hops, top_k)
            return []
    
    def dijkstra_paths(self, 
                       source: str, 
                       target: str, 
                       max_hops: int) -> List[Path]:
        """
        Modified Dijkstra's algorithm to find multiple paths 
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
                path = self.construct_path(path_nodes, path_edges)
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
            for neighbour in self.graph.successors(node_id):
                edge_data = self.graph.edges[node_id, neighbour]
                neighbours.append((neighbour, edge_data))
        
        return neighbours
    
    def calculate_edge_cost(self, edge_data: Dict) -> float:
        """Calculate cost of traversing an edge (lower is better)"""
        # Basic cost based on edge weight
        weight = edge_data.get('weight', 1.0)
        return 1.0 / (weight + 0.1)  # Avoid division by 0 by adding 0.1 to denominator 
    
    def construct_path(self, node_ids: List[str], edge_data_list: List[Dict]) -> Optional[Path]:
        """Construct a Path object from node IDs and edge data with textual chunks"""
        try:
            path = Path()
            
            # Add nodes with textual chunks (tv)
            for node_id in node_ids:
                if self.graph.has_node(node_id):
                    node_data = self.graph.nodes[node_id]
                    pathrag_node = TemporalPathRAGNode(
                        id=node_id,
                        entity_type=node_data.get('entity_type', 'UNKNOWN'),
                        name=node_data.get('name', node_id),
                        properties=node_data,
                        description=node_data.get('description', ''),
                        # tv will be auto-generated in __post_init__
                    )
                    path.add_node(pathrag_node)
            
            # Add edges with textual chunks (te)
            for i, edge_data in enumerate(edge_data_list):
                if i < len(node_ids) - 1:
                    pathrag_edge = TemporalPathRAGEdge(
                        source_id=node_ids[i],
                        target_id=node_ids[i + 1],
                        relation_type=edge_data.get('relation_type', 'RELATED'),
                        weight=edge_data.get('weight', 1.0),
                        description=edge_data.get('description', ''),
                        flow_capacity=edge_data.get('flow_capacity', 1.0)
                        # te will be auto-generated in __post_init__
                    )
                    path.add_edge(pathrag_edge)
            
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
        
        # Use GPU-accelerated semantic similarity if available
        if self.use_gpu_embeddings and self.sentence_transformer:
            return self.calculate_gpu_semantic_similarity(path)
        else:
            return self.calculate_rule_based_similarity(path)
    
    def calculate_gpu_semantic_similarity(self, path: Path) -> float:
        """
        GPU-accelerated semantic similarity using sentence transformers
        """
        try:
            # Extract semantic content from the path
            path_text = path.path_text if hasattr(path, 'path_text') and path.path_text else ""
            if not path_text:
                # Fallback option by constructing text from nodes and edges
                node_texts = [f"{node.name}: {getattr(node, 'description', '')}" for node in path.nodes]
                edge_texts = [edge.relation_type for edge in path.edges]
                path_text = " -> ".join([f"{node_texts[i]} --[{edge_texts[i]}]-->" if i < len(edge_texts) else node_texts[i] 
                                       for i in range(len(node_texts))])
            
            # Create a reference query for semantic similarity
            # Use the source and target nodes to create an expected semantic context
            source_context = f"{path.nodes[0].name} {getattr(path.nodes[0], 'description', '')}"
            target_context = f"{path.nodes[-1].name} {getattr(path.nodes[-1], 'description', '')}"
            reference_query = f"How is {source_context} related to {target_context}?"
            
            # Encode both texts using GPU
            with torch.no_grad():
                path_embedding = self.sentence_transformer.encode([path_text], 
                                                               convert_to_tensor=True,
                                                               device=self.device)
                query_embedding = self.sentence_transformer.encode([reference_query], 
                                                                convert_to_tensor=True,
                                                                device=self.device)
                
                # Calculate cosine similarity on GPU
                similarity = torch.nn.functional.cosine_similarity(
                    path_embedding, query_embedding, dim=1
                ).cpu().item()
                
                # Normalise to [0, 1] range as cosine similarity is [-1, 1]
                normalised_similarity = (similarity + 1) / 2
                
                # Apply path quality bonus (combine with rule-based factors)
                quality_bonus = self.calculate_path_quality_bonus(path)
                
                # Final score: 70% semantic similarity + 30% structural quality
                final_score = 0.7 * normalised_similarity + 0.3 * quality_bonus
                
                return min(final_score, 1.0)
                
        except Exception as e:
            print(f"Warning: GPU semantic similarity failed ({e}), falling back to rule-based")
            return self.calculate_rule_based_similarity(path)
    
    def calculate_path_quality_bonus(self, path: Path) -> float:
        """
        Calculate structural quality bonus for the path
        """
        # Bonus for high-quality relationship types
        high_quality_relations = {
            'DEVELOPED', 'INVENTED', 'DISCOVERED', 'FOUNDED', 'AWARDED',
            'COLLABORATED_WITH', 'INFLUENCED', 'AUTHORED'
        }
        
        if not path.edges:
            return 0.5
            
        quality_relations = sum(1 for edge in path.edges 
                              if edge.relation_type in high_quality_relations)
        relation_quality = quality_relations / len(path.edges)
        
        # Path length penalty (shorter paths are often better)
        length_penalty = 1.0 / (1.0 + len(path.nodes) * 0.1)
        
        return 0.6 * relation_quality + 0.4 * length_penalty
    
    def calculate_rule_based_similarity(self, path: Path) -> float:
        """
        Fallback rule-based semantic similarity (original implementation)
        """
        score = 0.5  # base score
        
        # Bonus for coherent entity type sequences
        entity_types = [node.entity_type for node in path.nodes]
        
        # Higher score for paths connecting related types
        coherent_patterns = [
            ['Person', 'Institution', 'Country'],  # person -> institution -> location
            ['Person', 'Award', 'Person'],         # person -> award -> person (shared award)
            ['Person', 'Concept', 'Person'],       # person -> concept -> person (shared work)
            ['Person', 'Publication', 'Concept'],  # person -> publication -> concept
            ['Concept', 'Person', 'Institution']   # concept -> person -> institution
        ]
        
        for pattern in coherent_patterns:
            if len(entity_types) >= len(pattern):
                for i in range(len(entity_types) - len(pattern) + 1):
                    if entity_types[i:i+len(pattern)] == pattern:
                        score += 0.2
                        break
        
        # Add quality bonus
        quality_bonus = self._calculate_path_quality_bonus(path)
        score += quality_bonus * 0.3
        
        return min(score, 1.0)  # cap at 1.0
    
    def batch_calculate_semantic_similarity(self, paths: List[Path]) -> List[float]:
        """
        Batch calculate semantic similarity for multiple paths using GPU acceleration
        """
        if not paths:
            return []
            
        # Use GPU batch processing if available
        if self.use_gpu_embeddings and self.sentence_transformer and len(paths) > 1:
            return self.batch_gpu_semantic_similarity(paths)
        else:
            # Fallback to individual calculations
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
    
    def calculate_temporal_coherence(self, path: Path) -> float:
        """
        Calculate temporal coherence score based on time constraints in the data
        """
        if not path.edges:
            return 0.5  # neutral score for nodes-only paths
        
        temporal_consistency = 0.5  # base score
        temporal_edges = 0
        
        # Check for temporal information in edges and nodes
        for edge in path.edges:
            if hasattr(edge, 'source_id') and hasattr(edge, 'target_id'):
                source_node = None
                target_node = None
                
                # Find the corresponding nodes
                for node in path.nodes:
                    if node.id == edge.source_id:
                        source_node = node
                    elif node.id == edge.target_id:
                        target_node = node
                
                if source_node and target_node:
                    # Check temporal constraints from graph data
                    source_data = self.graph.nodes.get(source_node.id, {})
                    target_data = self.graph.nodes.get(target_node.id, {})
                    edge_data = None
                    
                    # Get edge data from graph
                    if self.graph.has_edge(source_node.id, target_node.id):
                        edge_data = self.graph.edges[source_node.id, target_node.id]
                    
                    # Check temporal consistency
                    temporal_valid = self.check_temporal_consistency(
                        source_data, target_data, edge_data
                    )
                    
                    if temporal_valid is not None:
                        temporal_edges += 1
                        if temporal_valid:
                            temporal_consistency += 0.2
                        else:
                            temporal_consistency -= 0.1
        
        # Normalise based on number of temporal edges found
        if temporal_edges > 0:
            temporal_consistency = max(0.0, min(1.0, temporal_consistency))
        
        return temporal_consistency
    
    def check_temporal_consistency(self, source_data: dict, target_data: dict, edge_data: dict) -> Optional[bool]:
        """
        Check if temporal constraints are satisfied between nodes
        """
        if not edge_data:
            return None
        
        # Check for temporal information in edge
        edge_year = edge_data.get('year')
        edge_start = edge_data.get('start_year')
        edge_end = edge_data.get('end_year')
        
        # Check birth/death years for people
        source_birth = source_data.get('born_year')
        source_death = source_data.get('died_year')
        target_birth = target_data.get('born_year')
        target_death = target_data.get('died_year')
        
        # If we have edge timing and person lifespans, check consistency
        if edge_year and source_birth:
            # Person must be alive when edge event occurred
            if edge_year < source_birth:
                return False
            if source_death and edge_year > source_death:
                return False
        
        if edge_year and target_birth:
            if edge_year < target_birth:
                return False
            if target_death and edge_year > target_death:
                return False
        
        # Check start/end year constraints
        if edge_start and edge_end:
            if source_birth and source_birth > edge_end:
                return False
            if source_death and source_death < edge_start:
                return False
            if target_birth and target_birth > edge_end:
                return False
            if target_death and target_death < edge_start:
                return False
        
        # If we found temporal constraints and passed all checks
        if any([edge_year, edge_start, edge_end, source_birth, target_birth]):
            return True
        
        return None  # No temporal information found
    
    def flow_based_pruning(self, paths: List[Path], top_k: int) -> List[Path]:
        """
        Apply basic flow-based pruning inspired by PathRAG paper
        
        Groups paths by endpoints and applies capacity constraints
        """
        if not paths:
            return []
        
        # Group paths by source-target pairs
        path_groups = defaultdict(list)
        for path in paths:
            if path.nodes:
                key = (path.nodes[0].id, path.nodes[-1].id)
                path_groups[key].append(path)
        
        pruned_paths = []
        
        for (source, target), group_paths in path_groups.items():
            # Sort by score
            group_paths.sort(key=lambda p: p.score, reverse=True)
            
            # Apply flow capacity constraints
            selected_paths = []
            total_flow = 0.0
            max_flow = top_k  # Simple threshold
            
            for path in group_paths:
                path_flow = self.calculate_path_flow(path)
                if total_flow + path_flow <= max_flow:
                    selected_paths.append(path)
                    total_flow += path_flow
                
                if len(selected_paths) >= top_k // max(len(path_groups), 1):
                    break
            
            pruned_paths.extend(selected_paths)
        
        return sorted(pruned_paths, key=lambda p: p.score, reverse=True)
    
    def calculate_path_flow(self, path: Path) -> float:
        """Calculate flow capacity of a path"""
        if not path.edges:
            return 1.0
        
        # Flow is limited by minimum edge capacity
        min_capacity = min(edge.flow_capacity for edge in path.edges)
        
        # Length penalty
        length_penalty = 1.0 / (1.0 + len(path.edges) * 0.1)
        
        return min_capacity * length_penalty
    
    def explore_neighbourhood(self, 
                           source_node_id: str,
                           max_hops: int = 2,
                           top_k: int = 10) -> List[Path]:
        """
        Explore neighbourhood around a source node (useful for open-ended queries)
        """
        paths = []
        visited = set()
        
        # BFS exploration
        queue = deque([(source_node_id, [source_node_id], [])])
        
        while queue and len(paths) < top_k * 2:
            current_node, path_nodes, path_edges = queue.popleft()
            
            path_signature = tuple(path_nodes)
            if path_signature in visited:
                continue
            visited.add(path_signature)
            
            # Create path if long enough
            if len(path_nodes) > 1:
                path = self.construct_path(path_nodes, path_edges)
                if path:
                    paths.append(path)
            
            # Continue exploration
            if len(path_nodes) < max_hops + 1:
                neighbours = self.get_neighbours(current_node)
                
                for neighbour_id, edge_data in neighbours:
                    if neighbour_id not in path_nodes:  # Avoid cycles
                        new_path_nodes = path_nodes + [neighbour_id]
                        new_path_edges = path_edges + [edge_data]
                        queue.append((neighbour_id, new_path_nodes, new_path_edges))
        
        # Score and return top paths
        scored_paths = self.score_paths(paths)
        return scored_paths[:top_k]
    
    def find_shared_connections(self, 
                              entity1_id: str, 
                              entity2_id: str,
                              max_hops: int = 2) -> Dict[str, any]:
        """
        Find shared connections between two entities
        """
        # Get neighbourhood for both entities
        entity1_paths = self.explore_neighbourhood(entity1_id, max_hops=max_hops, top_k=20)
        entity2_paths = self.explore_neighbourhood(entity2_id, max_hops=max_hops, top_k=20)
        
        # Extract all nodes reachable from each entity
        entity1_nodes = set()
        entity1_path_map = {}
        for path in entity1_paths:
            for node in path.nodes:
                entity1_nodes.add(node.id)
                if node.id not in entity1_path_map:
                    entity1_path_map[node.id] = []
                entity1_path_map[node.id].append(path)
        
        entity2_nodes = set()
        entity2_path_map = {}
        for path in entity2_paths:
            for node in path.nodes:
                entity2_nodes.add(node.id)
                if node.id not in entity2_path_map:
                    entity2_path_map[node.id] = []
                entity2_path_map[node.id].append(path)
        
        # Debug: Print what nodes each entity can reach
        print(f"DEBUG: Nodes reachable from {entity1_id}:")
        for path in entity1_paths:
            node_names = [node.name for node in path.nodes]
            print(f"  Path: {' -> '.join(node_names)}")

        print(f"DEBUG: Nodes reachable from {entity2_id}:")
        for path in entity2_paths:
            node_names = [node.name for node in path.nodes]
            print(f"  Path: {' -> '.join(node_names)}")
        
        # Find shared nodes (not including the starting entities)
        shared_nodes = entity1_nodes.intersection(entity2_nodes)
        shared_nodes.discard(entity1_id)
        shared_nodes.discard(entity2_id)
        
        # Build result
        result = {
            'shared_nodes': [],
            'entity1_paths': {},
            'entity2_paths': {}
        }
        
        for node_id in shared_nodes:
            if self.graph.has_node(node_id):
                node_data = self.graph.nodes[node_id]
                result['shared_nodes'].append({
                    'id': node_id,
                    'name': node_data.get('name', node_id),
                    'entity_type': node_data.get('entity_type', 'UNKNOWN'),
                    'description': node_data.get('description', '')
                })
                
                # Get paths to this shared node
                result['entity1_paths'][node_id] = entity1_path_map.get(node_id, [])
                result['entity2_paths'][node_id] = entity2_path_map.get(node_id, [])
        
        return result
    
    def find_connecting_paths(self,
                            entity1_id: str,
                            entity2_id: str,
                            max_hops: int = 4) -> List[Path]:
        """
        Find paths that connect two entities through intermediate nodes
        """
        # First try direct paths
        direct_paths = self.find_paths(entity1_id, entity2_id, max_hops=max_hops, top_k=5)
        if direct_paths:
            return direct_paths
        
        # If no direct paths, look for paths through shared connections
        shared_info = self.find_shared_connections(entity1_id, entity2_id, max_hops=max_hops//2)
        
        connecting_paths = []
        
        for shared_node in shared_info['shared_nodes']:
            shared_id = shared_node['id']
            
            # Get paths from entity1 to shared node
            paths1 = self.find_paths(entity1_id, shared_id, max_hops=max_hops//2, top_k=3)
            
            # Get paths from entity2 to shared node  
            paths2 = self.find_paths(entity2_id, shared_id, max_hops=max_hops//2, top_k=3)
            
            # Create combined paths through thay specific shared node
            for path1 in paths1:
                for path2 in paths2:
                    # Create a conceptual 'connecting path' metadata
                    connection_info = {
                        'entity1': entity1_id,
                        'entity2': entity2_id,
                        'shared_connection': shared_id,
                        'path1': path1,
                        'path2': path2,
                        'connection_type': shared_node['entity_type']
                    }
                    
                    # Use path1 as the base and add metadata
                    combined_path = Path()
                    combined_path.nodes = path1.nodes.copy()
                    combined_path.edges = path1.edges.copy()
                    combined_path.score = (path1.score + path2.score) / 2
                    combined_path.metadata['connection_info'] = connection_info
                    
                    connecting_paths.append(combined_path)
        
        # Sort by score and return top paths
        connecting_paths.sort(key=lambda p: p.score, reverse=True)
        return connecting_paths[:5]
    
    def find_bidirectional_paths(self, 
                               source_node_id: str, 
                               target_node_id: str,
                               max_hops: int = 3,
                               top_k: int = 10) -> List[Dict]:
        """
        Find bidirectional paths that connect two nodes through shared intermediate nodes
        """
        # Find paths from source to intermediate nodes
        source_paths = self.explore_neighbourhood(source_node_id, max_hops=max_hops//2 + 1, top_k=top_k*2)
        
        # Find all nodes reachable from source
        source_reachable = set()
        source_path_map = {}
        for path in source_paths:
            for node in path.nodes[1:]:  # Skip source node itself
                source_reachable.add(node.id)
                if node.id not in source_path_map:
                    source_path_map[node.id] = []
                source_path_map[node.id].append(path)
        
        # Find paths from target to intermediate nodes  
        target_paths = self.explore_neighbourhood(target_node_id, max_hops=max_hops//2 + 1, top_k=top_k*2)
        
        # Find all nodes reachable from target
        target_reachable = set()
        target_path_map = {}
        for path in target_paths:
            for node in path.nodes[1:]:  # Skip target node itself
                target_reachable.add(node.id)
                if node.id not in target_path_map:
                    target_path_map[node.id] = []
                target_path_map[node.id].append(path)
        
        # Find shared intermediate nodes
        shared_nodes = source_reachable.intersection(target_reachable)
        
        if not shared_nodes:
            return []
        
        # Create bidirectional connection objects
        bidirectional_connections = []
        
        for shared_node in shared_nodes:
            source_to_shared = source_path_map.get(shared_node, [])
            target_to_shared = target_path_map.get(shared_node, [])
            
            if source_to_shared and target_to_shared:
                # Use the best path from each side (shortest first, then highest score)
                best_source_path = min(source_to_shared, 
                                     key=lambda p: (len(p.nodes), -p.score))
                best_target_path = min(target_to_shared, 
                                     key=lambda p: (len(p.nodes), -p.score))
                
                # Calculate connection quality score
                connection_score = self.calculate_bidirectional_score(
                    best_source_path, best_target_path, shared_node
                )
                
                # Create bidirectional connection info
                connection_info = {
                    'type': 'bidirectional',
                    'source_node': source_node_id,
                    'target_node': target_node_id,
                    'shared_node': shared_node,
                    'shared_node_data': self.graph.nodes.get(shared_node, {}),
                    'source_path': best_source_path,
                    'target_path': best_target_path,
                    'source_hops': len(best_source_path.edges),
                    'target_hops': len(best_target_path.edges),
                    'total_hops': len(best_source_path.edges) + len(best_target_path.edges),
                    'connection_score': connection_score,
                    'connection_text': self.generate_bidirectional_text(
                        best_source_path, best_target_path, shared_node
                    )
                }
                
                bidirectional_connections.append(connection_info)
        
        # Sort by connection score (higher is better)
        bidirectional_connections.sort(key=lambda x: x['connection_score'], reverse=True)
        
        return bidirectional_connections[:top_k]
    
    def calculate_bidirectional_score(self, 
                                    source_path: Path, 
                                    target_path: Path, 
                                    shared_node: str) -> float:
        """
        Calculate quality score for a bidirectional connection
        
        Higher scores indicate better connections (shorter paths, higher individual scores)
        """
        # Favour shorter total paths
        total_hops = len(source_path.edges) + len(target_path.edges)
        length_penalty = 1.0 / (1.0 + total_hops * 0.2)
        
        # Consider individual path quality
        avg_path_score = (source_path.score + target_path.score) / 2.0
        
        # Bonus for important node types (awards, institutions, concepts)
        shared_node_data = self.graph.nodes.get(shared_node, {})
        node_type = shared_node_data.get('entity_type', '').lower()
        
        importance_bonus = 1.0
        if 'award' in node_type:
            importance_bonus = 1.5  
        elif 'concept' in node_type or 'theory' in node_type:
            importance_bonus = 1.3  
        elif 'institution' in node_type:
            importance_bonus = 1.2  
        
        # Final score combines all factors
        final_score = length_penalty * avg_path_score * importance_bonus
        
        return final_score
    
    def generate_bidirectional_text(self, 
                                   source_path: Path, 
                                   target_path: Path, 
                                   shared_node: str) -> str:
        """
        Generate human-readable text describing the bidirectional connection
        """
        source_start = source_path.nodes[0].name if source_path.nodes else "Unknown"
        target_start = target_path.nodes[0].name if target_path.nodes else "Unknown" 
        shared_node_data = self.graph.nodes.get(shared_node, {})
        shared_name = shared_node_data.get('name', shared_node)
        
        # Create description based on node type
        node_type = shared_node_data.get('entity_type', '').lower()
        
        if 'award' in node_type:
            connection_text = f"Both {source_start} and {target_start} received the {shared_name}"
        elif 'concept' in node_type or 'theory' in node_type:
            connection_text = f"Both {source_start} and {target_start} are connected to {shared_name}"
        elif 'institution' in node_type:
            connection_text = f"Both {source_start} and {target_start} have connections to {shared_name}"
        elif 'event' in node_type:
            connection_text = f"Both {source_start} and {target_start} were involved in {shared_name}"
        else:
            connection_text = f"{source_start} and {target_start} share a connection through {shared_name}"
        
        # Add path details
        source_relation = source_path.edges[0].relation_type if source_path.edges else "connected to"
        target_relation = target_path.edges[0].relation_type if target_path.edges else "connected to"
        
        if source_relation == target_relation:
            connection_text += f" (both via {source_relation})"
        else:
            connection_text += f" ({source_start} via {source_relation}, {target_start} via {target_relation})"
        
        return connection_text
    
    def find_shared_connections(self, 
                              source_node_id: str, 
                              target_node_id: str,
                              max_hops: int = 2,
                              top_k: int = 5) -> Dict[str, any]:
        """
        Enhanced version of find_shared_connections using bidirectional search
        """
        bidirectional_paths = self.find_bidirectional_paths(
            source_node_id, target_node_id, max_hops=max_hops*2, top_k=top_k*2
        )
        
        # Convert to the original format for backwards compatibility
        shared_nodes = []
        source_paths_map = {}
        target_paths_map = {}
        
        for conn in bidirectional_paths:
            shared_node_id = conn['shared_node']
            shared_node_data = conn['shared_node_data']
            
            # Add to shared nodes list
            shared_nodes.append({
                'id': shared_node_id,
                'name': shared_node_data.get('name', shared_node_id),
                'entity_type': shared_node_data.get('entity_type', 'UNKNOWN'),
                'description': shared_node_data.get('description', ''),
                'connection_score': conn['connection_score'],
                'connection_text': conn['connection_text']
            })
            
            # Map paths for compatibility
            source_paths_map[shared_node_id] = [conn['source_path']]
            target_paths_map[shared_node_id] = [conn['target_path']]
        
        return {
            'shared_nodes': shared_nodes,
            'source_paths': source_paths_map,
            'target_paths': target_paths_map,
            'bidirectional_connections': bidirectional_paths
        }
    
    def find_paths_via_nodes(self, 
                           source_node_id: str, 
                           target_node_id: str,
                           via_nodes: List[str],
                           max_hops: int = 3,
                           top_k: int = 10) -> List[Path]:
        """
        Find paths that must pass through specified intermediate nodes
        """
        if not via_nodes:
            return self.find_paths(source_node_id, target_node_id, max_hops, top_k)
        
        # Check if via nodes exist in graph
        missing_nodes = [node for node in via_nodes if not self.graph.has_node(node)]
        if missing_nodes:
            print(f"Warning: Via nodes not found in graph: {missing_nodes}")
            via_nodes = [node for node in via_nodes if node not in missing_nodes]
            if not via_nodes:
                return []
        
        all_via_paths = []
        
        # For each via node, find paths through it
        for via_node in via_nodes:
            # Find paths from source to via node
            source_to_via = self.find_paths(source_node_id, via_node, max_hops//2, top_k)
            
            # Find paths from via node to target
            via_to_target = self.find_paths(via_node, target_node_id, max_hops//2, top_k)
            
            # Combine paths
            for path1 in source_to_via:
                for path2 in via_to_target:
                    combined_path = self.combine_paths(path1, path2, via_node)
                    if combined_path and len(combined_path.nodes) <= max_hops + 1:
                        all_via_paths.append(combined_path)
        
        # Score and sort paths
        scored_paths = self.score_paths(all_via_paths)
        
        # Apply flow-based pruning
        pruned_paths = self.flow_based_pruning(scored_paths, top_k)
        
        return pruned_paths[:top_k]
    
    def combine_paths(self, path1: Path, path2: Path, via_node: str) -> Optional[Path]:
        """
        Combine two paths that meet at a via node
        """
        try:
            # Verify paths meet at via node
            if not path1.nodes or not path2.nodes:
                return None
            
            if path1.nodes[-1].id != via_node or path2.nodes[0].id != via_node:
                return None
            
            # Create combined path
            combined_path = Path()
            
            # Add nodes from first path
            for node in path1.nodes:
                combined_path.add_node(node)
            
            # Add nodes from second path (skip the via node to avoid duplication)
            for node in path2.nodes[1:]:
                combined_path.add_node(node)
            
            # Add edges from first path
            for edge in path1.edges:
                combined_path.add_edge(edge)
            
            # Add edges from second path
            for edge in path2.edges:
                combined_path.add_edge(edge)
            
            # Calculate combined score
            combined_path.score = (path1.score + path2.score) / 2
            
            # Add metadata about the combination
            combined_path.metadata['via_node'] = via_node
            combined_path.metadata['source_path_score'] = path1.score
            combined_path.metadata['target_path_score'] = path2.score
            
            return combined_path
            
        except Exception as e:
            print(f"Error combining paths: {e}")
            return None
    
    def aggregate_multiple_paths(self, paths: List[Path], query_context: str = None) -> dict:
        """
        Aggregate multiple paths to provide better answers
        """
        if not paths:
            return {
                'summary': 'No paths found',
                'confidence': 0.0,
                'key_entities': [],
                'relationships': [],
                'evidence_strength': 'none'
            }
        
        # Extract key information from all paths
        all_entities = {}
        all_relationships = {}
        path_scores = [path.score for path in paths]
        
        # Aggregate entities with frequency and importance
        for path in paths:
            for node in path.nodes:
                if node.id not in all_entities:
                    all_entities[node.id] = {
                        'name': node.name,
                        'type': node.entity_type,
                        'frequency': 0,
                        'importance_score': 0.0,
                        'descriptions': set()
                    }
                
                all_entities[node.id]['frequency'] += 1
                all_entities[node.id]['importance_score'] += path.score
                if hasattr(node, 'description') and node.description:
                    all_entities[node.id]['descriptions'].add(node.description)
        
        # Aggregate relationships
        for path in paths:
            for edge in path.edges:
                rel_key = f"{edge.source_id}_{edge.relation_type}_{edge.target_id}"
                if rel_key not in all_relationships:
                    all_relationships[rel_key] = {
                        'source': edge.source_id,
                        'target': edge.target_id,
                        'relation': edge.relation_type,
                        'frequency': 0,
                        'strength': 0.0,
                        'descriptions': set()
                    }
                
                all_relationships[rel_key]['frequency'] += 1
                all_relationships[rel_key]['strength'] += path.score
                if hasattr(edge, 'description') and edge.description:
                    all_relationships[rel_key]['descriptions'].add(edge.description)
        
        # Calculate confidence based on path consensus
        confidence = self.calculate_path_consensus(paths)
        
        # Identify key entities (most frequent and important)
        key_entities = sorted(
            all_entities.items(),
            key=lambda x: (x[1]['frequency'], x[1]['importance_score']),
            reverse=True
        )[:5]  # top 5 entities
        
        # Identify key relationships
        key_relationships = sorted(
            all_relationships.items(),
            key=lambda x: (x[1]['frequency'], x[1]['strength']),
            reverse=True
        )[:5]  # top 5 relationships
        
        # Generate summary based on aggregated paths
        summary = self.generate_path_summary(paths, key_entities, key_relationships, query_context)
        
        # Determine evidence strength
        evidence_strength = self.assess_evidence_strength(paths, confidence)
        
        return {
            'summary': summary,
            'confidence': confidence,
            'key_entities': [
                {
                    'id': ent_id,
                    'name': ent_data['name'],
                    'type': ent_data['type'],
                    'frequency': ent_data['frequency'],
                    'importance': ent_data['importance_score'] / len(paths)
                }
                for ent_id, ent_data in key_entities
            ],
            'relationships': [
                {
                    'source': rel_data['source'],
                    'target': rel_data['target'],
                    'relation': rel_data['relation'],
                    'frequency': rel_data['frequency'],
                    'strength': rel_data['strength'] / len(paths)
                }
                for _, rel_data in key_relationships
            ],
            'evidence_strength': evidence_strength,
            'num_supporting_paths': len(paths),
            'avg_path_score': sum(path_scores) / len(path_scores) if path_scores else 0.0
        }
    
    def calculate_path_consensus(self, paths: List[Path]) -> float:
        """
        Calculate consensus score based on how much paths agree
        """
        if len(paths) <= 1:
            return 1.0 if paths else 0.0
        
        # Count shared entities and relationships across paths
        entity_appearances = {}
        relationship_appearances = {}
        
        for path in paths:
            # Count entity appearances
            for node in path.nodes:
                entity_appearances[node.id] = entity_appearances.get(node.id, 0) + 1
            
            # Count relationship appearances
            for edge in path.edges:
                rel_key = f"{edge.source_id}_{edge.relation_type}_{edge.target_id}"
                relationship_appearances[rel_key] = relationship_appearances.get(rel_key, 0) + 1
        
        # Calculate consensus based on overlap
        num_paths = len(paths)
        
        # Entity consensus (how many entities appear in multiple paths)
        entity_consensus = sum(1 for count in entity_appearances.values() if count > 1) / max(len(entity_appearances), 1)
        
        # Relationship consensus
        rel_consensus = sum(1 for count in relationship_appearances.values() if count > 1) / max(len(relationship_appearances), 1)
        
        # Combined consensus score
        consensus = (entity_consensus + rel_consensus) / 2
        
        # Bonus for high-scoring paths agreeing
        avg_score = sum(path.score for path in paths) / num_paths
        consensus = consensus * (0.7 + 0.3 * avg_score)
        
        return min(consensus, 1.0)
    
    def generate_path_summary(self, paths: List[Path], key_entities: list, key_relationships: list, query_context: str = None) -> str:
        """
        Generate a textual summary of aggregated paths
        """
        if not paths:
            return "No relevant paths found."
        
        # Extract most common entities
        entity_names = [ent_data['name'] for ent_id, ent_data in key_entities[:3]]
        
        # Extract most common relationships
        rel_types = [rel_data['relation'] for _, rel_data in key_relationships[:3]]
        
        if len(paths) == 1:
            path = paths[0]
            node_names = [node.name for node in path.nodes]
            return f"Found connection: {' → '.join(node_names)}"
        
        # Multi-path summary
        summary_parts = []
        
        if entity_names:
            summary_parts.append(f"Key entities involved: {', '.join(entity_names)}")
        
        if rel_types:
            unique_rel_types = list(set(rel_types))
            summary_parts.append(f"Main relationships: {', '.join(unique_rel_types)}")
        
        summary_parts.append(f"Found {len(paths)} supporting paths")
        
        # Add confidence indicator
        confidence = self.calculate_path_consensus(paths)
        if confidence > 0.8:
            summary_parts.append("(high confidence)")
        elif confidence > 0.5:
            summary_parts.append("(moderate confidence)")
        else:
            summary_parts.append("(low confidence)")
        
        return ". ".join(summary_parts) + "."
    
    def assess_evidence_strength(self, paths: List[Path], confidence: float) -> str:
        """
        Assess the strength of evidence based on paths and confidence
        """
        num_paths = len(paths)
        avg_score = sum(path.score for path in paths) / num_paths if paths else 0.0
        
        if num_paths >= 3 and confidence > 0.7 and avg_score > 0.7:
            return "strong"
        elif num_paths >= 2 and confidence > 0.5 and avg_score > 0.5:
            return "moderate"
        elif num_paths >= 1 and avg_score > 0.3:
            return "weak"
        else:
            return "insufficient"
    
    def validate_nodes(self, node_ids: List[str]) -> dict:
        """
        Validate that all specified nodes exist in the graph
        """
        missing_nodes = []
        existing_nodes = []
        
        for node_id in node_ids:
            if self.graph.has_node(node_id):
                existing_nodes.append(node_id)
            else:
                missing_nodes.append(node_id)
        
        return {
            'all_valid': len(missing_nodes) == 0,
            'missing_nodes': missing_nodes,
            'existing_nodes': existing_nodes,
            'missing_count': len(missing_nodes)
        }
    
    def handle_missing_nodes(self, source_node_id: str, target_node_id: str, 
                           validation_result: dict, fallback_search: bool,
                           max_hops: int, top_k: int) -> List[Path]:
        """
        Handle cases where some nodes are missing from the graph
        """
        missing = validation_result['missing_nodes']
        existing = validation_result['existing_nodes']
        
        print(f"Warning: Missing nodes in graph: {missing}")
        
        if not fallback_search:
            return []
        
        # If only one node is missing, try alternatives
        if len(missing) == 1:
            missing_node = missing[0]
            existing_node = existing[0] if existing else None
            
            if existing_node:
                # Find similar nodes or do neighbourhood exploration
                return self.find_similar_node_paths(existing_node, missing_node, max_hops, top_k)
        
        # If both nodes missing, return empty
        return []
    
    def find_similar_node_paths(self, existing_node: str, missing_node: str, 
                              max_hops: int, top_k: int) -> List[Path]:
        """
        Find paths using similar nodes when target node is missing
        """
        print(f"Attempting to find alternatives for missing node: {missing_node}")
        
        # Try neighbourhood exploration around existing node
        neighbourhood_paths = self.explore_neighbourhood(existing_node, max_hops=max_hops, top_k=top_k)
        
        if neighbourhood_paths:
            print(f"Found {len(neighbourhood_paths)} alternative paths via neighbourhood exploration")
        
        return neighbourhood_paths
    
    def fallback_path_search(self, source_node_id: str, target_node_id: str, max_hops: int) -> List[Path]:
        """
        Attempt alternative search strategies when direct path search fails
        """
        fallback_paths = []
        
        # Try bidirectional search
        try:
            bidirectional_results = self.find_bidirectional_paths(source_node_id, target_node_id, max_hops*2, top_k=5)
            for conn in bidirectional_results:
                if 'source_path' in conn and 'target_path' in conn:
                    fallback_paths.extend([conn['source_path'], conn['target_path']])
        except Exception as e:
            print(f"Bidirectional fallback failed: {e}")
        
        # Try via shared connections
        if not fallback_paths:
            try:
                shared_info = self.find_shared_connections(source_node_id, target_node_id, max_hops=max_hops//2)
                if shared_info.get('shared_nodes'):
                    # Get paths via first shared node
                    first_shared = shared_info['shared_nodes'][0]['id']
                    source_to_shared = self.find_paths(source_node_id, first_shared, max_hops//2, 3, fallback_search=False)
                    shared_to_target = self.find_paths(first_shared, target_node_id, max_hops//2, 3, fallback_search=False)
                    fallback_paths.extend(source_to_shared)
                    fallback_paths.extend(shared_to_target)
            except Exception as e:
                print(f"Shared connection fallback failed: {e}")
        
        return fallback_paths
    
    def graceful_fallback(self, source_node_id: str, target_node_id: str, 
                         max_hops: int, top_k: int) -> List[Path]:
        """
        Final fallback when all other methods fail
        """
        print(f"Graceful fallback: exploring neighbourhood around {source_node_id}")
        
        try:
            # Just explore neighbourhood of source
            if self.graph.has_node(source_node_id):
                return self.explore_neighbourhood(source_node_id, max_hops=max_hops//2, top_k=top_k)
            elif self.graph.has_node(target_node_id):
                return self.explore_neighbourhood(target_node_id, max_hops=max_hops//2, top_k=top_k)
        except Exception as e:
            print(f"Graceful fallback failed: {e}")
        
        return []