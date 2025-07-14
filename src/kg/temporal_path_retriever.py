"""
Temporal Path Retriever for TKG (Temporal Knowledge Graph)

This module is a Top-K retrieval system for temporally relevant paths, 
putting together the temporal flow-based pruning and ranking work

Key Features:
1. Retrieval of temporal paths
2. Scoring for temporal reliability and integraying flow-based pruning
3. Selection of Top-K paths and consideration of diversity constraints
5. Processing accelerated by GPU for TKGs, given large scale of dataset
"""

import math
import numpy as np
import torch
from collections import defaultdict, deque
from typing import List, Dict, Optional, Tuple, Set, Any
from datetime import datetime, timedelta
import networkx as nx
import heapq
import time

from .models import (
    TemporalPathRAGNode, TemporalPathRAGEdge, Path, TemporalQuery, 
    PerformanceMetrics, FlowPruningConfig
)
from .temporal_scoring import TemporalWeightingFunction, TemporalPathRanker, TemporalPath, TemporalRelevanceMode
from .temporal_flow_pruning import TemporalFlowPruning
from .path_traversal import TemporalPathTraversal
from .updated_temporal_scoring import UpdatedTemporalScorer


class TemporalPathRetriever:
    """
    Temporal path retrieval system for TKG
    """
    
    def __init__(self, 
                 graph: nx.MultiDiGraph,
                 temporal_weighting: TemporalWeightingFunction = None,
                 device: torch.device = None,
                 alpha: float = 0.01,
                 base_theta: float = 0.1,
                 diversity_threshold: float = 0.7,
                 updated_scorer: Optional[UpdatedTemporalScorer] = None):
        """
        Initialise temporal path retriever
        
        Args:
            graph: The temporal knowledge graph
            temporal_weighting: Temporal weighting function
            device: Computation device
            alpha: Temporal decay rate
            base_theta: Base pruning threshold
            diversity_threshold: Diversity threshold for path selection
            updated_scorer: Optional updated temporal scorer
        """
        self.graph = graph
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.diversity_threshold = diversity_threshold
        
        # Initialise temporal components
        if temporal_weighting is None:
            self.temporal_weighting = TemporalWeightingFunction(
                decay_rate=alpha,
                temporal_window=365,
                chronological_weight=0.3,
                proximity_weight=0.4,
                consistency_weight=0.3
            )
        else:
            self.temporal_weighting = temporal_weighting
            
        # Initialise path traversal system
        self.path_traversal = TemporalPathTraversal(
            graph=graph,
            device=device,
            temporal_mode=TemporalRelevanceMode.EXPONENTIAL_DECAY
        )
        
        # Initialise temporal flow pruning
        self.temporal_flow_pruning = TemporalFlowPruning(
            temporal_weighting=self.temporal_weighting,
            temporal_mode=TemporalRelevanceMode.EXPONENTIAL_DECAY,
            alpha=alpha,
            base_theta=base_theta
        )
        
        # Initialise temporal path ranker
        self.temporal_ranker = TemporalPathRanker(self.temporal_weighting)
        
        # Store updated scorer if provided
        self.updated_scorer = updated_scorer
        
        # Performance tracking
        self.retrieval_stats = {
            'total_queries': 0,
            'avg_retrieval_time': 0.0,
            'avg_paths_discovered': 0,
            'avg_paths_after_pruning': 0
        }
        
        print(f"TemporalPathRetriever initialised on device: {self.device}")
        if self.updated_scorer:
            print("Using updated temporal scoring in path retriever")
    
    def get_edge_data_for_multigraph(self, source_node: str, target_node: str) -> Dict:
        """
        Get edge data for MultiDiGraph, handling multiple edges between same nodes.
        Returns the most recent edge data based on timestamp, or first edge if no timestamps.
        """
        if not self.graph.has_edge(source_node, target_node):
            return {}
        
        # Get all edges between the two nodes
        edge_dict = self.graph.get_edge_data(source_node, target_node)
        
        if not edge_dict:
            return {}
        
        # If there's only one edge, return it directly
        if len(edge_dict) == 1:
            return list(edge_dict.values())[0]
        
        # If multiple edges, try to find the most recent one based on timestamp
        best_edge = None
        latest_timestamp = None
        
        for edge_key, edge_data in edge_dict.items():
            if 'timestamp' in edge_data:
                try:
                    # Try to parse timestamp for comparison
                    timestamp = edge_data['timestamp']
                    if latest_timestamp is None or timestamp > latest_timestamp:
                        latest_timestamp = timestamp
                        best_edge = edge_data
                except (ValueError, TypeError):
                    # If timestamp parsing fails, continue with next edge
                    continue
        
        # If we found a timestamped edge, return it
        if best_edge is not None:
            return best_edge
        
        # Otherwise, return the first edge (fallback)
        return list(edge_dict.values())[0]
    
    def retrieve_temporal_paths(self, 
                              query: TemporalQuery,
                              enable_flow_pruning: bool = True,
                              enable_diversity: bool = True,
                              verbose: bool = False) -> List[Tuple[Path, float]]:
        """
        Main retrieval method - discover and rank temporally relevant paths
        
        Implements retrieval pipeline:
        1. Temporal path discovery from TKG
        2. Flow-based pruning with temporal weighting
        3. Temporal reliability scoring
        4. Top-K selection with constraints
        """
        start_time = time.time()
        
        if verbose:
            print(f"Processing temporal query: {query.query_text}")
            print(f"Temporal constraints: {query.temporal_constraints}")
        
        # Stage 1: Temporal path discovery
        candidate_paths = self.discover_temporal_paths(query, verbose)
        
        if verbose:
            print(f"Discovered {len(candidate_paths)} candidate paths")
        
        # Stage 2: Flow-based pruning (if enabled)
        if enable_flow_pruning and candidate_paths:
            pruned_paths = self.apply_temporal_flow_pruning(
                candidate_paths, query, verbose
            )
        else:
            pruned_paths = candidate_paths
            
        if verbose:
            print(f"After flow pruning: {len(pruned_paths)} paths")
        
        # Stage 3: Temporal reliability scoring
        scored_paths = self.score_temporal_reliability(pruned_paths, query, verbose)
        
        # Stage 4: Top-K selection with diversity
        if enable_diversity:
            final_paths = self.select_diverse_top_k(scored_paths, query.top_k, verbose)
        else:
            final_paths = sorted(scored_paths, key=lambda x: x[1], reverse=True)[:query.top_k]
        
        # Update performance statistics
        retrieval_time = time.time() - start_time
        self.update_stats(retrieval_time, len(candidate_paths), len(pruned_paths))
        
        if verbose:
            print(f"Final selection: {len(final_paths)} paths")
            print(f"Retrieval completed in {retrieval_time:.3f}s")
        
        return final_paths
    
    def discover_temporal_paths(self, query: TemporalQuery, verbose: bool = False) -> List[Path]:
        """
        Stage 1: Discover candidate paths from TKG using temporal constraints
        """
        all_paths = []
        
        # Strategy 1: Entity-to-entity path finding
        if query.source_entities and query.target_entities:
            for source in query.source_entities:
                for target in query.target_entities:
                    if source != target:  # Avoid self-loops
                        paths = self.path_traversal.find_paths(
                            source_id=source,
                            target_id=target,
                            max_depth=query.max_hops,
                            max_paths=query.top_k * 3,  # Get more for better filtering
                            query_time=query.query_time
                        )
                        all_paths.extend(paths)
        
        # Strategy 2: Entity neighbourhood exploration (when targets unknown)
        elif query.source_entities:
            for source in query.source_entities:
                neighbourhood_paths = self.explore_entity_neighbourhood(
                    source, query.max_hops, query.top_k * 2, query.temporal_constraints
                )
                all_paths.extend(neighbourhood_paths)
        
        # Strategy 3: Temporal pattern-based discovery
        pattern_paths = self.discover_by_temporal_patterns(query)
        all_paths.extend(pattern_paths)
        
        # Remove duplicates while preserving order
        unique_paths = []
        seen_signatures = set()
        
        for path in all_paths:
            signature = self.get_path_signature(path)
            if signature not in seen_signatures:
                seen_signatures.add(signature)
                unique_paths.append(path)
        
        return unique_paths
    
    def explore_entity_neighbourhood(self, 
                                   entity_id: str, 
                                   max_hops: int, 
                                   max_paths: int,
                                   temporal_constraints: Dict) -> List[Path]:
        """Explore entity neighbourhood with temporal awareness"""
        paths = []
        
        if not self.graph.has_node(entity_id):
            return paths
        
        # BFS exploration with temporal filtering
        queue = deque([(entity_id, [entity_id], [], 0)])  # (node, path_nodes, path_edges, depth)
        visited_paths = set()
        
        while queue and len(paths) < max_paths:
            current_node, path_nodes, path_edges, depth = queue.popleft()
            
            # Check depth limit
            if depth >= max_hops:
                # Create path if it has meaningful length
                if len(path_nodes) > 1:
                    path = self.construct_path_from_exploration(path_nodes, path_edges)
                    if path and self.satisfies_temporal_constraints(path, temporal_constraints):
                        paths.append(path)
                continue
            
            # Explore neighbours
            if current_node in self.graph:
                for neighbour in self.graph.neighbors(current_node):
                    if neighbour not in path_nodes:  # Avoid cycles
                        edge_data = self.get_edge_data_for_multigraph(current_node, neighbour)
                        
                        # Check temporal constraints on edge
                        if self.edge_satisfies_temporal_constraints(edge_data, temporal_constraints):
                            new_path_nodes = path_nodes + [neighbour]
                            new_path_edges = path_edges + [(current_node, neighbour, edge_data)]
                            path_signature = tuple(new_path_nodes)
                            
                            if path_signature not in visited_paths:
                                visited_paths.add(path_signature)
                                queue.append((neighbour, new_path_nodes, new_path_edges, depth + 1))
        
        return paths
    
    def discover_by_temporal_patterns(self, query: TemporalQuery) -> List[Path]:
        """Discover paths based on temporal patterns in query"""
        pattern_paths = []
        temporal_direction = query.temporal_patterns.get('temporal_direction')
        
        if not temporal_direction:
            return pattern_paths
        
        # Find edges that match temporal direction pattern
        matching_edges = []
        
        for source, target, edge_data in self.graph.edges(data=True):
            edge_timestamp = edge_data.get('timestamp')
            
            if edge_timestamp and self.matches_temporal_direction(
                edge_timestamp, query.query_time, temporal_direction
            ):
                matching_edges.append((source, target, edge_data))
        
        # Build paths around matching edges
        for source, target, edge_data in matching_edges[:query.top_k * 2]:
            # Create simple 1-hop path
            path = self.create_simple_path(source, target, edge_data)
            if path:
                pattern_paths.append(path)
        
        return pattern_paths
    
    def apply_temporal_flow_pruning(self, 
                                   candidate_paths: List[Path], 
                                   query: TemporalQuery,
                                   verbose: bool = False) -> List[Path]:
        """
        Stage 2: Apply temporal flow-based pruning algorithm on TKG
        """
        if not candidate_paths:
            return []
        
        if verbose:
            print(f"Applying temporal flow pruning to {len(candidate_paths)} paths")
        
        # Apply the flow-based pruning with temporal weighting
        pruned_paths = self.temporal_flow_pruning.flow_based_pruning_with_temporal_weighting(
            paths=candidate_paths,
            top_k=min(query.top_k * 2, len(candidate_paths)),  # Keep 2x for diversity
            query_time=query.query_time
        )
        
        if verbose:
            print(f"Flow pruning reduced paths from {len(candidate_paths)} to {len(pruned_paths)}")
        
        return pruned_paths
    
    def score_temporal_reliability(self, 
                                  paths: List[Path], 
                                  query: TemporalQuery,
                                  verbose: bool = False) -> List[Tuple[Path, float]]:
        """
        Stage 3: Score paths using temporal reliability scoring
        """
        scored_paths = []
        
        if verbose:
            print(f"Scoring {len(paths)} paths for temporal reliability")
        
        # Use updated scorer if available
        if self.updated_scorer is not None:
            # Prepare query context
            query_context = {
                'query_text': query.query_text,
                'temporal_constraints': query.temporal_constraints
            }
            
            # Batch score all paths
            scores_and_components = self.updated_scorer.batch_score_paths(
                paths, query.query_time, query_context
            )
            
            for path, (score, components) in zip(paths, scores_and_components):
                scored_paths.append((path, score))
                
            if verbose:
                print(f"Used updated scoring for {len(paths)} paths")
        else:
            # Fall back to original scoring
            for path in paths:
                # Convert to TemporalPath for updated scoring
                temporal_path = self.convert_to_temporal_path(path, query.query_time)
                
                # Component 1: Updated temporal reliability score
                temporal_reliability = self.temporal_weighting.updated_reliability_score(
                    temporal_path, query.query_time, path.score
                )
                
                # Component 2: Flow-based score (bottleneck principle)
                flow_score = self.calculate_flow_score(path, query.query_time)
                
                # Component 3: Semantic relevance to query
                semantic_score = self.calculate_semantic_relevance(path, query)
                
                # Component 4: Temporal consistency score
                consistency_score = self.calculate_temporal_consistency(path, query)
                
                # Component 5: Query-specific temporal pattern matching
                pattern_score = self.calculate_pattern_matching_score(path, query)
                
                # Combined reliability score with learned weights
                reliability_score = (
                    0.35 * temporal_reliability +
                    0.25 * flow_score +
                    0.15 * semantic_score +
                    0.15 * consistency_score +
                    0.10 * pattern_score
                )
                
                scored_paths.append((path, reliability_score))
        
        return scored_paths
    
    def select_diverse_top_k(self, 
                            scored_paths: List[Tuple[Path, float]], 
                            k: int,
                            verbose: bool = False) -> List[Tuple[Path, float]]:
        """
        Stage 4: Select top-K paths with constraints
        """
        if not scored_paths:
            return []
        
        # Sort by score
        sorted_paths = sorted(scored_paths, key=lambda x: x[1], reverse=True)
        selected_paths = []
        
        for path, score in sorted_paths:
            if len(selected_paths) >= k:
                break
            
            # Check constraints
            if not selected_paths or self.is_sufficiently_diverse(path, selected_paths):
                selected_paths.append((path, score))
            elif len(selected_paths) < k // 2:  # Allow some similar paths initially
                selected_paths.append((path, score))
        
        if verbose:
            print(f"Selected {len(selected_paths)} diverse paths from {len(sorted_paths)} candidates")
        
        return selected_paths
    
    # Helper methods for implementation
    
    def get_path_signature(self, path: Path) -> str:
        """Generate unique signature for path de-duplication"""
        if not path.nodes:
            return ""
        
        node_ids = [node.id for node in path.nodes]
        edge_types = [edge.relation_type for edge in path.edges] if path.edges else []
        
        return f"{'-'.join(node_ids)}|{'-'.join(edge_types)}"
    
    def construct_path_from_exploration(self, 
                                       node_ids: List[str], 
                                       edge_data: List[Tuple]) -> Optional[Path]:
        """Construct Path object from exploration results"""
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
            
            # Add edges
            for source, target, data in edge_data:
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
    
    def satisfies_temporal_constraints(self, path: Path, constraints: Dict) -> bool:
        """Check if path satisfies temporal constraints"""
        if not constraints:
            return True
        
        # Check time range constraints
        if 'time_range' in constraints:
            start_time, end_time = constraints['time_range']
            for edge in path.edges:
                if edge.timestamp:
                    try:
                        edge_time = datetime.fromisoformat(edge.timestamp.replace('T', ' '))
                        if not (start_time <= edge_time <= end_time):
                            return False
                    except (ValueError, TypeError):
                        continue
        
        return True
    
    def edge_satisfies_temporal_constraints(self, edge_data: Dict, constraints: Dict) -> bool:
        """Check if edge satisfies temporal constraints"""
        if not constraints or 'time_range' not in constraints:
            return True
        
        timestamp = edge_data.get('timestamp')
        if not timestamp:
            return True  # Non-temporal edges pass by default
        
        try:
            edge_time = datetime.fromisoformat(timestamp.replace('T', ' '))
            start_time, end_time = constraints['time_range']
            return start_time <= edge_time <= end_time
        except (ValueError, TypeError):
            return True
    
    def matches_temporal_direction(self, edge_timestamp: str, query_time: str, direction: str) -> bool:
        """Check if edge timestamp matches temporal direction pattern"""
        try:
            edge_time = datetime.fromisoformat(edge_timestamp.replace('T', ' '))
            query_dt = datetime.fromisoformat(query_time.replace('T', ' '))
            
            if direction == 'before':
                return edge_time < query_dt
            elif direction == 'after':
                return edge_time > query_dt
            elif direction == 'during':
                # Within reasonable window (e.g., same year)
                return abs((edge_time - query_dt).days) <= 365
            
        except (ValueError, TypeError):
            pass
        
        return False
    
    def create_simple_path(self, source: str, target: str, edge_data: Dict) -> Optional[Path]:
        """Create simple path from single edge"""
        try:
            path = Path()
            
            # Add source node
            if self.graph.has_node(source):
                source_data = self.graph.nodes[source]
                source_node = TemporalPathRAGNode(
                    id=source,
                    entity_type=source_data.get('entity_type', 'Unknown'),
                    name=source_data.get('name', source),
                    description=source_data.get('description', ''),
                    properties=source_data
                )
                path.add_node(source_node)
            
            # Add target node
            if self.graph.has_node(target):
                target_data = self.graph.nodes[target]
                target_node = TemporalPathRAGNode(
                    id=target,
                    entity_type=target_data.get('entity_type', 'Unknown'),
                    name=target_data.get('name', target),
                    description=target_data.get('description', ''),
                    properties=target_data
                )
                path.add_node(target_node)
            
            # Add edge
            edge = TemporalPathRAGEdge(
                source_id=source,
                target_id=target,
                relation_type=edge_data.get('relation_type', 'related_to'),
                weight=edge_data.get('weight', 1.0),
                description=edge_data.get('description', ''),
                timestamp=edge_data.get('timestamp'),
                flow_capacity=edge_data.get('flow_capacity', 1.0),
                properties=edge_data
            )
            path.add_edge(edge)
            
            return path
            
        except Exception:
            return None
    
    def convert_to_temporal_path(self, path: Path, query_time: str) -> TemporalPath:
        """Convert Path to TemporalPath for updated scoring"""
        timestamps = []
        edges_with_timestamps = []
        
        for edge in path.edges:
            timestamp = edge.timestamp if edge.timestamp else query_time
            timestamps.append(timestamp)
            edges_with_timestamps.append((
                edge.source_id, edge.relation_type, edge.target_id, timestamp
            ))
        
        return TemporalPath(
            nodes=[node.id for node in path.nodes],
            edges=edges_with_timestamps,
            timestamps=timestamps,
            original_score=getattr(path, 'score', 0.5)
        )
    
    def calculate_flow_score(self, path: Path, query_time: str) -> float:
        """Calculate flow-based score using bottleneck principle"""
        return self.temporal_flow_pruning.calculate_temporal_path_flow(path, query_time)
    
    def calculate_semantic_relevance(self, path: Path, query: TemporalQuery) -> float:
        """Calculate semantic relevance between path and query"""
        # Use existing semantic similarity from path traversal
        return self.path_traversal.calculate_semantic_similarity(path)
    
    def calculate_temporal_consistency(self, path: Path, query: TemporalQuery) -> float:
        """Calculate temporal consistency score"""
        return self.path_traversal.calculate_temporal_coherence(path)
    
    def calculate_pattern_matching_score(self, path: Path, query: TemporalQuery) -> float:
        """Calculate how well path matches temporal patterns in query"""
        if not query.temporal_patterns.get('temporal_direction'):
            return 0.5  # Neutral score
        
        direction = query.temporal_patterns['temporal_direction']
        matching_edges = 0
        total_edges = len(path.edges)
        
        if total_edges == 0:
            return 0.5
        
        for edge in path.edges:
            if edge.timestamp and self.matches_temporal_direction(
                edge.timestamp, query.query_time, direction
            ):
                matching_edges += 1
        
        return matching_edges / total_edges
    
    def is_sufficiently_diverse(self, path: Path, selected_paths: List[Tuple[Path, float]]) -> bool:
        """Check if path is sufficiently diverse from already selected paths"""
        path_signature = self.get_path_signature(path)
        
        for selected_path, _ in selected_paths:
            selected_signature = self.get_path_signature(selected_path)
            
            # Calculate similarity (simple string-based for now) 
            # Change in the future (as of 4th July 2025) - keeping it simple for now
            similarity = self.calculate_path_similarity(path_signature, selected_signature)
            
            if similarity > (1.0 - self.diversity_threshold):
                return False
        
        return True
    
    def calculate_path_similarity(self, sig1: str, sig2: str) -> float:
        """Calculate similarity between path signatures"""
        if not sig1 or not sig2:
            return 0.0
        
        # Simple Jaccard similarity on path components
        # Change in the future (as of 4th July 2025) - keeping it simple for now
        set1 = set(sig1.split('|')[0].split('-'))  # Node IDs
        set2 = set(sig2.split('|')[0].split('-'))  # Node IDs
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    def update_stats(self, retrieval_time: float, paths_discovered: int, paths_after_pruning: int):
        """Update performance statistics"""
        self.retrieval_stats['total_queries'] += 1
        
        # Update running averages
        total_queries = self.retrieval_stats['total_queries']
        self.retrieval_stats['avg_retrieval_time'] = (
            (self.retrieval_stats['avg_retrieval_time'] * (total_queries - 1) + retrieval_time) / total_queries
        )
        self.retrieval_stats['avg_paths_discovered'] = (
            (self.retrieval_stats['avg_paths_discovered'] * (total_queries - 1) + paths_discovered) / total_queries
        )
        self.retrieval_stats['avg_paths_after_pruning'] = (
            (self.retrieval_stats['avg_paths_after_pruning'] * (total_queries - 1) + paths_after_pruning) / total_queries
        )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return self.retrieval_stats.copy()
    
    def validate_retrieval_system(self, test_queries: List[TemporalQuery]) -> Dict[str, Any]:
        """Validate the retrieval system with test queries"""
        validation_results = {
            'total_test_queries': len(test_queries),
            'successful_retrievals': 0,
            'average_paths_retrieved': 0.0,
            'average_retrieval_time': 0.0,
            'temporal_coverage': 0.0,
            'diversity_score': 0.0
        }
        
        total_paths = 0
        total_time = 0.0
        successful_queries = 0
        
        for query in test_queries:
            try:
                start_time = time.time()
                results = self.retrieve_temporal_paths(query, verbose=False)
                retrieval_time = time.time() - start_time
                
                if results:
                    successful_queries += 1
                    total_paths += len(results)
                    total_time += retrieval_time
                
            except Exception as e:
                print(f"Query failed: {e}")
        
        if successful_queries > 0:
            validation_results['successful_retrievals'] = successful_queries
            validation_results['average_paths_retrieved'] = total_paths / successful_queries
            validation_results['average_retrieval_time'] = total_time / successful_queries
        
        return validation_results