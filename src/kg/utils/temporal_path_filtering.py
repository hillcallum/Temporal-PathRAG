"""
Temporal Path Filtering module for filtering knowledge graph paths based on temporal constraints.
This module provides functionality to filter paths that are temporally relevant to a query.
"""

import logging
from typing import List, Tuple, Dict, Optional, Any
from datetime import datetime
import networkx as nx
from .graph_utils import safe_get_edge_data

logger = logging.getLogger(__name__)


class TemporalPathFilter:
    """Filter paths based on temporal constraints and relevance"""
    
    def __init__(self, temporal_weight: float = 0.5):
        """
        Initialise the temporal path filter.
        
        Args:
            temporal_weight: Weight for temporal relevance scoring (0-1)
        """
        self.temporal_weight = temporal_weight
        
    def filter_paths(
        self, 
        paths: List[List[Tuple[str, str, str]]], 
        graph: nx.Graph,
        temporal_context: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None
    ) -> List[List[Tuple[str, str, str]]]:
        """
        Filter paths based on temporal relevance
        
        Args:
            paths: List of paths, where each path is a list of (source, relation, target) triples
            graph: The knowledge graph
            temporal_context: Optional temporal context containing:
                - query_time: The time point or range of the query
                - start_time: Start of temporal range
                - end_time: End of temporal range
                - temporal_type: Type of temporal constraint ('point', 'range', 'before', 'after')
            top_k: Return only top k paths (if None, return all passing paths)
            
        Returns:
            Filtered list of temporally relevant paths
        """
        if not paths:
            return []
            
        # If no temporal context, return all paths
        if not temporal_context:
            logger.debug("No temporal context provided, returning all paths")
            return paths[:top_k] if top_k else paths
            
        # Score and filter paths
        scored_paths = []
        for path in paths:
            score = self.score_path_temporal_relevance(path, graph, temporal_context)
            if score > 0:  # Only keep paths with positive temporal relevance
                scored_paths.append((path, score))
                
        # Sort by score
        scored_paths.sort(key=lambda x: x[1], reverse=True)
        
        # Return top k or all
        filtered_paths = [path for path, score in scored_paths]
        if top_k:
            return filtered_paths[:top_k]
        return filtered_paths
        
    def score_path_temporal_relevance(
        self,
        path: List[Tuple[str, str, str]],
        graph: nx.Graph,
        temporal_context: Dict[str, Any]
    ) -> float:
        """
        Score a single path for temporal relevance
        
        Args:
            path: A path as list of (source, relation, target) triples
            graph: The knowledge graph
            temporal_context: Temporal context for scoring
            
        Returns:
            Temporal relevance score (0-1)
        """
        if not path:
            return 0.0
            
        total_score = 0.0
        temporal_edges = 0
        
        # Check each edge in the path
        for source, relation, target in path:
            edge_score = self.score_edge_temporal_relevance(
                source, relation, target, graph, temporal_context
            )
            total_score += edge_score
            if edge_score > 0:
                temporal_edges += 1
                
        # Also check node temporal validity
        nodes = set()
        for source, _, target in path:
            nodes.add(source)
            nodes.add(target)
            
        node_scores = []
        for node in nodes:
            node_score = self.score_node_temporal_relevance(node, graph, temporal_context)
            node_scores.append(node_score)
            
        # Combine edge and node scores
        if temporal_edges > 0:
            edge_avg = total_score / len(path)
        else:
            edge_avg = 0.0
            
        if node_scores:
            node_avg = sum(node_scores) / len(node_scores)
        else:
            node_avg = 0.0
            
        # Weighted combination
        final_score = (self.temporal_weight * edge_avg + 
                      (1 - self.temporal_weight) * node_avg)
        
        return final_score
        
    def score_edge_temporal_relevance(
        self,
        source: str,
        relation: str,
        target: str,
        graph: nx.Graph,
        temporal_context: Dict[str, Any]
    ) -> float:
        """Score temporal relevance of a single edge."""
        # Check if edge exists in graph
        if not graph.has_edge(source, target):
            return 0.0
            
        edge_data = safe_get_edge_data(graph, source, target)
        
        # Check for temporal validity attribute
        if 'te' in edge_data:  # temporal edge
            return self.check_temporal_overlap(edge_data['te'], temporal_context)
        elif 'timestamp' in edge_data:
            return self.check_temporal_overlap(edge_data['timestamp'], temporal_context)
        elif 'time' in edge_data:
            return self.check_temporal_overlap(edge_data['time'], temporal_context)
            
        # No temporal information - neutral score
        return 0.5
        
    def score_node_temporal_relevance(
        self,
        node: str,
        graph: nx.Graph,
        temporal_context: Dict[str, Any]
    ) -> float:
        """Score temporal relevance of a single node"""
        if node not in graph:
            return 0.0
            
        node_data = graph.nodes[node]
        
        # Check for temporal validity attribute
        if 'tv' in node_data:  # temporal validity
            return self.check_temporal_overlap(node_data['tv'], temporal_context)
        elif 'timestamp' in node_data:
            return self.check_temporal_overlap(node_data['timestamp'], temporal_context)
        elif 'time' in node_data:
            return self.check_temporal_overlap(node_data['time'], temporal_context)
            
        # No temporal information - neutral score
        return 0.5
        
    def check_temporal_overlap(
        self,
        temporal_value: Any,
        temporal_context: Dict[str, Any]
    ) -> float:
        """
        Check if a temporal value overlaps with the query temporal context
        
        Returns a score between 0 and 1 indicating degree of overlap
        """
        # Handle different temporal value formats
        if isinstance(temporal_value, (list, tuple)) and len(temporal_value) == 2:
            # Range format: (start, end)
            start, end = temporal_value
        elif isinstance(temporal_value, dict):
            start = temporal_value.get('start') or temporal_value.get('from')
            end = temporal_value.get('end') or temporal_value.get('to')
        else:
            # Single time point
            start = end = temporal_value
            
        # Get query temporal constraints
        query_type = temporal_context.get('temporal_type', 'point')
        
        if query_type == 'point':
            query_time = temporal_context.get('query_time')
            if not query_time:
                return 0.5  # No specific time, neutral score
                
            # Check if query time falls within the range
            if start and end:
                if self.compare_times(start, query_time) <= 0 and self.compare_times(query_time, end) <= 0:
                    return 1.0
            elif start and self.compare_times(start, query_time) == 0:
                return 1.0
                
        elif query_type == 'range':
            query_start = temporal_context.get('start_time')
            query_end = temporal_context.get('end_time')
            
            # Check for overlap between ranges
            if start and end and query_start and query_end:
                # Check if ranges overlap
                if (self.compare_times(start, query_end) <= 0 and 
                    self.compare_times(query_start, end) <= 0):
                    return 1.0
                    
        elif query_type == 'before':
            query_time = temporal_context.get('query_time')
            if end and query_time:
                if self.compare_times(end, query_time) < 0:
                    return 1.0
                    
        elif query_type == 'after':
            query_time = temporal_context.get('query_time')
            if start and query_time:
                if self.compare_times(start, query_time) > 0:
                    return 1.0
                    
        return 0.0
        
    def compare_times(self, time1: Any, time2: Any) -> int:
        """
        Compare two time values
        
        Returns:
            -1 if time1 < time2
            0 if time1 == time2
            1 if time1 > time2
        """
        # Handle None values
        if time1 is None or time2 is None:
            return 0
            
        # Try to convert to comparable format
        try:
            # If strings, try to parse as years or dates
            if isinstance(time1, str) and isinstance(time2, str):
                # Try year comparison first
                try:
                    year1 = int(time1[:4]) if len(time1) >= 4 else int(time1)
                    year2 = int(time2[:4]) if len(time2) >= 4 else int(time2)
                    if year1 < year2:
                        return -1
                    elif year1 > year2:
                        return 1
                    else:
                        return 0
                except:
                    pass
                    
            # Direct comparison for numbers
            if isinstance(time1, (int, float)) and isinstance(time2, (int, float)):
                if time1 < time2:
                    return -1
                elif time1 > time2:
                    return 1
                else:
                    return 0
                    
        except Exception as e:
            logger.debug(f"Error comparing times {time1} and {time2}: {e}")
            
        # Default to equal if can't compare
        return 0