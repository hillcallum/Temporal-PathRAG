"""
Updated Temporal Scoring module with enhanced scoring functions for temporal paths.
This module extends the base temporal scoring with improvements for better temporal reasoning.
"""

import logging
from typing import List, Tuple, Dict, Optional, Any, Union
import numpy as np
from datetime import datetime
import networkx as nx

# Import base temporal scoring functions
from .temporal_scoring import TemporalWeightingFunction
from ..utils.graph_utils import safe_get_edge_data

logger = logging.getLogger(__name__)


class UpdatedTemporalScorer:
    """Enhanced temporal scorer with improved path scoring algorithms."""
    
    def __init__(
        self,
        alpha: float = 10.0,
        beta: float = 1.0,
        use_decay: bool = True,
        temporal_weight: float = 0.3
    ):
        """
        Initialize the updated temporal scorer.
        
        Args:
            alpha: Exponential decay parameter for temporal distance
            beta: Scaling parameter for temporal scores
            use_decay: Whether to use exponential decay for temporal distance
            temporal_weight: Weight for temporal features vs semantic features
        """
        self.alpha = alpha
        self.beta = beta
        self.use_decay = use_decay
        self.temporal_weight = temporal_weight
        
        # Initialize base temporal weighting function
        self.temporal_weighting = TemporalWeightingFunction(decay_rate=alpha)
        
    def score_path(
        self,
        path: List[Tuple[str, str, str]],
        query: str,
        graph: nx.Graph,
        query_time: Optional[Union[str, int, datetime]] = None,
        semantic_scores: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Score a path based on temporal and semantic relevance.
        
        Args:
            path: List of (source, relation, target) triples
            query: The query string
            graph: The knowledge graph
            query_time: The time point or range for the query
            semantic_scores: Optional pre-computed semantic scores
            
        Returns:
            Combined temporal-semantic score
        """
        if not path:
            return 0.0
            
        # Compute temporal score
        temporal_score = self._compute_temporal_score(path, graph, query_time)
        
        # Compute semantic score
        semantic_score = self._compute_semantic_score(path, query, semantic_scores)
        
        # Combine scores
        combined_score = (
            self.temporal_weight * temporal_score + 
            (1 - self.temporal_weight) * semantic_score
        )
        
        # Apply path length penalty
        length_penalty = 1.0 / (1.0 + np.log(len(path)))
        
        return combined_score * length_penalty
        
    def score_paths(
        self,
        paths: List[List[Tuple[str, str, str]]],
        query: str,
        graph: nx.Graph,
        query_time: Optional[Union[str, int, datetime]] = None,
        top_k: Optional[int] = None
    ) -> List[Tuple[List[Tuple[str, str, str]], float]]:
        """
        Score multiple paths and return sorted results.
        
        Args:
            paths: List of paths to score
            query: The query string
            graph: The knowledge graph
            query_time: The time point or range for the query
            top_k: Return only top k paths
            
        Returns:
            List of (path, score) tuples sorted by score
        """
        scored_paths = []
        
        for path in paths:
            score = self.score_path(path, query, graph, query_time)
            scored_paths.append((path, score))
            
        # Sort by score (descending)
        scored_paths.sort(key=lambda x: x[1], reverse=True)
        
        if top_k:
            return scored_paths[:top_k]
        return scored_paths
        
    def _compute_temporal_score(
        self,
        path: List[Tuple[str, str, str]],
        graph: nx.Graph,
        query_time: Optional[Union[str, int, datetime]] = None
    ) -> float:
        """Compute temporal relevance score for a path."""
        if not query_time:
            # No temporal constraint - neutral score
            return 0.5
            
        temporal_scores = []
        
        # Score each edge
        for source, relation, target in path:
            if graph.has_edge(source, target):
                edge_data = safe_get_edge_data(graph, source, target)
                edge_time = self._extract_time(edge_data)
                
                if edge_time is not None:
                    # Compute temporal distance
                    distance = self._compute_time_distance(edge_time, query_time)
                    
                    # Apply temporal weighting
                    if self.use_decay:
                        score = np.exp(-self.alpha * distance)
                    else:
                        score = 1.0 / (1.0 + self.beta * distance)
                        
                    temporal_scores.append(score)
                    
        # Score nodes
        nodes = set()
        for source, _, target in path:
            nodes.add(source)
            nodes.add(target)
            
        for node in nodes:
            if node in graph:
                node_data = graph.nodes[node]
                node_time = self._extract_time(node_data)
                
                if node_time is not None:
                    distance = self._compute_time_distance(node_time, query_time)
                    
                    if self.use_decay:
                        score = np.exp(-self.alpha * distance)
                    else:
                        score = 1.0 / (1.0 + self.beta * distance)
                        
                    temporal_scores.append(score)
                    
        # Aggregate scores
        if temporal_scores:
            return np.mean(temporal_scores)
        return 0.5  # Neutral if no temporal information
        
    def _compute_semantic_score(
        self,
        path: List[Tuple[str, str, str]],
        query: str,
        semantic_scores: Optional[Dict[str, float]] = None
    ) -> float:
        """Compute semantic relevance score for a path."""
        if semantic_scores:
            # Use pre-computed scores if available
            path_key = str(path)
            if path_key in semantic_scores:
                return semantic_scores[path_key]
                
        # Simple heuristic based on query terms
        query_terms = set(query.lower().split())
        path_terms = set()
        
        for source, relation, target in path:
            # Add entity and relation terms
            path_terms.update(source.lower().split('_'))
            path_terms.update(relation.lower().split('_'))
            path_terms.update(target.lower().split('_'))
            
        # Compute overlap
        overlap = len(query_terms.intersection(path_terms))
        if len(query_terms) > 0:
            return overlap / len(query_terms)
        return 0.0
        
    def _extract_time(self, data: Dict[str, Any]) -> Optional[Any]:
        """Extract temporal information from node/edge data."""
        # Check various temporal attribute names
        for attr in ['te', 'tv', 'timestamp', 'time', 'date', 'year']:
            if attr in data:
                return data[attr]
        return None
        
    def _compute_time_distance(
        self,
        time1: Any,
        time2: Any
    ) -> float:
        """Compute normalized distance between two time points."""
        try:
            # Handle various time formats
            if isinstance(time1, (list, tuple)) and len(time1) == 2:
                # Time range - use midpoint
                start, end = time1
                if start and end:
                    time1 = (self._parse_time(start) + self._parse_time(end)) / 2
                elif start:
                    time1 = self._parse_time(start)
                else:
                    time1 = self._parse_time(end)
            else:
                time1 = self._parse_time(time1)
                
            time2 = self._parse_time(time2)
            
            # Compute absolute difference in years
            diff = abs(time1 - time2)
            
            # Normalize (assuming max relevant time span is 100 years)
            return diff / 100.0
            
        except Exception as e:
            logger.debug(f"Error computing time distance: {e}")
            return 1.0  # Maximum distance on error
            
    def _parse_time(self, time_value: Any) -> float:
        """Parse time value to numeric year representation."""
        if isinstance(time_value, (int, float)):
            return float(time_value)
            
        if isinstance(time_value, str):
            # Try to extract year
            try:
                # Handle YYYY-MM-DD format
                if '-' in time_value and len(time_value) >= 4:
                    year = int(time_value[:4])
                    return float(year)
                # Handle plain year
                elif time_value.isdigit() and len(time_value) == 4:
                    return float(time_value)
                # Try to parse as integer
                else:
                    return float(int(time_value))
            except:
                pass
                
        # Default to a middle value if parsing fails
        return 2000.0
        
    def get_temporal_context(
        self,
        query: str,
        default_time: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Extract temporal context from a query.
        
        Args:
            query: The query string
            default_time: Default time to use if none found in query
            
        Returns:
            Dictionary with temporal context information
        """
        temporal_context = {
            'temporal_type': 'point',
            'query_time': default_time
        }
        
        # Simple temporal keyword extraction
        query_lower = query.lower()
        
        # Check for temporal keywords
        if 'before' in query_lower:
            temporal_context['temporal_type'] = 'before'
        elif 'after' in query_lower:
            temporal_context['temporal_type'] = 'after'
        elif 'between' in query_lower or 'from' in query_lower:
            temporal_context['temporal_type'] = 'range'
        elif 'during' in query_lower or 'in' in query_lower:
            temporal_context['temporal_type'] = 'point'
            
        # Try to extract years from query
        import re
        year_pattern = r'\b(19\d{2}|20\d{2})\b'
        years = re.findall(year_pattern, query)
        
        if years:
            if len(years) >= 2 and temporal_context['temporal_type'] == 'range':
                temporal_context['start_time'] = years[0]
                temporal_context['end_time'] = years[1]
            else:
                temporal_context['query_time'] = years[0]
                
        return temporal_context