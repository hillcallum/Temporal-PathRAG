"""
Utility functions for handling graph operations and fixing compatibility issues
"""

import networkx as nx
from typing import Dict, Any, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def safe_get_edge_data(graph: nx.Graph, source: str, target: str) -> Dict[str, Any]:
    """
    Safely get edge data from a NetworkX graph, handling different edge data structures
    
    Args:
        graph: NetworkX graph
        source: Source node
        target: Target node
        
    Returns:
        Edge data dictionary or empty dict if edge doesn't exist
    """
    if not graph.has_edge(source, target):
        return {}
        
    edge_data = graph.get_edge_data(source, target)
    
    # Handle different edge data structures
    if edge_data is None:
        return {}
    elif isinstance(edge_data, dict):
        # Check if it's a multi-edge structure (dict of dicts)
        if edge_data and all(isinstance(v, dict) for v in edge_data.values()):
            # Multi-edge: return the first edge data or merge them
            first_key = next(iter(edge_data))
            return edge_data[first_key]
        else:
            # Simple edge data
            return edge_data
    else:
        # Unexpected structure
        logger.warning(f"Unexpected edge data structure: {type(edge_data)}")
        return {}


def safe_get_neighbours_with_edges(graph: nx.Graph, node: str) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Safely get all neighbours of a node with their edge data
    
    Args:
        graph: NetworkX graph
        node: Node to get neighbours for
        
    Returns:
        List of (neighbour, edge_data) tuples
    """
    neighbours = []
    
    if not graph.has_node(node):
        return neighbours
        
    for neighbour in graph.neighbors(node):
        edge_data = safe_get_edge_data(graph, node, neighbour)
        neighbours.append((neighbour, edge_data))
        
    return neighbours


def iterate_all_edges(graph: nx.Graph, node: str, neighbour: str) -> List[Dict[str, Any]]:
    """
    Iterate over all edges between two nodes (handling multi-edges)
    
    Args:
        graph: NetworkX graph
        node: Source node
        neighbour: Target node
        
    Returns:
        List of edge data dictionaries
    """
    if not graph.has_edge(node, neighbour):
        return []
        
    edge_data = graph.get_edge_data(node, neighbour)
    
    if edge_data is None:
        return []
    elif isinstance(edge_data, dict):
        # Check if it's a multi-edge structure
        if edge_data and all(isinstance(v, dict) for v in edge_data.values()):
            # Multi-edge: return all edge data
            return list(edge_data.values())
        else:
            # Simple edge: return as single-item list
            return [edge_data]
    else:
        return []


def normalise_edge_attributes(edge_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalise edge attributes to handle different naming conventions.
    
    Args:
        edge_data: Raw edge data
        
    Returns:
        Normalised edge data with standard attribute names
    """
    normalised = edge_data.copy()
    
    # Normalise temporal attributes
    temporal_mappings = {
        'timestamp': ['time', 'date', 'year', 't'],
        'te': ['temporal_edge', 'time_validity', 'temporal_validity'],
        'relation': ['predicate', 'rel', 'type', 'edge_type'],
        'weight': ['score', 'confidence', 'strength']
    }
    
    for standard_name, alternatives in temporal_mappings.items():
        if standard_name not in normalised:
            for alt in alternatives:
                if alt in edge_data:
                    normalised[standard_name] = edge_data[alt]
                    break
                    
    return normalised


def fix_graph_for_pathfinding(graph: nx.Graph) -> nx.Graph:
    """
    Fix a graph to ensure it's compatible with path finding algorithms.
    
    This handles multi-edges and ensures all edge data is properly structured.
    
    Args:
        graph: Original graph
        
    Returns:
        Fixed graph suitable for path finding
    """
    # Create a new graph with the same nodes
    fixed_graph = nx.Graph()
    fixed_graph.add_nodes_from(graph.nodes(data=True))
    
    # Add edges with fixed data structure
    for u, v, data in graph.edges(data=True):
        # Ensure edge data is a simple dict
        if isinstance(data, dict):
            fixed_graph.add_edge(u, v, **data)
        else:
            fixed_graph.add_edge(u, v)
            
    return fixed_graph