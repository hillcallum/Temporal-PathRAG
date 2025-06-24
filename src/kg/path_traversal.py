import networkx as nx
from collections import defaultdict, deque
from typing import List, Dict, Optional, Tuple
import heapq

from .models import PathRAGNode, PathRAGEdge, Path

class BasicPathTraversal:
    """
    Basic PathRAG path traversal implementation
    

    """
    
    def __init__(self, graph: nx.DiGraph):
        self.graph = graph
        self.path_cache: Dict[str, List[Path]] = {}
    
    def find_paths(self, 
                   source_node_id: str, 
                   target_node_id: str,
                   max_hops: int = 3,
                   top_k: int = 10) -> List[Path]:
        """
        Find paths between source and target nodes using our bassic PathRAG approach
            
        Returns:
            List of paths ranked by score
        """
        # Check cache first
        cache_key = f"{source_node_id}_{target_node_id}_{max_hops}"
        if cache_key in self.path_cache:
            return self.path_cache[cache_key][:top_k]
        
        # Find all paths
        all_paths = self.dijkstra_paths(source_node_id, target_node_id, max_hops)
        
        # Score paths
        scored_paths = self.score_paths(all_paths)
        
        # Apply basic flow-based pruning
        pruned_paths = self.flow_based_pruning(scored_paths, top_k)
        
        # Cache results
        self.path_cache[cache_key] = pruned_paths
        
        return pruned_paths[:top_k]
    
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
        """Construct a Path object from node IDs and edge data"""
        try:
            path = Path()
            
            # Add nodes
            for node_id in node_ids:
                if self.graph.has_node(node_id):
                    node_data = self.graph.nodes[node_id]
                    pathrag_node = PathRAGNode(
                        id=node_id,
                        entity_type=node_data.get('entity_type', 'UNKNOWN'),
                        name=node_data.get('name', node_id),
                        properties=node_data,
                        description=node_data.get('description', '')
                    )
                    path.add_node(pathrag_node)
            
            # Add edges
            for i, edge_data in enumerate(edge_data_list):
                if i < len(node_ids) - 1:
                    pathrag_edge = PathRAGEdge(
                        source_id=node_ids[i],
                        target_id=node_ids[i + 1],
                        relation_type=edge_data.get('relation_type', 'RELATED'),
                        weight=edge_data.get('weight', 1.0),
                        description=edge_data.get('description', ''),
                        flow_capacity=edge_data.get('flow_capacity', 1.0)
                    )
                    path.add_edge(pathrag_edge)
            
            return path
            
        except Exception as e:
            print(f"Error constructing path: {e}")
            return None
    
    def score_paths(self, paths: List[Path]) -> List[Path]:
        """Score paths based on basic PathRAG principles"""
        for path in paths:
            # Basic scoring considering path length and edge weights
            length_penalty = 1.0 / (len(path.nodes) + 1)
            
            edge_quality = 1.0
            if path.edges:
                edge_weights = [edge.weight for edge in path.edges]
                edge_quality = sum(edge_weights) / len(edge_weights)
            
            path.score = length_penalty * edge_quality
        
        return sorted(paths, key=lambda p: p.score, reverse=True)
    
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