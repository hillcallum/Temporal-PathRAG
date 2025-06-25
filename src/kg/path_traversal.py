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
        """Construct a Path object from node IDs and edge data with textual chunks"""
        try:
            path = Path()
            
            # Add nodes with textual chunks (tv)
            for node_id in node_ids:
                if self.graph.has_node(node_id):
                    node_data = self.graph.nodes[node_id]
                    pathrag_node = PathRAGNode(
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
                    pathrag_edge = PathRAGEdge(
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
        Find bidirectional paths that connect two nodes through shared intermediate nodes.
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
        Calculate quality score for a bidirectional connection.
        
        Higher scores indicate better connections (shorter paths, higher individual scores).
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
        Generate human-readable text describing the bidirectional connection.
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
        Enhanced version of find_shared_connections using bidirectional search.
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