"""
Basic Multi-hop QA Graph Demo

Demonstrates the core PathRAG path traversal on a toy knowledge graph
inspired by multi-hop question answering tasks outlined in HotpotQA

"""

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from data.expanded_toy_graph import ExpandedToyGraphBuilder
from src.kg.path_traversal import BasicPathTraversal

def main():
    """Main demo function"""
    print("="*60)
    print("Multi-hop QA Toy Graph Demo")
    print("="*60)
    
    # 1. Create expanded toy KG
    print("\n1. Creating expanded toy knowledge graph")
    print("-" * 30)
    builder = ExpandedToyGraphBuilder()
    graph = builder.get_graph()
    print(f"Graph loaded: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    
    # 2. Initialise path traversal
    print("\n2. Initialising path traversal")
    print("-" * 30)
    traversal = BasicPathTraversal(graph)
    print("BasicPathTraversal initialised")
    
    # 3. Test sample queries from expanded toy graph
    print("\n3. Running challenging queries with PathRAG textual chunks and bidirectional search")
    print("-" * 30)
    
    # Get all sample queries from the expanded toy graph
    queries = builder.get_sample_queries()
    
    # Filter to show a mix of difficulties
    selected_queries = {
        "query_1": queries["query_1"],  # Easy
        "query_4": queries["query_4"],  # Bidirectional
        "query_5": queries["query_5"],  # Medium  
        "query_11": queries["query_11"], # Bidirectional
        "query_13": queries["query_13"]  # Hard
    }
    
    for key, query in selected_queries.items():
        print(f"\n{key.upper()}: {query['description']}")
        print(f"Difficulty: {query['difficulty']}, Expected hops: {query['expected_hops']}")
        
        source = query['source']
        target = query.get('target')
        max_hops = query.get('expected_hops', 3) + 1
        query_type = query.get('query_type', 'unidirectional')

        if query_type == 'bidirectional':
            # Use bidirectional search for shared connection queries
            print(f"Bidirectional search: {source} ↔ {target}")
            
            bidirectional_results = traversal.find_bidirectional_paths(source, target, max_hops=max_hops*2, top_k=3)
            
            if bidirectional_results:
                print(f"Found {len(bidirectional_results)} shared connections:")
                for i, conn in enumerate(bidirectional_results, 1):
                    shared_node = conn['shared_node']
                    shared_name = conn['shared_node_data'].get('name', shared_node)
                    print(f" {i}. {conn['connection_text']}")
                    print(f" Via: {shared_name} (score: {conn['connection_score']:.3f})")
                    print(f" Paths: {conn['source_hops']} + {conn['target_hops']} hops")
            else:
                print(f"No shared connections found")
                
        elif target:
            # Regular unidirectional path finding
            print(f"→ Path search: {source} → {target}")
            
            paths = traversal.find_paths(source, target, max_hops=max_hops, top_k=3)
            
            if paths:
                print(f"Found {len(paths)} paths:")
                for i, path in enumerate(paths, 1):
                    node_names = [node.name for node in path.nodes]
                    relations = [edge.relation_type for edge in path.edges]
                    print(f" {i}. {' -> '.join(node_names)} (score: {path.score:.3f})")
                    if relations:
                        print(f"Relations: {' -> '.join(relations)}")
                    # Show PathRAG textual chunks (truncated)
                    print(f"PathRAG text: {path.path_text[:120]}")
            else:
                print(f"No paths found")
        else:
            # Neighbourhood exploration
            print(f"Neighbourhood exploration: {source}")
            paths = traversal.explore_neighbourhood(source, max_hops=max_hops, top_k=5)
            print(f"Found {len(paths)} neighbouring paths:")
            for i, path in enumerate(paths):
                node_names = [node.name for node in path.nodes]
                print(f"{i + 1}. {' -> '.join(node_names)} (score: {path.score:.3f})")
        print()

    # 4. Demonstrate flow-based pruning
    print("\n4. Demonstrating flow-based pruning")
    print("-" * 30)
    source, target = "albert_einstein", "photoelectric_effect"
    paths_before = traversal.dijkstra_paths(source, target, max_hops=3)
    paths_after = traversal.find_paths(source, target, max_hops=3, top_k=3)

    print(f"Before pruning: {len(paths_before)} paths")
    print(f"After pruning: {len(paths_after)} paths")
    for i, path in enumerate(paths_after):
        node_names = [node.name for node in path.nodes]
        flow = traversal.calculate_path_flow(path)
        print(f"Path {i+1}: {' -> '.join(node_names)} (score: {path.score:.3f}, flow: {flow:.3f})")
    print()

    # 5. Demonstrate ability to answer Q3
    print("\n5. Attempt at fixing Q3")
    print("-" * 30)
    
    # Find shared connections between Einstein and Curie
    print("\n5a. Finding shared connections between Einstein and Curie:")
    shared_info = traversal.find_shared_connections("albert_einstein", "marie_curie", max_hops=2)
    
    if shared_info['shared_nodes']:
        print("Shared connections:")
        for node in shared_info['shared_nodes']:
            print(f" - {node['name']} ({node['entity_type']})")
            
        # Show paths to the first shared connection (Nobel Prize)
        first_shared = shared_info['shared_nodes'][0]
        shared_id = first_shared['id']
        print(f"\n Paths to {first_shared['name']}:")
        
        einstein_paths = traversal.find_paths("albert_einstein", shared_id, max_hops=1, top_k=3)
        curie_paths = traversal.find_paths("marie_curie", shared_id, max_hops=1, top_k=3)
        
        print("From Einstein:")
        for path in einstein_paths:
            node_names = [node.name for node in path.nodes]
            relations = [edge.relation_type for edge in path.edges]
            print(f" {' -> '.join(node_names)} ({' -> '.join(relations)})")
            
        print("From Curie:")
        for path in curie_paths:
            node_names = [node.name for node in path.nodes]
            relations = [edge.relation_type for edge in path.edges]
            print(f" {' -> '.join(node_names)} ({' -> '.join(relations)})")
    else:
        print("No shared connections found within 2 hops.")
    
    # Find connecting paths
    print("\n5b. Finding connecting paths between Einstein and Curie:")
    connecting_paths = traversal.find_connecting_paths("albert_einstein", "marie_curie", max_hops=4)
    
    if connecting_paths:
        print("Connection paths:")
        for i, path in enumerate(connecting_paths):
            if 'connection_info' in path.metadata:
                conn_info = path.metadata['connection_info']
                shared_name = None
                if graph.has_node(conn_info['shared_connection']):
                    shared_name = graph.nodes[conn_info['shared_connection']]['name']
                print(f"Path {i+1}: Connected through {shared_name} ({conn_info['connection_type']})")
            else:
                node_names = [node.name for node in path.nodes]
                print(f"Path {i+1}: {' -> '.join(node_names)}")
    else:
        print("No connecting paths found")
    
    print()

if __name__ == "__main__":
    main()