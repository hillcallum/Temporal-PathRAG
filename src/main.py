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
from src.utils.device import setup_device_and_logging, optimise_for_pathrag

def main():
    """Main demo function"""
    print("="*60)
    print("Temporal PathRAG Demo using GPU capabilities")
    print("="*60)
    
    # 0. Setup GPU device and optimisation
    print("\n0. Setting up GPU acceleration")
    print("-" * 30)
    device = setup_device_and_logging()
    device = optimise_for_pathrag()
    
    # 1. Create expanded toy KG
    print("\n1. Creating expanded toy knowledge graph")
    print("-" * 30)
    builder = ExpandedToyGraphBuilder()
    graph = builder.get_graph()
    print(f"Graph loaded: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    
    # 2. Initialise GPU-accelerated path traversal
    print("\n2. Initialising GPU-accelerated path traversal")
    print("-" * 30)
    traversal = BasicPathTraversal(graph, device=device)
    print(f"BasicPathTraversal initialised with device: {device}")
    
    # Show initial GPU memory usage
    gpu_info = traversal.get_gpu_memory_usage()
    if 'allocated_gb' in gpu_info:
        print(f"GPU Memory: {gpu_info['allocated_gb']:.2f}GB allocated, {gpu_info['utilisation_percent']:.1f}% utilisation")
    
    # 3. Test sample queries from expanded toy graph
    print("\n3. Running enhanced PathRAG queries with new features")
    print("-" * 30)
    
    # Get all sample queries from the expanded toy graph
    queries = builder.get_sample_queries()
    
    # Test queries with new enhanced features
    enhanced_queries = {
        "query_1": {**queries["query_1"], "test_features": ["basic", "error_handling"]},  # Easy
        "query_4": {**queries["query_4"], "test_features": ["bidirectional", "multi_path_aggregation"]},  # Bidirectional
        "query_5": {**queries["query_5"], "test_features": ["via_nodes", "semantic_scoring"]},  # Via nodes
        "query_11": {**queries["query_11"], "test_features": ["bidirectional", "temporal_scoring"]}, # Temporal
        "missing_node_test": {
            "description": "Test error handling with missing node",
            "source": "nonexistent_person", 
            "target": "albert_einstein",
            "expected_hops": 2,
            "difficulty": "error_test",
            "test_features": ["error_handling", "graceful_degradation"]
        }
    }
    
    for key, query in enhanced_queries.items():
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
        
        # Demonstrate new features (26th June) for this query
        if 'test_features' in query:
            print(f"Testing features: {', '.join(query['test_features'])}")
            
            # Test via nodes if specified
            if 'via_nodes' in query['test_features'] and 'via' in query:
                print(f"Via-node search through: {query['via']}")
                via_paths = traversal.find_paths(source, target, max_hops=max_hops, top_k=3, via_nodes=[query['via']])
                print(f"Found {len(via_paths)} via-node paths")
                
            # Test multi-path aggregation
            if 'multi_path_aggregation' in query['test_features'] and target:
                all_paths = traversal.find_paths(source, target, max_hops=max_hops, top_k=5)
                if all_paths:
                    aggregation = traversal.aggregate_multiple_paths(all_paths, query.get('description'))
                    print(f"Multi-path aggregation: {aggregation['summary']}")
                    print(f"Confidence: {aggregation['confidence']:.3f}, Evidence: {aggregation['evidence_strength']}")
        
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
    
    # 6. Demonstrate new scoring features
    print("\\n6. Enhanced scoring demonstration")
    print("-" * 30)
    
    # Test semantic and temporal scoring
    einstein_paths = traversal.find_paths("albert_einstein", "quantum_mechanics", max_hops=3, top_k=5)
    if einstein_paths:
        print("Enhanced scoring for Einstein -> Quantum Mechanics paths:")
        for i, path in enumerate(einstein_paths[:3]):
            node_names = [node.name for node in path.nodes]
            print(f"{i+1}. {' -> '.join(node_names)}")
            
            # Show scoring breakdown
            semantic_score = traversal.calculate_semantic_similarity(path)
            temporal_score = traversal.calculate_temporal_coherence(path)
            print(f"   Semantic score: {semantic_score:.3f}, Temporal score: {temporal_score:.3f}")
            print(f"   Combined score: {path.score:.3f}")
    
    # 7. Error handling and graceful degradation demo
    print("\\n7. Error handling demonstration")
    print("-" * 30)
    
    # Test with missing node
    print("Testing with missing source node:")
    missing_paths = traversal.find_paths("missing_person", "albert_einstein", max_hops=3, top_k=3)
    print(f"Paths found: {len(missing_paths)}")
    
    # Test validation
    validation = traversal.validate_nodes(["albert_einstein", "missing_person", "marie_curie"])
    print(f"Node validation: {validation['missing_count']} missing out of 3 nodes")
    print(f"Missing nodes: {validation['missing_nodes']}")
    
    print()
    
    # GPU cleanup and final memory report
    print("\n8. GPU Memory Summary")
    print("-" * 30)
    final_gpu_info = traversal.get_gpu_memory_usage()
    if 'allocated_gb' in final_gpu_info:
        print(f"Final GPU Memory: {final_gpu_info['allocated_gb']:.2f}GB allocated, {final_gpu_info['utilisation_percent']:.1f}% utilisation")
        print("Cleaning up GPU memory...")
        traversal.cleanup_gpu_memory()

if __name__ == "__main__":
    main()