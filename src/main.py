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

from data.toy_graph import ToyGraphBuilder
from src.kg.path_traversal import BasicPathTraversal

def main():
    """Main demo function"""
    print("="*60)
    print("Multi-hop QA Toy Graph Demo")
    print("="*60)
    
    # 1. Create toy KG
    print("\n1. Creating toy knowledge graph")
    print("-" * 30)
    builder = ToyGraphBuilder()
    graph = builder.get_graph()
    builder.print_graph_info()
    
    # 2. Initialise path traversal
    print("\n2. Initialising path traversal")
    print("-" * 30)
    traversal = BasicPathTraversal(graph)
    print("BasicPathTraversal initialised")
    
    # 3. Test sample queries
    print("\n3. Running multi-hop queries")
    print("-" * 30)
    queries = builder.get_sample_queries()
    
    for key, query in queries.items():
        print(f"\n{key}: {query['description']}")
        source = query['source']
        target = query.get('target')
        max_hops = query.get('expected_hops', 3)

        if target:
            paths = traversal.find_paths(source, target, max_hops=max_hops, top_k=5)
            print(f" Found {len(paths)} paths:")
            for i, path in enumerate(paths):
                node_names = [node.name for node in path.nodes]
                relations = [edge.relation_type for edge in path.edges]
                print(f" Path {i + 1}: {' -> '.join(node_names)} (score: {path.score:.3f})")
                if relations:
                    print(f" Relations: {' -> '.join(relations)}")
        else:
            # Neighbourhood query
            paths = traversal.explore_neighbourhood(source, max_hops=max_hops, top_k=5)
            print(f" Found {len(paths)} neighbouring paths:")
            for i, path in enumerate(paths):
                node_names = [node.name for node in path.nodes]
                print(f" Path {i + 1}: {' -> '.join(node_names)} (score: {path.score:.3f})")
        print()

    # 4. Demonstrate flow-based pruning
    print("\n4. Demonstrating flow-based pruning")
    print("-" * 30)
    source, target = "albert_einstein", "photoelectric_effect"
    paths_before = traversal.dijkstra_paths(source, target, max_hops=3)
    paths_after = traversal.find_paths(source, target, max_hops=3, top_k=3)

    print(f" Before pruning: {len(paths_before)} paths")
    print(f" After pruning: {len(paths_after)} paths")
    for i, path in enumerate(paths_after):
        node_names = [node.name for node in path.nodes]
        flow = traversal.calculate_path_flow(path)
        print(f" Path {i+1}: {' -> '.join(node_names)} (score: {path.score:.3f}, flow: {flow:.3f})")
    print()

if __name__ == "__main__":
    main()