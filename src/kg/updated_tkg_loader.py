"""
Updated TKG loader that adds textual representations to graphs
"""

import networkx as nx
from typing import Dict, Any, Optional
import logging
from .entity_resolution import enhance_graph_with_textual_representations, EntityResolver
from src.utils.dataset_loader import load_dataset as original_load_dataset

logger = logging.getLogger(__name__)


def load_enhanced_dataset(dataset_name: str, 
                         config_override: Optional[Dict[str, Any]] = None, 
                         use_cache: bool = True) -> nx.DiGraph:
    """
    Load a dataset and enhance it with textual representations
    """
    # Load the base graph
    graph = original_load_dataset(dataset_name, config_override, use_cache)
    
    # Check if already enhanced
    sample_nodes = list(graph.nodes)[:10]
    already_enhanced = all('tv' in graph.nodes[n] for n in sample_nodes if n in graph.nodes)
    
    if already_enhanced:
        logger.info(f"Graph already has textual representations")
        return graph
    
    # Enhance the graph
    logger.info(f"Enhancing {dataset_name} graph with textual representations")
    enhanced_graph = enhance_graph_with_textual_representations(graph)
    
    # Log some examples
    logger.info("Sample enhanced nodes:")
    for node in sample_nodes[:3]:
        if node in enhanced_graph.nodes:
            tv = enhanced_graph.nodes[node].get('tv', 'No tv')
            logger.info(f"  {node} -> {tv}")
    
    return enhanced_graph


def create_enhanced_entity_resolver(graph: nx.DiGraph) -> EntityResolver:
    """
    Create an entity resolver for the graph
    """
    return EntityResolver(graph)


def test_entity_resolution(graph: nx.DiGraph, test_mentions: list) -> None:
    """
    Test entity resolution on sample mentions
    """
    resolver = EntityResolver(graph)
    
    print("\nEntity Resolution Tests:")
    
    for mention in test_mentions:
        resolved = resolver.resolve(mention)
        if resolved:
            info = resolver.get_entity_info(resolved)
            print(f"'{mention}' -> '{resolved}'")
            print(f"Variations: {info['variations']}")
            print(f"Connections: {info['total_connections']}")
        else:
            print(f"'{mention}' -> Not Found")
        print()


def demonstrate_enhancements():
    """Demonstrate the enhancements on a small sample"""
    
    # Create a small test graph
    test_graph = nx.DiGraph()
    
    # Add some nodes
    test_graph.add_node("Cabinet_/Council_of_Ministers/Advisors(Denmark)", 
                       node_type="entity")
    test_graph.add_node("Danish_Ministry", 
                       node_type="entity")
    test_graph.add_node("Denmark", 
                       node_type="entity")
    test_graph.add_node("Royal_Castle(Copenhagen)", 
                       node_type="entity")
    
    # Add some edges
    test_graph.add_edge("Cabinet_/Council_of_Ministers/Advisors(Denmark)", 
                       "Denmark",
                       relation="located_in",
                       timestamp="1990-01-01")
    
    print("Original graph nodes:")
    for node in test_graph.nodes:
        print(f"{node}: {dict(test_graph.nodes[node])}")
    
    # Enhance the graph
    enhanced = enhance_graph_with_textual_representations(test_graph)
    
    print("\nEnhanced graph nodes:")
    for node in enhanced.nodes:
        data = dict(enhanced.nodes[node])
        print(f"{node}:")
        print(f"tv: {data.get('tv', 'Missing')}")
    
    print("\nEnhanced edges:")
    for u, v, data in enhanced.edges(data=True):
        print(f"{u} -> {v}:")
        print(f"relation: {data.get('relation')}")
        print(f"te: {data.get('te', 'Missing')}")
    
    # Test entity resolution
    test_mentions = [
        "Danish Ministry",
        "Cabinet of Denmark",
        "Council of Ministers",
        "Denmark",
        "Royal Castle",
        "Copenhagen Castle"
    ]
    
    test_entity_resolution(enhanced, test_mentions)


if __name__ == "__main__":
    # Run demonstration
    demonstrate_enhancements()