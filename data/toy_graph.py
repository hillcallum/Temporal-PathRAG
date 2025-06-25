import networkx as nx
from typing import Dict, Any

class ToyGraphBuilder:
    """
    Creates a simple knowledge graph for multi-hop QA testing
    
    Graph structure:
    - 6 entities: People, Locations, Title
    - 2 hops needed to answer some set sample questions
    """
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.create_toy_data()
    
    def create_toy_data(self):
        # Nodes: id, type, name, description
        nodes = [
            # People
            {'id': 'albert_einstein', 'entity_type': 'Person', 'name': 'Albert Einstein', 'description': 'Theoretical physicist known for the theory of relativity'},
            {'id': 'marie_curie', 'entity_type': 'Person', 'name': 'Marie Curie', 'description': 'Physicist and Chemist known for the discovery of Radium'},
            # Locations
            {'id': 'ulm', 'entity_type': 'Location', 'name': 'German City', 'description': 'City where Einstein was born'},
            {'id': 'warsaw', 'entity_type': 'Location', 'name': 'Warsaw', 'description': 'City where Marie Curie was born'},
            # Awards - specific instances
            {'id': 'nobel_prize_einstein', 'entity_type': 'Award', 'name': 'Nobel Prize in Physics (Einstein)', 'description': 'Award given to Einstein in 1921'},
            {'id': 'nobel_prize_curie', 'entity_type': 'Award', 'name': 'Nobel Prize in Physics (Curie)', 'description': 'Award given to Curie in 1903'},
            # Awards - general category
            {'id': 'nobel_prize', 'entity_type': 'Award', 'name': 'Nobel Prize', 'description': 'Prestigious international award'},
            # Concepts
            {'id': 'photoelectric_effect', 'entity_type': 'Concept', 'name': 'Photoelectric Effect', 'description': 'Phenomenon explained by Einstein, leading to Nobel Prize'}           
        ]
        
        # Add nodes to graph
        for node in nodes:
            self.graph.add_node(node['id'], **node)
        
        # Edges: source, target, relation_type, description, weight
        edges = [
            # Birth locations
            {'source': 'albert_einstein', 'target': 'ulm', 'relation_type': 'BORN_IN', 'description': 'Einstein was born in Ulm', 'weight': 1.0},
            {'source': 'marie_curie', 'target': 'warsaw', 'relation_type': 'BORN_IN', 'description': 'Marie Curie was born in Warsaw', 'weight': 1.0},
            # Specific Nobel Prize awards
            {'source': 'albert_einstein', 'target': 'nobel_prize_einstein', 'relation_type': 'AWARDED', 'description': 'Einstein received the Nobel Prize', 'weight': 1.0},
            {'source': 'marie_curie', 'target': 'nobel_prize_curie', 'relation_type': 'AWARDED', 'description': 'Curie received a Nobel Prize', 'weight': 1.0},
            # Connect specific prizes to general Nobel Prize category
            {'source': 'nobel_prize_einstein', 'target': 'nobel_prize', 'relation_type': 'INSTANCE_OF', 'description': 'Einstein Nobel Prize is an instance of Nobel Prize', 'weight': 1.0},
            {'source': 'nobel_prize_curie', 'target': 'nobel_prize', 'relation_type': 'INSTANCE_OF', 'description': 'Curie Nobel Prize is an instance of Nobel Prize', 'weight': 1.0},
            # Concept connection
            {'source': 'nobel_prize_einstein', 'target': 'photoelectric_effect', 'relation_type': 'AWARDED_FOR', 'description': 'Einstein received the prize for explaining the photoelectric effect', 'weight': 1.0}
        ]
        
        # Add edges to graph
        for edge in edges:
            self.graph.add_edge(
                edge['source'], 
                edge['target'],
                relation_type=edge['relation_type'],
                description=edge['description'],
                weight=edge['weight']
            )
    
    def get_graph(self) -> nx.DiGraph:
        """Return the toy graph"""
        return self.graph
    
    def print_graph_info(self):
        """Print information about the toy graph"""
        print("TOY GRAPH INFORMATION")
        print(f"Nodes: {self.graph.number_of_nodes()}")
        print(f"Edges: {self.graph.number_of_edges()}")
        print()
        
        print("NODES:")
        for node_id, data in self.graph.nodes(data=True):
            print(f" {node_id}: {data['name']} ({data['entity_type']})")
        print()
        
        print("EDGES:")
        for source, target, data in self.graph.edges(data=True):
            print(f" {source} --[{data['relation_type']}]--> {target}")
        print()

        # Debugging - checking for reverse edges
        print("Debug - All edges in both directions:")
        for node in self.graph.nodes():
            print(f"{node} successors: {list(self.graph.successors(node))}")
            print(f"{node} predecessors: {list(self.graph.predecessors(node))}")
        print()
    
    def get_sample_queries(self) -> Dict[str, Dict[str, Any]]:
        """Return sample queries for testing"""
        return {
            "query_1": {
                "description": "Where was Albert Einstein born?",
                "source": "albert_einstein",
                "target": "ulm",
                "expected_hops": 1
            },
            "query_2": {
                "description": "What concept earned Einstein the Nobel Prize?",
                "source": "albert_einstein",
                "target": "photoelectric_effect",
                "expected_hops": 2  # Einstein -> Nobel Prize -> Photoelectric Effect
            },
            "query_3": {
                "description": "What is a shared award between Einstein and Curie?",
                "source": "albert_einstein",
                "target": "marie_curie",
                "via": "nobel_prize",
                "expected_hops": 2
            }
        }

def create_toy_graph() -> nx.DiGraph:
    builder = ToyGraphBuilder()
    return builder.get_graph()