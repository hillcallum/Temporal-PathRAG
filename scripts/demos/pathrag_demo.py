#!/usr/bin/env python3
"""
PathRAG Demo with LLM Integration
"""

import sys
import os
import logging
from typing import List

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.llm import llm_manager
from src.kg.path_traversal import BasicPathTraversal
from datasets.toy.expanded_toy_graph import ExpandedToyGraphBuilder

# Configure logging
logging.basicConfig(level=logging.WARNING)  # Reduce noise

class PathRAGDemo:
    """PathRAG demonstration that it works with LLM integration"""
    
    def __init__(self):
        """Initialise PathRAG demo"""
        # Create expanded knowledge graph
        self.builder = ExpandedToyGraphBuilder()
        self.graph = self.builder.get_graph()
        self.traversal = BasicPathTraversal(self.graph)
        
        print("PathRAG Demo Initialised")
        print(f" Knowledge Graph: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
        print(f" LLM Status: {llm_manager.get_status()['active_client'] or 'No LLM available'}")
        print()
    
    def answer_question(self, question: str, source_entity: str = None, target_entity: str = None, query_type: str = "unidirectional", via_nodes: List[str] = None) -> dict:
        """
        Answer a question using PathRAG pipeline
        """
        print(f"Question: {question}")
        
        # Step 1: Find relevant paths based on query type
        paths = []
        bidirectional_results = []
        
        if query_type == "bidirectional" and source_entity and target_entity:
            # Use bidirectional search for shared connection queries
            print(f"Using bidirectional search: {source_entity} ↔ {target_entity}")
            bidirectional_results = self.traversal.find_bidirectional_paths(source_entity, target_entity, max_hops=4, top_k=5)
            # Convert bidirectional results to paths for LLM processing
            for conn in bidirectional_results:
                paths.extend([conn['source_path'], conn['target_path']])
        elif source_entity and target_entity:
            # Direct path finding with optional via nodes
            paths = self.traversal.find_paths(source_entity, target_entity, max_hops=3, top_k=5, via_nodes=via_nodes)
        elif source_entity:
            # Explore from source
            paths = self.traversal.explore_neighbourhood(source_entity, max_hops=2, top_k=5)
        else:
            # For demo, use Einstein as default
            paths = self.traversal.explore_neighbourhood("albert_einstein", max_hops=2, top_k=5)
        
        print(f"Found {len(paths)} paths in knowledge graph")
        
        # Step 2: Display paths with PathRAG textual chunks
        if bidirectional_results:
            print("Bidirectional Connections Found:")
            for i, conn in enumerate(bidirectional_results[:3], 1):
                print(f"{i}. {conn['connection_text']}")
                print(f"Via: {conn['shared_node_data'].get('name', conn['shared_node'])} (score: {conn['connection_score']:.3f})")
                print(f"Paths: {conn['source_hops']} + {conn['target_hops']} hops")
                print()
        
        if paths:
            print("Knowledge Graph Paths (with PathRAG textual chunks):")
            for i, path in enumerate(paths[:3]):  # Show top 3
                node_names = [node.name for node in path.nodes]
                relations = [edge.relation_type for edge in path.edges] if path.edges else []
                
                if relations:
                    path_str = node_names[0]
                    for j, rel in enumerate(relations):
                        if j + 1 < len(node_names):
                            path_str += f" --{rel}--> {node_names[j + 1]}"
                else:
                    path_str = " -> ".join(node_names)
                
                print(f" {i+1}. {path_str} (score: {path.score:.3f})")
                # Show PathRAG textual representation
                if hasattr(path, 'path_text') and path.path_text:
                    print(f" PathRAG text: {path.path_text[:150]}")
                print()
        
        # Step 3: Generate LLM answer
        llm_answer = None
        try:
            llm_answer = llm_manager.answer_question_with_paths(question, paths, fallback=True)
            print(f"LLM Answer: {llm_answer}")
        except Exception as e:
            print(f"LLM unavailable: {str(e)[:50]}")
        
        # Step 4: Enhanced analysis with new features (26th June)
        if paths and len(paths) > 1:
            # Demonstrate multi-path aggregation
            aggregation = self.traversal.aggregate_multiple_paths(paths, question)
            print(f"Multi-path Analysis:")
            print(f"Confidence: {aggregation['confidence']:.3f}")
            print(f"Evidence Strength: {aggregation['evidence_strength']}")
            print(f"Key entities: {[e['name'] for e in aggregation['key_entities'][:3]]}")
        
        # Step 5: Create structured answer
        structured_answer = self.create_structured_answer(question, paths, llm_answer)
        print(f"Structured Answer: {structured_answer['answer']}")
        print()
        
        return structured_answer
    
    def create_structured_answer(self, question: str, paths: List, llm_answer: str = None) -> dict:
        """Create structured answer from paths and LLM response"""
        
        if llm_answer:
            # Use LLM answer if available
            answer = llm_answer
            answer_type = "llm_generated"
        elif paths:
            # Create rule-based answer from paths
            answer = self.generate_rule_based_answer(question, paths)
            answer_type = "rule_based"
        else:
            # No information found
            answer = "I couldn't find relevant information to answer this question."
            answer_type = "no_info"
        
        return {
            "question": question,
            "answer": answer,
            "answer_type": answer_type,
            "num_paths": len(paths),
            "paths": [self.serialise_path(path) for path in paths[:3]],
            "llm_available": llm_answer is not None
        }
    
    def generate_rule_based_answer(self, question: str, paths: List) -> str:
        """Generate rule-based answer when LLM is not available"""
        
        if not paths:
            return "No relevant information found."
        
        # Extract key information from paths
        entities = set()
        relationships = set()
        
        for path in paths:
            for node in path.nodes:
                entities.add(node.name)
            for edge in path.edges:
                relationships.add(edge.relation_type)
        
        # Create simple rule-based responses
        if "born" in question.lower() or "birth" in question.lower():
            for path in paths:
                if any("BORN_IN" in edge.relation_type for edge in path.edges):
                    source = path.nodes[0].name
                    target = path.nodes[-1].name
                    return f"Based on the knowledge graph, {source} was born in {target}."
        
        elif "award" in question.lower() or "prize" in question.lower():
            for path in paths:
                if any("AWARDED" in edge.relation_type for edge in path.edges):
                    source = path.nodes[0].name
                    if "Nobel" in path.nodes[-1].name:
                        return f"{source} received a Nobel Prize."
        
        # Generic answer
        top_path = paths[0]
        node_names = [node.name for node in top_path.nodes]
        return f"Based on the knowledge graph: {' is connected to '.join(node_names)}."
    
    def serialise_path(self, path) -> dict:
        """Serialise path for JSON output"""
        return {
            "nodes": [{"id": node.id, "name": node.name, "type": node.entity_type} for node in path.nodes],
            "edges": [{"relation": edge.relation_type, "description": edge.description} for edge in path.edges],
            "score": path.score
        }
    
    def run_demo_questions(self):
        """Run a set of demo questions"""
        print("PathRAG Demo - Multi-hop Question Answering")
        print("=" * 60)
        print()
        
        # Demo questions using sample queries from expanded toy graph
        sample_queries = self.builder.get_sample_queries()
        
        questions = [
            {
                "question": sample_queries["query_1"]["description"],
                "source": sample_queries["query_1"]["source"],
                "target": sample_queries["query_1"]["target"],
                "query_type": "unidirectional"
            },
            {
                "question": sample_queries["query_4"]["description"], 
                "source": sample_queries["query_4"]["source"],
                "target": sample_queries["query_4"]["target"],
                "query_type": "bidirectional"
            },
            {
                "question": sample_queries["query_5"]["description"] + " (via Princeton University)",
                "source": sample_queries["query_5"]["source"],
                "target": sample_queries["query_5"]["target"],
                "via_nodes": [sample_queries["query_5"]["via"]],
                "query_type": "via_nodes"
            },
            {
                "question": "Test error handling: What connects a missing person to Einstein?",
                "source": "nonexistent_scientist",
                "target": "albert_einstein",
                "query_type": "error_test"
            }
        ]
        
        results = []
        for q in questions:
            result = self.answer_question(
                q["question"], 
                source_entity=q["source"], 
                target_entity=q.get("target"),
                query_type=q["query_type"],
                via_nodes=q.get("via_nodes")
            )
            results.append(result)
        
        return results

def main():
    """Main demo function"""
    demo = PathRAGDemo()
    demo.run_demo_questions()

if __name__ == "__main__":
    main()